// static/app.js
let gpus = [];
let deployments = [];
let savedConfigs = [];
let endpoints = {};
let gpuGroupCollapsed = new Set();
const epImages = new Map();       // worker_id -> images[] cache
const pullInProgress = new Set(); // worker_ids currently pulling an image
const epModels = new Map();       // worker_id -> models[] cache
const modelDownloadInProgress = new Set();
const _endpointActions = new Map();
const _stopping = new Set(); // keys: dep id or "depId:gpu" for replicas
let versionInfo = null;      // last GET /api/version payload (worker versions panel)
let frontendInfo = null;     // last GET /api/frontend payload (Dynamo frontend)
let hostGroupCollapsed = new Set(); // collapsed host sections on the dashboard

// Default images per engine. Dynamo instances are served through the frontend;
// `vllm` is the legacy path (its own HTTPS endpoint, not behind the frontend).
const ENGINE_IMAGE = {
    dynamo: 'nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.2',
    vllm: 'vllm/vllm-openai:latest',
};
// Fixed KV block size — the worker and the frontend must agree on it
// (`--kv-cache-block-size`), so it is shown read-only in the deploy form.
const KV_BLOCK_SIZE = 64;

document.addEventListener('DOMContentLoaded', () => {
    // Self-rescheduling poll (re-armed in finally) so a slow response can't
    // overlap the next refresh.
    (async function pollStatus() {
        try {
            await fetchStatus();
        } finally {
            setTimeout(pollStatus, 5000);
        }
    })();

    // Worker versions panel (endpoints page only): separate, slower poll.
    // Same self-rescheduling pattern as pollStatus so a slow response can't
    // overlap the next refresh.
    const versionsContainer = document.getElementById('worker-versions-container');
    if (versionsContainer) {
        (async function pollVersion() {
            try {
                await fetchVersionInfo();
            } finally {
                setTimeout(pollVersion, 15000);
            }
        })();
        // Delegated listener on the container so buttons keep working across
        // the periodic re-render (no inline onclick with interpolated ids).
        versionsContainer.addEventListener('click', (e) => {
            const rowBtn = e.target.closest('button[data-worker-id]');
            if (rowBtn) {
                rowBtn.disabled = true;
                updateWorker(rowBtn.dataset.workerId);
                return;
            }
            const allBtn = e.target.closest('#update-all-drifted-btn');
            if (allBtn) {
                allBtn.disabled = true;
                updateAllWorkers();
            }
        });
    }

    // Stop the view from jumping when a GPU card is clicked: the hidden Bootstrap
    // .btn-check input gets focus on label click and the browser scrolls that
    // clipped element into view (yanking the deploy page to the top, or the modal
    // grid). Snapshot scroll on pointerdown, restore it right after focus moves.
    // Listeners are on document so they survive the periodic grid rebuild.
    let _gpuScrollSnap = null;
    document.addEventListener('pointerdown', (e) => {
        const grid = e.target.closest && e.target.closest('#deployGpusGrid');
        if (grid) _gpuScrollSnap = { win: window.scrollY, grid, top: grid.scrollTop };
    }, true);
    document.addEventListener('focusin', (e) => {
        if (_gpuScrollSnap && e.target.classList && e.target.classList.contains('gpu-checkbox')) {
            const s = _gpuScrollSnap;
            _gpuScrollSnap = null;
            requestAnimationFrame(() => {
                window.scrollTo(0, s.win);
                if (s.grid) s.grid.scrollTop = s.top;
            });
        }
    });

    if (new URLSearchParams(window.location.search).get('deployed') === '1') {
        history.replaceState({}, '', '/');
        setTimeout(() => showAlert('success', '배포가 성공적으로 시작되었습니다.'), 300);
    }

    // Populate image datalist when the deploy modal opens
    document.getElementById('deployModal')?.addEventListener('show.bs.modal', () => {
        fetchAllWorkerImages();
    });

    // Deploy form: apply the engine-dependent hints once on load, and fill the
    // read-only KV block size from the single source of truth above.
    if (document.getElementById('deployEngine') && typeof window.handleEngineChange === 'function') {
        document.querySelectorAll('.kv-block-size').forEach(el => { el.value = KV_BLOCK_SIZE; });
        window.handleEngineChange();
    }

    // Gateway test panel API key: prefill from localStorage, persist on change
    const gwApiKeyEl = document.getElementById('gw-api-key');
    if (gwApiKeyEl) {
        gwApiKeyEl.value = localStorage.getItem('gw_api_key') || '';
        gwApiKeyEl.addEventListener('change', () => {
            localStorage.setItem('gw_api_key', gwApiKeyEl.value);
        });
    }
});

async function fetchStatus() {
    try {
        const [gpusRes, depsRes, confsRes, endpRes, frontRes] = await Promise.all([
            fetch('/api/gpus'),
            fetch('/api/deployments'),
            fetch('/api/configs'),
            fetch('/api/endpoints'),
            fetch('/api/frontend').catch(() => null)
        ]);
        gpus = await gpusRes.json();
        deployments = await depsRes.json();
        savedConfigs = await confsRes.json();
        endpoints = await endpRes.json();
        // The frontend summary is optional: on a transient failure we keep the
        // previous payload and the UI degrades to "—" rather than blanking out.
        if (frontRes && frontRes.ok) {
            frontendInfo = await frontRes.json().catch(() => frontendInfo);
        }

        if (document.getElementById('frontend-status')) renderFrontendStatus();
        if (document.getElementById('gpu-cards-container') || document.getElementById('deployGpusGrid')) renderGPUs();
        if (document.getElementById('deployments-container')) renderDeployments();
        if (document.getElementById('saved-configs-container')) renderConfigs();
        if (document.getElementById('pending-endpoints-container')) renderEndpoints();
        if (document.getElementById('gw-models-list')) renderGateway();
    } catch (err) {
        console.error("Failed to fetch status", err);
    }
}

function renderGPUs() {
    const list = document.getElementById('gpu-cards-container');
    const gpuGrid = document.getElementById('deployGpusGrid');
    const nodeSelector = document.getElementById('deployTargetNode');

    // Save currently checked GPUs to prevent polling from erasing selections
    let selectedGpuIds = new Set();
    // Preserve scroll so the 5s polling refresh doesn't yank the view to the top
    // (the modal grid scrolls internally; the full deploy page scrolls the window).
    const _prevGridScroll = gpuGrid ? gpuGrid.scrollTop : 0;
    const _prevWinScroll = window.scrollY;
    if (gpuGrid) {
        gpuGrid.querySelectorAll('input:checked').forEach(cb => selectedGpuIds.add(cb.value));
        gpuGrid.innerHTML = '';
    }

    if (list) list.innerHTML = '';

    // Setup Node Selector
    if (nodeSelector) {
        const prevNode = nodeSelector.value;
        nodeSelector.innerHTML = '<option value="">-- Select a Node --</option>';
        Object.values(endpoints).filter(ep => ep.status === 'active').forEach(ep => {
            nodeSelector.innerHTML += `<option value="${ep.id}">${ep.name} (${ep.gpus.length} GPUs)</option>`;
        });
        nodeSelector.value = prevNode;
    }

    if (gpus.length === 0) {
        if (list) list.innerHTML = '<p class="text-muted mb-0 p-2">No active GPUs available. Accept a pending host first.</p>';
        return;
    }

    if (list) {
        // Group by worker
        const workerGroups = new Map();
        gpus.forEach(gpu => {
            if (!workerGroups.has(gpu.worker_id)) workerGroups.set(gpu.worker_id, { name: gpu.worker_name, gpus: [] });
            workerGroups.get(gpu.worker_id).gpus.push(gpu);
        });
        const sortedWorkers = [...workerGroups.entries()].sort(([, a], [, b]) => a.name.localeCompare(b.name));

        sortedWorkers.forEach(([wid, group]) => {
            const gid = 'gpugrp_' + wid.replace(/[^a-zA-Z0-9]/g, '_');
            const isCollapsed = gpuGroupCollapsed.has(wid);
            const cardsDisplay = isCollapsed ? 'display:none' : '';
            const iconTransform = isCollapsed ? 'transform:rotate(-90deg)' : '';
            const avgMem = Math.round(group.gpus.reduce((s, g) => s + Math.round((g.memory_used / g.memory_total) * 100), 0) / group.gpus.length);
            const avgColor = avgMem > 85 ? '#ef4444' : avgMem > 60 ? '#f59e0b' : '#22c55e';

            const gpuRows = group.gpus.map(gpu => {
                const memPercent = Math.min(100, Math.round((gpu.memory_used / gpu.memory_total) * 100));
                const barColor = memPercent > 85 ? '#ef4444' : memPercent > 60 ? '#f59e0b' : '#3b82f6';
                return `
                    <div class="gpu-list-row">
                        <span class="fw-semibold small">GPU ${gpu.local_id}</span>
                        <span class="gpu-list-name" title="${escapeHtml(gpu.name)}">${escapeHtml(gpu.name)}</span>
                        <div class="gpu-list-membar" title="${(gpu.memory_used / 1024).toFixed(1)} GB / ${(gpu.memory_total / 1024).toFixed(0)} GB (${memPercent}%)">
                            <div style="width:${memPercent}%;background:${barColor}"></div>
                        </div>
                        <span class="gpu-list-mem"><span class="fw-semibold" style="color:${barColor}">${memPercent}%</span> · ${(gpu.memory_used / 1024).toFixed(1)} / ${(gpu.memory_total / 1024).toFixed(0)} GB</span>
                        <span class="gpu-list-util">${gpu.utilization != null ? `Compute ${gpu.utilization}%` : '&mdash;'}</span>
                    </div>`;
            }).join('');

            list.innerHTML += `
                <div class="card gpu-list-group">
                    <div class="gpu-list-head" data-wid="${escapeHtml(wid)}" onclick="toggleGpuGroup(this.dataset.wid)">
                        <i id="${gid}-icon" class="fa-solid fa-chevron-down" style="font-size:.72rem;color:#6b7280;transition:transform .2s;${iconTransform}"></i>
                        <span class="fw-semibold">${escapeHtml(group.name)}</span>
                        <span class="badge bg-secondary" style="font-size:.75rem">${group.gpus.length} GPU${group.gpus.length > 1 ? 's' : ''}</span>
                        <span style="font-size:.8rem;color:${avgColor};font-weight:600">${avgMem}% avg mem</span>
                    </div>
                    <div class="gpu-list-body" id="${gid}-cards" style="${cardsDisplay}">
                        ${gpuRows}
                    </div>
                </div>`;
        });
    }

    // Deploy Page GPU Checkboxes — grouped by worker (endpoint) with a full-width
    // header per node so endpoints are visually separated and the node name is shown
    // in full (instead of being truncated on every card).
    if (gpuGrid) {
        const deployGroups = new Map();
        gpus.forEach(gpu => {
            if (!deployGroups.has(gpu.worker_id)) deployGroups.set(gpu.worker_id, { name: gpu.worker_name, gpus: [] });
            deployGroups.get(gpu.worker_id).gpus.push(gpu);
        });
        const deploySorted = [...deployGroups.entries()].sort(([, a], [, b]) => a.name.localeCompare(b.name));

        deploySorted.forEach(([wid, grp]) => {
            // Full-width node separator header (forces a new row in the flex-wrap grid)
            gpuGrid.innerHTML += `
                <div style="flex:0 0 100%;width:100%;display:flex;align-items:center;gap:8px;margin-top:6px;padding:0 2px">
                    <span class="fw-semibold" style="font-size:.8rem;color:#374151;white-space:nowrap" title="${escapeHtml(grp.name)}">${escapeHtml(grp.name)}</span>
                    <span class="badge bg-secondary" style="font-size:.66rem">${grp.gpus.length} GPU${grp.gpus.length > 1 ? 's' : ''}</span>
                    <div style="flex:1;height:1px;background:#d1d5db"></div>
                </div>`;

            grp.gpus.forEach(gpu => {
                const memPercent = Math.min(100, Math.round((gpu.memory_used / gpu.memory_total) * 100));
                const isChecked = selectedGpuIds.has(gpu.id) ? 'checked' : '';
                const barColor = memPercent > 85 ? '#ef4444' : memPercent > 60 ? '#f59e0b' : '#3b82f6';

                // Find running deployments on this GPU
                const runningOnGpu = deployments.filter(d =>
                    d.status === 'running' && d.gpus && d.gpus.includes(gpu.id)
                );
                const runningHtml = runningOnGpu.length > 0
                    ? runningOnGpu.map(d => `<div style="font-size:.72rem;color:#dc2626;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-top:2px" title="${d.name}: ${d.served_model_name || d.model}">▶ ${d.name}</div>`).join('')
                    : '';

                gpuGrid.innerHTML += `
                    <input type="checkbox" class="btn-check gpu-checkbox" id="gpu-btn-${gpu.id}" value="${gpu.id}" data-node="${gpu.worker_id}" autocomplete="off" onchange="validateDeployGpus()" ${isChecked}>
                    <label for="gpu-btn-${gpu.id}" class="gpu-checkbox-label" title="${escapeHtml(grp.name)} · GPU ${gpu.local_id}">
                        <div style="font-size:.82rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">GPU ${gpu.local_id}</div>
                        <div style="height:3px;border-radius:2px;background:#e5e7eb;margin:4px 0 2px">
                            <div style="width:${memPercent}%;height:100%;border-radius:2px;background:${barColor}"></div>
                        </div>
                        <div style="font-size:.78rem;color:#9ca3af">${memPercent}%</div>
                        ${runningHtml}
                    </label>
                `;
            });
        });
    }

    // Run filter depending on current mode
    toggleDeployModeUI();

    // Restore scroll positions captured before the rebuild, so selecting a GPU
    // (or the periodic refresh) no longer jumps the view back to the top.
    if (gpuGrid) gpuGrid.scrollTop = _prevGridScroll;
    if (gpuGrid && _prevWinScroll) window.scrollTo(0, _prevWinScroll);
}

window.toggleGpuGroup = function(wid) {
    const gid = 'gpugrp_' + wid.replace(/[^a-zA-Z0-9]/g, '_');
    const cardsEl = document.getElementById(`${gid}-cards`);
    const iconEl = document.getElementById(`${gid}-icon`);
    if (gpuGroupCollapsed.has(wid)) {
        gpuGroupCollapsed.delete(wid);
        if (cardsEl) cardsEl.style.display = '';
        if (iconEl) iconEl.style.transform = '';
    } else {
        gpuGroupCollapsed.add(wid);
        if (cardsEl) cardsEl.style.display = 'none';
        if (iconEl) iconEl.style.transform = 'rotate(-90deg)';
    }
};

window.toggleDeployModeUI = function () {
    const isTp = document.getElementById('typeTp').checked;
    const nodeSelector = document.getElementById('tpNodeSelector');
    const helpText = document.getElementById('deployGpusHelp');
    const tpContainer = document.getElementById('tpDisplayContainer');
    const checkboxes = document.querySelectorAll('.gpu-checkbox');

    // Highlight selected serving type card
    const labelReplicas = document.getElementById('label-replicas');
    const labelTp = document.getElementById('label-tp');
    if (labelReplicas && labelTp) {
        labelReplicas.style.borderColor = isTp ? '#e5e7eb' : '#3b82f6';
        labelReplicas.style.background  = isTp ? '' : '#eff6ff';
        labelTp.style.borderColor       = isTp ? '#3b82f6' : '#e5e7eb';
        labelTp.style.background        = isTp ? '#eff6ff' : '';
    }

    if (isTp) {
        nodeSelector.style.display = 'block';
        tpContainer.style.display = 'block';
        helpText.textContent = "1, 2, 4, 8개 GPU를 선택하세요. TP 수가 자동으로 계산됩니다.";
        filterGpusByNode();
    } else {
        nodeSelector.style.display = 'none';
        tpContainer.style.display = 'none';
        helpText.textContent = "GPU를 선택하세요. 각 GPU에 독립적인 레플리카가 실행됩니다.";
        checkboxes.forEach(cb => cb.nextElementSibling.style.display = 'inline-block');
    }
    validateDeployGpus();
}

window.filterGpusByNode = function () {
    const selectedNode = document.getElementById('deployTargetNode').value;
    const checkboxes = document.querySelectorAll('.gpu-checkbox');

    checkboxes.forEach(cb => {
        if (!selectedNode || cb.getAttribute('data-node') === selectedNode) {
            cb.nextElementSibling.style.display = 'inline-block';
        } else {
            cb.nextElementSibling.style.display = 'none';
            cb.checked = false;
        }
    });
    validateDeployGpus();
}

window.validateDeployGpus = function () {
    const isTp = document.getElementById('typeTp').checked;
    const selectedCount = document.querySelectorAll('.gpu-checkbox:checked').length;
    const deployBtn = document.querySelector('button[onclick="submitDeployment()"]');

    if (isTp) {
        const tpInput = document.getElementById('deployTp');
        const warning = document.getElementById('tpWarningText');

        tpInput.value = selectedCount || 1;

        const validTp = [1, 2, 4, 8].includes(selectedCount);
        if (!validTp && selectedCount > 0) {
            warning.style.display = 'block';
            deployBtn.disabled = true;
        } else {
            warning.style.display = 'none';
            deployBtn.disabled = false;
        }
    } else {
        deployBtn.disabled = false;
    }
}

function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    }[ch]));
}



// ─── Dynamo frontend status (dashboard) ────────────────────────────────
function fmtNum(v) {
    if (v === null || v === undefined || v === '') return '—';
    const n = Number(v);
    if (!isFinite(n)) return '—';
    if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e4) return (n / 1e3).toFixed(1) + 'k';
    return String(Math.round(n));
}

// Base URL a client outside should call. Through the gateway the dashboard and
// /v1 share one origin, so central reports that origin as `public_url` (derived
// from the X-Forwarded-* headers) and every example on this page uses it — the
// domain the operator typed, not an internal IP. Falls back to central's view of
// the frontend when the page is opened directly on central.
function frontendBaseUrl() {
    const pub = frontendInfo && frontendInfo.public_url;
    if (pub) return String(pub).replace(/\/+$/, '');
    const raw = frontendInfo && frontendInfo.url;
    if (raw) {
        try {
            const u = new URL(raw);
            const local = /^(127\.|0\.0\.0\.0$|localhost$|\[::1\]$|::1$)/.test(u.hostname);
            const port = u.port || '11434';
            if (!local) return `${u.protocol}//${u.host}`.replace(/\/+$/, '');
            return `${u.protocol}//${window.location.hostname}:${port}`;
        } catch (e) { /* fall through to the default below */ }
    }
    return `http://${window.location.hostname}:11434`;
}

function renderFrontendStatus() {
    const el = document.getElementById('frontend-status');
    if (!el) return;
    const fe = frontendInfo;
    const models = (fe && Array.isArray(fe.models)) ? fe.models : [];
    const metrics = (fe && fe.metrics) || {};
    const instanceCount = models.reduce((s, m) => s + (Number(m.instances) || 0), 0);
    const healthy = fe ? !!fe.healthy : false;
    const color = !fe ? '#9ca3af' : healthy ? '#22c55e' : '#ef4444';
    const label = !fe ? 'Unknown' : healthy ? 'Healthy' : 'Unreachable';

    const modelRows = models.length === 0
        ? '<div class="text-muted small px-3 pb-3">No models registered on the frontend.</div>'
        : `<div class="fe-model-list">${models.map(m => `
            <div class="fe-model-row">
                <span class="status-dot" style="background:${m.ready ? '#22c55e' : '#f59e0b'}"></span>
                <span class="font-monospace small text-truncate" title="${escapeHtml(m.id)}">${escapeHtml(m.id)}</span>
                <span class="text-muted small font-monospace">${escapeHtml(m.namespace ?? '—')}</span>
                <span class="badge bg-light text-secondary border">${fmtNum(m.instances)} inst</span>
            </div>`).join('')}</div>`;

    el.innerHTML = `
        <div class="stat-strip">
            <div class="stat-item">
                <div class="val" style="color:${color}"><span class="status-dot" style="background:${color}"></span>${escapeHtml(label)}</div>
                <div class="lbl">Frontend</div>
            </div>
            <div class="stat-item">
                <div class="val">${escapeHtml((fe && fe.router_mode) || '—')}</div>
                <div class="lbl">Router mode</div>
            </div>
            <div class="stat-item">
                <div class="val">${models.length}</div>
                <div class="lbl">Models</div>
            </div>
            <div class="stat-item">
                <div class="val">${instanceCount}</div>
                <div class="lbl">Instances</div>
            </div>
            <div class="stat-item">
                <div class="val">${fmtNum(metrics.active_requests)}</div>
                <div class="lbl">Active req</div>
            </div>
            <div class="stat-item">
                <div class="val">${fmtNum(metrics.queued_requests)}</div>
                <div class="lbl">Queued req</div>
            </div>
            <div class="stat-item stat-item-url">
                <div class="val font-monospace" style="font-size:.92rem">${escapeHtml(frontendBaseUrl())}/v1</div>
                <div class="lbl">Base URL</div>
            </div>
        </div>
        ${modelRows}`;
}

// ─── Deployments (per host) ────────────────────────────────────────────
// Global GPU ids are `<worker_id>-<local_index>`.
function widOfGpu(globalGpuId) {
    const g = String(globalGpuId || '');
    const i = g.lastIndexOf('-');
    return i >= 0 ? g.substring(0, i) : g;
}
function gpuIndexOf(globalGpuId) {
    const g = String(globalGpuId || '');
    const i = g.lastIndexOf('-');
    return i >= 0 ? g.substring(i + 1) : g;
}
function hostDisplayName(wid) {
    const ep = endpoints && endpoints[wid];
    return (ep && ep.name) || wid;
}
// Stable per-host key: used both for the "stop all on host" action map and as a
// row class, so stopping a host can fade its rows immediately.
function hostKeyOf(wid) {
    return ('h_' + String(wid)).replace(/[^a-zA-Z0-9_]/g, '_');
}
// Container names are `<engine>_<deployId>_<worker>_<gpu>` (worker/manager.py);
// the prefix follows the deployment engine or dynamo instances never match and
// show as "Starting" with no logs.
function enginePrefix(engine) {
    const e = String(engine || 'dynamo').toLowerCase();
    return e === 'dynamo' ? 'dynamo' : e === 'ollama' ? 'ollama' : 'vllm';
}
function engineBadge(engine) {
    const e = String(engine || 'dynamo').toUpperCase();
    if (e === 'DYNAMO') return '<span class="badge bg-success" title="Served through the Dynamo frontend">DYNAMO</span>';
    if (e === 'VLLM') return '<span class="badge bg-secondary" title="Legacy: direct endpoint, not served by the frontend">VLLM</span>';
    return `<span class="badge bg-light text-secondary border">${escapeHtml(e)}</span>`;
}
function statusDot(isRunning) {
    return isRunning
        ? '<span class="st-dot-wrap"><span class="status-dot" style="background:#22c55e"></span><span class="small">Running</span></span>'
        : '<span class="st-dot-wrap"><span class="status-dot" style="background:#f59e0b"></span><span class="small text-muted">Starting</span></span>';
}

function renderDeployments() {
    const container = document.getElementById('deployments-container');
    if (!container) return;

    if (!deployments || deployments.length === 0) {
        container.innerHTML = '<div class="card"><div class="p-3 text-muted mb-0">No active deployments.</div></div>';
        return;
    }

    // One entry per instance: a replica group has one container per GPU, a TP
    // deployment has a single container spanning all of its GPUs.
    const instances = [];
    deployments.forEach(dep => {
        const gpuList = dep.gpus || [];
        const multi = (dep.deployment_type || '').toLowerCase() === 'replicas' && gpuList.length > 1;
        if (multi) {
            gpuList.forEach(gpu => instances.push({ dep, gpu, gpus: [gpu], wid: widOfGpu(gpu), multi: true }));
        } else {
            const first = gpuList[0] || '';
            instances.push({ dep, gpu: first, gpus: gpuList, wid: widOfGpu(first), multi: false });
        }
    });

    // host -> (deployment id -> instances)
    const hosts = new Map();
    instances.forEach(inst => {
        if (!hosts.has(inst.wid)) hosts.set(inst.wid, new Map());
        const byDep = hosts.get(inst.wid);
        if (!byDep.has(inst.dep.id)) byDep.set(inst.dep.id, []);
        byDep.get(inst.dep.id).push(inst);
    });
    const sortedHosts = [...hosts.entries()]
        .sort((a, b) => hostDisplayName(a[0]).localeCompare(hostDisplayName(b[0])));

    container.innerHTML = sortedHosts.map(([wid, byDep]) => {
        const epKey = hostKeyOf(wid);
        const hostInstances = [...byDep.values()].flat();
        _endpointActions.set(epKey, hostInstances);
        const collapsed = hostGroupCollapsed.has(wid);
        const body = collapsed ? 'display:none' : '';
        const caret = collapsed ? 'transform:rotate(-90deg)' : '';
        const depCount = byDep.size;
        const healthy = hostInstances.filter(i => instanceHealthy(i)).length;

        const groups = [...byDep.values()]
            .sort((a, b) => (a[0].dep.name || '').localeCompare(b[0].dep.name || ''))
            .map(list => renderDeploymentGroup(list))
            .join('');

        return `
            <div class="card dep-host-group mb-3">
                <div class="dep-host-head" data-wid="${escapeHtml(wid)}">
                    <i class="fa-solid fa-chevron-down" style="font-size:.72rem;color:#6b7280;transition:transform .2s;${caret}"></i>
                    <span class="fw-semibold">${escapeHtml(hostDisplayName(wid))}</span>
                    <span class="badge bg-secondary" style="font-size:.72rem">${depCount} deployment${depCount > 1 ? 's' : ''}</span>
                    <span class="badge bg-light text-secondary border" style="font-size:.72rem">${healthy}/${hostInstances.length} instances running</span>
                    <button class="btn btn-outline-danger btn-sm ms-auto" data-stop-host="${escapeHtml(epKey)}" title="이 호스트의 모든 인스턴스 종료">
                        <i class="fa-solid fa-stop me-1"></i>Stop all on host
                    </button>
                </div>
                <div class="dep-host-body" style="${body}">
                    <table class="table table-sm inst-table mb-0">
                        <thead class="table-light">
                            <tr>
                                <th class="small text-muted fw-semibold">모델 / Served name</th>
                                <th class="small text-muted fw-semibold d-none d-md-table-cell">엔진</th>
                                <th class="small text-muted fw-semibold">상태</th>
                                <th class="small text-muted fw-semibold">GPU</th>
                                <th class="small text-muted fw-semibold text-end">액션</th>
                            </tr>
                        </thead>
                        <tbody>${groups}</tbody>
                    </table>
                </div>
            </div>`;
    }).join('');

    container.querySelectorAll('.dep-host-head').forEach(el => {
        el.addEventListener('click', (e) => {
            if (e.target.closest('button')) return;
            toggleHostGroup(el.dataset.wid);
        });
    });
    container.querySelectorAll('button[data-stop-host]').forEach(btn => {
        btn.addEventListener('click', () => stopEndpoint(btn.dataset.stopHost, btn));
    });
}

// Instance health comes from the deployment node whose container name matches
// this instance (`<engine>_<deployId>_<worker>_<gpu>`); single-container
// deployments (TP, single replica) use the deployment's only node.
function instanceContainerName(inst) {
    if (!inst.multi) {
        const node = (inst.dep.nodes || [])[0];
        return node ? node.name : null;
    }
    return `${enginePrefix(inst.dep.engine)}_${inst.dep.id}_${widOfGpu(inst.gpu)}_${gpuIndexOf(inst.gpu)}`;
}
function instanceHealthy(inst) {
    const nodes = inst.dep.nodes || [];
    if (!inst.multi) return nodes.length > 0 ? !!nodes[0].is_healthy : false;
    const name = instanceContainerName(inst);
    const node = nodes.find(n => n.name === name);
    return node ? !!node.is_healthy : false;
}

function renderDeploymentGroup(list) {
    const dep = list[0].dep;
    const dtype = (dep.deployment_type || '').toUpperCase();
    const dtypeColor = dtype === 'TP' ? 'bg-primary' : 'bg-secondary';
    const depStopping = _stopping.has(dep.id);

    const meta = [];
    if (dep.max_len) meta.push(`max_len ${escapeHtml(dep.max_len)}`);
    if (dep.gpu_util) meta.push(`gpu_util ${escapeHtml(dep.gpu_util)}`);
    if (dep.block_size) meta.push(`block ${escapeHtml(dep.block_size)}`);
    if (dep.reasoning_parser) meta.push(`reasoning ${escapeHtml(dep.reasoning_parser)}`);
    if (dep.tool_call_parser) meta.push(`tools ${escapeHtml(dep.tool_call_parser)}`);
    if (dep.image) meta.push(escapeHtml(dep.image));

    const depStopBtn = depStopping
        ? '<button class="btn btn-sm btn-outline-danger" disabled><i class="fa-solid fa-spinner fa-spin"></i></button>'
        : `<button class="btn btn-sm btn-outline-danger" data-stop-dep="${escapeHtml(dep.id)}"
               title="배포 전체 종료 (모든 호스트)"><i class="fa-solid fa-stop me-1"></i>Stop deployment</button>`;

    const head = `
        <tr class="dep-head-row">
            <td colspan="5">
                <div class="dep-head">
                    <span class="fw-semibold">${escapeHtml(dep.name)}</span>
                    <span class="font-monospace text-muted" style="font-size:.78rem">${escapeHtml(dep.id)}</span>
                    ${engineBadge(dep.engine)}
                    <span class="badge ${dtypeColor}">${escapeHtml(dtype || 'REPLICAS')}</span>
                    ${dep.is_embedding ? '<span class="badge bg-info text-dark">EMBEDDING</span>' : ''}
                    <span class="ms-auto">${depStopBtn}</span>
                </div>
                ${meta.length ? `<div class="dep-head-meta font-monospace">${meta.join(' · ')}</div>` : ''}
            </td>
        </tr>`;

    const rows = list.map(inst => {
        const dep2 = inst.dep;
        const running = instanceHealthy(inst);
        const container = instanceContainerName(inst);
        const stopKey = inst.multi ? `${dep2.id}:${inst.gpu}` : dep2.id;
        const stopping = _stopping.has(stopKey);
        const rowStyle = stopping ? 'opacity:.3;pointer-events:none' : '';
        const stopBtn = stopping
            ? '<button class="btn btn-sm btn-outline-danger" disabled><i class="fa-solid fa-spinner fa-spin"></i></button>'
            : inst.multi
                ? `<button class="btn btn-sm btn-outline-danger" data-stop-gpu="${escapeHtml(dep2.id)}" data-gpu="${escapeHtml(inst.gpu)}" title="이 GPU의 인스턴스만 종료"><i class="fa-solid fa-stop"></i></button>`
                : `<button class="btn btn-sm btn-outline-danger" data-stop-dep="${escapeHtml(dep2.id)}" title="이 인스턴스(=배포) 종료"><i class="fa-solid fa-stop"></i></button>`;
        const served = dep2.served_model_name || dep2.model;
        const gpuBadges = (inst.gpus || []).map(g =>
            `<span class="badge bg-light text-secondary border me-1" style="font-size:.78rem">GPU ${escapeHtml(gpuIndexOf(g))}</span>`).join('');

        return `
            <tr class="ep-${escapeHtml(hostKeyOf(inst.wid))}" style="${rowStyle}">
                <td>
                    <div class="small text-break" title="${escapeHtml(dep2.model)}">${escapeHtml(dep2.model)}</div>
                    <div class="small text-break" style="color:#94a3b8" title="${escapeHtml(served)}">↳ ${escapeHtml(served)}</div>
                </td>
                <td class="d-none d-md-table-cell">${engineBadge(dep2.engine)}</td>
                <td>${statusDot(running)}</td>
                <td>${gpuBadges || '<span class="text-muted">—</span>'}</td>
                <td class="text-end text-nowrap">
                    <button class="btn btn-sm btn-outline-secondary me-1" data-logs-dep="${escapeHtml(dep2.id)}"
                        data-container="${escapeHtml(container || '')}" title="Logs"><i class="fa-solid fa-terminal"></i></button>
                    ${stopBtn}
                </td>
            </tr>`;
    }).join('');

    return head + rows;
}

window.toggleHostGroup = function (wid) {
    if (hostGroupCollapsed.has(wid)) hostGroupCollapsed.delete(wid);
    else hostGroupCollapsed.add(wid);
    renderDeployments();
};

// Delegated actions for the deployment list (rebuilt on every poll, so no
// per-row listeners are kept around).
document.addEventListener('click', (e) => {
    const logsBtn = e.target.closest && e.target.closest('button[data-logs-dep]');
    if (logsBtn) {
        viewLogs(logsBtn.dataset.logsDep, logsBtn.dataset.container || undefined);
        return;
    }
    const depBtn = e.target.closest && e.target.closest('button[data-stop-dep]');
    if (depBtn) {
        stopDeployment(depBtn.dataset.stopDep, depBtn);
        return;
    }
    const gpuBtn = e.target.closest && e.target.closest('button[data-stop-gpu]');
    if (gpuBtn) stopReplica(gpuBtn.dataset.stopGpu, gpuBtn.dataset.gpu, gpuBtn);
});



// ─── API page (Dynamo frontend) ────────────────────────────────────────
// Models come from GET /api/frontend (the ids clients pass in the `model`
// field). Capability is inferred from the model name — the frontend's
// /v1/models does not expose a task per model.
function gwEndpointKind(model) {
    const m = (model || '').toLowerCase();
    return /(embed|embedding|sentence-transformers|bge[-_]|e5[-_]|gte[-_])/.test(m)
        ? 'embedding' : 'chat';
}
let gwSelectedModelId = null;
let gwExampleLang = 'curl';
let gwTestAbort = null;

// Frontend models, enriched with what central knows about the deployments
// behind each served name (engine, replica health). Falls back to the
// deployment list alone when /api/frontend is unavailable.
function gwModels() {
    const byServed = new Map();
    (deployments || []).filter(d => d.status === 'running').forEach(d => {
        const key = d.served_model_name || d.model;
        if (!byServed.has(key)) byServed.set(key, []);
        byServed.get(key).push(d);
    });

    const feModels = (frontendInfo && Array.isArray(frontendInfo.models)) ? frontendInfo.models : null;
    const rows = (feModels || [...byServed.keys()].map(id => ({ id, instances: null, ready: null, namespace: null })))
        .map(m => {
            const deps = byServed.get(m.id) || [];
            const nodes = deps.flatMap(d => d.nodes || []);
            return {
                id: m.id,
                namespace: m.namespace ?? null,
                ready: m.ready ?? null,
                instances: m.instances ?? nodes.length,
                healthy: nodes.filter(n => n.is_healthy).length,
                total: nodes.length,
                engine: deps.length ? (deps[0].engine || 'dynamo') : null,
                kind: gwEndpointKind(m.id + ' ' + deps.map(d => d.model).join(' ')),
            };
        });
    rows.sort((a, b) => a.id.localeCompare(b.id));
    return rows;
}

// Legacy vLLM deployments never appear on the frontend — they serve their own
// HTTPS endpoint. Listed separately so the page doesn't imply they are routed.
function gwLegacyDeployments() {
    const feIds = new Set(((frontendInfo && frontendInfo.models) || []).map(m => m.id));
    return (deployments || []).filter(d =>
        d.status === 'running' &&
        String(d.engine || 'dynamo').toLowerCase() === 'vllm' &&
        !feIds.has(d.served_model_name || d.model));
}

function renderGateway() {
    const listEl = document.getElementById('gw-models-list');
    if (!listEl) return;

    const baseEl = document.getElementById('gw-base-url');
    if (baseEl) baseEl.textContent = `${frontendBaseUrl()}/v1`;
    const stateEl = document.getElementById('gw-frontend-state');
    if (stateEl) {
        const fe = frontendInfo;
        const ok = fe ? !!fe.healthy : false;
        stateEl.className = 'badge ' + (!fe ? 'bg-secondary' : ok ? 'bg-success' : 'bg-danger');
        stateEl.textContent = !fe ? 'frontend: unknown' : ok ? 'frontend: healthy' : 'frontend: unreachable';
    }

    const models = gwModels();
    const legacy = gwLegacyDeployments();
    if (models.length === 0) {
        listEl.innerHTML = '<div class="text-muted small">No models are registered on the frontend.</div>'
            + renderLegacyList(legacy);
        document.getElementById('gw-example-card').style.display = 'none';
        document.getElementById('gw-test-card').style.display = 'none';
        return;
    }

    const groups = { chat: [], embedding: [] };
    for (const m of models) groups[m.kind].push(m);
    const groupMeta = {
        chat:      { label: 'Chat completions', icon: 'fa-comments',      path: '/v1/chat/completions' },
        embedding: { label: 'Embeddings',       icon: 'fa-vector-square', path: '/v1/embeddings' },
    };
    const sections = Object.keys(groups)
        .filter(k => groups[k].length > 0)
        .sort((a, b) => groups[b].length - groups[a].length);

    listEl.innerHTML = sections.map(kind => {
        const meta = groupMeta[kind];
        const rows = groups[kind].map(m => {
            const sel = m.id === gwSelectedModelId ? 'selected' : '';
            const unhealthy = m.total > 0 && m.healthy < m.total;
            const health = m.total > 0 ? `${m.healthy}/${m.total}` : `${fmtNum(m.instances)}`;
            return `<div class="gw-deploy-row ${sel}" data-name="${encodeURIComponent(m.id)}" onclick="gwSelectModel(decodeURIComponent(this.dataset.name))">
                <div class="gw-deploy-name">${escapeHtml(m.id)}</div>
                <div class="gw-deploy-meta">
                    <span title="${escapeHtml(m.namespace || '')}">${escapeHtml(m.namespace || '—')}</span>
                    <span><span class="gw-deploy-status ${unhealthy ? 'unhealthy' : ''}"></span>${escapeHtml(health)} instances</span>
                </div>
            </div>`;
        }).join('');
        return `<div class="gw-deploy-group">
            <div class="gw-group-head">
                <i class="fa-solid ${meta.icon}"></i><span>${meta.label}</span>
                <span class="badge bg-light text-secondary border">${meta.path}</span>
                <span class="ms-auto text-muted">${groups[kind].length}</span>
            </div>
            ${rows}
        </div>`;
    }).join('') + renderLegacyList(legacy);

    if (!gwSelectedModelId || !models.some(m => m.id === gwSelectedModelId)) {
        gwSelectModel(models[0].id);
    } else {
        gwRefreshExample();
    }
}

function renderLegacyList(legacy) {
    if (!legacy || legacy.length === 0) return '';
    const rows = legacy.map(d => {
        const served = d.served_model_name || d.model;
        const nodes = d.nodes || [];
        const urls = nodes.map(n => `${n.host}:${n.port}`).join(', ');
        return `<div class="gw-deploy-row" style="cursor:default;opacity:.85">
            <div class="gw-deploy-name">${escapeHtml(served)}</div>
            <div class="gw-deploy-meta"><span title="${escapeHtml(urls)}">${escapeHtml(urls || '—')}</span></div>
        </div>`;
    }).join('');
    return `<div class="gw-deploy-group">
        <div class="gw-group-head">
            <i class="fa-solid fa-triangle-exclamation"></i><span>Legacy (vllm)</span>
            <span class="badge bg-light text-secondary border">direct endpoint — no gateway</span>
            <span class="ms-auto text-muted">${legacy.length}</span>
        </div>
        ${rows}
    </div>`;
}

window.gwSelectModel = function (id) {
    gwSelectedModelId = id;
    document.querySelectorAll('.gw-deploy-row').forEach(el => {
        el.classList.toggle('selected', decodeURIComponent(el.dataset.name || '') === id);
    });
    document.getElementById('gw-example-card').style.display = 'block';
    document.getElementById('gw-test-card').style.display = 'block';
    gwRefreshExample();
}
// Backwards-compatible alias (older markup/bookmarks call this name).
window.gwSelectDeployment = window.gwSelectModel;

window.gwSelectExampleTab = function (lang) {
    gwExampleLang = lang;
    document.querySelectorAll('.gw-tab-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.lang === lang);
    });
    gwRefreshExample();
}

// Default prompts that match the endpoint kind. Each user edit is preserved
// per-kind so switching back and forth doesn't blow away typed input.
const GW_DEFAULT_PROMPT = {
    chat: 'Hello, can you introduce yourself?',
    embedding: 'The quick brown fox jumps over the lazy dog.',
};
const gwPromptCache = { chat: null, embedding: null };
let gwLastKind = null;

function gwRefreshExample() {
    const model = gwSelectedModelId;
    if (!model) return;
    const kind = gwEndpointKind(model);
    const base = frontendBaseUrl();
    const code = document.getElementById('gw-example-code');
    if (!code) return;

    // Swap the prompt textarea to a kind-appropriate default the first time we
    // visit each kind. If the user has typed something, that edit is captured
    // before the swap and restored when they come back.
    const inputEl = document.getElementById('gw-test-input');
    if (inputEl) {
        if (gwLastKind && gwLastKind !== kind) {
            gwPromptCache[gwLastKind] = inputEl.value;
        }
        if (gwLastKind !== kind) {
            inputEl.value = gwPromptCache[kind] ?? GW_DEFAULT_PROMPT[kind];
            gwLastKind = kind;
        }
    }
    const examplePrompt = GW_DEFAULT_PROMPT[kind];
    if (gwExampleLang === 'curl') {
        if (kind === 'embedding') {
            code.textContent = `curl -X POST ${base}/v1/embeddings \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model}",
    "input": "${examplePrompt}"
  }'`;
        } else {
            code.textContent = `curl -X POST ${base}/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "${model}",
    "messages": [
      {"role": "user", "content": "${examplePrompt}"}
    ]
  }'`;
        }
    } else {
        if (kind === 'embedding') {
            code.textContent = `from openai import OpenAI

client = OpenAI(
    base_url="${base}/v1",
    api_key="EMPTY",
)

resp = client.embeddings.create(
    model="${model}",
    input="${examplePrompt}",
)
print(resp.data[0].embedding[:8], "...")`;
        } else {
            code.textContent = `from openai import OpenAI

client = OpenAI(
    base_url="${base}/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="${model}",
    messages=[
        {"role": "user", "content": "${examplePrompt}"},
    ],
)
print(resp.choices[0].message.content)`;
        }
    }
    const modelEl = document.getElementById('gw-test-model');
    if (modelEl) modelEl.textContent = model;
}

window.gwCopyExample = function () {
    const text = document.getElementById('gw-example-code').textContent;
    navigator.clipboard?.writeText(text).then(() => {
        const btn = document.querySelector('.gw-copy-btn');
        if (!btn) return;
        const orig = btn.innerHTML;
        btn.innerHTML = '<i class="fa-solid fa-check me-1"></i>Copied';
        setTimeout(() => { btn.innerHTML = orig; }, 1100);
    });
}

// In-browser load test, pointed at the Dynamo frontend.
window.gwRunTest = async function () {
    const modelId = gwSelectedModelId;
    if (!modelId) return;
    const kind = gwEndpointKind(modelId);
    const input = document.getElementById('gw-test-input').value;
    // Hard caps. A browser opens only a handful of connections per origin, so a
    // large "concurrency" here does not load the server — it queues inside the
    // tab and starves the dashboard's own polling, which looks like the whole UI
    // freezing. Real load tests belong on the CLI against the gateway.
    const MAX_TOTAL = 500, MAX_CONCURRENCY = 32;
    const total = Math.min(MAX_TOTAL, Math.max(1, parseInt(document.getElementById('gw-test-total').value, 10) || 1));
    const concurrency = Math.min(MAX_CONCURRENCY, total,
        Math.max(1, parseInt(document.getElementById('gw-test-concurrency').value, 10) || 1));
    document.getElementById('gw-test-total').value = String(total);
    document.getElementById('gw-test-concurrency').value = String(concurrency);
    const apiKey = (document.getElementById('gw-api-key')?.value || '').trim();

    const btn = document.getElementById('gw-test-btn');
    const abortBtn = document.getElementById('gw-test-abort');
    const progressBar = document.getElementById('gw-progress-bar');
    const progressText = document.getElementById('gw-progress-text');
    const progressRps = document.getElementById('gw-progress-rps');
    const sentEl = document.getElementById('gw-stat-sent');
    const successEl = document.getElementById('gw-stat-success');
    const failEl = document.getElementById('gw-stat-fail');
    const latEl = document.getElementById('gw-stat-latency');
    const responseEl = document.getElementById('gw-test-response');

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Running…';
    abortBtn.disabled = false;

    progressBar.style.width = '0%';
    sentEl.textContent = '0';
    successEl.textContent = '0';
    failEl.textContent = '0';
    latEl.textContent = '—';
    responseEl.textContent = '';

    const abortController = new AbortController();
    gwTestAbort = abortController;

    // Same-origin proxy: the Dynamo frontend sends no CORS headers, so a direct
    // browser fetch to :11434 fails with "Failed to fetch". Central forwards it.
    const url = `/api/gateway${kind === 'embedding' ? '/v1/embeddings' : '/v1/chat/completions'}`;
    const buildPayload = () => kind === 'embedding'
        ? { model: modelId, input }
        : { model: modelId, messages: [{ role: 'user', content: input }] };

    let completed = 0, success = 0, fail = 0;
    const latencies = [];
    const start = Date.now();
    let lastSampleResponse = null;
    let inFlight = 0, dispatched = 0;

    function updateUi() {
        const pct = total === 0 ? 0 : (completed / total) * 100;
        progressBar.style.width = pct.toFixed(1) + '%';
        progressText.textContent = `${completed} / ${total}  ·  ${inFlight} in flight`;
        const elapsedSec = (Date.now() - start) / 1000;
        const rps = elapsedSec > 0 ? completed / elapsedSec : 0;
        progressRps.textContent = `${rps.toFixed(1)} req/s`;
        sentEl.textContent = String(dispatched);
        successEl.textContent = String(success);
        failEl.textContent = String(fail);
        if (latencies.length > 0) {
            const avg = latencies.reduce((s, v) => s + v, 0) / latencies.length;
            latEl.textContent = avg.toFixed(2) + ' s';
        }
    }

    async function dispatchOne(idx) {
        inFlight++;
        dispatched++;
        updateUi();
        const t0 = performance.now();
        try {
            const res = await fetch(url, {
                method: 'POST',
                signal: abortController.signal,
                headers: {
                    'Content-Type': 'application/json',
                    ...(apiKey ? { 'Authorization': `Bearer ${apiKey}` } : {}),
                },
                body: JSON.stringify(buildPayload()),
            });
            const text = await res.text();
            const dt = (performance.now() - t0) / 1000;
            latencies.push(dt);
            if (res.ok) {
                success++;
                if (!lastSampleResponse) {
                    try {
                        const parsed = JSON.parse(text);
                        if (kind === 'embedding') {
                            const vec = parsed.data?.[0]?.embedding || [];
                            lastSampleResponse = `[embedding length=${vec.length}] ${JSON.stringify(vec.slice(0, 8))}…`;
                        } else {
                            lastSampleResponse = parsed.choices?.[0]?.message?.content || text;
                        }
                    } catch { lastSampleResponse = text; }
                    responseEl.textContent = `--- Sample successful response (#${idx + 1}) ---\n${lastSampleResponse}`;
                }
            } else {
                fail++;
                if (fail <= 3) {
                    responseEl.textContent += `\n\n--- Request #${idx + 1} failed (HTTP ${res.status}) ---\n${text.slice(0, 500)}`;
                }
            }
        } catch (err) {
            if (err.name === 'AbortError') { return; }
            fail++;
            if (fail <= 3) {
                responseEl.textContent += `\n\n--- Request #${idx + 1} network error ---\n${err.message}`;
            }
        } finally {
            inFlight--;
            if (!abortController.signal.aborted) completed++;
            updateUi();
        }
    }

    // Bounded concurrency: keep up to `concurrency` requests in flight, fed off a
    // shared index counter, until `total` have been dispatched.
    let nextIdx = 0;
    async function worker() {
        while (true) {
            if (abortController.signal.aborted) return;
            const my = nextIdx++;
            if (my >= total) return;
            await dispatchOne(my);
        }
    }
    const workers = Array.from({ length: concurrency }, () => worker());
    await Promise.all(workers);

    progressText.textContent = abortController.signal.aborted
        ? `Aborted at ${completed} / ${total}`
        : `Completed ${completed} / ${total} in ${((Date.now() - start)/1000).toFixed(2)}s`;
    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-paper-plane me-2"></i>Send requests';
    abortBtn.disabled = true;
    gwTestAbort = null;
}

window.gwAbortTest = function () {
    if (gwTestAbort) gwTestAbort.abort();
}

function renderConfigs() {
    const list = document.getElementById('saved-configs-container');
    if (!list) return;
    list.innerHTML = '';

    if (savedConfigs.length === 0) {
        list.innerHTML = '<div class="col-12 text-muted">No saved configurations.</div>';
        return;
    }

    const rows = savedConfigs.map(conf => {
        const dtype = (conf.deployment_type || conf.mode || 'replicas').toUpperCase();
        const dtypeColor = dtype === 'TP' ? 'bg-primary' : 'bg-secondary';
        // `image` is the contract field; `vllm_image` is what older saved
        // configs used, so both are accepted when reading one back.
        const image = conf.image || conf.vllm_image;
        const imageTag = image
            ? `<div class="font-monospace text-muted text-break mt-1" style="font-size:.78rem" title="${escapeHtml(image)}"><i class="fa-brands fa-docker me-1"></i>${escapeHtml(image)}</div>`
            : '';
        const extras = [];
        if (conf.served_model_name) extras.push(`served: ${escapeHtml(conf.served_model_name)}`);
        if (conf.reasoning_parser) extras.push(`reasoning: ${escapeHtml(conf.reasoning_parser)}`);
        if (conf.tool_call_parser) extras.push(`tools: ${escapeHtml(conf.tool_call_parser)}`);
        const extraTag = extras.length
            ? `<div class="text-muted text-break mt-1" style="font-size:.78rem">${extras.join(' · ')}</div>`
            : '';
        return `
            <tr class="config-row">
                <td>
                    <div class="fw-semibold small">${escapeHtml(conf.name)}</div>
                    <div class="font-monospace text-muted text-break" style="font-size:.82rem" title="${escapeHtml(conf.model)}">${escapeHtml(conf.model)}</div>
                    ${imageTag}
                    ${extraTag}
                </td>
                <td class="text-nowrap">
                    <span class="badge ${dtypeColor} me-1">${dtype}</span>
                    ${engineBadge(conf.engine)}
                    ${conf.is_embedding ? '<span class="badge bg-info text-dark ms-1">EMBEDDING</span>' : ''}
                </td>
                <td class="text-end text-nowrap">
                    <button class="btn btn-sm btn-outline-danger me-1" onclick="deleteConfig('${conf.name}')" title="Delete"><i class="fa-solid fa-trash"></i></button>
                    <button class="btn btn-sm btn-outline-secondary me-1" onclick="loadConfig('${conf.name}')"><i class="fa-solid fa-pen me-1"></i>Edit</button>
                    <button class="btn btn-sm btn-primary" onclick="loadConfig('${conf.name}')"><i class="fa-solid fa-play me-1"></i>Run</button>
                </td>
            </tr>`;
    }).join('');

    list.innerHTML = `
        <div class="col-12">
            <div class="card table-card">
                <table class="table table-hover">
                    <thead class="table-light">
                        <tr>
                            <th class="small text-muted fw-semibold">이름 / 모델</th>
                            <th class="small text-muted fw-semibold">모드</th>
                            <th class="small text-muted fw-semibold text-end">액션</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>`;

    // Populate modal dropdown
    const selector = document.getElementById('configSelector');
    if (selector) {
        const prevConfig = selector.value;
        selector.innerHTML = '<option value="">-- Select a Config (Optional) --</option>';
        savedConfigs.forEach(conf => {
            const dtype = (conf.deployment_type || conf.mode || 'replicas').toUpperCase();
            selector.innerHTML += `<option value="${conf.name}">${conf.name} [${dtype}]</option>`;
        });
        selector.value = prevConfig;
    }
}

window.applySelectedConfig = function () {
    const selector = document.getElementById('configSelector');
    if (selector && selector.value) {
        loadConfig(selector.value);
    }
}

function renderEndpoints() {
    const pendingList = document.getElementById('pending-endpoints-container');
    const activeList = document.getElementById('active-endpoints-container');
    if (!pendingList || !activeList) return;

    // Preserve typed text in accept-name inputs
    const inputCache = {};
    document.querySelectorAll('[id^=accept-name-]').forEach(el => { inputCache[el.id] = el.value; });
    const activeId = document.activeElement?.id || null;
    let selStart = null, selEnd = null;
    if (activeId && document.activeElement.tagName === 'INPUT') {
        try { selStart = document.activeElement.selectionStart; selEnd = document.activeElement.selectionEnd; } catch (e) {}
    }

    pendingList.innerHTML = '';
    activeList.innerHTML = '';
    let hasPending = false, hasActive = false;
    const activeRows = [];

    Object.values(endpoints).forEach(ep => {
        if (ep.status === 'pending') {
            hasPending = true;
            pendingList.innerHTML += `
                <div class="col-md-6 col-lg-4">
                    <div class="card" style="border-left:3px solid #f59e0b">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <div>
                                    <div class="fw-semibold">신규 노드 감지됨</div>
                                    <div class="text-muted small">${ep.gpus.length}개 GPU · ${escapeHtml(ep.host)}:${escapeHtml(ep.port)}</div>
                                </div>
                                <span class="badge bg-warning text-dark">Pending</span>
                            </div>
                            <div class="text-muted mb-3 font-monospace small">${escapeHtml(ep.id)}</div>
                            <div class="input-group">
                                <input type="text" id="accept-name-${escapeHtml(ep.id)}" class="form-control" placeholder="노드 이름 지정" value="${escapeHtml(inputCache[`accept-name-${ep.id}`] || ep.id)}">
                                <button class="btn btn-success" onclick="acceptEndpoint('${escapeHtml(ep.id)}')">
                                    <i class="fa-solid fa-check me-1"></i>Accept
                                </button>
                            </div>
                        </div>
                    </div>
                </div>`;
        } else if (ep.status === 'active') {
            hasActive = true;
            const safeId = escapeHtml(ep.id);
            const epUrlId = encodeURIComponent(ep.id);
            activeRows.push(`
                <div class="ep-row" style="border-left:3px solid #22c55e">
                    <div class="ep-row-info">
                        <div class="ep-row-title">
                            <span style="width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block"></span>
                            <span class="fw-semibold">${escapeHtml(ep.name)}</span>
                            <span class="badge bg-success" style="font-size:.72rem">Active</span>
                            <span class="badge bg-secondary" style="font-size:.72rem">${ep.gpus.length} GPUs</span>
                        </div>
                        <div class="ep-row-meta">
                            <span class="text-muted font-monospace">${escapeHtml(ep.host)}:${escapeHtml(ep.port)}</span>
                            <span class="text-muted font-monospace" style="word-break:break-all">${safeId}</span>
                        </div>
                    </div>
                    <div class="ep-row-actions">
                        <button class="btn btn-sm btn-outline-secondary" data-id="${encodeURIComponent(ep.id)}" data-name="${encodeURIComponent(ep.name)}" onclick="renameEndpoint(decodeURIComponent(this.dataset.id), decodeURIComponent(this.dataset.name))">
                            <i class="fa-solid fa-pen me-1"></i>이름 변경
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="resetEndpoint('${escapeHtml(ep.id)}')">
                            <i class="fa-solid fa-trash me-1"></i>제거
                        </button>
                        <a href="/endpoints/${epUrlId}/images" class="btn btn-sm btn-outline-primary">
                            <i class="fa-brands fa-docker me-1"></i>Images
                        </a>
                        <a href="/endpoints/${epUrlId}/models" class="btn btn-sm btn-outline-success">
                            <i class="fa-solid fa-robot me-1"></i>Models
                        </a>
                    </div>
                </div>`);
        }
    });

    if (!hasPending) pendingList.innerHTML = '<div class="col-12 text-muted">등록 대기 중인 호스트가 없습니다.</div>';
    if (!hasActive) {
        activeList.innerHTML = '<p class="text-muted p-3 mb-0">활성화된 호스트가 없습니다.</p>';
    } else {
        activeList.innerHTML = activeRows.join('');
    }

    if (activeId) {
        const el = document.getElementById(activeId);
        if (el) {
            el.focus();
            if (selStart !== null && el.setSelectionRange) { try { el.setSelectionRange(selStart, selEnd); } catch (e) {} }
        }
    }
}

// ---- Worker versions / updates panel (endpoints page) ----

async function fetchVersionInfo() {
    if (!document.getElementById('worker-versions-container')) return;
    try {
        const res = await fetch('/api/version');
        if (!res.ok) throw new Error(res.statusText);
        versionInfo = await res.json();
    } catch (err) {
        console.error('Failed to fetch version info', err);
        return; // keep last rendered state on transient errors
    }
    renderWorkerVersions();
}

function renderWorkerVersions() {
    const container = document.getElementById('worker-versions-container');
    if (!container || !versionInfo) return;

    const target = versionInfo.target;
    const branch = versionInfo.branch;
    const autoUpdate = !!versionInfo.auto_update;
    const workers = versionInfo.workers || [];
    // With no computable target (no repo mounted on central) drift is
    // meaningless — don't flag it, but updates are still allowed.
    const hasDrift = !!target && workers.some(w => w.drift && !w.updating && w.status === 'active');

    const headerHtml = `
        <div class="d-flex align-items-center flex-wrap gap-2 p-3 border-bottom">
            ${target
                ? `<span class="text-muted small">Target:</span>
                   <span class="font-monospace small">${escapeHtml(String(target).slice(0, 12))}</span>`
                : '<span class="text-muted small fst-italic">target version unavailable</span>'}
            ${branch ? `<span class="badge bg-light text-secondary border"><i class="fa-solid fa-code-branch me-1"></i>${escapeHtml(branch)}</span>` : ''}
            <span class="badge ${autoUpdate ? 'bg-success' : 'bg-secondary'}">Auto-update: ${autoUpdate ? 'on' : 'off'}</span>
            <button id="update-all-drifted-btn" class="btn btn-sm btn-outline-primary ms-auto" ${hasDrift ? '' : 'disabled'}>
                <i class="fa-solid fa-rotate me-1"></i>Update all drifted
            </button>
        </div>`;

    if (workers.length === 0) {
        container.innerHTML = headerHtml + '<p class="text-muted p-3 mb-0">No workers reported.</p>';
        return;
    }

    const rows = workers.map(w => {
        const commit = String(w.commit ?? 'unknown');
        const commitHtml = commit === 'unmanaged'
            ? '<span class="text-warning">unmanaged</span>'
            : `<span class="font-monospace">${escapeHtml(commit)}</span>`;

        let stateHtml;
        if (w.updating) {
            stateHtml = '<span class="badge bg-info text-dark"><i class="fa-solid fa-rotate fa-spin me-1"></i>updating</span>';
        } else if (w.up_to_date) {
            stateHtml = '<span class="badge bg-success"><i class="fa-solid fa-check me-1"></i>up to date</span>';
        } else if (w.drift && target) {
            stateHtml = '<span class="badge bg-warning text-dark"><i class="fa-solid fa-triangle-exclamation me-1"></i>drift</span>';
        } else {
            stateHtml = '<span class="badge bg-light text-muted border">&mdash; unknown</span>';
        }

        const statusHtml = w.status === 'active'
            ? '<span class="badge bg-success">Active</span>'
            : `<span class="badge bg-warning text-dark">${escapeHtml(w.status)}</span>`;

        const disabled = w.up_to_date || w.updating || w.status !== 'active';
        let updatedAt = '';
        if (w.updated_at) {
            const d = new Date(w.updated_at);
            updatedAt = isNaN(d) ? String(w.updated_at) : d.toLocaleString();
        }

        return `<tr>
            <td>
                <div class="fw-semibold">${escapeHtml(w.name || w.worker_id)}</div>
                <div class="text-muted small font-monospace">${escapeHtml(w.worker_id)}</div>
            </td>
            <td>${statusHtml}</td>
            <td>${commitHtml}</td>
            <td>${stateHtml}</td>
            <td class="text-muted small">${escapeHtml(updatedAt)}</td>
            <td class="text-end">
                <button class="btn btn-sm btn-outline-primary" data-worker-id="${escapeHtml(w.worker_id)}" ${disabled ? 'disabled' : ''}>
                    <i class="fa-solid fa-arrow-up me-1"></i>Update
                </button>
            </td>
        </tr>`;
    }).join('');

    container.innerHTML = `${headerHtml}
        <div style="overflow-x:auto">
            <table class="table table-hover">
                <thead class="table-light">
                    <tr>
                        <th>Worker</th>
                        <th>Status</th>
                        <th>Commit</th>
                        <th>State</th>
                        <th>Updated</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>${rows}</tbody>
            </table>
        </div>`;
}

async function updateWorker(workerId) {
    try {
        const res = await fetch(`/api/workers/${encodeURIComponent(workerId)}/update`, { method: 'POST' });
        const body = await res.json().catch(() => ({}));
        if (res.ok) {
            showAlert('success', `Update triggered for ${escapeHtml(workerId)}`);
        } else {
            // 409 (mid-deploy) and other errors carry a {detail} message
            showAlert('danger', `Update failed for ${escapeHtml(workerId)}: ${escapeHtml(body.detail || res.statusText)}`);
        }
    } catch (err) {
        showAlert('danger', 'Network error while triggering update');
    }
    setTimeout(fetchVersionInfo, 3000);
}

async function updateAllWorkers() {
    try {
        const res = await fetch('/api/workers/update_all', { method: 'POST' });
        const body = await res.json().catch(() => ({}));
        if (res.ok) {
            const summary = Object.entries(body.results || {})
                .map(([wid, r]) => `${escapeHtml(wid)}: ${escapeHtml(r)}`)
                .join('<br>');
            showAlert('info', 'Update all triggered.' + (summary ? '<br>' + summary : ''));
        } else {
            showAlert('danger', `Update all failed: ${escapeHtml(body.detail || res.statusText)}`);
        }
    } catch (err) {
        showAlert('danger', 'Network error while triggering update all');
    }
    setTimeout(fetchVersionInfo, 3000);
}

async function renderImagePanelBody(wid, bodyEl) {
    // Show loading while fetching
    if (!epImages.has(wid)) {
        bodyEl.innerHTML = '<div class="text-muted small py-1">Loading images...</div>';
        try {
            const resp = await fetch(`/api/endpoints/${wid}/images`);
            if (resp.ok) {
                epImages.set(wid, await resp.json());
            } else if (resp.status === 501) {
                bodyEl.innerHTML = `
                    <div class="text-warning small py-2">
                        <i class="fa-solid fa-triangle-exclamation me-1"></i>
                        이 호스트의 워커 에이전트가 이미지 관리를 지원하지 않습니다. 워커를 업데이트하세요.
                    </div>`;
                return;
            } else {
                epImages.set(wid, []);
            }
        } catch (e) {
            epImages.set(wid, []);
        }
    }

    const images = epImages.get(wid) || [];
    const safeWid = escapeHtml(wid);

    const imageRows = _renderImageRows(images, safeWid);

    bodyEl.innerHTML = `
        <div class="mb-2" style="font-size:.8rem;font-weight:600;color:#374151;margin-top:.25rem">
            <i class="fa-brands fa-docker me-1"></i>Pulled engine images
        </div>
        <div class="mb-3" id="img-list-${safeWid}">${imageRows}</div>
        <div style="font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.4rem">Pull an image</div>
        <div class="d-flex flex-wrap gap-1 mb-2">
            <button class="btn btn-sm btn-outline-success" style="font-size:.72rem;padding:.1rem .45rem"
                data-wid="${safeWid}" data-image="${escapeHtml(ENGINE_IMAGE.dynamo)}" onclick="fillPullImage(this)">
                Dynamo engine image
            </button>
            <button class="btn btn-sm btn-outline-secondary" style="font-size:.72rem;padding:.1rem .45rem"
                data-wid="${safeWid}" data-image="${escapeHtml(ENGINE_IMAGE.vllm)}" onclick="fillPullImage(this)">
                vLLM (legacy)
            </button>
        </div>
        <div class="input-group input-group-sm mb-2">
            <input type="text" class="form-control font-monospace" id="pull-image-${safeWid}"
                placeholder="${escapeHtml(ENGINE_IMAGE.dynamo)}" value="${escapeHtml(ENGINE_IMAGE.dynamo)}" style="font-size:.8rem">
            <button class="btn btn-primary" id="pull-btn-${safeWid}" data-wid="${safeWid}"
                onclick="pullImageFromInput(this.dataset.wid)">
                <i class="fa-solid fa-download me-1"></i>Pull
            </button>
        </div>
        <div id="pull-output-wrap-${safeWid}" style="display:none">
            <div class="d-flex justify-content-between align-items-center mb-1">
                <span style="font-size:.75rem;color:#6b7280">Pull output</span>
                <button class="btn btn-sm" style="font-size:.7rem;padding:.05rem .4rem;color:#9ca3af"
                    data-wid="${safeWid}" onclick="clearPullOutput(this.dataset.wid)">
                    <i class="fa-solid fa-xmark me-1"></i>Clear
                </button>
            </div>
            <pre id="pull-output-${safeWid}" class="mb-0"
                style="max-height:220px;overflow-y:auto;background:#111;color:#86efac;font-size:.75rem;border-radius:.4rem;padding:.6rem .8rem"></pre>
        </div>
    `;
}

async function refreshImageList(wid) {
    const safeWid = escapeHtml(wid);
    const listEl = document.getElementById(`img-list-${safeWid}`);
    if (!listEl) return;

    epImages.delete(wid);
    listEl.innerHTML = '<div class="text-muted small py-1">Refreshing...</div>';
    try {
        const resp = await fetch(`/api/endpoints/${wid}/images`);
        if (resp.ok) epImages.set(wid, await resp.json());
        else epImages.set(wid, []);
    } catch (e) {
        epImages.set(wid, []);
    }

    const images = epImages.get(wid) || [];
    listEl.innerHTML = _renderImageRows(images, safeWid);
}

// Image list rows. The Dynamo runtime image is what `engine=dynamo` instances
// run, so it is called out as the engine image; vllm/vllm-openai is legacy.
function _renderImageRows(images, safeWid) {
    if (!images || images.length === 0) {
        return '<div class="text-muted small py-1">No engine images found on this host.</div>';
    }
    return images.map(img => {
        const safeName = escapeHtml(img.name);
        const name = String(img.name || '');
        let tag = '';
        if (name.startsWith('nvcr.io/nvidia/ai-dynamo/vllm-runtime')) {
            tag = '<span class="badge bg-success ms-2" style="font-size:.66rem">ENGINE IMAGE (DYNAMO)</span>';
        } else if (name.startsWith('vllm/vllm-openai')) {
            tag = '<span class="badge bg-secondary ms-2" style="font-size:.66rem">LEGACY (VLLM)</span>';
        }
        return `
            <div class="d-flex align-items-center justify-content-between gap-2 py-1 flex-wrap" style="border-bottom:1px solid #f1f5f9">
                <div class="text-break">
                    <span class="font-monospace small fw-semibold">${safeName}</span>
                    <span class="text-muted ms-2" style="font-size:.78rem">${escapeHtml(img.size || '')}</span>
                    ${tag}
                </div>
                <button class="btn btn-sm btn-outline-secondary" style="font-size:.72rem;padding:.1rem .45rem"
                    data-wid="${safeWid}" data-image="${safeName}"
                    onclick="pullImageFromBtn(this)">
                    <i class="fa-solid fa-arrow-rotate-right me-1"></i>Update
                </button>
            </div>`;
    }).join('');
}

window.clearPullOutput = function(wid) {
    const safeWid = escapeHtml(wid);
    const wrap = document.getElementById(`pull-output-wrap-${safeWid}`);
    const pre = document.getElementById(`pull-output-${safeWid}`);
    if (pre) pre.textContent = '';
    if (wrap) wrap.style.display = 'none';
};

async function renderModelPanelBody(wid, bodyEl) {
    if (!epModels.has(wid)) {
        bodyEl.innerHTML = '<div class="text-muted small py-1">Loading models...</div>';
        try {
            const resp = await fetch(`/api/endpoints/${wid}/models`);
            if (resp.ok) {
                epModels.set(wid, await resp.json());
            } else if (resp.status === 501) {
                bodyEl.innerHTML = `<div class="text-warning small py-2"><i class="fa-solid fa-triangle-exclamation me-1"></i>이 호스트의 워커 에이전트가 모델 관리를 지원하지 않습니다. 워커를 업데이트하세요.</div>`;
                return;
            } else {
                epModels.set(wid, []);
            }
        } catch (e) {
            epModels.set(wid, []);
        }
    }

    const models = epModels.get(wid) || [];
    const safeWid = escapeHtml(wid);

    const modelRows = _renderModelRows(models, safeWid);

    bodyEl.innerHTML = `
        <div class="mb-2" style="font-size:.8rem;font-weight:600;color:#374151;margin-top:.25rem">
            <i class="fa-solid fa-robot me-1"></i>Cached HF Models
        </div>
        <div class="mb-3" id="model-list-${safeWid}">${modelRows}</div>
        <div style="font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.4rem">Download Model</div>
        <div class="input-group input-group-sm mb-2">
            <input type="text" class="form-control font-monospace" id="model-id-${safeWid}"
                placeholder="Qwen/Qwen3-30B-A3B" style="font-size:.8rem">
            <button class="btn btn-primary" data-wid="${safeWid}" onclick="downloadModelFromInput(this.dataset.wid)">
                <i class="fa-solid fa-download me-1"></i>Download
            </button>
        </div>
        <div id="model-output-wrap-${safeWid}" style="display:none">
            <div class="d-flex justify-content-between align-items-center mb-1">
                <span style="font-size:.75rem;color:#6b7280">Download output</span>
                <button class="btn btn-sm" style="font-size:.7rem;padding:.05rem .4rem;color:#9ca3af"
                    data-wid="${safeWid}" onclick="clearModelOutput(this.dataset.wid)">
                    <i class="fa-solid fa-xmark me-1"></i>Clear
                </button>
            </div>
            <pre id="model-output-${safeWid}"
                style="max-height:220px;overflow-y:auto;background:#111;color:#86efac;font-size:.75rem;border-radius:.4rem;padding:.6rem .8rem;margin-bottom:0"></pre>
        </div>
    `;

    // Auto-reconnect to any active/recent download job
    _reconnectDownloadIfActive(wid);
}

async function refreshModelList(wid) {
    const safeWid = escapeHtml(wid);
    const listEl = document.getElementById(`model-list-${safeWid}`);
    if (!listEl) return;

    epModels.delete(wid);
    listEl.innerHTML = '<div class="text-muted small py-1">Refreshing...</div>';
    try {
        const resp = await fetch(`/api/endpoints/${wid}/models`);
        if (resp.ok) epModels.set(wid, await resp.json());
        else epModels.set(wid, []);
    } catch (e) {
        epModels.set(wid, []);
    }

    const models = epModels.get(wid) || [];
    listEl.innerHTML = _renderModelRows(models, safeWid);
}

function _renderModelRows(models, safeWid) {
    if (!models || models.length === 0) {
        return '<div class="text-muted small py-1">No HuggingFace models cached on this host.</div>';
    }
    return models.map(m => {
        const safeId = escapeHtml(m.repo_id);
        return `
            <div class="d-flex align-items-center justify-content-between py-1" style="border-bottom:1px solid #f1f5f9">
                <div>
                    <span class="font-monospace small fw-semibold">${safeId}</span>
                    <span class="text-muted ms-2" style="font-size:.78rem">${escapeHtml(m.size || '')}</span>
                </div>
                <div class="d-flex gap-1">
                    <button class="btn btn-sm btn-outline-primary" style="font-size:.72rem;padding:.1rem .45rem"
                        data-wid="${safeWid}" data-model="${safeId}" onclick="checkModelUpdate(this)"
                        title="Check for a newer revision and update only if available">
                        <i class="fa-solid fa-rotate me-1"></i>Check for update
                    </button>
                    <button class="btn btn-sm btn-outline-danger" style="font-size:.72rem;padding:.1rem .45rem"
                        data-wid="${safeWid}" data-model="${safeId}" onclick="forceRedownloadModel(this)"
                        title="Re-download every file from scratch, ignoring the cache">
                        <i class="fa-solid fa-arrow-rotate-right me-1"></i>Force Redownload
                    </button>
                </div>
            </div>`;
    }).join('');
}

window.downloadModelFromInput = async function(wid) {
    const safeWid = escapeHtml(wid);
    const modelId = document.getElementById(`model-id-${safeWid}`)?.value.trim();
    if (!modelId) return;
    await _doDownloadModel(wid, modelId, false);
};

window.checkModelUpdate = async function(btn) {
    await _doDownloadModel(btn.dataset.wid, btn.dataset.model, false);
};

window.forceRedownloadModel = async function(btn) {
    if (!confirm(`Force re-download "${btn.dataset.model}"?\nThis re-downloads every file from scratch, ignoring the cache.`)) return;
    await _doDownloadModel(btn.dataset.wid, btn.dataset.model, true);
};

async function _doDownloadModel(wid, modelId, force = false) {
    if (modelDownloadInProgress.has(wid)) return;

    const safeWid = escapeHtml(wid);
    const outputEl = document.getElementById(`model-output-${safeWid}`);
    const outputWrap = document.getElementById(`model-output-wrap-${safeWid}`);

    if (outputEl) outputEl.textContent = `${force ? 'Force re-downloading' : 'Checking'}: ${modelId}...\n`;
    if (outputWrap) outputWrap.style.display = '';

    let job_id;
    try {
        const resp = await fetch(`/api/endpoints/${wid}/models/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId, force: force })
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            if (outputEl) outputEl.textContent += `\n[Error] ${err.detail || resp.statusText}\n`;
            return;
        }
        const data = await resp.json();
        job_id = data.job_id;
    } catch (e) {
        if (outputEl) outputEl.textContent += `\n[Error] ${e}\n`;
        return;
    }

    localStorage.setItem(`model_job_${wid}`, JSON.stringify({ job_id, model_id: modelId }));
    await _streamJobLogs(wid, job_id, 0);
}

async function _streamJobLogs(wid, job_id, offset) {
    modelDownloadInProgress.add(wid);
    const safeWid = escapeHtml(wid);
    const outputEl = document.getElementById(`model-output-${safeWid}`);
    const outputWrap = document.getElementById(`model-output-wrap-${safeWid}`);
    if (outputWrap) outputWrap.style.display = '';
    document.querySelectorAll(`#model-panel-${safeWid} button`).forEach(el => { el.disabled = true; });

    try {
        const resp = await fetch(`/api/endpoints/${wid}/models/jobs/${job_id}/logs?offset=${offset}`);
        if (!resp.ok) {
            if (outputEl) outputEl.textContent += `\n[Error] HTTP ${resp.status}\n`;
            return;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (outputEl) {
                outputEl.textContent += decoder.decode(value, { stream: true });
                outputEl.scrollTop = outputEl.scrollHeight;
            }
        }
        await refreshModelList(wid);
        localStorage.removeItem(`model_job_${wid}`);
    } catch (e) {
        if (outputEl) outputEl.textContent += `\n[Error] ${e}\n`;
    } finally {
        modelDownloadInProgress.delete(wid);
        document.querySelectorAll(`#model-panel-${safeWid} button`).forEach(el => { el.disabled = false; });
    }
}

async function _reconnectDownloadIfActive(wid) {
    const stored = localStorage.getItem(`model_job_${wid}`);
    if (!stored) return;
    let jobInfo;
    try { jobInfo = JSON.parse(stored); } catch { localStorage.removeItem(`model_job_${wid}`); return; }

    const safeWid = escapeHtml(wid);
    const outputEl = document.getElementById(`model-output-${safeWid}`);
    const outputWrap = document.getElementById(`model-output-wrap-${safeWid}`);

    try {
        const resp = await fetch(`/api/endpoints/${wid}/models/jobs`);
        if (!resp.ok) { localStorage.removeItem(`model_job_${wid}`); return; }
        const jobs = await resp.json();
        const job = jobs.find(j => j.job_id === jobInfo.job_id);
        if (!job) { localStorage.removeItem(`model_job_${wid}`); return; }

        if (outputEl) outputEl.textContent = '';
        if (outputWrap) outputWrap.style.display = '';

        if (job.status === 'running') {
            if (outputEl) outputEl.textContent = `[Reconnecting to download: ${jobInfo.model_id}]\n`;
        }
        // Stream all buffered lines (offset=0), works for both running and finished jobs
        await _streamJobLogs(wid, jobInfo.job_id, 0);
    } catch (e) {
        localStorage.removeItem(`model_job_${wid}`);
    }
}

window.clearModelOutput = function(wid) {
    const safeWid = escapeHtml(wid);
    const wrap = document.getElementById(`model-output-wrap-${safeWid}`);
    const pre = document.getElementById(`model-output-${safeWid}`);
    if (pre) pre.textContent = '';
    if (wrap) wrap.style.display = 'none';
};

window.pullImageFromBtn = async function(btn) {
    const wid = btn.dataset.wid;
    const image = btn.dataset.image;
    await _doPullImage(wid, image);
};

window.fillPullImage = function(btn) {
    const input = document.getElementById(`pull-image-${btn.dataset.wid}`);
    if (input) input.value = btn.dataset.image;
};

window.pullImageFromInput = async function(wid) {
    const el = document.getElementById(`pull-image-${wid}`);
    const image = (el?.value.trim()) || ENGINE_IMAGE.dynamo;
    await _doPullImage(wid, image);
};

async function _doPullImage(wid, image) {
    if (pullInProgress.has(wid)) return;
    pullInProgress.add(wid);

    const safeWid = escapeHtml(wid);
    const outputEl = document.getElementById(`pull-output-${safeWid}`);
    const outputWrap = document.getElementById(`pull-output-wrap-${safeWid}`);

    if (outputEl) { outputEl.textContent = `Pulling ${image}...\n`; }
    if (outputWrap) { outputWrap.style.display = ''; }

    // Disable interactive elements in this panel
    document.querySelectorAll(`#img-panel-${safeWid} button`).forEach(el => { el.disabled = true; });

    try {
        const resp = await fetch(`/api/endpoints/${wid}/images/pull`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image })
        });
        if (!resp.ok) {
            if (outputEl) outputEl.textContent += `\n[Error] HTTP ${resp.status}\n`;
            return;
        }
        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            if (outputEl) { outputEl.textContent += decoder.decode(value, { stream: true }); outputEl.scrollTop = outputEl.scrollHeight; }
        }
        // Refresh only the image list — output stays visible
        await refreshImageList(wid);
    } catch (e) {
        if (outputEl) outputEl.textContent += `\n[Error] ${e}\n`;
    } finally {
        pullInProgress.delete(wid);
        document.querySelectorAll(`#img-panel-${safeWid} button`).forEach(el => { el.disabled = false; });
    }
}

// Suggestions for the (optional) image override: every image already pulled on
// an active host, plus the two defaults.
async function fetchAllWorkerImages() {
    const activeWorkers = Object.values(endpoints).filter(ep => ep.status === 'active');
    const allImages = new Set([ENGINE_IMAGE.dynamo, ENGINE_IMAGE.vllm]);
    await Promise.all(activeWorkers.map(async ep => {
        try {
            const resp = await fetch(`/api/endpoints/${ep.id}/images`);
            if (resp.ok) {
                const imgs = await resp.json();
                imgs.forEach(img => allImages.add(img.name));
            }
        } catch (e) {}
    }));
    const datalist = document.getElementById('deploy-image-datalist');
    if (datalist) datalist.innerHTML = [...allImages].map(n => `<option value="${escapeHtml(n)}">`).join('');
}

window.handleEngineChange = function() {
    const engine = document.getElementById('deployEngine')?.value || 'dynamo';
    const imgEl = document.getElementById('deployImage');
    if (imgEl) imgEl.placeholder = ENGINE_IMAGE[engine] || ENGINE_IMAGE.dynamo;
    // The legacy vLLM engine serves its own endpoint and is not routed by the
    // Dynamo frontend — say so instead of silently deploying something the API
    // page won't list.
    document.querySelectorAll('.deploy-legacy-warning').forEach(el => {
        el.style.display = engine === 'vllm' ? '' : 'none';
    });
    document.querySelectorAll('.deploy-dynamo-only').forEach(el => {
        el.style.display = engine === 'dynamo' ? '' : 'none';
    });
    fetchAllWorkerImages();
};

async function acceptEndpoint(id) {
    const btn = event ? event.currentTarget : null;
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Working...';
    }
    try {
        const res = await fetch(`/api/endpoints/${id}/accept`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ custom_name: id })
        });
        if (res.ok) {
            showAlert("success", `Host ${id} accepted successfully!`);
            fetchStatus();
        } else {
            showAlert("danger", "Failed to accept host");
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<i class="fa-solid fa-check me-1"></i> Accept';
            }
        }
    } catch (err) {
        console.error(err);
        showAlert("danger", "Network error while accepting endpoint");
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-check me-1"></i> Accept';
        }
    }
}

function resetDeployModal() {
    document.getElementById('deployForm').reset();
}

async function loadConfig(name) {
    // If not on the deploy page, navigate there with config param
    if (window.location.pathname !== '/deploy') {
        window.location.href = `/deploy?config=${encodeURIComponent(name)}`;
        return;
    }

    const conf = savedConfigs.find(c => c.name === name);
    if (!conf) return;

    document.getElementById('deployName').value = conf.name;
    document.getElementById('deployModel').value = conf.model;
    document.getElementById('deployServedModel').value = conf.served_model_name || '';

    const engineEl = document.getElementById('deployEngine');
    if (engineEl) engineEl.value = conf.engine || 'dynamo';

    const dtype = conf.deployment_type || conf.mode || 'replicas';
    if (dtype === 'tp') {
        document.getElementById('typeTp').checked = true;
    } else {
        document.getElementById('typeReplicas').checked = true;
    }

    document.getElementById('deployMaxLen').value = conf.max_len || '';
    document.getElementById('deployGpuUtil').value = conf.gpu_util || 0.9;
    document.getElementById('deployExtraArgs').value = conf.extra_args || '';
    // `image` is the contract field; `vllm_image` is what pre-Dynamo configs
    // stored, so saved configs from either era round-trip.
    const imgEl = document.getElementById('deployImage');
    if (imgEl) imgEl.value = conf.image || conf.vllm_image || '';
    const embedEl = document.getElementById('deployIsEmbedding');
    if (embedEl) embedEl.checked = !!conf.is_embedding;
    const reasonEl = document.getElementById('deployReasoningParser');
    if (reasonEl) reasonEl.value = conf.reasoning_parser || '';
    const toolEl = document.getElementById('deployToolCallParser');
    if (toolEl) toolEl.value = conf.tool_call_parser || '';
    if (typeof window.handleEngineChange === 'function') window.handleEngineChange();

    if (gpus.length === 0) await fetchStatus();

    const checkboxes = document.querySelectorAll('.gpu-checkbox');
    checkboxes.forEach(cb => { cb.checked = false; });

    if (typeof window.toggleDeployModeUI === 'function') window.toggleDeployModeUI();
    if (typeof window.validateDeployGpus === 'function') window.validateDeployGpus();
}

function getFormData() {
    const name = document.getElementById('deployName').value;
    if (!name) { showAlert("warning", "Please enter a deployment name."); return null; }

    const gpus = Array.from(document.querySelectorAll('.gpu-checkbox:checked')).map(cb => cb.value);

    if (gpus.length === 0) {
        showAlert("warning", "Please select at least one GPU.");
        return null;
    }

    const isTp = document.getElementById('typeTp').checked;
    if (isTp && ![1, 2, 4, 8].includes(gpus.length)) {
        showAlert("warning", "Tensor Parallelism requires 1, 2, 4, or 8 GPUs.");
        return null;
    }

    // POST /api/deploy body (docs/omniserve-ui-redesign.md, "API 계약").
    return {
        name: name,
        deployment_type: isTp ? "tp" : "replicas",

        model: document.getElementById('deployModel').value,
        served_model_name: document.getElementById('deployServedModel').value.trim() || null,
        engine: document.getElementById('deployEngine')?.value || 'dynamo',
        gpus: gpus,
        tp: isTp ? gpus.length : 1,
        max_len: parseInt(document.getElementById('deployMaxLen').value) || null,
        gpu_util: parseFloat(document.getElementById('deployGpuUtil').value) || 0.9,
        extra_args: document.getElementById('deployExtraArgs').value.trim() || null,
        image: document.getElementById('deployImage')?.value.trim() || null,
        is_embedding: !!document.getElementById('deployIsEmbedding')?.checked,
        // Empty = inferred from the model name by the worker.
        reasoning_parser: document.getElementById('deployReasoningParser')?.value.trim() || null,
        tool_call_parser: document.getElementById('deployToolCallParser')?.value.trim() || null
    };
}

async function saveConfiguration() {
    // We repurpose the deployment form data
    const config = getFormData();
    if (!config) return;

    const name = config.name;

    try {
        const res = await fetch('/api/configs', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name, config: config })
        });
        if (res.ok) {
            showAlert("success", "Configuration saved!");
            fetchStatus();
        }
    } catch (err) {
        console.error(err);
        showAlert("danger", "Failed to save configuration");
    }
}

// Global scope attachment for the HTML button
window.triggerSaveConfig = saveConfiguration;

async function submitDeployment() {
    const data = getFormData();
    if (!data) return;

    const modal = document.getElementById('deployModal');
    const isModal = !!(modal && modal.classList.contains('show'));

    // Disable every interactive element in scope
    const scope = isModal ? modal : document.body;
    const interactives = Array.from(scope.querySelectorAll('button, input, select, textarea, a.nav-link'));
    const prevDisabled = interactives.map(el => el.disabled);
    interactives.forEach(el => { el.disabled = true; });

    // Overlay — prevents any interaction and gives clear feedback
    const overlay = document.createElement('div');
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.45);z-index:99999;display:flex;align-items:center;justify-content:center';
    overlay.innerHTML = `
        <div style="background:#fff;border-radius:.75rem;padding:2rem 2.5rem;text-align:center;box-shadow:0 8px 32px rgba(0,0,0,.25);min-width:220px">
            <i class="fa-solid fa-spinner fa-spin" style="font-size:1.75rem;color:#2563eb"></i>
            <div style="margin-top:.85rem;font-weight:600;color:#111">배포 중...</div>
            <div style="margin-top:.3rem;font-size:.82rem;color:#6b7280">잠시만 기다려 주세요</div>
        </div>`;
    document.body.appendChild(overlay);

    const cleanup = () => {
        overlay.remove();
        interactives.forEach((el, i) => { el.disabled = prevDisabled[i]; });
    };

    try {
        const res = await fetch('/api/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (res.ok) {
            cleanup();
            if (window.location.pathname === '/deploy') {
                window.location.href = '/?deployed=1';
            } else {
                showAlert("success", "Deployment created successfully!");
                bootstrap.Modal.getInstance(modal)?.hide();
                fetchStatus();
            }
        } else {
            const err = await res.json();
            cleanup();
            showAlert("danger", "Failed to deploy: " + (err.detail || res.statusText));
        }
    } catch (err) {
        console.error(err);
        cleanup();
        showAlert("danger", "Network error while deploying");
    }
}

function _fadeRow(row) {
    if (row) { row.style.transition = 'opacity .2s'; row.style.opacity = '.3'; row.style.pointerEvents = 'none'; }
}
function _unfadeRow(row) {
    if (row) { row.style.opacity = ''; row.style.pointerEvents = ''; }
}

window.stopDeployment = async function(id, btn) {
    _stopping.add(id);
    const row = btn.closest('tr');
    _fadeRow(row);
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
    try {
        await fetch(`/api/stop/${id}`, { method: 'POST' });
    } catch(e) {
        console.error(e);
    } finally {
        await fetchStatus();
        _stopping.delete(id);
    }
};

window.stopReplica = async function(deployId, globalGpuId, btn) {
    const key = `${deployId}:${globalGpuId}`;
    _stopping.add(key);
    const row = btn.closest('tr');
    _fadeRow(row);
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
    try {
        await fetch(`/api/stop/${deployId}/gpu/${globalGpuId}`, { method: 'POST' });
    } catch(e) {
        console.error(e);
    } finally {
        await fetchStatus();
        _stopping.delete(key);
    }
};

window.stopEndpoint = async function(epKey, btn) {
    const actions = _endpointActions.get(epKey) || [];
    if (actions.length === 0) return;

    // Register all keys in _stopping so polling re-renders keep the faded state
    const keys = actions.map(({ dep, gpu, multi }) => multi ? `${dep.id}:${gpu}` : dep.id);
    keys.forEach(k => _stopping.add(k));

    document.querySelectorAll(`.ep-${epKey}`).forEach(r => _fadeRow(r));
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';

    try {
        await Promise.all(actions.map(({ dep, gpu, multi }) =>
            multi
                ? fetch(`/api/stop/${dep.id}/gpu/${gpu}`, { method: 'POST' })
                : fetch(`/api/stop/${dep.id}`, { method: 'POST' })
        ));
    } catch(e) {
        console.error(e);
    } finally {
        await fetchStatus();
        keys.forEach(k => _stopping.delete(k));
    }
};

window.viewLogs = function (deployId, containerName) {
    const url = containerName ? `/logs/${deployId}?container=${encodeURIComponent(containerName)}` : `/logs/${deployId}`;
    window.open(url, '_blank');
}

function exportConfigs() {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(savedConfigs, null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "omniserve_configs.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
}

function showAlert(type, message) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    document.getElementById('alert-container').appendChild(alertDiv);
    setTimeout(() => alertDiv.remove(), 5000);
}

async function deleteConfig(name) {
    if (!confirm('Are you sure you want to delete config: ' + name + '?')) return;
    try {
        const res = await fetch(`/api/configs/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (res.ok) {
            showAlert('success', 'Config deleted');
            fetchStatus();
        } else {
            showAlert('danger', 'Failed to delete config');
        }
    } catch (err) {
        showAlert('danger', 'Error: ' + err.message);
    }
}

async function resetEndpoint(workerId) {
    if (!confirm('Are you sure you want to reset and remove this endpoint?')) return;
    try {
        const res = await fetch(`/api/endpoints/${encodeURIComponent(workerId)}`, { method: 'DELETE' });
        if (res.ok) {
            showAlert('success', 'Endpoint removed');
            fetchStatus();
        } else {
            showAlert('danger', 'Failed to remove endpoint');
        }
    } catch (err) {
        showAlert('danger', 'Error: ' + err.message);
    }
}

async function renameEndpoint(workerId, currentName) {
    const newName = prompt('Enter new name for this endpoint:', currentName);
    if (!newName) return;
    try {
        const res = await fetch(`/api/endpoints/${workerId}/accept`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ custom_name: newName })
        });
        if (res.ok) {
            showAlert('success', 'Endpoint renamed to ' + newName);
            fetchStatus();
        } else {
            showAlert('danger', 'Failed to rename endpoint');
        }
    } catch (err) {
        showAlert('danger', 'Error: ' + err.message);
    }
}
