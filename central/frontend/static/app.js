// static/app.js
let gpus = [];
let deployments = [];
let savedConfigs = [];
let endpoints = {};
let proxyStats = {};

document.addEventListener('DOMContentLoaded', () => {
    fetchStatus();
    setInterval(fetchStatus, 5000);
    fetchProxyStats();
    setInterval(fetchProxyStats, 2000);

    if (new URLSearchParams(window.location.search).get('deployed') === '1') {
        history.replaceState({}, '', '/');
        setTimeout(() => showAlert('success', '배포가 성공적으로 시작되었습니다.'), 300);
    }
});

async function fetchProxyStats() {
    try {
        const res = await fetch('/api/proxy_stats');
        proxyStats = await res.json();
        if (document.getElementById('model-stats-container')) renderModelStats();
    } catch (err) {
        // proxy may not be up yet, ignore
    }
}

async function fetchStatus() {
    try {
        const [gpusRes, depsRes, confsRes, endpRes] = await Promise.all([
            fetch('/api/gpus'),
            fetch('/api/deployments'),
            fetch('/api/configs'),
            fetch('/api/endpoints')
        ]);
        gpus = await gpusRes.json();
        deployments = await depsRes.json();
        savedConfigs = await confsRes.json();
        endpoints = await endpRes.json();

        if (document.getElementById('gpu-cards-container') || document.getElementById('deployGpusGrid')) renderGPUs();
        if (document.getElementById('deployments-table-body')) renderDeployments();
        if (document.getElementById('saved-configs-container')) renderConfigs();
        if (document.getElementById('pending-endpoints-container')) renderEndpoints();
        if (document.getElementById('haproxy-map-list')) renderHaproxyMap();
        if (document.getElementById('gateway-models-list')) renderGateway();
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
        if (list) list.innerHTML = '<div class="col-12 text-muted">No active GPUs available. Make sure to accept pending endpoints.</div>';
        return;
    }

    gpus.forEach(gpu => {
        let memPercent = Math.min(100, Math.round((gpu.memory_used / gpu.memory_total) * 100));
        const barColor = memPercent > 85 ? '#ef4444' : memPercent > 60 ? '#f59e0b' : '#3b82f6';
        // Global GPU Card
        if (list) {
            list.innerHTML += `
                <div class="col-md-4 col-lg-3 mb-3">
                    <div class="card h-100">
                        <div class="card-body py-3 px-3">
                            <div class="d-flex justify-content-between align-items-start mb-1">
                                <div>
                                    <span class="fw-semibold small">GPU ${gpu.local_id}</span>
                                    <span class="badge bg-secondary ms-1" style="font-size:.8rem">${gpu.worker_name}</span>
                                </div>
                                <span class="fw-bold small" style="color:${barColor}">${memPercent}%</span>
                            </div>
                            <div class="text-muted mb-2" style="font-size:.82rem">${gpu.name}</div>
                            <div class="progress" style="height:5px;border-radius:3px;background:#e5e7eb">
                                <div style="width:${memPercent}%;background:${barColor};height:100%;border-radius:3px;transition:width .4s"></div>
                            </div>
                            <div class="d-flex justify-content-between mt-1" style="font-size:.82rem;color:#9ca3af">
                                <span>${(gpu.memory_used / 1024).toFixed(1)} GB used</span>
                                <span>${(gpu.memory_total / 1024).toFixed(0)} GB total</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Deploy Page GPU Checkboxes
        if (gpuGrid) {
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
                <label for="gpu-btn-${gpu.id}" class="gpu-checkbox-label">
                    <div style="font-size:.82rem;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">GPU ${gpu.local_id}</div>
                    <div style="font-size:.78rem;color:#6b7280;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${gpu.worker_name}</div>
                    <div style="height:3px;border-radius:2px;background:#e5e7eb;margin:4px 0 2px">
                        <div style="width:${memPercent}%;height:100%;border-radius:2px;background:${barColor}"></div>
                    </div>
                    <div style="font-size:.78rem;color:#9ca3af">${memPercent}%</div>
                    ${runningHtml}
                </label>
            `;
        }
    });

    // Run filter depending on current mode
    toggleDeployModeUI();
}

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

function renderModelStats() {
    const container = document.getElementById('model-stats-container');
    if (!container) return;

    const models = Object.keys(proxyStats);
    if (models.length === 0) {
        container.innerHTML = '<div class="col-12 text-muted small">No inference data yet.</div>';
        return;
    }

    const rows = models.map(name => {
        const s = proxyStats[name];
        const active = s.active_requests || 0;
        const rps = (s.req_per_sec || 0).toFixed(1);
        const activeBadge = active > 0
            ? `<span class="badge stat-active">${active} 진행 중</span>`
            : `<span class="badge stat-idle">대기</span>`;
        const rpsBadge = `<span class="badge stat-rps">${rps} req/s</span>`;
        return `
            <tr>
                <td class="font-monospace fw-semibold small text-truncate" style="max-width:260px" title="${name}">${name}</td>
                <td class="text-center">${activeBadge}</td>
                <td class="text-center">${rpsBadge}</td>
            </tr>`;
    }).join('');

    container.innerHTML = `
        <div class="col-12">
            <div class="card table-card">
                <table class="table table-hover">
                    <thead class="table-light">
                        <tr>
                            <th class="small text-muted fw-semibold">모델 (Served Name)</th>
                            <th class="small text-muted fw-semibold text-center">추론 진행</th>
                            <th class="small text-muted fw-semibold text-center">처리량</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        </div>`;
}

function renderDeployments() {
    const list = document.getElementById('deployments-table-body');
    if (!list) return;
    list.innerHTML = '';

    if (deployments.length === 0) {
        list.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No active deployments.</td></tr>';
        return;
    }

    deployments.forEach(dep => {
        const isRunning = dep.status === 'running';
        const statusDot = isRunning
            ? `<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block"></span><span class="small">Running</span></span>`
            : `<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:7px;height:7px;border-radius:50%;background:#f59e0b;display:inline-block"></span><span class="small text-muted">Starting</span></span>`;
        const dtype = (dep.deployment_type || '').toUpperCase();
        const engine = (dep.engine || 'vllm').toUpperCase();
        const dtypeColor = dtype === 'TP' ? 'bg-primary' : 'bg-secondary';
        const engineColor = engine === 'OLLAMA' ? 'bg-warning text-dark' : 'bg-dark';
        const gpuList = dep.gpus.map(g => `<span class="badge bg-light text-secondary border me-1" style="font-size:.8rem">${g}</span>`).join('');

        list.innerHTML += `
            <tr>
                <td>
                    <div class="fw-semibold small">${dep.name}</div>
                    <div class="font-monospace" style="font-size:.8rem;color:#9ca3af">${dep.id}</div>
                </td>
                <td>
                    <div class="small text-truncate" style="max-width:200px" title="${dep.model}">${dep.model}</div>
                    ${dep.served_model_name && dep.served_model_name !== dep.model ? `<div class="font-monospace text-muted text-truncate" style="font-size:.8rem;max-width:200px" title="${dep.served_model_name}">→ ${dep.served_model_name}</div>` : ''}
                </td>
                <td class="text-nowrap">
                    <span class="badge ${engineColor} me-1">${engine}</span>
                    <span class="badge ${dtypeColor}">${dtype}</span>
                </td>
                <td>${statusDot}</td>
                <td><div class="d-flex flex-wrap gap-1">${gpuList}</div></td>
                <td class="text-end text-nowrap">
                    <button onclick="viewLogs('${dep.id}')" class="btn btn-sm btn-outline-secondary me-1" title="Logs"><i class="fa-solid fa-terminal"></i></button>
                    <button id="stop-btn-${dep.id}" onclick="stopDeployment('${dep.id}')" class="btn btn-sm btn-outline-danger" title="Stop"><i class="fa-solid fa-stop"></i></button>
                </td>
                </td>
            </tr>
        `;
    });
}



function renderGateway() {
    const list = document.getElementById('gateway-models-list');
    const example = document.getElementById('gateway-curl-example');
    if (!list || !example) return;

    list.innerHTML = '';

    // Aggregate unique models
    const activeModels = [...new Set(deployments.filter(d => d.status === 'running').map(d => d.model))];

    if (activeModels.length === 0) {
        list.innerHTML = '<li class="list-group-item text-muted">No models currenly deployed.</li>';
        document.getElementById('gateway-curl-example').textContent = "No models available.";
        return;
    }

    activeModels.forEach(model => {
        // Sum the actual node counts (e.g. 2 replicas = 2 nodes)
        let count = 0;
        deployments.filter(d => d.model === model && d.status === 'running').forEach(d => {
            count += d.nodes ? d.nodes.length : 0;
        });

        list.innerHTML += `
            <li class="list-group-item d-flex justify-content-between align-items-center cursor-pointer list-group-item-action" onclick="updateCurlExample('${model}')">
                <span class="font-monospace fw-bold">${model}</span>
                <span class="badge bg-primary rounded-pill" title="${count} nodes serving this model">${count} Nodes</span>
            </li>
        `;
    });

    // Auto-select first if none selected or current is gone
    const currentSelection = document.getElementById('gateway-test-model-name')?.textContent;
    if (activeModels.length > 0) {
        if (!currentSelection || !activeModels.includes(currentSelection)) {
            updateCurlExample(activeModels[0]);
        }
    } else {
        const testCard = document.getElementById('gateway-test-card');
        if (testCard) testCard.style.display = 'none';
    }
}

window.updateCurlExample = function (modelName) {
    const pre = document.getElementById('gateway-curl-example');
    pre.textContent = `curl -X POST http://${window.location.hostname}:11434/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer bislaprom3#" \\
  -d '{
    "model": "${modelName}",
    "messages": [
      {"role": "user", "content": "How are you?"}
    ]
  }'`;

    // Update interactive Gateway UI
    document.getElementById('gateway-proxy-info').textContent = `Proxy at: ${window.location.hostname}:11434`;
    document.getElementById('gateway-test-card').style.display = 'block';
    document.getElementById('gateway-test-model-name').textContent = modelName;
}

window.testGatewayEndpoint = async function () {
    const modelName = document.getElementById('gateway-test-model-name').textContent;
    const prompt = document.getElementById('gateway-test-prompt').value;
    const concurrencyInput = document.getElementById('gateway-test-concurrency');
    const concurrency = parseInt(concurrencyInput ? concurrencyInput.value : 1) || 1;
    const responseEl = document.getElementById('gateway-test-response');
    const btn = document.getElementById('gateway-test-btn');

    if (!prompt) return;

    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Sending...';
    responseEl.textContent = `Dispatching ${concurrency} concurrent request(s)...`;

    const payload = {
        model: modelName,
        messages: [{ role: "user", content: prompt }]
    };

    const startTime = Date.now();
    const requests = [];

    for (let i = 0; i < concurrency; i++) {
        requests.push(
            fetch(`http://${window.location.hostname}:11434/v1/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer bislaprom3#'
                },
                body: JSON.stringify(payload)
            }).then(async res => {
                const text = await res.text();
                let parsed = null;
                try { parsed = JSON.parse(text); } catch (e) { }

                if (res.ok) {
                    if (parsed && parsed.choices && parsed.choices.length > 0) {
                        return { ok: true, id: i + 1, type: 'success', content: parsed.choices[0].message.content };
                    }
                    return { ok: true, id: i + 1, type: 'raw', content: text };
                }
                return { ok: false, id: i + 1, type: 'error', status: res.status, content: text };
            }).catch(err => {
                return { ok: false, id: i + 1, type: 'network_error', content: err.message };
            })
        );
    }

    try {
        const results = await Promise.all(requests);
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);

        let outputText = `[Batch Completed in ${elapsed}s]\n\n`;
        results.forEach(r => {
            if (r.ok) {
                outputText += `--- Request #${r.id} (Success) ---\n${r.content}\n\n`;
            } else {
                outputText += `--- Request #${r.id} (FAILED) ---\n${r.status ? 'Status: ' + r.status : 'Network Error'}\n${r.content}\n\n`;
            }
        });

        responseEl.textContent = outputText.trim();

    } catch (err) {
        responseEl.textContent = `Critical Error running batch: ${err.message}`;
    }

    btn.disabled = false;
    btn.innerHTML = '<i class="fa-solid fa-paper-plane me-2"></i>Send Request(s)';
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
        const engine = (conf.engine || 'vllm').toUpperCase();
        const dtypeColor = dtype === 'TP' ? 'bg-primary' : 'bg-secondary';
        const engineColor = engine === 'OLLAMA' ? 'bg-warning text-dark' : 'bg-dark';
        return `
            <tr class="config-row">
                <td>
                    <div class="fw-semibold small">${conf.name}</div>
                    <div class="font-monospace text-muted text-truncate" style="font-size:.82rem;max-width:320px" title="${conf.model}">${conf.model}</div>
                </td>
                <td class="text-nowrap">
                    <span class="badge ${dtypeColor} me-1">${dtype}</span>
                    <span class="badge ${engineColor}">${engine}</span>
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

    pendingList.innerHTML = '';
    activeList.innerHTML = '';

    let hasPending = false;
    let hasActive = false;

    // Cache user's active inputs for accept-name fields before replacing DOM to prevent wipeouts
    const inputCache = {};
    document.querySelectorAll('[id^=accept-name-]').forEach(el => {
        inputCache[el.id] = el.value;
    });

    const activeId = document.activeElement ? document.activeElement.id : null;
    let selectionStart = null, selectionEnd = null;
    if (activeId && document.activeElement.tagName === 'INPUT') {
        try {
            selectionStart = document.activeElement.selectionStart;
            selectionEnd = document.activeElement.selectionEnd;
        } catch (e) { }
    }

    pendingList.innerHTML = '';
    activeList.innerHTML = '';

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
                                    <div class="text-muted small">${ep.gpus.length}개 GPU · ${ep.host}:${ep.port}</div>
                                </div>
                                <span class="badge bg-warning text-dark">Pending</span>
                            </div>
                            <div class="text-muted mb-3 font-monospace small">${ep.id}</div>
                            <div class="input-group">
                                <input type="text" id="accept-name-${ep.id}" class="form-control" placeholder="노드 이름 지정" value="${inputCache[`accept-name-${ep.id}`] || ep.id}">
                                <button class="btn btn-success" onclick="acceptEndpoint('${ep.id}')">
                                    <i class="fa-solid fa-check me-1"></i>Accept
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else if (ep.status === 'active') {
            hasActive = true;
            activeRows.push(`
                <tr>
                    <td>
                        <span class="d-inline-flex align-items-center gap-2">
                            <span style="width:8px;height:8px;border-radius:50%;background:#22c55e;flex-shrink:0;display:inline-block"></span>
                            <span class="fw-semibold">${ep.name}</span>
                        </span>
                    </td>
                    <td class="font-monospace text-muted">${ep.host}:${ep.port}</td>
                    <td><span class="badge bg-secondary">${ep.gpus.length} GPUs</span></td>
                    <td class="font-monospace text-muted small">${ep.id}</td>
                    <td class="text-end text-nowrap">
                        <button class="btn btn-sm btn-outline-secondary me-1" onclick="renameEndpoint('${ep.id}', '${ep.name}')">
                            <i class="fa-solid fa-pen me-1"></i>이름 변경
                        </button>
                        <button class="btn btn-sm btn-outline-danger" onclick="resetEndpoint('${ep.id}')">
                            <i class="fa-solid fa-trash me-1"></i>제거
                        </button>
                    </td>
                </tr>
            `);
        }
    });

    if (!hasPending) {
        pendingList.innerHTML = '<div class="col-12 text-muted">등록 대기 중인 노드가 없습니다.</div>';
    }

    if (!hasActive) {
        activeList.innerHTML = '<p class="text-muted p-3 mb-0">활성화된 엔드포인트가 없습니다.</p>';
    } else {
        activeList.innerHTML = `
            <table class="table table-hover mb-0">
                <thead class="table-light">
                    <tr>
                        <th class="text-muted fw-semibold small">노드 이름</th>
                        <th class="text-muted fw-semibold small">호스트</th>
                        <th class="text-muted fw-semibold small">GPU</th>
                        <th class="text-muted fw-semibold small">ID</th>
                        <th class="text-muted fw-semibold small text-end">액션</th>
                    </tr>
                </thead>
                <tbody>${activeRows.join('')}</tbody>
            </table>
        `;
    }

    // Restore focus and cursor position if the user was typing
    if (activeId) {
        const el = document.getElementById(activeId);
        if (el) {
            el.focus();
            if (selectionStart !== null && selectionEnd !== null && el.setSelectionRange) {
                try {
                    el.setSelectionRange(selectionStart, selectionEnd);
                } catch (e) { }
            }
        }
    }
}

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
            showAlert("success", `Endpoint ${id} accepted successfully!`);
            fetchStatus();
        } else {
            showAlert("danger", "Failed to accept endpoint");
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
    if (engineEl) engineEl.value = conf.engine || 'vllm';

    const dtype = conf.deployment_type || conf.mode || 'replicas';
    if (dtype === 'tp') {
        document.getElementById('typeTp').checked = true;
    } else {
        document.getElementById('typeReplicas').checked = true;
    }

    document.getElementById('deployMaxLen').value = conf.max_len || '';
    document.getElementById('deployGpuUtil').value = conf.gpu_util || 0.9;
    document.getElementById('deployExtraArgs').value = conf.extra_args || '';

    if (gpus.length === 0) await fetchStatus();

    const checkboxes = document.querySelectorAll('.gpu-checkbox');
    const gpusToSelect = conf.gpus || conf.gpuIds || [];
    checkboxes.forEach(cb => {
        cb.checked = gpusToSelect.includes(cb.value);
    });

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

    return {
        name: name,
        deployment_type: isTp ? "tp" : "replicas",

        model: document.getElementById('deployModel').value,
        served_model_name: document.getElementById('deployServedModel').value.trim() || null,
        engine: document.getElementById('deployEngine') ? document.getElementById('deployEngine').value : 'vllm',
        gpus: gpus,
        tp: isTp ? gpus.length : 1,
        max_len: parseInt(document.getElementById('deployMaxLen').value) || null,
        gpu_util: parseFloat(document.getElementById('deployGpuUtil').value) || 0.9,
        extra_args: document.getElementById('deployExtraArgs').value.trim() || null
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

    const btn = document.querySelector('button[onclick="submitDeployment()"]');
    const originalText = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin me-2"></i>Deploying...';

    try {
        const res = await fetch('/api/deploy', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        if (res.ok) {
            if (window.location.pathname === '/deploy') {
                window.location.href = '/?deployed=1';
            } else {
                showAlert("success", "Deployment created successfully!");
                bootstrap.Modal.getInstance(document.getElementById('deployModal'))?.hide();
                fetchStatus();
            }
        } else {
            const err = await res.json();
            showAlert("danger", "Failed to deploy: " + (err.detail || res.statusText));
        }
    } catch (err) {
        console.error(err);
        showAlert("danger", "Network error while deploying");
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

window.stopDeployment = async function (id) {
    if (!confirm("Are you sure you want to stop this deployment?")) return;

    const btn = document.getElementById(`stop-btn-${id}`);
    if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i>';
    }

    try {
        await fetch(`/api/stop/${id}`, { method: 'POST' });
        setTimeout(fetchStatus, 1000);
    } catch (err) {
        console.error(err);
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<i class="fa-solid fa-stop"></i>';
        }
    }
}

window.viewLogs = function (deployId) {
    window.open(`/logs/${deployId}`, '_blank');
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
