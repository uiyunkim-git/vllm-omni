// static/app.js
let gpus = [];
let deployments = [];
let savedConfigs = [];
let endpoints = {};
let proxyStats = {};
let depGroupCollapsed = new Set();
let gpuGroupCollapsed = new Set();
const epImages = new Map();       // worker_id -> images[] cache
const pullInProgress = new Set(); // worker_ids currently pulling an image
const epModels = new Map();       // worker_id -> models[] cache
const modelDownloadInProgress = new Set();
const _endpointActions = new Map();
const _stopping = new Set(); // keys: dep id or "depId:gpu" for replicas
let throughputChart = null;
let selectedThroughputModel = null; // null = show all models

document.addEventListener('DOMContentLoaded', () => {
    fetchStatus();
    setInterval(fetchStatus, 5000);
    fetchProxyStats();
    setInterval(fetchProxyStats, 2000);

    if (new URLSearchParams(window.location.search).get('deployed') === '1') {
        history.replaceState({}, '', '/');
        setTimeout(() => showAlert('success', '배포가 성공적으로 시작되었습니다.'), 300);
    }

    // Populate image datalist when deploy modal opens
    document.getElementById('deployModal')?.addEventListener('show.bs.modal', () => {
        if (document.getElementById('deployEngine')?.value === 'vllm') fetchAllWorkerImages();
    });
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

            const gpuCards = group.gpus.map(gpu => {
                const memPercent = Math.min(100, Math.round((gpu.memory_used / gpu.memory_total) * 100));
                const barColor = memPercent > 85 ? '#ef4444' : memPercent > 60 ? '#f59e0b' : '#3b82f6';
                return `
                    <div class="col-md-4 col-lg-3 mb-3">
                        <div class="card h-100">
                            <div class="card-body py-3 px-3">
                                <div class="d-flex justify-content-between align-items-start mb-1">
                                    <span class="fw-semibold small">GPU ${gpu.local_id}</span>
                                    <span class="fw-bold small" style="color:${barColor}">${memPercent}%</span>
                                </div>
                                <div class="text-muted mb-2" style="font-size:.82rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${gpu.name}">${gpu.name}</div>
                                <div class="progress mb-1" style="height:5px;border-radius:3px;background:#e5e7eb">
                                    <div style="width:${memPercent}%;background:${barColor};height:100%;border-radius:3px;transition:width .4s"></div>
                                </div>
                                <div class="d-flex justify-content-between mt-1" style="font-size:.78rem;color:#9ca3af">
                                    <span>${(gpu.memory_used / 1024).toFixed(1)} GB used</span>
                                    <span>${(gpu.memory_total / 1024).toFixed(0)} GB total</span>
                                </div>
                                ${gpu.utilization != null ? `<div class="mt-1" style="font-size:.78rem;color:#9ca3af">Compute: ${gpu.utilization}%</div>` : ''}
                            </div>
                        </div>
                    </div>`;
            }).join('');

            list.innerHTML += `
                <div class="col-12 mb-1">
                    <div class="d-flex align-items-center gap-2 px-1 mb-2" style="cursor:pointer;user-select:none" data-wid="${escapeHtml(wid)}" onclick="toggleGpuGroup(this.dataset.wid)">
                        <i id="${gid}-icon" class="fa-solid fa-chevron-down" style="font-size:.72rem;color:#6b7280;transition:transform .2s;${iconTransform}"></i>
                        <span class="fw-semibold">${escapeHtml(group.name)}</span>
                        <span class="badge bg-secondary" style="font-size:.75rem">${group.gpus.length} GPU${group.gpus.length > 1 ? 's' : ''}</span>
                        <span style="font-size:.8rem;color:${avgColor};font-weight:600">${avgMem}% avg mem</span>
                        <div style="flex:1;height:1px;background:#e2e8f0"></div>
                    </div>
                    <div class="row" id="${gid}-cards" style="${cardsDisplay}">
                        ${gpuCards}
                    </div>
                </div>`;
        });
    }

    gpus.forEach(gpu => {
        let memPercent = Math.min(100, Math.round((gpu.memory_used / gpu.memory_total) * 100));

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

function formatRate(value) {
    const numeric = Number(value) || 0;
    return numeric >= 100 ? numeric.toFixed(0) : numeric.toFixed(1);
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

function getThroughputHistory(stats) {
    if (Array.isArray(stats?.throughput_history)) return stats.throughput_history;
    if (Array.isArray(stats?.history)) return stats.history;
    return [];
}

function chartTimestampToMs(value) {
    const numeric = Number(value) || 0;
    return numeric > 100000000000 ? numeric : numeric * 1000;
}

function formatChartTime(value) {
    return new Date(Number(value)).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function renderModelStats() {
    const container = document.getElementById('model-stats-container');
    if (!container) return;

    const readyModelNames = new Set(
        deployments.flatMap(d => (d.nodes || []).some(n => n.is_healthy)
            ? [d.served_model_name || d.model]
            : []
        )
    );
    const models = Object.keys(proxyStats).filter(m => readyModelNames.has(m)).sort((a, b) => a.localeCompare(b));
    if (models.length === 0) {
        if (throughputChart) { throughputChart.destroy(); throughputChart = null; }
        selectedThroughputModel = null;
        container.innerHTML = '<div class="col-12 text-muted small">No inference data yet.</div>';
        return;
    }

    // Reset selection if the selected model is no longer in the list
    if (selectedThroughputModel && !models.includes(selectedThroughputModel)) {
        selectedThroughputModel = null;
        if (throughputChart) { throughputChart.destroy(); throughputChart = null; }
    }

    if (!document.getElementById('model-stats-table-body') || !document.getElementById('throughput-chart')) {
        if (throughputChart) { throughputChart.destroy(); throughputChart = null; }
        container.innerHTML = `
            <div class="col-12 col-xl-5 mb-3 mb-xl-0">
                <div class="card table-card h-100" style="overflow:hidden">
                    <div class="card-header d-flex justify-content-between align-items-center py-2">
                        <span class="small fw-semibold text-muted">모델 목록</span>
                        <button id="throughput-all-btn" onclick="selectThroughputModel(null)" class="btn btn-sm" style="font-size:.72rem;padding:.15rem .55rem">전체 보기</button>
                    </div>
                    <table class="table table-hover mb-0">
                        <thead class="table-light">
                            <tr>
                                <th class="small text-muted fw-semibold">모델 (Served Name)</th>
                                <th class="small text-muted fw-semibold text-center">추론 진행</th>
                                <th class="small text-muted fw-semibold text-center">처리량<br><span class="fw-normal">30초 평균</span></th>
                            </tr>
                        </thead>
                        <tbody id="model-stats-table-body"></tbody>
                    </table>
                </div>
            </div>
            <div class="col-12 col-xl-7">
                <div class="card throughput-chart-card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <span id="throughput-chart-title">처리량 추이</span>
                        <span class="small text-muted">최근 15분 · 30초 평균</span>
                    </div>
                    <div class="card-body">
                        <div class="throughput-chart-wrap">
                            <canvas id="throughput-chart"></canvas>
                        </div>
                    </div>
                </div>
            </div>`;
    }

    // Update "전체 보기" button style
    const allBtn = document.getElementById('throughput-all-btn');
    if (allBtn) {
        if (selectedThroughputModel === null) {
            allBtn.className = 'btn btn-sm btn-primary';
        } else {
            allBtn.className = 'btn btn-sm btn-outline-secondary';
        }
    }

    // Update chart title
    const chartTitle = document.getElementById('throughput-chart-title');
    if (chartTitle) {
        chartTitle.textContent = selectedThroughputModel ?? '처리량 추이';
    }

    const rows = models.map(name => {
        const s = proxyStats[name] || {};
        const active = s.active_requests || 0;
        const avgRps = s.req_per_sec_avg_30s ?? s.req_per_sec ?? 0;
        const safeName = escapeHtml(name);
        const isSelected = selectedThroughputModel === name;
        const rowStyle = isSelected
            ? 'cursor:pointer;background:#eff6ff;border-left:3px solid #2563eb'
            : 'cursor:pointer';
        const activeBadge = active > 0
            ? `<span class="badge stat-active">${active} 진행 중</span>`
            : `<span class="badge stat-idle">대기</span>`;
        const rpsBadge = `<span class="badge stat-rps">${formatRate(avgRps)} req/s</span>`;
        return `
            <tr style="${rowStyle}" data-model="${safeName}" onclick="selectThroughputModel(this.dataset.model)">
                <td class="font-monospace fw-semibold small text-truncate" style="max-width:220px" title="${safeName}">${safeName}</td>
                <td class="text-center">${activeBadge}</td>
                <td class="text-center">${rpsBadge}</td>
            </tr>`;
    }).join('');

    const tbody = document.getElementById('model-stats-table-body');
    if (tbody) tbody.innerHTML = rows;

    const chartModels = selectedThroughputModel ? [selectedThroughputModel] : models;
    renderThroughputChart(chartModels);
}

window.selectThroughputModel = function(name) {
    if (selectedThroughputModel === name) return;
    selectedThroughputModel = name;
    if (throughputChart) { throughputChart.destroy(); throughputChart = null; }
    renderModelStats();
};

function renderThroughputChart(models) {
    const canvas = document.getElementById('throughput-chart');
    if (!canvas || typeof Chart === 'undefined') return;

    const now = Date.now();
    const historyWindowMs = 15 * 60 * 1000;
    const colors = ['#2563eb', '#059669', '#dc2626', '#7c3aed', '#ea580c', '#0891b2', '#be123c', '#4d7c0f'];

    const latestSampleAt = models.reduce((latest, name) => {
        const history = getThroughputHistory(proxyStats[name] || {});
        const last = history.length ? chartTimestampToMs(history[history.length - 1].ts ?? history[history.length - 1].timestamp) : 0;
        return Math.max(latest, last || 0);
    }, 0);
    const xMax = Math.max(now, latestSampleAt) + 1000;
    const xMin = xMax - historyWindowMs;

    const datasets = models.map((name, idx) => {
        const stats = proxyStats[name] || {};
        const history = getThroughputHistory(stats);
        let points = history.map(sample => ({
            x: chartTimestampToMs(sample.ts ?? sample.timestamp),
            y: Number(sample.req_per_sec_avg_30s ?? sample.req_per_sec ?? 0) || 0
        })).filter(point => point.x >= xMin && point.x <= xMax);

        points.sort((a, b) => a.x - b.x);

        if (points.length === 0) {
            points = [
                { x: xMin, y: 0 },
                { x: xMax, y: Number(stats.req_per_sec_avg_30s ?? stats.req_per_sec ?? 0) || 0 }
            ];
        } else if (points[0].x > xMin) {
            const firstPoint = points[0];
            const zeroBeforeFirst = Math.max(xMin, firstPoint.x - 1000);
            points.unshift({ x: zeroBeforeFirst, y: 0 });
            if (zeroBeforeFirst > xMin) points.unshift({ x: xMin, y: 0 });
        }

        const lastPoint = points[points.length - 1];
        if (lastPoint.x < xMax) {
            points.push({ x: xMax, y: lastPoint.y });
        }

        const color = colors[idx % colors.length];
        const showPoints = points.length < 120;
        return {
            label: name,
            data: points,
            borderColor: color,
            backgroundColor: color,
            borderWidth: 2.5,
            pointRadius: showPoints ? 2 : 0,
            pointHoverRadius: 4,
            pointHitRadius: 8,
            tension: 0.25,
            spanGaps: true
        };
    });

    const chartData = { datasets };

    if (throughputChart) {
        throughputChart.data = chartData;
        throughputChart.options.scales.x.min = xMin;
        throughputChart.options.scales.x.max = xMax;
        throughputChart.update('none');
        return;
    }

    throughputChart = new Chart(canvas.getContext('2d'), {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            parsing: false,
            normalized: true,
            interaction: { mode: 'nearest', intersect: false },
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { boxWidth: 10, usePointStyle: true }
                },
                tooltip: {
                    callbacks: {
                        title: items => items.length ? formatChartTime(items[0].parsed.x) : '',
                        label: item => `${item.dataset.label}: ${formatRate(item.parsed.y)} req/s`
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: xMin,
                    max: xMax,
                    grid: { color: '#eef2f7' },
                    ticks: {
                        maxTicksLimit: 6,
                        callback: value => formatChartTime(value)
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: { color: '#eef2f7' },
                    ticks: {
                        precision: 0,
                        callback: value => `${formatRate(value)} req/s`
                    }
                }
            }
        }
    });
}

function renderDeployments() {
    const list = document.getElementById('deployments-table-body');
    if (!list) return;
    list.innerHTML = '';

    if (deployments.length === 0) {
        list.innerHTML = '<tr><td colspan="6" class="text-center text-muted py-4">No active deployments.</td></tr>';
        return;
    }

    // Group by served_model_name (fallback to model)
    const groupMap = new Map();
    deployments.forEach(dep => {
        const key = dep.served_model_name || dep.model;
        if (!groupMap.has(key)) groupMap.set(key, []);
        groupMap.get(key).push(dep);
    });

    groupMap.forEach((deps, servedName) => {
        // Deterministic gid so collapse state survives re-renders
        const gid = 'dg_' + servedName.replace(/[^a-zA-Z0-9]/g, '_');
        const collapsed = depGroupCollapsed.has(gid);
        const rowDisplay = collapsed ? 'display:none' : '';

        const allRunning = deps.every(d => (d.nodes || []).length > 0 && (d.nodes || []).every(n => n.is_healthy));
        const totalGpuCount = [...new Set(deps.flatMap(d => d.gpus || []))].length;
        const depIds = deps.map(d => d.id).join(',');
        const replicaCount = deps.reduce((sum, d) =>
            (d.deployment_type || '').toLowerCase() === 'replicas' ? sum + (d.gpus || []).length : sum + 1, 0);

        const groupStatusHtml = allRunning
            ? `<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block"></span><span class="small">Running</span></span>`
            : `<span style="display:inline-flex;align-items:center;gap:4px"><span style="width:7px;height:7px;border-radius:50%;background:#f59e0b;display:inline-block"></span><span class="small text-muted">Starting</span></span>`;

        const iconTransform = collapsed ? 'transform:rotate(-90deg)' : '';
        list.innerHTML += `
            <tr style="background:#f1f5f9;cursor:pointer" onclick="toggleDepGroup('${gid}')">
                <td colspan="3">
                    <div class="d-flex align-items-center gap-2">
                        <i class="fa-solid fa-chevron-down" id="${gid}-icon" style="font-size:.72rem;transition:transform .2s;color:#6b7280;${iconTransform}"></i>
                        <span class="fw-semibold">${servedName}</span>
                        <span class="badge bg-secondary" style="font-size:.75rem">${replicaCount} replica${replicaCount > 1 ? 's' : ''}</span>
                        ${groupStatusHtml}
                    </div>
                </td>
                <td class="text-muted small">${totalGpuCount} GPU${totalGpuCount !== 1 ? 's' : ''} total</td>
                <td></td>
                <td></td>
            </tr>
        `;

        // Flatten instances to GPU level, tag each with its worker (endpoint)
        const instances = [];
        deps.forEach(dep => {
            const isMultiReplica = (dep.deployment_type || '').toLowerCase() === 'replicas' && (dep.gpus || []).length > 1;
            if (isMultiReplica) {
                (dep.gpus || []).forEach(gpu => {
                    const lastDash = gpu.lastIndexOf('-');
                    const wid = lastDash >= 0 ? gpu.substring(0, lastDash) : gpu;
                    instances.push({ dep, gpu, wid, multi: true });
                });
            } else {
                const firstGpu = (dep.gpus || [])[0] || '';
                const lastDash = firstGpu.lastIndexOf('-');
                const wid = lastDash >= 0 ? firstGpu.substring(0, lastDash) : firstGpu;
                instances.push({ dep, gpu: firstGpu, wid, multi: false });
            }
        });
        instances.sort((a, b) => a.wid.localeCompare(b.wid));

        // Group by endpoint
        const endpointMap = new Map();
        instances.forEach(inst => {
            if (!endpointMap.has(inst.wid)) endpointMap.set(inst.wid, []);
            endpointMap.get(inst.wid).push(inst);
        });

        // Render: one section per endpoint
        endpointMap.forEach((epInsts, wid) => {
            const epKey = `${gid}_${wid}`.replace(/[^a-zA-Z0-9_]/g, '_');
            _endpointActions.set(epKey, epInsts);

            // Endpoint separator row with Stop All button
            list.innerHTML += `
                <tr class="${gid}-row ep-${epKey}" style="${rowDisplay}">
                    <td colspan="6" style="padding:.25rem .5rem .25rem 2rem;background:#f8fafc;border-top:1px solid #e2e8f0">
                        <div style="display:flex;align-items:center;justify-content:space-between">
                            <span style="font-size:.75rem;font-weight:700;color:#94a3b8;letter-spacing:.06em;text-transform:uppercase">${wid}</span>
                            <button onclick="stopEndpoint('${epKey}',this)" class="btn btn-outline-danger" style="font-size:.72rem;padding:.1rem .45rem;line-height:1.5" title="이 endpoint의 모든 replica 종료">
                                <i class="fa-solid fa-stop me-1"></i>Stop All
                            </button>
                        </div>
                    </td>
                </tr>
            `;

            // Individual instance rows
            epInsts.forEach(({ dep, gpu, multi }) => {
                const dtype = (dep.deployment_type || '').toUpperCase();
                const engine = (dep.engine || 'vllm').toUpperCase();
                const dtypeColor = dtype === 'TP' ? 'bg-primary' : 'bg-secondary';
                const engineColor = engine === 'OLLAMA' ? 'bg-warning text-dark' : 'bg-dark';

                if (multi) {
                    const lastDash = gpu.lastIndexOf('-');
                    const gpuWid = gpu.substring(0, lastDash);
                    const gpuGid = gpu.substring(lastDash + 1);
                    const containerName = `vllm_${dep.id}_${gpuWid}_${gpuGid}`;
                    const node = (dep.nodes || []).find(n => n.name === containerName);
                    const isRunning = node ? !!node.is_healthy : false;
                    const statusDot = isRunning
                        ? `<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block"></span><span class="small">Running</span></span>`
                        : `<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:7px;height:7px;border-radius:50%;background:#f59e0b;display:inline-block"></span><span class="small text-muted">Starting</span></span>`;
                    const isStopping = _stopping.has(`${dep.id}:${gpu}`);
                    const rowStyle = (rowDisplay || '') + (isStopping ? ';opacity:.3;pointer-events:none' : '');
                    const stopBtn = isStopping
                        ? `<button class="btn btn-sm btn-outline-danger" disabled><i class="fa-solid fa-spinner fa-spin"></i></button>`
                        : `<button onclick="stopReplica('${dep.id}','${gpu}',this)" class="btn btn-sm btn-outline-danger" title="Stop this replica"><i class="fa-solid fa-stop"></i></button>`;
                    list.innerHTML += `
                        <tr class="${gid}-row ep-${epKey}" style="${rowStyle}">
                            <td style="padding-left:3rem">
                                <div class="fw-semibold small">${dep.name}</div>
                                <div class="font-monospace" style="font-size:.8rem;color:#9ca3af">${dep.id}</div>
                            </td>
                            <td>
                                <div class="small text-truncate" style="max-width:200px" title="${dep.model}">${dep.model}</div>
                                ${dep.served_model_name && dep.served_model_name !== dep.model ? `<div class="small text-truncate" style="max-width:200px;color:#94a3b8" title="${dep.served_model_name}">↳ ${dep.served_model_name}</div>` : ''}
                            </td>
                            <td class="text-nowrap">
                                <span class="badge ${engineColor} me-1">${engine}</span>
                                <span class="badge ${dtypeColor}">${dtype}</span>
                            </td>
                            <td>${statusDot}</td>
                            <td><span class="badge bg-light text-secondary border" style="font-size:.8rem">${gpu}</span></td>
                            <td class="text-end text-nowrap">
                                <button onclick="viewLogs('${dep.id}','${containerName}')" class="btn btn-sm btn-outline-secondary me-1" title="Logs"><i class="fa-solid fa-terminal"></i></button>
                                ${stopBtn}
                            </td>
                        </tr>
                    `;
                } else {
                    const singleNode = (dep.nodes || [])[0];
                    const isRunning = singleNode ? !!singleNode.is_healthy : false;
                    const statusDot = isRunning
                        ? `<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:7px;height:7px;border-radius:50%;background:#22c55e;display:inline-block"></span><span class="small">Running</span></span>`
                        : `<span style="display:inline-flex;align-items:center;gap:5px"><span style="width:7px;height:7px;border-radius:50%;background:#f59e0b;display:inline-block"></span><span class="small text-muted">Starting</span></span>`;
                    const isStopping = _stopping.has(dep.id);
                    const rowStyle = (rowDisplay || '') + (isStopping ? ';opacity:.3;pointer-events:none' : '');
                    const stopBtn = isStopping
                        ? `<button class="btn btn-sm btn-outline-danger" disabled><i class="fa-solid fa-spinner fa-spin"></i></button>`
                        : `<button onclick="stopDeployment('${dep.id}',this)" class="btn btn-sm btn-outline-danger" title="Stop"><i class="fa-solid fa-stop"></i></button>`;
                    const gpuList = (dep.gpus || []).map(g => `<span class="badge bg-light text-secondary border me-1" style="font-size:.8rem">${g}</span>`).join('');
                    list.innerHTML += `
                        <tr class="${gid}-row ep-${epKey}" style="${rowStyle}">
                            <td style="padding-left:3rem">
                                <div class="fw-semibold small">${dep.name}</div>
                                <div class="font-monospace" style="font-size:.8rem;color:#9ca3af">${dep.id}</div>
                            </td>
                            <td>
                                <div class="small text-truncate" style="max-width:200px" title="${dep.model}">${dep.model}</div>
                                ${dep.served_model_name && dep.served_model_name !== dep.model ? `<div class="small text-truncate" style="max-width:200px;color:#94a3b8" title="${dep.served_model_name}">↳ ${dep.served_model_name}</div>` : ''}
                            </td>
                            <td class="text-nowrap">
                                <span class="badge ${engineColor} me-1">${engine}</span>
                                <span class="badge ${dtypeColor}">${dtype}</span>
                            </td>
                            <td>${statusDot}</td>
                            <td><div class="d-flex flex-wrap gap-1">${gpuList}</div></td>
                            <td class="text-end text-nowrap">
                                <button onclick="viewLogs('${dep.id}')" class="btn btn-sm btn-outline-secondary me-1" title="Logs"><i class="fa-solid fa-terminal"></i></button>
                                ${stopBtn}
                            </td>
                        </tr>
                    `;
                }
            });
        });
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
        const imageTag = conf.vllm_image
            ? `<div class="font-monospace text-muted text-truncate mt-1" style="font-size:.78rem;max-width:320px" title="${escapeHtml(conf.vllm_image)}"><i class="fa-brands fa-docker me-1"></i>${escapeHtml(conf.vllm_image)}</div>`
            : '';
        return `
            <tr class="config-row">
                <td>
                    <div class="fw-semibold small">${escapeHtml(conf.name)}</div>
                    <div class="font-monospace text-muted text-truncate" style="font-size:.82rem;max-width:320px" title="${escapeHtml(conf.model)}">${escapeHtml(conf.model)}</div>
                    ${imageTag}
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
                </div>`;
        } else if (ep.status === 'active') {
            hasActive = true;
            const safeId = escapeHtml(ep.id);
            activeRows.push(`
                <div class="col-md-6 col-xl-4 mb-3">
                    <div class="card h-100" style="border-left:3px solid #22c55e">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-start mb-1">
                                <div>
                                    <span style="width:8px;height:8px;border-radius:50%;background:#22c55e;display:inline-block;margin-right:6px"></span>
                                    <span class="fw-semibold">${escapeHtml(ep.name)}</span>
                                    <span class="badge bg-success ms-1" style="font-size:.72rem">Active</span>
                                </div>
                                <span class="badge bg-secondary" style="font-size:.75rem">${ep.gpus.length} GPUs</span>
                            </div>
                            <div class="text-muted small mb-1 font-monospace">${ep.host}:${ep.port}</div>
                            <div class="text-muted mb-3" style="font-size:.78rem;word-break:break-all">${safeId}</div>
                            <div class="d-flex gap-2 flex-wrap">
                                <button class="btn btn-sm btn-outline-secondary" onclick="renameEndpoint('${ep.id}', '${escapeHtml(ep.name)}')">
                                    <i class="fa-solid fa-pen me-1"></i>이름 변경
                                </button>
                                <button class="btn btn-sm btn-outline-danger" onclick="resetEndpoint('${ep.id}')">
                                    <i class="fa-solid fa-trash me-1"></i>제거
                                </button>
                                <a href="/endpoints/${ep.id}/images" class="btn btn-sm btn-outline-primary">
                                    <i class="fa-brands fa-docker me-1"></i>Images
                                </a>
                                <a href="/endpoints/${ep.id}/models" class="btn btn-sm btn-outline-success">
                                    <i class="fa-solid fa-robot me-1"></i>Models
                                </a>
                            </div>
                        </div>
                    </div>
                </div>`);
        }
    });

    if (!hasPending) pendingList.innerHTML = '<div class="col-12 text-muted">등록 대기 중인 노드가 없습니다.</div>';
    if (!hasActive) {
        activeList.innerHTML = '<p class="text-muted p-3 mb-0">활성화된 엔드포인트가 없습니다.</p>';
    } else {
        activeList.innerHTML = `<div class="row g-0 p-3">${activeRows.join('')}</div>`;
    }

    if (activeId) {
        const el = document.getElementById(activeId);
        if (el) {
            el.focus();
            if (selStart !== null && el.setSelectionRange) { try { el.setSelectionRange(selStart, selEnd); } catch (e) {} }
        }
    }
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
                        이 워커는 이미지 관리를 지원하지 않습니다. 워커 코드를 업데이트하세요.
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

    const imageRows = images.length > 0
        ? images.map(img => {
            const safeName = escapeHtml(img.name);
            return `
                <div class="d-flex align-items-center justify-content-between py-1" style="border-bottom:1px solid #f1f5f9">
                    <div>
                        <span class="font-monospace small fw-semibold">${safeName}</span>
                        <span class="text-muted ms-2" style="font-size:.78rem">${escapeHtml(img.size || '')}</span>
                    </div>
                    <button class="btn btn-sm btn-outline-secondary" style="font-size:.72rem;padding:.1rem .45rem"
                        data-wid="${safeWid}" data-image="${safeName}"
                        onclick="pullImageFromBtn(this)">
                        <i class="fa-solid fa-arrow-rotate-right me-1"></i>Update
                    </button>
                </div>`;
        }).join('')
        : '<div class="text-muted small py-1">No vLLM images found on this worker.</div>';

    bodyEl.innerHTML = `
        <div class="mb-2" style="font-size:.8rem;font-weight:600;color:#374151;margin-top:.25rem">
            <i class="fa-brands fa-docker me-1"></i>Pulled vLLM Images
        </div>
        <div class="mb-3" id="img-list-${safeWid}">${imageRows}</div>
        <div style="font-size:.8rem;font-weight:600;color:#374151;margin-bottom:.4rem">Pull New Image</div>
        <div class="input-group input-group-sm mb-2">
            <span class="input-group-text font-monospace" style="font-size:.8rem">vllm/vllm-openai:</span>
            <input type="text" class="form-control font-monospace" id="pull-tag-${safeWid}"
                placeholder="latest" value="latest" style="font-size:.8rem">
            <button class="btn btn-primary" id="pull-btn-${safeWid}" data-wid="${safeWid}"
                onclick="pullImageFromTag(this.dataset.wid)">
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
    listEl.innerHTML = images.length > 0
        ? images.map(img => {
            const safeName = escapeHtml(img.name);
            return `
                <div class="d-flex align-items-center justify-content-between py-1" style="border-bottom:1px solid #f1f5f9">
                    <div>
                        <span class="font-monospace small fw-semibold">${safeName}</span>
                        <span class="text-muted ms-2" style="font-size:.78rem">${escapeHtml(img.size || '')}</span>
                    </div>
                    <button class="btn btn-sm btn-outline-secondary" style="font-size:.72rem;padding:.1rem .45rem"
                        data-wid="${safeWid}" data-image="${safeName}"
                        onclick="pullImageFromBtn(this)">
                        <i class="fa-solid fa-arrow-rotate-right me-1"></i>Update
                    </button>
                </div>`;
        }).join('')
        : '<div class="text-muted small py-1">No vLLM images found on this worker.</div>';
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
                bodyEl.innerHTML = `<div class="text-warning small py-2"><i class="fa-solid fa-triangle-exclamation me-1"></i>이 워커는 모델 관리를 지원하지 않습니다. 워커 코드를 업데이트하세요.</div>`;
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

    const modelRows = models.length > 0
        ? models.map(m => {
            const safeId = escapeHtml(m.repo_id);
            return `
                <div class="d-flex align-items-center justify-content-between py-1" style="border-bottom:1px solid #f1f5f9">
                    <div>
                        <span class="font-monospace small fw-semibold">${safeId}</span>
                        <span class="text-muted ms-2" style="font-size:.78rem">${escapeHtml(m.size || '')}</span>
                    </div>
                    <button class="btn btn-sm btn-outline-secondary" style="font-size:.72rem;padding:.1rem .45rem"
                        data-wid="${safeWid}" data-model="${safeId}" onclick="reDownloadModel(this)">
                        <i class="fa-solid fa-arrow-rotate-right me-1"></i>Re-download
                    </button>
                </div>`;
        }).join('')
        : '<div class="text-muted small py-1">No HuggingFace models cached on this worker.</div>';

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
    listEl.innerHTML = models.length > 0
        ? models.map(m => {
            const safeId = escapeHtml(m.repo_id);
            return `
                <div class="d-flex align-items-center justify-content-between py-1" style="border-bottom:1px solid #f1f5f9">
                    <div>
                        <span class="font-monospace small fw-semibold">${safeId}</span>
                        <span class="text-muted ms-2" style="font-size:.78rem">${escapeHtml(m.size || '')}</span>
                    </div>
                    <button class="btn btn-sm btn-outline-secondary" style="font-size:.72rem;padding:.1rem .45rem"
                        data-wid="${safeWid}" data-model="${safeId}" onclick="reDownloadModel(this)">
                        <i class="fa-solid fa-arrow-rotate-right me-1"></i>Re-download
                    </button>
                </div>`;
        }).join('')
        : '<div class="text-muted small py-1">No HuggingFace models cached on this worker.</div>';
}

window.downloadModelFromInput = async function(wid) {
    const safeWid = escapeHtml(wid);
    const modelId = document.getElementById(`model-id-${safeWid}`)?.value.trim();
    if (!modelId) return;
    await _doDownloadModel(wid, modelId);
};

window.reDownloadModel = async function(btn) {
    await _doDownloadModel(btn.dataset.wid, btn.dataset.model);
};

async function _doDownloadModel(wid, modelId) {
    if (modelDownloadInProgress.has(wid)) return;

    const safeWid = escapeHtml(wid);
    const outputEl = document.getElementById(`model-output-${safeWid}`);
    const outputWrap = document.getElementById(`model-output-wrap-${safeWid}`);

    if (outputEl) outputEl.textContent = `Starting download: ${modelId}...\n`;
    if (outputWrap) outputWrap.style.display = '';

    let job_id;
    try {
        const resp = await fetch(`/api/endpoints/${wid}/models/download`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: modelId })
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

window.pullImageFromTag = async function(wid) {
    const tagEl = document.getElementById(`pull-tag-${wid}`);
    const tag = (tagEl?.value.trim()) || 'latest';
    await _doPullImage(wid, `vllm/vllm-openai:${tag}`);
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

async function fetchAllWorkerImages() {
    const activeWorkers = Object.values(endpoints).filter(ep => ep.status === 'active');
    const allImages = new Set(['vllm/vllm-openai:latest']);
    await Promise.all(activeWorkers.map(async ep => {
        try {
            const resp = await fetch(`/api/endpoints/${ep.id}/images`);
            if (resp.ok) {
                const imgs = await resp.json();
                imgs.forEach(img => allImages.add(img.name));
            }
        } catch (e) {}
    }));
    const datalist = document.getElementById('vllm-image-datalist');
    if (datalist) datalist.innerHTML = [...allImages].map(n => `<option value="${escapeHtml(n)}">`).join('');
}

window.handleEngineChange = function() {
    const engine = document.getElementById('deployEngine')?.value;
    const section = document.getElementById('vllm-image-section');
    if (section) section.style.display = engine === 'vllm' ? '' : 'none';
    if (engine === 'vllm') fetchAllWorkerImages();
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
    const imgEl = document.getElementById('deployVllmImage');
    if (imgEl) imgEl.value = conf.vllm_image || '';

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
        extra_args: document.getElementById('deployExtraArgs').value.trim() || null,
        vllm_image: document.getElementById('deployVllmImage')?.value.trim() || null
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

window.toggleDepGroup = function(gid) {
    const rows = document.querySelectorAll(`.${gid}-row`);
    const icon = document.getElementById(`${gid}-icon`);
    const isVisible = rows.length > 0 && rows[0].style.display !== 'none';
    rows.forEach(r => { r.style.display = isVisible ? 'none' : ''; });
    if (icon) icon.style.transform = isVisible ? 'rotate(-90deg)' : '';
    if (isVisible) depGroupCollapsed.add(gid); else depGroupCollapsed.delete(gid);
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
