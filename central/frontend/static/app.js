// static/app.js
let gpus = [];
let deployments = [];
let savedConfigs = [];
let endpoints = {};

document.addEventListener('DOMContentLoaded', () => {
    fetchStatus();
    setInterval(fetchStatus, 5000);
});

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
        // Global GPU Card
        if (list) {
            list.innerHTML += `
                <div class="col-md-4 mb-3">
                    <div class="card shadow-sm h-100">
                        <div class="card-body">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <h5 class="card-title mb-0">GPU ${gpu.local_id} <span class="badge bg-secondary">${gpu.worker_name}</span></h5>
                                <small class="text-muted font-monospace">${gpu.name}</small>
                            </div>
                            <div class="progress mb-2" style="height: 10px;">
                                <div class="progress-bar ${memPercent > 80 ? 'bg-danger' : 'bg-primary'}" role="progressbar" style="width: ${memPercent}%"></div>
                            </div>
                            <div class="d-flex justify-content-between text-muted small">
                                <span>${memPercent}% VRAM</span>
                                <span>${(gpu.memory_used / 1024).toFixed(1)}GB / ${(gpu.memory_total / 1024).toFixed(1)}GB</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        // Deploy Modal Checkboxes
        if (gpuGrid) {
            const isChecked = selectedGpuIds.has(gpu.id) ? 'checked' : '';
            gpuGrid.innerHTML += `
                <input type="checkbox" class="btn-check gpu-checkbox" id="gpu-btn-${gpu.id}" value="${gpu.id}" data-node="${gpu.worker_id}" autocomplete="off" onchange="validateDeployGpus()" ${isChecked}>
                <label class="btn btn-outline-info text-start p-2" for="gpu-btn-${gpu.id}" style="width: 200px;">
                    <div class="fw-bold text-truncate" title="GPU ${gpu.local_id}">GPU ${gpu.local_id} <span class="badge bg-secondary ms-1">${gpu.worker_name}</span></div>
                    <div class="small mt-1">${memPercent}% Used</div>
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

    if (isTp) {
        nodeSelector.style.display = 'block';
        tpContainer.style.display = 'block';
        helpText.textContent = "Select 1, 2, 4, or 8 GPUs. Tensor Parallelism is automatically calculated.";
        filterGpusByNode();
    } else {
        nodeSelector.style.display = 'none';
        tpContainer.style.display = 'none';
        helpText.textContent = "Select multiple GPUs. Each selected GPU will run a separate 1-GPU replica.";
        // Show all GPUs
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

function renderDeployments() {
    const list = document.getElementById('deployments-table-body');
    if (!list) return;
    list.innerHTML = '';

    if (deployments.length === 0) {
        list.innerHTML = '<tr><td colspan="8" class="text-center text-muted py-4">No active deployments.</td></tr>';
        return;
    }

    deployments.forEach(dep => {
        let statusBadge = dep.status === 'running'
            ? '<span class="badge bg-success"><i class="fa-solid fa-play me-1"></i>Running</span>'
            : '<span class="badge bg-warning text-dark"><i class="fa-solid fa-spinner fa-spin me-1"></i>Starting</span>';

        list.innerHTML += `
            <tr>
                <td class="font-monospace">${dep.id}</td>
                <td class="fw-bold">${dep.name}</td>
                <td><span class="text-muted small">${dep.model}</span></td>
                <td><span class="badge bg-dark">${(dep.engine || 'vllm').toUpperCase()}</span></td>
                <td><span class="text-muted small">${dep.served_model_name || '-'}</span></td>
                <td><span class="badge bg-secondary">${(dep.deployment_type || '').toUpperCase()}</span></td>
                <td>${statusBadge}</td>
                <td><small>${dep.gpus.join(', ')}</small></td>
                <td>
                    <div class="btn-group">
                        <button onclick="viewLogs('${dep.id}')" class="btn btn-sm btn-outline-secondary" title="View Logs">
                            <i class="fa-solid fa-terminal"></i>
                        </button>
                        <button id="stop-btn-${dep.id}" onclick="stopDeployment('${dep.id}')" class="btn btn-sm btn-outline-danger" title="Stop">
                            <i class="fa-solid fa-stop"></i>
                        </button>
                    </div>
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

    savedConfigs.forEach(conf => {
        list.innerHTML += `
            <div class="col-md-4 mb-3">
                <div class="card shadow-sm h-100">
                    <div class="card-body">
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <h5 class="card-title mb-0">${conf.name}</h5>
                            <span class="badge bg-info text-dark">${(conf.deployment_type || conf.mode || 'REPLICAS').toUpperCase()}</span>
                        </div>
                        <p class="card-text small text-muted text-truncate" title="${conf.model}">${conf.model}</p>
                    </div>
                    </div>
                    <div class="card-footer bg-transparent border-top-0 d-flex justify-content-between text-end">
                        <button class="btn btn-sm btn-outline-danger" onclick="deleteConfig('${conf.name}')" title="Delete Config"><i class="fa-solid fa-trash"></i></button>
                        <div>
                            <button class="btn btn-sm btn-outline-primary me-2" onclick="loadConfig('${conf.name}')"><i class="fa-solid fa-edit me-1"></i>Edit</button>
                            <button class="btn btn-sm btn-primary" onclick="loadConfig('${conf.name}')"><i class="fa-solid fa-play me-1"></i>Run</button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });

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

    Object.values(endpoints).forEach(ep => {
        if (ep.status === 'pending') {
            hasPending = true;
            pendingList.innerHTML += `
                <div class="col-md-6 mb-3">
                    <div class="card border-warning shadow-sm">
                        <div class="card-body">
                            <h5 class="card-title text-warning"><i class="fa-solid fa-triangle-exclamation me-2"></i>New Node Detected</h5>
                            <p class="mb-1 text-muted small">ID: <span class="font-monospace">${ep.id}</span></p>
                            <p class="mb-3 text-muted small">Host: <span class="font-monospace">${ep.host}:${ep.port}</span> (${ep.gpus.length} GPUs)</p>

                            <div class="input-group input-group-sm">
                                <input type="text" id="accept-name-${ep.id}" class="form-control" placeholder="Assign Name (e.g. server-room-1)" value="${inputCache[`accept-name-${ep.id}`] || ep.id}">
                                <button class="btn btn-success" onclick="acceptEndpoint('${ep.id}')">Accept</button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        } else if (ep.status === 'active') {
            hasActive = true;
            activeList.innerHTML += `
                <div class="col-md-4 mb-3">
                    <div class="card border-success shadow-sm">
                        <div class="card-body bg-light">
                            <div class="d-flex justify-content-between">
                                <h5 class="card-title text-success"><i class="fa-solid fa-circle-check me-2"></i>${ep.name}</h5>
                                <span class="badge bg-secondary">${ep.gpus.length} GPUs</span>
                            </div>
                            <p class="mb-0 text-muted small mt-2">Host: <span class="font-monospace">${ep.host}:${ep.port}</span></p>
                            <p class="mb-0 text-muted small">ID: <span class="font-monospace">${ep.id}</span></p>
                        </div>
                        <div class="card-footer bg-transparent border-top-0 d-flex justify-content-end gap-2 text-end">
                            <button class="btn btn-sm btn-outline-info" onclick="renameEndpoint('${ep.id}', '${ep.name}')"><i class="fa-solid fa-i-cursor me-1"></i>Rename</button>
                            <button class="btn btn-sm btn-outline-danger" onclick="resetEndpoint('${ep.id}')"><i class="fa-solid fa-trash me-1"></i>Remove</button>
                        </div>
                    </div>
                </div>
            `;
        }
    });

    if (!hasPending) pendingList.innerHTML = '<div class="col-12 text-muted small">No pending nodes.</div>';
    if (!hasActive) activeList.innerHTML = '<div class="col-12 text-muted small">No active nodes.</div>';

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
    const conf = savedConfigs.find(c => c.name === name);
    if (!conf) return;

    document.getElementById('deployName').value = conf.name;
    document.getElementById('deployModel').value = conf.model;
    document.getElementById('deployServedModel').value = conf.served_model_name || '';
    
    // Set Engine explicitly, defaulting to vllm if missing for backwards compatibility
    const engineEl = document.getElementById('deployEngine');
    if (engineEl) {
        engineEl.value = conf.engine || 'vllm';
    }

    const dtype = conf.deployment_type || conf.mode || 'replicas';
    if (dtype === 'tp') {
        document.getElementById('typeTp').checked = true;
    } else {
        document.getElementById('typeReplicas').checked = true;
    }

    document.getElementById('deployIsEmbedding').checked = !!conf.is_embedding;

    document.getElementById('deployMaxLen').value = conf.max_len || '';
    document.getElementById('deployGpuUtil').value = conf.gpu_util || 0.9;
    document.getElementById('deployExtraArgs').value = conf.extra_args || '';

    // Make sure GPUs are loaded
    if (gpus.length === 0) {
        await fetchStatus();
    }

    // Select GPUs in grid
    const checkboxes = document.querySelectorAll('.gpu-checkbox');
    const gpusToSelect = conf.gpus || conf.gpuIds || [];
    checkboxes.forEach(cb => {
        cb.checked = gpusToSelect.includes(cb.value);
    });

    // Trigger UI updates
    if (typeof window.toggleDeployModeUI === 'function') window.toggleDeployModeUI();
    if (typeof window.validateDeployGpus === 'function') window.validateDeployGpus();

    const modalEl = document.getElementById('deployModal');
    if (!modalEl.classList.contains('show')) {
        const modal = bootstrap.Modal.getOrCreateInstance(modalEl);
        modal.show();
    }
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
        is_embedding: document.getElementById('deployIsEmbedding').checked,
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
            showAlert("success", "Deployment created successfully!");
            bootstrap.Modal.getInstance(document.getElementById('deployModal')).hide();
            fetchStatus();
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
