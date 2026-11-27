(() => {
  const role = document.body.dataset.role || 'viewer';
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || '';
  const targets = Array.isArray(window.CANCER_TARGETS) ? window.CANCER_TARGETS : [];

  const REGION_VI = {
    brain: 'Não', lung: 'Phổi', liver: 'Gan', breast: 'Vú', kidney: 'Thận',
    thyroid: 'Tuyến giáp', stomach: 'Dạ dày', colorectal: 'Đại trực tràng',
    cervical: 'Cổ tử cung', prostate: 'Tuyến tiền liệt', pancreas: 'Tụy',
  };

  const state = {
    cases: [],
    currentCaseId: null,
    currentConversationId: null,
    uploadedPath: null,
    currentTarget: 'brain',
    zoom: 1,
    panX: 0,
    panY: 0,
    dragging: false,
    dragStartX: 0,
    dragStartY: 0,
    inverted: false,
    cliniciansLoaded: false,
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    sidebar: $('sidebar'),
    resultCol: $('resultCol'),
    resizerLeft: $('resizerLeft'),
    resizerRight: $('resizerRight'),
    toggleLeftBtn: $('toggleLeftBtn'),
    toggleRightBtn: $('toggleRightBtn'),
    themeToggleBtn: $('themeToggleBtn'),
    caseCrumb: $('caseCrumb'),
    systemStatus: $('systemStatus'),
    demoBanner: $('demoBanner'),
    caseCount: $('caseCount'),
    caseSearch: $('caseSearch'),
    caseList: $('caseList'),
    newCaseBtn: $('newCaseBtn'),
    viewportStage: $('viewportStage'),
    dropzone: $('dropzone'),
    viewerLookupBox: $('viewerLookupBox'),
    fileInput: $('fileInput'),
    pickFileBtn: $('pickFileBtn'),
    previewImage: $('previewImage'),
    viewportTopRight: $('viewportTopRight'),
    changeImageBtn: $('changeImageBtn'),
    pacsToolbar: $('pacsToolbar'),
    zoomOutBtn: $('zoomOutBtn'),
    zoomInBtn: $('zoomInBtn'),
    invertBtn: $('invertBtn'),
    resetZoomBtn: $('resetZoomBtn'),
    zoomLabel: $('zoomLabel'),
    analyzeStrip: $('analyzeStrip'),
    targetSelect: $('targetSelect'),
    modalitySelect: $('modalitySelect'),
    patientCodeInput: $('patientCodeInput'),
    promptInput: $('promptInput'),
    analyzeBtn: $('analyzeBtn'),
    uploadError: $('uploadError'),
    analyzeError: $('analyzeError'),
    publicCodeInput: $('publicCodeInput'),
    lookupPublicBtn: $('lookupPublicBtn'),
    tabResult: $('tabResult'),
    tabChat: $('tabChat'),
    resultEmpty: $('resultEmpty'),
    resultContent: $('resultContent'),
    riskCard: $('riskCard'),
    resRiskText: $('resRiskText'),
    resPatientCode: $('resPatientCode'),
    resModality: $('resModality'),
    resRegion: $('resRegion'),
    resConfidence: $('resConfidence'),
    resStatus: $('resStatus'),
    resReplyText: $('resReplyText'),
    open3dBtn: $('open3dBtn'),
    downloadPdfBtn: $('downloadPdfBtn'),
    reviewFold: $('reviewFold'),
    reviewForm: $('reviewForm'),
    revRisk: $('revRisk'),
    revMalignant: $('revMalignant'),
    revNote: $('revNote'),
    adminFold: $('adminFold'),
    assignClinicianSelect: $('assignClinicianSelect'),
    assignBtn: $('assignBtn'),
    linkViewerSelect: $('linkViewerSelect'),
    linkViewerBtn: $('linkViewerBtn'),
    approveBtn: $('approveBtn'),
    publicCodeBanner: $('publicCodeBanner'),
    chatLog: $('chatLog'),
    chatInput: $('chatInput'),
    chatSendBtn: $('chatSendBtn'),
    actionFeedback: $('actionFeedback'),
    anatomyModal: $('anatomyModal'),
    anatomyTitle: $('anatomyTitle'),
    anatomyCanvasWrap: $('anatomyCanvasWrap'),
    anatomyOrganList: $('anatomyOrganList'),
    close3dBtn: $('close3dBtn'),
    anatomy3dWireBtn: $('anatomy3dWireBtn'),
    anatomy3dResetBtn: $('anatomy3dResetBtn'),
    settingsBtns: document.querySelectorAll('.settings-btn'),
    settingsModal: $('settingsModal'),
    closeSettingsBtn: $('closeSettingsBtn'),
    settingsForm: $('settingsForm'),
    setTheme: $('setTheme'),
    setLang: $('setLang'),
    settingsFeedback: $('settingsFeedback'),
  };

  async function api(url, opts = {}) {
    const headers = { ...(opts.headers || {}) };
    if (opts.method && opts.method !== 'GET') headers['X-CSRF-Token'] = csrf;
    const res = await fetch(url, { ...opts, headers });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Lỗi ${res.status}`);
    return data;
  }

  function showAlert(node, msg, isError = true) {
    if (!node) return;
    if (!msg) { node.classList.add('hidden'); return; }
    node.textContent = msg;
    node.className = `alert ${isError ? 'alert-error' : 'alert-ok'}`;
  }

  function normalizeRisk(risk) {
    if (!risk) return 'uncertain';
    const r = String(risk).toLowerCase();
    if (r.includes('high')) return 'high';
    if (r.includes('medium') || r.includes('indeterminate')) return 'medium';
    if (r.includes('low')) return 'low';
    return 'uncertain';
  }

  function riskLabel(risk) {
    return { high: 'NGUY CƠ CAO', medium: 'CẦN THEO DÕI', low: 'NGUY CƠ THẤP', uncertain: 'CHƯA RÕ' }[normalizeRisk(risk)];
  }

  function reviewLabel(status) {
    return { pending: 'Chờ duyệt', submitted: 'Đã gửi duyệt', approved: 'Đã phê duyệt' }[status] || (status || 'Mới phân tích');
  }

  function regionLabel(region) {
    if (!region) return '-';
    return REGION_VI[region.toLowerCase()] || region;
  }

  // Theme
  function applyTheme(theme) {
    const resolved = theme === 'system'
      ? (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark')
      : (theme === 'light' ? 'light' : 'dark');
    document.documentElement.dataset.theme = resolved;
    localStorage.setItem('theme', theme);
    $('themeIconSun')?.classList.toggle('hidden', resolved === 'light');
    $('themeIconMoon')?.classList.toggle('hidden', resolved === 'dark');
  }
  applyTheme(localStorage.getItem('theme') || 'dark');
  el.themeToggleBtn?.addEventListener('click', () => {
    applyTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark');
  });

  // Settings Modal
  el.settingsBtns.forEach((button) => {
    button.addEventListener('click', async () => {
      showAlert(el.settingsFeedback, '');
      el.settingsModal?.classList.add('open');
      try {
        const s = await api('/api/settings');
        if (el.setTheme) el.setTheme.value = s.theme || localStorage.getItem('theme') || 'dark';
        if (el.setLang) el.setLang.value = s.language || 'vi';
      } catch { /* fallback to local */ }
    });
  });
  el.closeSettingsBtn?.addEventListener('click', () => el.settingsModal?.classList.remove('open'));
  el.settingsModal?.addEventListener('click', (e) => {
    if (e.target === el.settingsModal) el.settingsModal.classList.remove('open');
  });
  el.settingsForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const theme = el.setTheme?.value || 'dark';
    const language = el.setLang?.value || 'vi';
    const fd = new FormData();
    fd.append('theme', theme);
    fd.append('language', language);
    try {
      await api('/api/settings', { method: 'POST', body: fd });
      applyTheme(theme);
      showAlert(el.settingsFeedback, 'Đã lưu cài đặt.', false);
    } catch (err) {
      showAlert(el.settingsFeedback, err.message, true);
    }
  });

  // Panel Collapse / Expand
  function togglePanel(panel, resizer, key) {
    if (!panel) return;
    const collapsed = panel.classList.toggle('collapsed');
    resizer?.classList.toggle('hidden', collapsed);
    localStorage.setItem(key, collapsed ? '1' : '0');
  }
  if (localStorage.getItem('pacs_left_collapsed') === '1') {
    el.sidebar?.classList.add('collapsed');
    el.resizerLeft?.classList.add('hidden');
  }
  if (localStorage.getItem('pacs_right_collapsed') === '1') {
    el.resultCol?.classList.add('collapsed');
    el.resizerRight?.classList.add('hidden');
  }
  el.toggleLeftBtn?.addEventListener('click', () => togglePanel(el.sidebar, el.resizerLeft, 'pacs_left_collapsed'));
  el.toggleRightBtn?.addEventListener('click', () => togglePanel(el.resultCol, el.resizerRight, 'pacs_right_collapsed'));

  // Resizable panels
  function initResizer(handle, targetEl, side, storageKey, minW, maxW) {
    if (!handle || !targetEl) return;
    const saved = Number(localStorage.getItem(storageKey));
    if (saved >= minW && saved <= maxW) targetEl.style.width = `${saved}px`;

    handle.addEventListener('mousedown', (e) => {
      e.preventDefault();
      handle.classList.add('active');
      targetEl.classList.add('resizing');
      const startX = e.clientX;
      const startW = targetEl.getBoundingClientRect().width;
      const onMove = (ev) => {
        const delta = side === 'left' ? (ev.clientX - startX) : (startX - ev.clientX);
        targetEl.style.width = `${Math.max(minW, Math.min(maxW, startW + delta))}px`;
      };
      const onUp = () => {
        handle.classList.remove('active');
        targetEl.classList.remove('resizing');
        localStorage.setItem(storageKey, String(Math.round(targetEl.getBoundingClientRect().width)));
        window.removeEventListener('mousemove', onMove);
        window.removeEventListener('mouseup', onUp);
      };
      window.addEventListener('mousemove', onMove);
      window.addEventListener('mouseup', onUp);
    });
  }
  initResizer(el.resizerLeft, el.sidebar, 'left', 'pacs_left_w', 200, 400);
  initResizer(el.resizerRight, el.resultCol, 'right', 'pacs_right_w', 280, 560);

  // Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      const isChat = btn.dataset.tab === 'chat';
      el.tabResult?.classList.toggle('hidden', isChat);
      el.tabChat?.classList.toggle('hidden', !isChat);
    });
  });

  function updateTransform() {
    if (!el.previewImage) return;
    el.previewImage.style.transform = `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
    el.previewImage.classList.toggle('inverted', state.inverted);
    if (el.zoomLabel) el.zoomLabel.textContent = `${Math.round(state.zoom * 100)}%`;
  }

  function showImageInViewport(src, showAnalyze) {
    el.dropzone?.classList.add('hidden');
    el.viewerLookupBox?.classList.add('hidden');
    el.previewImage.classList.remove('hidden');
    el.viewportTopRight?.classList.remove('hidden');
    el.pacsToolbar?.classList.remove('hidden');
    el.analyzeStrip?.classList.toggle('hidden', !showAnalyze);
    state.zoom = 1; state.panX = 0; state.panY = 0; state.inverted = false;
    updateTransform();
    if (src) el.previewImage.src = src;
  }

  function resetToNewCase() {
    state.uploadedPath = null;
    state.currentCaseId = null;
    state.currentConversationId = null;
    el.dropzone?.classList.remove('hidden');
    el.viewerLookupBox?.classList.remove('hidden');
    el.previewImage.classList.add('hidden');
    el.viewportTopRight?.classList.add('hidden');
    el.pacsToolbar?.classList.add('hidden');
    el.analyzeStrip?.classList.add('hidden');
    el.resultContent.classList.add('hidden');
    el.resultEmpty.classList.remove('hidden');
    el.caseCrumb.innerHTML = '<strong>Ca mới</strong><span>· Chưa chọn phim</span>';
    showAlert(el.uploadError, '');
    showAlert(el.analyzeError, '');
    document.querySelectorAll('.case-item').forEach((n) => n.classList.remove('active'));
  }

  function populateTargets() {
    if (!el.targetSelect) return;
    el.targetSelect.innerHTML = targets.map((t) =>
      `<option value="${t.key}">${t.label}${t.model_ready ? '' : ' (Chưa sẵn sàng)'}</option>`
    ).join('');
  }

  async function loadStatus() {
    try {
      const s = await api('/api/status');
      const txt = el.systemStatus.querySelector('span:last-child');
      if (s.demo_mode) {
        txt.textContent = 'Demo Mode';
        el.systemStatus.className = 'status-pill demo';
        el.demoBanner.textContent = s.demo_warning || 'Hệ thống đang chạy chế độ mô phỏng.';
        el.demoBanner.classList.remove('hidden');
      } else if (s.model_ready) {
        txt.textContent = 'Sẵn sàng';
        el.systemStatus.className = 'status-pill ready';
      } else {
        txt.textContent = 'Chưa có model';
      }
    } catch {
      el.systemStatus.querySelector('span:last-child').textContent = 'Ngoại tuyến';
    }
  }

  async function loadCases() {
    try {
      const data = await api('/api/cases?limit=100');
      state.cases = data.cases || [];
      renderCaseList();
    } catch {
      el.caseList.innerHTML = '<div class="empty-state">Không tải được danh sách.</div>';
    }
  }

  function renderCaseList() {
    const q = (el.caseSearch?.value || '').trim().toLowerCase();
    const filtered = state.cases.filter((c) => {
      if (!q) return true;
      return `${c.patient_code} ${c.body_region || ''} ${regionLabel(c.body_region)} ${c.modality || ''}`.toLowerCase().includes(q);
    });
    if (el.caseCount) el.caseCount.textContent = String(filtered.length);
    if (!filtered.length) {
      el.caseList.innerHTML = '<div class="empty-state">Chưa có ca bệnh nào.</div>';
      return;
    }
    el.caseList.innerHTML = filtered.map((c) => {
      const rCls = normalizeRisk(c.risk_level);
      const mod = c.modality ? `<span class="mod-tag">${c.modality}</span>` : '';
      const reg = c.body_region ? regionLabel(c.body_region) : 'Chưa phân loại';
      return `
        <button type="button" class="case-item ${c.case_id === state.currentCaseId ? 'active' : ''}" data-id="${c.case_id}">
          <div class="case-item-top">
            <span>${c.patient_code || `Ca #${c.case_id}`} <small style="color:var(--text-muted);font-weight:400;">#${c.case_id}</small></span>
            <span class="risk-dot ${rCls}" title="${riskLabel(c.risk_level)}"></span>
          </div>
          <div class="case-item-sub">
            <span style="display:flex;align-items:center;gap:5px;">${mod}<span>${reg}</span></span>
            <span>${reviewLabel(c.review_status)}</span>
          </div>
        </button>
      `;
    }).join('');

    el.caseList.querySelectorAll('.case-item').forEach((btn) => {
      btn.addEventListener('click', () => selectCase(Number(btn.dataset.id)));
    });
  }

  async function selectCase(caseId) {
    try {
      const data = await api(`/api/cases/${caseId}`);
      const c = data.case;
      state.currentCaseId = c.case_id;
      state.currentTarget = c.body_region || state.currentTarget;

      el.caseCrumb.innerHTML = `<strong>Ca: ${c.patient_code || `#${c.case_id}`}</strong><span>· ${[c.modality?.toUpperCase(), regionLabel(c.body_region)].filter(Boolean).join(' - ')}</span>`;
      showImageInViewport(`/api/cases/${c.case_id}/image`, false);

      const conf = c.average_confidence ? `${Math.round(c.average_confidence * 100)}%` : '-';
      showResult({
        caseId: c.case_id,
        patientCode: c.patient_code,
        riskLevel: c.risk_level,
        modality: c.modality || '-',
        region: regionLabel(c.body_region),
        confidence: conf,
        status: reviewLabel(c.review_status),
        replyText: c.recommendation || 'Đã hoàn tất sàng lọc hình ảnh.',
        target: c.body_region,
        publicCode: c.public_code,
        pdfUrl: `/api/cases/${c.case_id}/pdf`,
      });

      if (role === 'admin') await loadAdminOptions(c);
      renderCaseList();
      if (window.innerWidth <= 860) el.sidebar?.classList.add('collapsed');
    } catch (err) {
      showAlert(el.actionFeedback, err.message);
    }
  }

  function showResult({ caseId, patientCode, riskLevel, modality, region, confidence, status, replyText, target, publicCode, pdfUrl }) {
    el.resultEmpty.classList.add('hidden');
    el.resultContent.classList.remove('hidden');
    showAlert(el.actionFeedback, '');

    if (el.resultCol?.classList.contains('collapsed')) {
      el.resultCol.classList.remove('collapsed');
      el.resizerRight?.classList.remove('hidden');
    }

    el.riskCard.className = `risk-card ${normalizeRisk(riskLevel)}`;
    el.resRiskText.textContent = riskLabel(riskLevel);
    el.resPatientCode.textContent = patientCode || '-';
    el.resModality.textContent = (modality || '-').toUpperCase();
    el.resRegion.textContent = region || '-';
    el.resConfidence.textContent = confidence || '-';
    el.resStatus.textContent = status || '-';
    el.resReplyText.textContent = replyText || '-';

    if (pdfUrl && el.downloadPdfBtn) {
      el.downloadPdfBtn.href = pdfUrl;
      el.downloadPdfBtn.classList.remove('hidden');
    } else {
      el.downloadPdfBtn?.classList.add('hidden');
    }

    if (target && el.open3dBtn) {
      state.currentTarget = target;
      el.open3dBtn.classList.remove('hidden');
    } else {
      el.open3dBtn?.classList.add('hidden');
    }

    if (role === 'clinician' && el.reviewFold && caseId) {
      el.reviewFold.classList.remove('hidden');
      el.revRisk.value = normalizeRisk(riskLevel);
    }
    if (role === 'admin' && el.adminFold && caseId) {
      el.adminFold.classList.remove('hidden');
      if (publicCode) {
        el.publicCodeBanner.textContent = `Mã tra cứu công khai: ${publicCode}`;
        el.publicCodeBanner.classList.remove('hidden');
      } else {
        el.publicCodeBanner.classList.add('hidden');
      }
    }
  }

  async function loadAdminOptions(c) {
    if (!state.cliniciansLoaded) {
      try {
        const [cl, vw] = await Promise.all([api('/api/clinicians'), api('/api/viewers')]);
        el.assignClinicianSelect.innerHTML = '<option value="">-- Chọn bác sĩ --</option>' +
          (cl.clinicians || []).map((u) => `<option value="${u.username}">${u.username}</option>`).join('');
        el.linkViewerSelect.innerHTML = '<option value="">-- Chọn bệnh nhân --</option>' +
          (vw.viewers || []).map((u) => `<option value="${u.username}">${u.username}</option>`).join('');
        state.cliniciansLoaded = true;
      } catch { /* ignore */ }
    }
    if (c.assigned_to) el.assignClinicianSelect.value = c.assigned_to;
    if (c.patient_username) el.linkViewerSelect.value = c.patient_username;
  }

  async function handleFileUpload(file) {
    if (!file) return;
    showAlert(el.uploadError, '');
    showAlert(el.analyzeError, '');
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await api('/api/upload', { method: 'POST', body: fd });
      state.uploadedPath = res.stored_path;
      state.currentCaseId = null;

      const src = file.type.startsWith('image/') ? URL.createObjectURL(file) : '';
      showImageInViewport(src, true);
      el.caseCrumb.innerHTML = `<strong>${res.filename}</strong><span>· Sẵn sàng phân tích</span>`;

      if (res.detected_target && el.targetSelect) el.targetSelect.value = res.detected_target;
      if (res.detected_modality && el.modalitySelect) el.modalitySelect.value = res.detected_modality;
    } catch (err) {
      showAlert(el.uploadError, err.message);
    }
  }

  function appendChatBubble(sender, text) {
    if (!el.chatLog) return;
    const empty = el.chatLog.querySelector('.empty-state');
    if (empty) empty.remove();
    const div = document.createElement('div');
    div.className = `chat-bubble ${sender}`;
    const label = document.createElement('div');
    label.className = 'chat-sender';
    label.textContent = sender === 'user' ? 'Bác sĩ / Quản trị' : 'OncoVision AI';
    const body = document.createElement('div');
    body.textContent = text;
    div.appendChild(label);
    div.appendChild(body);
    el.chatLog.appendChild(div);
    el.chatLog.scrollTop = el.chatLog.scrollHeight;
  }

  async function runAnalyze() {
    if (!state.uploadedPath) return;
    el.analyzeBtn.disabled = true;
    el.analyzeBtn.textContent = 'Đang phân tích...';
    showAlert(el.analyzeError, '');
    try {
      const fd = new FormData();
      fd.append('image_path', state.uploadedPath);
      fd.append('target_key', el.targetSelect.value || 'brain');
      fd.append('modality', el.modalitySelect.value || '');
      fd.append('patient_code', el.patientCodeInput.value.trim() || 'WEB');
      fd.append('user_prompt', el.promptInput.value.trim());
      if (state.currentConversationId) fd.append('conversation_id', String(state.currentConversationId));

      const res = await api('/api/analyze', { method: 'POST', body: fd });
      const meta = res.metadata || {};
      state.currentConversationId = res.conversation_id || null;
      state.currentCaseId = meta.medical_case_id || null;

      if (res.attachment_path && role === 'admin') {
        const clean = res.attachment_path.replace(/^\/+/, '');
        el.previewImage.src = clean.startsWith('output/') ? `/${clean}` : `/output/${clean}`;
      }

      const pc = meta.patient_code || el.patientCodeInput.value.trim() || 'WEB';
      const mod = meta.detected_modality || meta.modality || '-';
      const reg = meta.suspected_cancer || el.targetSelect.value || '-';
      const conf = meta.average_confidence ? `${Math.round(meta.average_confidence * 100)}%` : '-';

      el.caseCrumb.innerHTML = `<strong>Ca: ${pc}</strong><span>· ${mod.toUpperCase()} - ${regionLabel(reg)}</span>`;
      showResult({
        caseId: state.currentCaseId,
        patientCode: pc,
        riskLevel: meta.risk_level || 'uncertain',
        modality: mod,
        region: regionLabel(reg),
        confidence: conf,
        status: 'Mới phân tích',
        replyText: res.reply_text,
        target: reg,
        pdfUrl: state.currentCaseId ? `/api/cases/${state.currentCaseId}/pdf` : null,
      });

      if (role === 'admin' && el.chatLog) {
        el.chatLog.innerHTML = '';
        if (el.promptInput.value.trim()) appendChatBubble('user', el.promptInput.value.trim());
        appendChatBubble('assistant', res.reply_text);
      }
      await loadCases();
    } catch (err) {
      showAlert(el.analyzeError, err.message);
    } finally {
      el.analyzeBtn.disabled = false;
      el.analyzeBtn.textContent = 'Phân tích AI';
    }
  }

  // --- 3D GLB Viewer ---
  // Map cancer target key → GLB filename in /static/models/
  const GLB_MAP = {
    brain: 'brain_cancer.glb',
    lung: 'lung_cancer.glb',
    liver: 'liver_cancer.glb',
    breast: 'breast_cancer.glb',
    kidney: 'kidney_cancer.glb',
    thyroid: 'thyroid_cancer.glb',
    stomach: 'stomach_cancer.glb',
    colorectal: 'colon_rectum_cancer.glb',
    colon_rectum: 'colon_rectum_cancer.glb',
    cervical: 'cervix_cancer.glb',
    cervix: 'cervix_cancer.glb',
    prostate: 'prostate_cancer.glb',
    pancreas: 'pancreas_cancer.glb',
  };

  let _3d = null; // current viewer state

  function dispose3D() {
    if (!_3d) return;
    cancelAnimationFrame(_3d.rafId);
    _3d.renderer.dispose();
    _3d = null;
  }

  async function open3dViewer() {
    el.anatomyModal.classList.add('open');
    el.anatomyCanvasWrap.innerHTML = '<div class="anatomy-loading" id="anatomy3dLoading">Đang tải model 3D...</div>';
    el.anatomyOrganList.innerHTML = '';
    dispose3D();

    // Load organ list from API
    const target = state.currentTarget || 'brain';
    try {
      const data = await api(`/api/anatomy/organs?target=${encodeURIComponent(target)}&lang=vi`);
      el.anatomyTitle.textContent = `Giải phẫu 3D: ${data.label || regionLabel(target)}`;
      const organs = data.organs_to_highlight || [];
      if (organs.length) {
        el.anatomyOrganList.innerHTML = organs
          .map((o) => `<span class="anatomy-organ-tag">${o.label || o.name || o}</span>`)
          .join('');
      }
    } catch { /* non-critical */ }

    // Wait for ESM modules to be ready (injected by <script type="module">)
    let tries = 0;
    while (!window._THREE && tries++ < 20) await new Promise((r) => setTimeout(r, 100));

    if (!window._THREE) {
      el.anatomyCanvasWrap.innerHTML = '<div class="anatomy-loading">Three.js chưa tải xong. Vui lòng thử lại.</div>';
      return;
    }

    const THREE = window._THREE;
    const GLTFLoader = window._GLTFLoader;
    const OrbitControls = window._OrbitControls;
    const DRACOLoader = window._DRACOLoader;

    const wrap = el.anatomyCanvasWrap;
    const w = wrap.clientWidth || 800;
    const h = wrap.clientHeight || 520;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05080f);

    const camera = new THREE.PerspectiveCamera(45, w / h, 0.01, 1000);
    camera.position.set(0, 0, 3);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dir1 = new THREE.DirectionalLight(0xffffff, 0.9);
    dir1.position.set(5, 10, 7);
    dir1.castShadow = true;
    scene.add(dir1);
    const dir2 = new THREE.DirectionalLight(0x90d4ff, 0.4);
    dir2.position.set(-5, 5, -7);
    scene.add(dir2);

    // Replace loading div with canvas
    wrap.innerHTML = '';
    wrap.appendChild(renderer.domElement);

    const glbFile = GLB_MAP[target] || GLB_MAP['brain'];
    const glbUrl = `/static/models/${glbFile}`;

    const loader = new GLTFLoader();
    if (DRACOLoader) {
      const draco = new DRACOLoader();
      draco.setDecoderPath('/static/js/vendor/esm/');
      loader.setDRACOLoader(draco);
    }

    // Loading indicator overlay
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'anatomy-loading';
    loadingDiv.textContent = `Đang tải ${glbFile}...`;
    wrap.appendChild(loadingDiv);

    loader.load(
      glbUrl,
      (gltf) => {
        loadingDiv.remove();
        const model = gltf.scene;

        // Apply materials: highlight organ mesh, rest semi-transparent
        model.traverse((child) => {
          if (!child.isMesh) return;
          child.castShadow = true;
          child.receiveShadow = true;
          if (child.material) {
            child.material = child.material.clone();
            child.material.metalness = 0.05;
            child.material.roughness = 0.75;
          }
        });

        // Auto-fit camera
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        const scale = maxDim > 0 ? 2 / maxDim : 1;
        model.scale.setScalar(scale);
        model.position.sub(center.clone().multiplyScalar(scale));
        scene.add(model);
        controls.target.set(0, 0, 0);
        controls.update();

        _3d = { renderer, scene, camera, controls, model, rafId: 0, wireframe: false };

        const animate = () => {
          if (!el.anatomyModal.classList.contains('open')) { dispose3D(); return; }
          controls.update();
          renderer.render(scene, camera);
          _3d.rafId = requestAnimationFrame(animate);
        };
        animate();
      },
      undefined,
      (err) => {
        loadingDiv.textContent = `Không tải được model: ${err.message || glbFile}`;
        console.error('GLB load error:', err);
      }
    );

    // Wireframe toggle
    el.anatomy3dWireBtn?.addEventListener('click', () => {
      if (!_3d) return;
      _3d.wireframe = !_3d.wireframe;
      _3d.model?.traverse((child) => {
        if (child.isMesh && child.material) child.material.wireframe = _3d.wireframe;
      });
      if (el.anatomy3dWireBtn) el.anatomy3dWireBtn.textContent = _3d.wireframe ? 'Solid' : 'Wireframe';
    }, { once: true });

    // Reset camera
    el.anatomy3dResetBtn?.addEventListener('click', () => {
      if (!_3d) return;
      _3d.camera.position.set(0, 0, 3);
      _3d.controls.target.set(0, 0, 0);
      _3d.controls.update();
    }, { once: true });

    // Resize observer
    const ro = new ResizeObserver(() => {
      if (!_3d) return;
      const nw = wrap.clientWidth, nh = wrap.clientHeight;
      _3d.renderer.setSize(nw, nh);
      _3d.camera.aspect = nw / nh;
      _3d.camera.updateProjectionMatrix();
    });
    ro.observe(wrap);
  }

  // Events
  el.newCaseBtn?.addEventListener('click', resetToNewCase);
  el.caseSearch?.addEventListener('input', renderCaseList);
  el.pickFileBtn?.addEventListener('click', (e) => { e.stopPropagation(); el.fileInput?.click(); });
  el.dropzone?.addEventListener('click', () => el.fileInput?.click());
  el.changeImageBtn?.addEventListener('click', () => { if (role !== 'viewer') el.fileInput?.click(); });
  el.fileInput?.addEventListener('change', (e) => handleFileUpload(e.target.files?.[0]));

  if (el.dropzone) {
    el.dropzone.addEventListener('dragover', (e) => { e.preventDefault(); el.dropzone.classList.add('dragover'); });
    el.dropzone.addEventListener('dragleave', () => el.dropzone.classList.remove('dragover'));
    el.dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      el.dropzone.classList.remove('dragover');
      handleFileUpload(e.dataTransfer.files?.[0]);
    });
  }

  el.zoomInBtn?.addEventListener('click', () => { state.zoom = Math.min(5, state.zoom + 0.25); updateTransform(); });
  el.zoomOutBtn?.addEventListener('click', () => { state.zoom = Math.max(0.5, state.zoom - 0.25); updateTransform(); });
  el.invertBtn?.addEventListener('click', () => { state.inverted = !state.inverted; updateTransform(); });
  el.resetZoomBtn?.addEventListener('click', () => {
    state.zoom = 1; state.panX = 0; state.panY = 0; state.inverted = false; updateTransform();
  });

  el.viewportStage?.addEventListener('wheel', (e) => {
    if (el.previewImage.classList.contains('hidden')) return;
    e.preventDefault();
    state.zoom = Math.max(0.5, Math.min(5, state.zoom + (e.deltaY < 0 ? 0.15 : -0.15)));
    updateTransform();
  }, { passive: false });

  el.previewImage?.addEventListener('mousedown', (e) => {
    state.dragging = true;
    state.dragStartX = e.clientX - state.panX;
    state.dragStartY = e.clientY - state.panY;
  });
  window.addEventListener('mousemove', (e) => {
    if (!state.dragging) return;
    state.panX = e.clientX - state.dragStartX;
    state.panY = e.clientY - state.dragStartY;
    updateTransform();
  });
  window.addEventListener('mouseup', () => { state.dragging = false; });

  el.analyzeBtn?.addEventListener('click', runAnalyze);
  el.open3dBtn?.addEventListener('click', open3dViewer);
  el.close3dBtn?.addEventListener('click', () => el.anatomyModal.classList.remove('open'));
  el.anatomyModal?.addEventListener('click', (e) => {
    if (e.target === el.anatomyModal) el.anatomyModal.classList.remove('open');
  });

  el.lookupPublicBtn?.addEventListener('click', async () => {
    const code = el.publicCodeInput.value.trim();
    if (!code) return;
    try {
      const data = await api(`/api/public/cases/${encodeURIComponent(code)}`);
      const c = data.case;
      showImageInViewport(`/api/cases/${c.case_id}/image`, false);
      showResult({
        caseId: c.case_id,
        patientCode: c.patient_code,
        riskLevel: c.risk_level,
        modality: c.modality || '-',
        region: regionLabel(c.body_region),
        confidence: c.average_confidence ? `${Math.round(c.average_confidence * 100)}%` : '-',
        status: reviewLabel(c.review_status),
        replyText: c.recommendation || '-',
        target: c.body_region,
        pdfUrl: `/api/public/cases/${encodeURIComponent(code)}/pdf`,
      });
    } catch (err) {
      showAlert(el.uploadError, err.message);
    }
  });

  el.reviewForm?.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!state.currentCaseId) return;
    const fd = new FormData();
    fd.append('risk_level', el.revRisk.value);
    fd.append('suspected_malignant', el.revMalignant.value);
    fd.append('recommendation', el.revNote.value.trim());
    try {
      await api(`/api/cases/${state.currentCaseId}/review`, { method: 'POST', body: fd });
      showAlert(el.actionFeedback, 'Đã gửi đánh giá lâm sàng.', false);
      await loadCases();
    } catch (err) { showAlert(el.actionFeedback, err.message, true); }
  });

  el.assignBtn?.addEventListener('click', async () => {
    if (!state.currentCaseId || !el.assignClinicianSelect.value) return;
    const fd = new FormData();
    fd.append('username', el.assignClinicianSelect.value);
    try {
      await api(`/api/cases/${state.currentCaseId}/assign`, { method: 'POST', body: fd });
      showAlert(el.actionFeedback, 'Đã phân công bác sĩ.', false);
    } catch (err) { showAlert(el.actionFeedback, err.message, true); }
  });

  el.linkViewerBtn?.addEventListener('click', async () => {
    if (!state.currentCaseId || !el.linkViewerSelect.value) return;
    const fd = new FormData();
    fd.append('viewer_username', el.linkViewerSelect.value);
    try {
      await api(`/api/cases/${state.currentCaseId}/link-patient`, { method: 'POST', body: fd });
      showAlert(el.actionFeedback, 'Đã gắn hồ sơ cho bệnh nhân.', false);
    } catch (err) { showAlert(el.actionFeedback, err.message, true); }
  });

  el.approveBtn?.addEventListener('click', async () => {
    if (!state.currentCaseId) return;
    try {
      const res = await api(`/api/cases/${state.currentCaseId}/approve`, { method: 'POST' });
      el.publicCodeBanner.textContent = `Đã phê duyệt. Mã tra cứu công khai: ${res.public_code}`;
      el.publicCodeBanner.classList.remove('hidden');
      await loadCases();
    } catch (err) { showAlert(el.actionFeedback, err.message, true); }
  });

  async function sendChatMessage() {
    const text = el.chatInput?.value.trim();
    if (!text) return;
    if (!state.currentConversationId) {
      appendChatBubble('user', text);
      appendChatBubble('assistant', 'Hãy tải và phân tích một phim chụp trước để gắn ghi chú vào hồ sơ.');
      el.chatInput.value = '';
      return;
    }
    const fd = new FormData();
    fd.append('sender', 'user');
    fd.append('text', text);
    try {
      await api(`/api/conversations/${state.currentConversationId}/messages`, { method: 'POST', body: fd });
      appendChatBubble('user', text);
      el.chatInput.value = '';
    } catch (err) { showAlert(el.actionFeedback, err.message, true); }
  }
  el.chatSendBtn?.addEventListener('click', sendChatMessage);
  el.chatInput?.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); sendChatMessage(); }
  });

  populateTargets();
  loadStatus();
  loadCases();
})();
