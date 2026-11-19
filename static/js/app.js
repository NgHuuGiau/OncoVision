/**
 * OncoVision AI Workstation Client Application
 */

const CANCER_TARGETS = window.CANCER_TARGETS || [];
let conversations = [];
let activeConvId = null;
let activeConvIndex = 0;
let pendingAttachments = [];
let currentLang = localStorage.getItem('lang') || 'vi';
let isRecording = false;
let recognition = null;
let isAnalyzing = false;

const TR = {
  vi: {
    new_chat: 'Ca chẩn đoán mới',
    search: 'Tìm ca bệnh/bệnh nhân...',
    history: 'Hồ Sơ Ca Lâm Sàng',
    settings: 'Cấu hình hệ thống',
    app_name: 'OncoVision AI™',
    greeting_title: 'Hệ Thống Trợ Lý Chẩn Đoán',
    greeting_text: 'Xin chào! Hãy tải phim chụp lên bảng điều khiển trung tâm hoặc đặt câu hỏi y khoa dưới đây để nhận hỗ trợ sàng lọc tự động.',
    input_placeholder: 'Nhập câu hỏi lâm sàng tại đây...',
    disclaimer: 'Kết quả phân tích từ AI chỉ mang tính tham khảo hỗ trợ, vui lòng xác chẩn với bác sĩ chuyên khoa lâm sàng.',
    attach: 'Đính kèm',
    camera: 'Mở camera',
    choose_image: 'Chọn ảnh chụp',
    medical_target: 'Nhóm bệnh học',
    medical_modality: 'Loại ảnh (Modality)',
    today: 'Hôm nay',
    medical_analyzing: 'Đang phân tích tổn thương y khoa...',
    medical_pending: 'Đang chạy pipeline sàng lọc y dược...',
    system_reply_text: 'Tôi đã nhận nội dung và sẽ xử lý trong ngữ cảnh ca bệnh này.',
  },
  en: {
    new_chat: 'New Diagnostic Case',
    search: 'Search case/patient...',
    history: 'Clinical Case Logs',
    settings: 'System Settings',
    app_name: 'OncoVision AI™',
    greeting_title: 'Diagnostic Assistant System',
    greeting_text: 'Welcome! Upload a medical scan to the center view or write a clinical query below for automated analysis support.',
    input_placeholder: 'Enter clinical query here...',
    disclaimer: 'AI results are for diagnostic reference only. Please confirm with clinical specialists.',
    attach: 'Attach',
    camera: 'Open camera',
    choose_image: 'Choose image',
    medical_target: 'Cancer Targets',
    medical_modality: 'Imaging Modality',
today: 'Today',
    medical_analyzing: 'Analyzing medical imaging lesion...',
    medical_pending: 'Medical analysis pipeline is running...',
    system_reply_text: 'I received the content and will process it locally in this case context.',
  }
};

function t(key) {
  return (TR[currentLang] || TR.vi)[key] || key;
}

function applyLanguage(lang) {
  currentLang = lang;
  localStorage.setItem('lang', lang);
  fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body: `language=${encodeURIComponent(lang)}`
  }).catch(() => {});
  translateUI();
}

function translateUI() {
  const newChatSpan = document.querySelector('.new-chat-btn span');
  if (newChatSpan) newChatSpan.textContent = t('new_chat');
  const searchInput = document.getElementById('searchInput');
  if (searchInput) searchInput.placeholder = t('search');
  const histTitle = document.querySelector('.history-title');
  if (histTitle) histTitle.textContent = t('history');
  const settingsBtnSpan = document.querySelector('.settings-btn[onclick="openSettings()"] span');
  if (settingsBtnSpan) settingsBtnSpan.textContent = t('settings');
  const msgInput = document.getElementById('messageInput');
  if (msgInput) msgInput.placeholder = t('input_placeholder');
  const greetingH3 = document.querySelector('.greeting-card h3');
  if (greetingH3) greetingH3.textContent = t('greeting_title');
  const greetingP = document.querySelector('.greeting-card p');
  if (greetingP) greetingP.textContent = t('greeting_text');
  const disclaimer = document.querySelector('.disclaimer');
  if (disclaimer) disclaimer.textContent = t('disclaimer');
  const labels = document.querySelectorAll('.control-group label');
  if (labels.length >= 2) {
    labels[0].textContent = t('medical_target');
    labels[1].textContent = t('medical_modality');
  }
}

let currentChatWidthMode = localStorage.getItem('chatWidthMode') || 'normal';
let isChatPanelCollapsed = localStorage.getItem('chatCollapsed') === '1';

function toggleSidebar(forceOpen) {
  const sb = document.getElementById('sidebar');
  const appContainer = document.querySelector('.app-container');
  const toggleBtn = document.getElementById('sidebarToggle');
  if (!sb) return;

  const willCollapse = forceOpen !== undefined ? !forceOpen : !sb.classList.contains('collapsed');
  sb.classList.toggle('collapsed', willCollapse);
  if (appContainer) {
    appContainer.classList.toggle('sidebar-collapsed', willCollapse);
  }
  if (toggleBtn) {
    toggleBtn.title = willCollapse ? 'Mở thanh bên (Ctrl+B)' : 'Thu gọn thanh bên (Ctrl+B)';
  }
  localStorage.setItem('sidebarCollapsed', willCollapse ? '1' : '0');
}

function setChatWidth(mode, customPx) {
  const appContainer = document.querySelector('.app-container');
  const label = document.getElementById('chatSizeLabel');
  const btn = document.getElementById('chatSizeBtn');
  if (!appContainer) return;

  appContainer.classList.remove('chat-narrow', 'chat-normal', 'chat-wide');

  if (mode === 'custom' && customPx) {
    appContainer.style.setProperty('--chat-width', `${customPx}px`);
    currentChatWidthMode = 'custom';
    if (label) label.textContent = `${customPx}px`;
    if (btn) btn.title = `Độ rộng tùy chỉnh (${customPx}px). Nhấn để chuyển Hẹp/Vừa/Rộng (Alt+W)`;
  } else if (mode === 'narrow') {
    appContainer.style.removeProperty('--chat-width');
    appContainer.classList.add('chat-narrow');
    currentChatWidthMode = 'narrow';
    if (label) label.textContent = 'Hẹp';
    if (btn) btn.title = 'Đang ở chế độ Hẹp (320px). Nhấn để đổi sang Vừa (Alt+W)';
  } else if (mode === 'wide') {
    appContainer.style.removeProperty('--chat-width');
    appContainer.classList.add('chat-wide');
    currentChatWidthMode = 'wide';
    if (label) label.textContent = 'Rộng';
    if (btn) btn.title = 'Đang ở chế độ Rộng (580px). Nhấn để đổi sang Hẹp (Alt+W)';
  } else {
    appContainer.style.removeProperty('--chat-width');
    appContainer.classList.add('chat-normal');
    currentChatWidthMode = 'normal';
    if (label) label.textContent = 'Vừa';
    if (btn) btn.title = 'Đang ở chế độ Vừa (410px). Nhấn để đổi sang Rộng (Alt+W)';
  }
  localStorage.setItem('chatWidthMode', currentChatWidthMode);
}

function cycleChatWidth() {
  if (isChatPanelCollapsed) {
    toggleChatCollapse(false);
  }
  if (currentChatWidthMode === 'narrow') {
    setChatWidth('normal');
  } else if (currentChatWidthMode === 'normal') {
    setChatWidth('wide');
  } else {
    setChatWidth('narrow');
  }
}

function toggleChatCollapse(collapseState) {
  const appContainer = document.querySelector('.app-container');
  const assistant = document.getElementById('aiAssistant') || document.querySelector('.ai-assistant');
  const chatOpenBtn = document.getElementById('chatOpenBtn');
  const chatCollapseBtn = document.getElementById('chatCollapseBtn');
  if (!appContainer || !assistant) return;

  const willCollapse = collapseState !== undefined ? collapseState : !isChatPanelCollapsed;
  isChatPanelCollapsed = willCollapse;

  appContainer.classList.toggle('chat-collapsed', willCollapse);
  assistant.classList.toggle('collapsed', willCollapse);

  if (chatOpenBtn) {
    chatOpenBtn.style.display = willCollapse ? 'inline-flex' : 'none';
  }
  if (chatCollapseBtn) {
    chatCollapseBtn.title = willCollapse ? 'Mở lại khung chat (Ctrl+J)' : 'Thu gọn khung chat (Ctrl+J)';
  }
  localStorage.setItem('chatCollapsed', willCollapse ? '1' : '0');
}

function initChatResizer() {
  const resizer = document.getElementById('chatResizer');
  const appContainer = document.querySelector('.app-container');
  if (!resizer || !appContainer) return;

  let isDragging = false;
  let startX = 0;
  let startWidth = 0;

  resizer.addEventListener('mousedown', (e) => {
    if (isChatPanelCollapsed) return;
    isDragging = true;
    startX = e.clientX;
    const assistant = document.getElementById('aiAssistant') || document.querySelector('.ai-assistant');
    startWidth = assistant ? assistant.getBoundingClientRect().width : 410;
    document.body.classList.add('is-resizing-chat');
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const diff = startX - e.clientX;
    let newWidth = Math.round(startWidth + diff);
    if (newWidth < 280) newWidth = 280;
    const maxW = Math.round(window.innerWidth * 0.65);
    if (newWidth > maxW) newWidth = maxW;
    setChatWidth('custom', newWidth);
  });

  window.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      document.body.classList.remove('is-resizing-chat');
      const assistant = document.getElementById('aiAssistant') || document.querySelector('.ai-assistant');
      if (assistant) {
        const finalW = Math.round(assistant.getBoundingClientRect().width);
        localStorage.setItem('chatCustomWidth', String(finalW));
      }
    }
  });
}

function initKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    const isCtrlOrCmd = e.ctrlKey || e.metaKey;
    if (isCtrlOrCmd && (e.key === 'b' || e.key === 'B')) {
      e.preventDefault();
      toggleSidebar();
    } else if (isCtrlOrCmd && (e.key === 'j' || e.key === 'J')) {
      e.preventDefault();
      toggleChatCollapse();
    } else if (e.altKey && (e.key === 'w' || e.key === 'W')) {
      e.preventDefault();
      cycleChatWidth();
    }
  });
}

// -----------------------------------------------------------------------------
// ĐIỀU KHIỂN PACS TƯƠNG TÁC (ZOOM, PAN, KÉO DI CHUYỂN, ÂM BẢN)
// -----------------------------------------------------------------------------
let pacsZoom = 1.0;
let pacsPanX = 0;
let pacsPanY = 0;
let pacsIsDragging = false;
let pacsStartX = 0;
let pacsStartY = 0;
let pacsIsInverted = false;

function pacsApplyTransform() {
  const wrapper = document.getElementById('pacsImageWrapper');
  const badge = document.getElementById('pacsZoomBadge');
  if (wrapper) {
    wrapper.style.transform = `translate(${pacsPanX}px, ${pacsPanY}px) scale(${pacsZoom})`;
  }
  if (badge) {
    badge.textContent = `${Math.round(pacsZoom * 100)}%`;
  }
}

function pacsZoomIn() {
  pacsZoom = Math.min(4.0, Math.round((pacsZoom + 0.25) * 100) / 100);
  pacsApplyTransform();
}

function pacsZoomOut() {
  pacsZoom = Math.max(0.5, Math.round((pacsZoom - 0.25) * 100) / 100);
  pacsApplyTransform();
}

function pacsResetView() {
  pacsZoom = 1.0;
  pacsPanX = 0;
  pacsPanY = 0;
  pacsApplyTransform();
}

function pacsToggleInvert() {
  const img = document.getElementById('pacsActiveImage');
  const btn = document.getElementById('pacsInvertBtn');
  pacsIsInverted = !pacsIsInverted;
  if (img) img.classList.toggle('inverted', pacsIsInverted);
  if (btn) btn.classList.toggle('active', pacsIsInverted);
}

function initPacsInteractions() {
  const container = document.getElementById('pacsScreen');
  const wrapper = document.getElementById('pacsImageWrapper');
  if (!container || !wrapper) return;

  // Lăn chuột để zoom ảnh
  container.addEventListener('wheel', (e) => {
    const activeImg = document.getElementById('pacsActiveImage');
    if (!activeImg || !activeImg.src || activeImg.src.endsWith('/')) return;
    e.preventDefault();
    if (e.deltaY < 0) {
      pacsZoomIn();
    } else {
      pacsZoomOut();
    }
  }, { passive: false });

  // Kéo chuột trái để di chuyển ảnh (Pan)
  wrapper.addEventListener('mousedown', (e) => {
    const activeImg = document.getElementById('pacsActiveImage');
    if (!activeImg || !activeImg.src || activeImg.src.endsWith('/')) return;
    pacsIsDragging = true;
    pacsStartX = e.clientX - pacsPanX;
    pacsStartY = e.clientY - pacsPanY;
    wrapper.classList.add('is-dragging');
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (!pacsIsDragging) return;
    pacsPanX = e.clientX - pacsStartX;
    pacsPanY = e.clientY - pacsStartY;
    pacsApplyTransform();
  });

  window.addEventListener('mouseup', () => {
    if (pacsIsDragging) {
      pacsIsDragging = false;
      if (wrapper) wrapper.classList.remove('is-dragging');
    }
  });

  // Nhấp đúp chuột vào ảnh để đổi nhanh giữa 100% và 175%
  wrapper.addEventListener('dblclick', (e) => {
    e.preventDefault();
    if (pacsZoom > 1.0) {
      pacsResetView();
    } else {
      pacsZoom = 1.75;
      pacsApplyTransform();
    }
  });

  // Nhấp đúp vào thanh kéo resizer để đặt lại độ rộng chuẩn (410px)
  const resizer = document.getElementById('chatResizer');
  if (resizer) {
    resizer.addEventListener('dblclick', () => {
      setChatWidth('normal');
    });
  }
}

// -----------------------------------------------------------------------------
// ĐỊNH DẠNG BÁO CÁO Y KHOA GỌN GÀNG, ĐẸP MẮT TRONG CHAT
// -----------------------------------------------------------------------------
function formatMedicalAnalysisContent(txt, meta) {
  if (!txt.includes('Kết quả phân tích') && !txt.includes('📋') && !(meta && meta.medical_case_id)) {
    return formatGenericMarkdown(txt);
  }

  const bnMatch = txt.match(/Mã BN:\s*([^\n\r]+)/i);
  const patientCode = bnMatch ? bnMatch[1].trim() : (meta ? meta.patient_code : '');

  const regionMatch = txt.match(/Vùng chọn:\s*([^|\n\r]+)/i);
  const modalityMatch = txt.match(/Modality:\s*([^\n\r]+)/i);
  const region = regionMatch ? regionMatch[1].trim() : (meta ? meta.body_region : '');
  const modality = modalityMatch ? modalityMatch[1].trim() : (meta ? meta.modality : '');

  const clinicalMatch = txt.match(/Đánh giá lâm sàng:\s*([^\n\r]+)/i);
  const clinicalStatus = clinicalMatch ? clinicalMatch[1].trim() : 'chưa thực hiện';

  const timeMatch = txt.match(/⏱\s*([0-9.]+s)/);
  const elapsed = timeMatch ? timeMatch[1] : '';

  let warnings = [];
  const warnMatch = txt.match(/⚠️\s*Cảnh báo:\s*([\s\S]+?)(?=\n\n|\n💡|\n⏱|\n\*|$)/);
  if (warnMatch) {
    const rawWarn = warnMatch[1].trim();
    warnings = rawWarn.split(';').map(w => w.trim()).filter(Boolean);
  }

  let recommendation = '';
  const recMatch = txt.match(/💡\s*([\s\S]+?)(?=\n⏱|\n\*|$)/);
  if (recMatch) {
    recommendation = recMatch[1].trim();
  }

  let disclaimer = '';
  const discMatch = txt.match(/\*Cảnh báo y khoa:[\s\S]*?\*/);
  if (discMatch) {
    disclaimer = discMatch[0].replace(/\*/g, '').trim();
  }

  let html = `<div class="medical-report-block">`;

  // Header Banner
  html += `
    <div class="report-header-banner">
      <div class="title">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
          <line x1="16" y1="13" x2="8" y2="13"></line>
          <line x1="16" y1="17" x2="8" y2="17"></line>
        </svg>
        <span>Kết quả phân tích y khoa</span>
      </div>
      ${patientCode ? `<span class="badge">Mã BN: ${escHtml(patientCode)}</span>` : ''}
    </div>
  `;

  // Metadata Chips
  html += `<div class="report-meta-chips">`;
  if (region) html += `<span class="report-chip chip-target">🎯 Vùng: <strong>${escHtml(region)}</strong></span>`;
  if (modality) html += `<span class="report-chip">📷 Modality: <strong>${escHtml(modality)}</strong></span>`;
  html += `<span class="report-chip">⚪ Lâm sàng: <strong>${escHtml(clinicalStatus)}</strong></span>`;
  if (elapsed) html += `<span class="report-chip chip-time">⏱ ${escHtml(elapsed)}</span>`;
  html += `</div>`;

  // Warnings Alert Box
  if (warnings.length > 0) {
    html += `
      <div class="warning-alert-card">
        <div class="alert-title">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
            <line x1="12" y1="9" x2="12" y2="13"></line>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
          </svg>
          <span>Cảnh báo chất lượng ảnh & kiểm định:</span>
        </div>
        <ul>
          ${warnings.map(w => `<li>${escHtml(w)}</li>`).join('')}
        </ul>
      </div>
    `;
  }

  // Recommendation Callout
  if (recommendation) {
    html += `
      <div class="recommendation-callout">
        <strong>💡 Đề xuất hỗ trợ:</strong> ${escHtml(recommendation)}
      </div>
    `;
  }

  // Disclaimer Box
  if (disclaimer) {
    html += `
      <div class="medical-disclaimer-box">
        <strong>* Cảnh báo y khoa:</strong> ${escHtml(disclaimer.replace(/\n+/g, ' '))}
      </div>
    `;
  }

  html += `</div>`;
  return html;
}

function formatGenericMarkdown(txt) {
  let safe = escHtml(txt);
  safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  safe = safe.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  safe = safe.replace(/\n/g, '<br>');
  return safe;
}

function initTargets() {
  const s = document.getElementById('targetSelect');
  if (!s || !CANCER_TARGETS.length) return;
  s.innerHTML = '';
  CANCER_TARGETS.forEach(target => {
    const o = document.createElement('option');
    o.value = target.key;
    o.disabled = !target.model_ready;
    o.textContent = target.label + (target.model_ready ? '' : ' — chưa sẵn sàng');
    s.appendChild(o);
  });
  const firstReady = CANCER_TARGETS.find(target => target.model_ready);
  if (!firstReady) {
    s.disabled = true;
    const th = document.getElementById('targetHint');
    if (th) th.textContent = 'Model phân tích chưa sẵn sàng; cần bổ sung model tương thích trước khi xử lý ảnh.';
    return;
  }
  s.value = firstReady.key;
  s.onchange = updateTargetSelection;
  updateTargetSelection();
}

function selectedTarget() {
  const s = document.getElementById('targetSelect');
  return CANCER_TARGETS.find(target => target.key === (s ? s.value : ''));
}

function updateTargetSelection() {
  const target = selectedTarget();
  if (!target) return;
  updateModalityList(target.modalities);
  const th = document.getElementById('targetHint');
  if (th) th.textContent = 'Modality được đối chiếu với tệp; việc này chưa xác nhận model đã được huấn luyện trên loại ảnh đó.';
}

function updateModalityList(modalities) {
  const s = document.getElementById('modalitySelect');
  if (!s) return;
  s.innerHTML = '';
  (modalities || []).forEach(m => {
    const o = document.createElement('option');
    o.value = m;
    o.textContent = m;
    s.appendChild(o);
  });
  s.onchange = () => {
    const mh = document.getElementById('modalityHint');
    if (mh) mh.textContent = 'Modality khai báo: ' + s.value;
  };
  s.onchange();
}

function applyThemeSetting(v) {
  setTheme(v);
}

function setTheme(mode) {
  localStorage.setItem('theme', mode);
  const light = mode === 'light'
    || (mode === 'system' && window.matchMedia('(prefers-color-scheme: light)').matches);
  document.documentElement.classList.toggle('light', light);
  document.body.classList.toggle('light', light);
  const btn = document.getElementById('themeToggleBtn');
  if (btn) btn.textContent = light ? '◑ Giao diện tối' : '◑ Giao diện sáng';
}

function toggleTheme() {
  setTheme(document.documentElement.classList.contains('light') ? 'dark' : 'light');
}

setTheme(localStorage.getItem('theme') || 'system');

function closeAllModals() {
  document.querySelectorAll('.modal-overlay.active').forEach(m => m.classList.remove('active'));
  const preview = document.getElementById('imagePreviewModal');
  if (preview) {
    preview.classList.remove('active');
    preview.style.display = 'none';
  }
}

function openSettings() {
  closeAllModals();
  const m = document.getElementById('settingsModal');
  if (m) m.classList.add('active');
}

function closeSettings() {
  const m = document.getElementById('settingsModal');
  if (m) m.classList.remove('active');
}

async function loadConversations() {
  try {
    const r = await fetch('/api/conversations');
    const d = await r.json();
    conversations = d.conversations || [];
  } catch (e) {
    conversations = [];
  }
  if (!conversations.length) {
    conversations = [{ id: null, title: t('new_chat'), subtitle: t('today'), messages: [] }];
  }
  renderHistory();
  if (activeConvId === null && conversations[0]) {
    selectConv(0);
  }
}

function renderHistory() {
  const searchEl = document.getElementById('searchInput');
  const q = (searchEl ? searchEl.value : '').toLowerCase().trim();
  const p = document.getElementById('historyPanel');
  if (!p) return;
  p.innerHTML = '';
  conversations.forEach((c, i) => {
    if (q && !c.title.toLowerCase().includes(q) && !(c.subtitle || '').toLowerCase().includes(q)) return;
    const d = document.createElement('div');
    d.className = 'history-item' + (i === activeConvIndex ? ' active' : '');
    d.innerHTML = `
      <div class="history-item-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
        </svg>
      </div>
      <div class="history-item-text">
        <div class="history-item-title">${escHtml(c.title || t('new_chat'))}</div>
        <div class="history-item-subtitle">${escHtml(c.subtitle || '')}</div>
      </div>
    `;
    d.onclick = () => selectConv(i);
    d.oncontextmenu = e => {
      e.preventDefault();
      if (confirm('Xóa trò chuyện này?')) deleteConv(i);
    };
    p.appendChild(d);
  });
}

function filterHistory() {
  renderHistory();
}

async function newChat() {
  try {
    const r = await fetch('/api/conversations', { method: 'POST' });
    const d = await r.json();
    conversations.unshift({ id: d.conversation_id, title: t('new_chat'), subtitle: t('today'), messages: [] });
    selectConv(0);
  } catch (e) {
    conversations.unshift({ id: null, title: t('new_chat'), subtitle: t('today'), messages: [] });
    selectConv(0);
  }
}

async function selectConv(i) {
  if (i < 0 || i >= conversations.length) return;
  activeConvIndex = i;
  const c = conversations[i];
  if (c.id) {
    try {
      const r = await fetch(`/api/conversations/${c.id}`);
      const d = await r.json();
      if (d.ok) {
        c.messages = d.conversation.messages;
      }
    } catch (e) {}
  }
  activeConvId = c.id;
  renderHistory();
  renderMessages();
  syncPACSView();
}

async function deleteConv(i) {
  const c = conversations[i];
  if (c && c.id) {
    try {
      await fetch(`/api/conversations/${c.id}`, { method: 'DELETE' });
    } catch (e) {}
  }
  conversations.splice(i, 1);
  if (!conversations.length) {
    conversations.push({ id: null, title: t('new_chat'), subtitle: t('today'), messages: [] });
  }
  selectConv(Math.min(activeConvIndex, conversations.length - 1));
}

function syncPACSView() {
  const c = conversations[activeConvIndex];
  let lastImage = null;
  let patientCode = 'CHƯA CÓ';

  if (c && c.messages) {
    for (let j = c.messages.length - 1; j >= 0; j--) {
      const msg = c.messages[j];
      if (patientCode === 'CHƯA CÓ' && msg.metadata_json) {
        try {
          const meta = JSON.parse(msg.metadata_json);
          if (meta && meta.patient_code) patientCode = meta.patient_code;
        } catch (e) {}
      }
      if (!lastImage && msg.attachment_path && (msg.attachment_kind === 'image' || msg.attachment_kind === 'camera')) {
        lastImage = msg.attachment_path;
      }
    }
  }

  const emptyDropzone = document.getElementById('emptyDropzone');
  const pacsImageContainer = document.getElementById('pacsImageContainer');
  const pacsActiveImage = document.getElementById('pacsActiveImage');
  const pacsPatientCodeEl = document.getElementById('pacsPatientCode');
  const pacsDateEl = document.getElementById('pacsDate');
  const pacsToolsOverlay = document.getElementById('pacsToolsOverlay');

  if (pacsPatientCodeEl) pacsPatientCodeEl.textContent = patientCode;

  if (lastImage) {
    if (pacsActiveImage) pacsActiveImage.src = attachmentUrl(lastImage);
    if (emptyDropzone) emptyDropzone.style.display = 'none';
    if (pacsImageContainer) pacsImageContainer.style.display = 'flex';
    if (pacsToolsOverlay) pacsToolsOverlay.style.display = 'flex';
    if (pacsDateEl) pacsDateEl.textContent = new Date().toLocaleDateString('vi-VN');
    pacsResetView();
  } else {
    if (pacsActiveImage) pacsActiveImage.src = '';
    if (emptyDropzone) emptyDropzone.style.display = 'flex';
    if (pacsImageContainer) pacsImageContainer.style.display = 'none';
    if (pacsToolsOverlay) pacsToolsOverlay.style.display = 'none';
    if (pacsDateEl) pacsDateEl.textContent = '-';
  }
}

function renderMessages() {
  const c = conversations[activeConvIndex];
  const hasMsg = c && c.messages && c.messages.length > 0;
  const greeting = document.getElementById('greetingCard');
  const container = document.getElementById('messagesContainer');
  if (greeting) greeting.style.display = hasMsg ? 'none' : 'block';
  if (container) container.style.display = hasMsg ? 'block' : 'none';
  if (!hasMsg || !container) return;
  container.innerHTML = '';
  c.messages.forEach(msg => addBubble(msg, false));
  const area = document.getElementById('messagesArea');
  if (area) area.scrollTop = area.scrollHeight;
}

function attachmentUrl(p) {
  if (!p) return '';
  const norm = String(p).replace(/\\/g, '/');
  // Luôn dùng /output/ prefix để đi qua endpoint serve_output_file
  const i = norm.toLowerCase().lastIndexOf('/output/');
  if (i >= 0) return norm.slice(i);
  // Nếu đường dẫn không chứa /output/, tìm output_dir từ tên file
  // và wrap vào /output/ prefix
  const lower = norm.toLowerCase();
  if (lower.includes('web_uploads') || lower.includes('processed_images') ||
      lower.includes('gradcam') || lower.includes('overlay') ||
      lower.includes('reports') || lower.includes('medical')) {
    // Tìm phần tương đối từ output dir
    const outputIdx = lower.lastIndexOf('output');
    if (outputIdx >= 0) {
      return '/output/' + norm.slice(outputIdx + 6).replace(/^\/+/, '');
    }
    // Legacy: web_uploads/... relative path (no output/ prefix)
    if (lower.startsWith('web_uploads/') || lower.startsWith('processed_images/') ||
        lower.startsWith('gradcam_') || lower.startsWith('overlay_')) {
      return '/output/' + norm.replace(/^\/+/, '');
    }
  }
  // Desktop attachments: absolute paths outside output/ (e.g. D:\...\data\datasets\...)
  // These cannot be served by web server — return empty to avoid 404
  if (/^[a-zA-Z]:\//.test(norm)) {
    return '';
  }
  return '/' + norm.replace(/^\/+/, '');
}

function addBubble(msg, animate = true) {
  const container = document.getElementById('messagesContainer');
  if (!container) return;
  const row = document.createElement('div');
  row.className = 'message-row ' + (msg.sender || 'assistant');
  if (!animate) row.style.animation = 'none';
  let html = '<div class="bubble">';
  if (msg.attachment_path && msg.attachment_kind) {
    const fname = msg.attachment_path.split('/').pop().split('\\').pop();
    html += `<div class="attachment-name">📎 ${escHtml(fname)}</div>`;
    if (msg.attachment_kind === 'image' || msg.attachment_kind === 'camera') {
      html += `<img src="${attachmentUrl(msg.attachment_path)}" onclick="previewImage(this.src)" loading="lazy">`;
    }
  }
  const txt = msg.text || '';
  let meta = null;
  try {
    meta = JSON.parse(msg.metadata_json || 'null');
  } catch (e) {}
  if (/Cannot read|does not support image/i.test(txt)) {
    html += escHtml('Đã nhận ảnh: ' + (msg.attachment_path || ''));
  } else if (txt.startsWith('Phân tích lỗi:') || txt.startsWith('Error:')) {
    html += `<div class="error-text">${escHtml(txt)}</div>`;
  } else {
    html += formatMedicalAnalysisContent(txt, meta);
  }
  if (meta && meta.medical_case_id) {
    // Tách mỗi detection thành 1 thẻ riêng để dễ nhìn
    const dets = meta.detections || [];
    if (dets.length > 1) {
      // Nhiều tọa độ → mỗi tọa độ 1 thẻ riêng
      dets.forEach((det, idx) => {
        html += buildSingleDetectionCard(meta, det, msg.image_name, idx, dets.length);
      });
    } else {
      // 1 tọa độ hoặc không có → dùng thẻ tổng hợp như cũ
      html += buildResultCard(meta, msg.image_name);
    }
  }
  html += '</div>';
  row.innerHTML = html;
  container.appendChild(row);
  if (animate) {
    setTimeout(() => {
      const area = document.getElementById('messagesArea');
      if (area) area.scrollTop = area.scrollHeight;
    }, 50);
  }
}

function showAILoadingIndicator() {
  hideAILoadingIndicator();
  const container = document.getElementById('messagesContainer');
  if (!container) return;
  const row = document.createElement('div');
  row.id = 'aiLoadingRow';
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="ai-loading-bubble">
      <div class="ai-loading-dots">
        <span></span><span></span><span></span>
      </div>
      <span>${t('medical_analyzing')}</span>
    </div>
  `;
  container.appendChild(row);
  const area = document.getElementById('messagesArea');
  if (area) area.scrollTop = area.scrollHeight;
}

function hideAILoadingIndicator() {
  const row = document.getElementById('aiLoadingRow');
  if (row && row.parentNode) {
    row.parentNode.removeChild(row);
  }
}

// ── Build a single detection card (1 tọa độ = 1 thẻ) ──
function buildSingleDetectionCard(m, det, imageName, detIndex, totalDets) {
  const risk = (m.risk_level || 'uncertain').toLowerCase();
  const bodyRegion = m.body_region || m.requested_target || null;
  const modality = m.modality || m.requested_modality || null;
  const modelName = m.model_name || null;

  // ── Cancer label mapping ──
  const cancerLabels = {
    brain: 'Não', thyroid: 'Tuyến giáp', lung: 'Phổi', breast: 'Vú',
    stomach: 'Dạ dày', liver: 'Gan', pancreas: 'Tụy',
    kidney: 'Thận', colorectal: 'Đại trực tràng', prostate: 'Tuyến tiền liệt',
    cervical: 'Cổ tử cung'
  };
  const cancerLabel = bodyRegion ? (cancerLabels[bodyRegion] || escHtml(bodyRegion)) : null;

  // ── Risk level label ──
  const riskLabels = { high: 'Cao', medium: 'Trung bình', low: 'Thấp', uncertain: 'Chưa rõ' };
  const riskLabel = riskLabels[risk] || risk;

  // ── Image name ──
  const displayName = imageName || (m.image_path ? m.image_path.split('/').pop().split('\\').pop() : null);

  // ── Context info ──
  let contextHtml = '';
  const contextItems = [];
  if (cancerLabel) contextItems.push({ icon: '🎯', label: 'Vùng', value: cancerLabel });
  if (modality) contextItems.push({ icon: '📷', label: 'Modality', value: escHtml(modality) });
  if (modelName) contextItems.push({ icon: '🤖', label: 'Model', value: escHtml(modelName) });
  if (contextItems.length > 0) {
    contextHtml = `<div class="result-context"><div class="context-grid">${contextItems.map(item => `<div class="context-item"><span class="context-icon">${item.icon}</span><div class="context-text"><span class="context-label">${item.label}</span><span class="context-value">${item.value}</span></div></div>`).join('')}</div></div>`;
  }

  // ── Position string ──
  let position = '-';
  const bboxCoords = det.bbox || det.bbox_source || null;
  if (det.bbox) {
    const [x1, y1, x2, y2] = det.bbox;
    position = `(${x1}, ${y1}) → (${x2}, ${y2})`;
    if (det.bbox_image_size) {
      position += ` [${det.bbox_image_size[0]}×${det.bbox_image_size[1]}]`;
    }
  } else if (det.bbox_source) {
    const [x1, y1, x2, y2] = det.bbox_source;
    position = `gốc(${x1}, ${y1}) → (${x2}, ${y2})`;
    if (det.bbox_source_image_size) {
      position += ` [${det.bbox_source_image_size[0]}×${det.bbox_source_image_size[1]}]`;
    }
  }
  if (!det.bbox_reliable) position += ' ⚠️ vị trí không đáng tin';
  const confPct = (det.confidence * 100).toFixed(1);

  // ── Crop image URL ──
  // Dùng processed_image_path (ảnh canvas có vẽ bbox) thay vì source_image
  // vì tọa độ bbox nằm trong hệ canvas (640x640)
  // Dùng attachmentUrl() để chuyển đổi path đúng cách, giống như các ảnh khác
  let cropImageHtml = '';
  const rawImgPath = m.processed_image_path || m.image_path || '';
  const isReliable = det.bbox_reliable !== false;  // Mặc định true nếu không có
  if (rawImgPath && bboxCoords && isReliable) {
    // Dùng attachmentUrl() để chuyển absolute path thành relative path từ /output/
    const imgUrl = attachmentUrl(rawImgPath);
    if (imgUrl) {
      // attachmentUrl() trả về "/output/medical/processed_images/..."
      // Bỏ "/output/" prefix để lấy relative path từ OUTPUT_DIR
      const relPath = imgUrl.startsWith('/output/') ? imgUrl.slice(8) : imgUrl.replace(/^\/+/, '');
      const [x1, y1, x2, y2] = bboxCoords;
      const cropUrl = `/api/crop?image_path=${encodeURIComponent(relPath)}&bbox=${x1},${y1},${x2},${y2}`;
      cropImageHtml = `<div class="detection-crop-image"><img src="${cropUrl}" onclick="previewImage(this.src)" loading="lazy" alt="Vùng phát hiện"></div>`;
    }
  }

  // ── Case info tags ──
  let caseInfoHtml = '';
  const caseFields = [];
  if (m.patient_code) caseFields.push({ label: 'Mã BN', value: escHtml(m.patient_code) });
  if (m.patient_age) caseFields.push({ label: 'Tuổi', value: m.patient_age + ' tuổi' });
  if (m.patient_sex) caseFields.push({ label: 'Giới tính', value: escHtml(m.patient_sex) });
  if (caseFields.length > 0) {
    caseInfoHtml = `<div class="result-case-info"><b>Thông tin ca</b><div class="case-info-tags">${caseFields.map(f => `<span class="case-info-tag"><span class="case-info-label">${f.label}:</span><span class="case-info-value">${f.value}</span></span>`).join('')}</div></div>`;
  }

  // ── Recommendation (only on first detection card) ──
  let recommendationHtml = '';
  if (detIndex === 0 && m.recommendation) {
    recommendationHtml = `<div class="result-recommendation"><b>💡 Đề xuất hỗ trợ</b><p>${escHtml(m.recommendation)}</p></div>`;
  }

  // ── Quality warnings (only on first detection card) ──
  let warningsHtml = '';
  const qualityWarnings = m.quality_warnings || [];
  if (detIndex === 0 && qualityWarnings.length > 0) {
    warningsHtml = `<div class="result-warnings"><b>⚠️ Cảnh báo chất lượng</b><ul class="warnings-list">${qualityWarnings.map(w => `<li>${escHtml(w)}</li>`).join('')}</ul></div>`;
  }

  // ── Malignancy (only on first detection card) ──
  let malignancyHtml = '';
  if (detIndex === 0 && m.clinical_assessment_available) {
    if (m.suspected_malignant) {
      malignancyHtml = `<div class="result-malignancy malignant"><span class="malignancy-icon">⚠️</span><div class="malignancy-text"><b>Nghi ngờ ác tính</b><span>Cần hội chẩn và xét nghiệm bổ sung ngay</span></div></div>`;
    } else {
      malignancyHtml = `<div class="result-malignancy benign"><span class="malignancy-icon">✅</span><div class="malignancy-text"><b>Không nghi ngờ ác tính</b><span>Tiếp tục theo dõi định kỳ</span></div></div>`;
    }
  }

  // ── Actions (only on last detection card) ──
  let actionsHtml = '';
  if (detIndex === totalDets - 1) {
    actionsHtml = `<div class="result-actions">
        <a class="result-btn" href="/api/cases/${m.medical_case_id}/pdf" download>⬇ Xuất PDF</a>
        <button class="result-btn" onclick="openCompare(${m.medical_case_id})">⇄ So sánh</button>
      </div>`;
  }

  // ── Grad-CAM (only on first detection card) ──
  let gradcamHtml = '';
  const gradcams = (m.gradcam_overlays || []).filter(Boolean);
  if (detIndex === 0 && gradcams.length) {
    gradcamHtml = `<p class="workflow-muted">Grad-CAM chỉ minh họa vùng ảnh ảnh hưởng đến dự đoán, không phải ranh giới tổn thương đã xác nhận.</p><div class="gradcam-strip">${gradcams.map(p => `<img src="${attachmentUrl(p)}" onclick="previewImage(this.src)" title="Grad-CAM heatmap" loading="lazy">`).join('')}</div>`;
  }

  return `
    <div class="result-card detection-card">
      <div class="result-card-head">
        <div class="result-card-title">
          <span class="result-card-icon">🔍</span>
          <b>Tọa độ ${detIndex + 1}/${totalDets}</b>
          ${displayName ? `<span class="result-image-name">📷 ${escHtml(displayName)}</span>` : ''}
        </div>
        <span class="risk-badge risk-${risk}">${m.clinical_assessment_available ? escHtml(riskLabel) : 'Chưa đánh giá lâm sàng'}</span>
      </div>
      ${contextHtml}
      ${caseInfoHtml}
      <div class="result-stats">
        <div class="result-stat"><b>${escHtml(det.label)}</b><span>Nhãn dự đoán</span></div>
        <div class="result-stat"><b>${confPct}%</b><span>Điểm tin cậy</span></div>
        <div class="result-stat"><span class="conf-bar-wrap"><span class="conf-bar-fill" style="width:${Math.round(det.confidence * 100)}%"></span></span></div>
      </div>
      ${cropImageHtml}
      <div class="result-section">
        <b class="result-section-title">📐 Vị trí phát hiện</b>
        <div class="detection-position" style="font-size:14px;font-family:monospace;padding:12px;background:var(--bg-secondary);border-radius:8px;text-align:center;">${position}</div>
      </div>
      ${malignancyHtml}
      ${warningsHtml}
      ${recommendationHtml}
      ${gradcamHtml}
      ${actionsHtml}
    </div>
  `;
}

function buildResultCard(m, imageName) {
  const risk = (m.risk_level || 'uncertain').toLowerCase();
  const dets = m.detections || [];
  const gradcams = (m.gradcam_overlays || []).filter(Boolean);
  const avgConf = m.average_confidence != null ? (m.average_confidence * 100).toFixed(1) : null;
  const bodyRegion = m.body_region || m.requested_target || null;
  const modality = m.modality || m.requested_modality || null;
  const modelName = m.model_name || null;
  const qualityWarnings = m.quality_warnings || [];
  const recommendation = m.recommendation || '';
  const inputDim = m.input_dimension || '2d';

  // ── Cancer label mapping (tiếng Việt) ──
  const cancerLabels = {
    brain: 'Não', thyroid: 'Tuyến giáp', lung: 'Phổi', breast: 'Vú',
    stomach: 'Dạ dày', liver: 'Gan', pancreas: 'Tụy',
    kidney: 'Thận', colorectal: 'Đại trực tràng', prostate: 'Tuyến tiền liệt',
    cervical: 'Cổ tử cung'
  };
  const cancerLabel = bodyRegion ? (cancerLabels[bodyRegion] || escHtml(bodyRegion)) : null;

  // ── Risk level label ──
  const riskLabels = {
    high: 'Cao', medium: 'Trung bình', low: 'Thấp', uncertain: 'Chưa rõ'
  };
  const riskLabel = riskLabels[risk] || risk;

  // ── Confidence level indicator ──
  let confLevel = '';
  let confLevelClass = '';
  if (avgConf != null) {
    const conf = parseFloat(avgConf);
    if (conf >= 80) { confLevel = '🟢 Cao'; confLevelClass = 'conf-high'; }
    else if (conf >= 50) { confLevel = '🟡 Trung bình'; confLevelClass = 'conf-medium'; }
    else { confLevel = '🔴 Thấp'; confLevelClass = 'conf-low'; }
  }

  // ── Malignancy indicator ──
  let malignancyHtml = '';
  if (m.clinical_assessment_available) {
    if (m.suspected_malignant) {
      malignancyHtml = `<div class="result-malignancy malignant"><span class="malignancy-icon">⚠️</span><div class="malignancy-text"><b>Nghi ngờ ác tính</b><span>Cần hội chẩn và xét nghiệm bổ sung ngay</span></div></div>`;
    } else {
      malignancyHtml = `<div class="result-malignancy benign"><span class="malignancy-icon">✅</span><div class="malignancy-text"><b>Không nghi ngờ ác tính</b><span>Tiếp tục theo dõi định kỳ</span></div></div>`;
    }
  }

  // ── Quality warnings ──
  let warningsHtml = '';
  if (qualityWarnings.length > 0) {
    warningsHtml = `<div class="result-warnings"><b>⚠️ Cảnh báo chất lượng</b><ul class="warnings-list">${qualityWarnings.map(w => `<li>${escHtml(w)}</li>`).join('')}</ul></div>`;
  }

  // ── Recommendation ──
  let recommendationHtml = '';
  if (recommendation) {
    recommendationHtml = `<div class="result-recommendation"><b>💡 Đề xuất hỗ trợ</b><p>${escHtml(recommendation)}</p></div>`;
  }

  // ── Detections table ──
  let detRows = dets.slice(0, 10).map((d, idx) => {
    let position = '-';
    if (d.bbox) {
      const [x1, y1, x2, y2] = d.bbox;
      position = `(${x1},${y1})-(${x2},${y2})`;
      if (d.bbox_image_size) {
        position += ` [${d.bbox_image_size[0]}x${d.bbox_image_size[1]}]`;
      }
    } else if (d.bbox_source) {
      const [x1, y1, x2, y2] = d.bbox_source;
      position = `gốc(${x1},${y1})-(${x2},${y2})`;
      if (d.bbox_source_image_size) {
        position += ` [${d.bbox_source_image_size[0]}x${d.bbox_source_image_size[1]}]`;
      }
    }
    if (!d.bbox_reliable) position += ' ⚠️';
    const confPct = (d.confidence * 100).toFixed(1);
    return `<tr><td>#${idx + 1}</td><td>${escHtml(d.label)}</td><td><span class="conf-bar-wrap"><span class="conf-bar-fill" style="width:${Math.round(d.confidence * 100)}%"></span></span>${confPct}%</td><td style="font-size:11px;font-family:monospace">${position}</td></tr>`;
  }).join('');
  if (!detRows) detRows = '<tr><td colspan="4">Model không trả về nhãn dự đoán.</td></tr>';

  // ── Grad-CAM ──
  const gradcamHtml = gradcams.length
    ? `<p class="workflow-muted">Grad-CAM chỉ minh họa vùng ảnh ảnh hưởng đến dự đoán, không phải ranh giới tổn thương đã xác nhận.</p><div class="gradcam-strip">${gradcams.map(p => `<img src="${attachmentUrl(p)}" onclick="previewImage(this.src)" title="Grad-CAM heatmap" loading="lazy">`).join('')}</div>`
    : '';

  // ── Case info tags ──
  let caseInfoHtml = '';
  const caseFields = [];
  if (m.patient_code) caseFields.push({ label: 'Mã BN', value: escHtml(m.patient_code) });
  if (m.patient_age) caseFields.push({ label: 'Tuổi', value: m.patient_age + ' tuổi' });
  if (m.patient_sex) caseFields.push({ label: 'Giới tính', value: escHtml(m.patient_sex) });
  if (m.image_path) {
    const imgName = m.image_path.split('/').pop();
    caseFields.push({ label: 'Ảnh', value: escHtml(imgName) });
  }
  if (caseFields.length > 0) {
    caseInfoHtml = `<div class="result-case-info"><b>Thông tin ca</b><div class="case-info-tags">${caseFields.map(f => `<span class="case-info-tag"><span class="case-info-label">${f.label}:</span><span class="case-info-value">${f.value}</span></span>`).join('')}</div></div>`;
  }

  // ── Context info (vùng, modality, model, dimension) ──
  let contextHtml = '';
  const contextItems = [];
  if (cancerLabel) contextItems.push({ icon: '🎯', label: 'Vùng', value: cancerLabel });
  if (modality) contextItems.push({ icon: '📷', label: 'Modality', value: escHtml(modality) });
  if (modelName) contextItems.push({ icon: '🤖', label: 'Model', value: escHtml(modelName) });
  if (inputDim === '3d') contextItems.push({ icon: '📊', label: 'Chiều', value: '3D Volume' });
  if (contextItems.length > 0) {
    contextHtml = `<div class="result-context"><div class="context-grid">${contextItems.map(item => `<div class="context-item"><span class="context-icon">${item.icon}</span><div class="context-text"><span class="context-label">${item.label}</span><span class="context-value">${item.value}</span></div></div>`).join('')}</div></div>`;
  }

  // ── Stats row ──
  const statsItems = [];
  statsItems.push({ value: dets.length, label: 'Nhãn dự đoán' });
  if (avgConf != null) statsItems.push({ value: avgConf + '%', label: 'Độ tin cậy TB' });
  if (confLevel) statsItems.push({ value: confLevel, label: 'Mức độ tin cậy', class: confLevelClass });
  if (m.clinical_assessment_available) {
    statsItems.push({ value: m.suspected_malignant ? 'Có' : 'Không', label: 'Nghi ngờ ác tính' });
  }
  statsItems.push({ value: riskLabel, label: 'Mức độ rủi ro', class: 'risk-' + risk });

  // ── Image name in header ──
  const displayName = imageName || (m.image_path ? m.image_path.split('/').pop().split('\\').pop() : null);

  return `
    <div class="result-card">
      <div class="result-card-head">
        <div class="result-card-title">
          <span class="result-card-icon">📋</span>
          <b>Kết quả phân tích AI</b>
          ${displayName ? `<span class="result-image-name">📷 ${escHtml(displayName)}</span>` : ''}
        </div>
        <span class="risk-badge risk-${risk}">${m.clinical_assessment_available ? escHtml(riskLabel) : 'Chưa đánh giá lâm sàng'}</span>
      </div>
      ${contextHtml}
      ${caseInfoHtml}
      <div class="result-stats">
        ${statsItems.map(s => `<div class="result-stat ${s.class || ''}"><b>${s.value}</b><span>${s.label}</span></div>`).join('')}
      </div>
      ${malignancyHtml}
      ${warningsHtml}
      ${recommendationHtml}
      <div class="result-section">
        <b class="result-section-title">🔬 Chi tiết phát hiện</b>
        <table class="result-detections"><thead><tr><th>#</th><th>Nhãn</th><th>Điểm tin cậy</th><th>Vị trí</th></tr></thead><tbody>${detRows}</tbody></table>
      </div>
      ${gradcamHtml}
      <div class="result-actions">
        <a class="result-btn" href="/api/cases/${m.medical_case_id}/pdf" download>⬇ Xuất PDF</a>
        <button class="result-btn" onclick="openCompare(${m.medical_case_id})">⇄ So sánh</button>
      </div>
    </div>
  `;
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

function togglePlusMenu(e) {
  const m = document.getElementById('plusMenu');
  if (!m) return;
  if (m.style.display === 'block') {
    m.style.display = 'none';
    return;
  }
  m.style.display = 'block';
  m.innerHTML = `
    <button onclick="pickImage()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><circle cx="8.5" cy="8.5" r="1.5"/><polyline points="21 15 16 10 5 21"/>
      </svg>
      ${t('choose_image')}
    </button>
    <button onclick="openCamera()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>
      </svg>
      ${t('camera')}
    </button>
  `;
  const rect = e.currentTarget.getBoundingClientRect();
  // Menu mở ngược lên khi không đủ chỗ bên dưới, và không tràn mép phải
  const menuHeight = m.offsetHeight;
  const menuWidth = m.offsetWidth;
  const top = rect.bottom + 4 + menuHeight > window.innerHeight
    ? Math.max(8, rect.top - menuHeight - 4)
    : rect.bottom + 4;
  m.style.top = top + 'px';
  m.style.left = Math.max(8, Math.min(rect.left, window.innerWidth - menuWidth - 8)) + 'px';
}

document.addEventListener('click', e => {
  const m = document.getElementById('plusMenu');
  if (m && !e.target.closest('.plus-btn') && !e.target.closest('.plus-menu')) {
    m.style.display = 'none';
  }
});

function pickImage() {
  const m = document.getElementById('plusMenu');
  if (m) m.style.display = 'none';
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = 'image/*,video/mp4,video/quicktime,.avi,.mkv,.dcm,.nii,.nii.gz,.mha,.mhd';
  inp.onchange = async () => {
    if (!inp.files[0]) return;
    uploadSingleImage(inp.files[0]);
  };
  inp.click();
}

function pickMultipleImages() {
  const m = document.getElementById('plusMenu');
  if (m) m.style.display = 'none';
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.multiple = true;
  inp.accept = 'image/*,video/mp4,video/quicktime,.avi,.mkv,.dcm,.nii,.nii.gz,.mha,.mhd';
  inp.onchange = async () => {
    if (!inp.files || inp.files.length === 0) return;
    uploadMultipleImages(Array.from(inp.files));
  };
  inp.click();
}

async function uploadSingleImage(file) {
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    const d = await r.json();
    if (d.ok) {
      addAttachment(d.stored_path, 'image', file.name);
      applyDetectedMetadata(d);
      showPacsImage(d.stored_path, 'WEB_UPLOAD');
    }
  } catch (e) {
    alert('Không thể tải file lên: ' + e);
  }
}

async function uploadMultipleImages(files) {
  const ed = document.getElementById('emptyDropzone');
  if (ed) ed.style.display = 'none';
  const pic = document.getElementById('pacsImageContainer');
  if (pic) pic.style.display = 'flex';
  const pd = document.getElementById('pacsDate');
  if (pd) pd.textContent = new Date().toLocaleDateString('vi-VN');

  let firstImage = null;
  for (let i = 0; i < files.length; i++) {
    const f = files[i];
    const fd = new FormData();
    fd.append('file', f);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (d.ok) {
        addAttachment(d.stored_path, 'image', f.name);
        if (i === 0) {
          applyDetectedMetadata(d);
          firstImage = d.stored_path;
        }
      }
    } catch (e) {
      console.error('Lỗi tải file ' + f.name + ':', e);
    }
  }
  if (firstImage) {
    showPacsImage(firstImage, 'MULTI_UPLOAD');
  }
}

function applyDetectedMetadata(d) {
  if (d.detected_target) {
    const ts = document.getElementById('targetSelect');
    if (ts) {
      ts.value = d.detected_target;
      ts.dispatchEvent(new Event('change'));
    }
  }
  if (d.detected_modality) {
    setTimeout(() => {
      const ms = document.getElementById('modalitySelect');
      if (ms) {
        const detected = d.detected_modality.toLowerCase();
        const options = Array.from(ms.options);
        const modalityMap = {
          'ct': 'ct',
          'mri': 'mri',
          'x-quang': 'x-quang',
          'x-quang ngực': 'x-quang',
          'siêu âm': 'siêu âm',
          'nội soi': 'nội soi',
        };
        let keyword = null;
        for (const [key, val] of Object.entries(modalityMap)) {
          if (detected.includes(key)) {
            keyword = val;
            break;
          }
        }
        if (keyword) {
          const match = options.find(o => o.value.toLowerCase().includes(keyword));
          if (match) {
            ms.value = match.value;
            ms.dispatchEvent(new Event('change'));
          }
        }
      }
    }, 100);
  }
}

function showPacsImage(storedPath, sourceCode) {
  const ed = document.getElementById('emptyDropzone');
  if (ed) ed.style.display = 'none';
  const pacsActiveImage = document.getElementById('pacsActiveImage');
  if (pacsActiveImage) pacsActiveImage.src = attachmentUrl(storedPath);
  const pic = document.getElementById('pacsImageContainer');
  if (pic) pic.style.display = 'flex';
  const ppc = document.getElementById('pacsPatientCode');
  if (ppc) ppc.textContent = sourceCode;
  const pd = document.getElementById('pacsDate');
  if (pd) pd.textContent = new Date().toLocaleDateString('vi-VN');
}

function addAttachment(path, kind, name) {
  pendingAttachments.push({ path, kind, name });
  renderPreviews();
}

function renderPreviews() {
  const a = document.getElementById('imagePreviewArea');
  if (!a) return;
  a.innerHTML = '';
  if (pendingAttachments.length === 0) {
    a.classList.add('hidden');
    return;
  }
  a.classList.remove('hidden');
  pendingAttachments.forEach((att, i) => {
    const d = document.createElement('div');
    d.className = 'preview-thumb';
    const src = attachmentUrl(att.path);
    d.innerHTML = `<img src="${src}"><button class="remove-btn" onclick="removeAttachment(${i})">✕</button>`;
    a.appendChild(d);
  });
}

function removeAttachment(i) {
  pendingAttachments.splice(i, 1);
  renderPreviews();
  if (pendingAttachments.length === 0) {
    syncPACSView();
  }
}

async function triggerAIScreening() {
  if (isAnalyzing) return;
  const input = document.getElementById('messageInput');
  const selected = selectedTarget();
  const modalitySelect = document.getElementById('modalitySelect');
  const modality = modalitySelect ? modalitySelect.value : '';
  if (!selected) {
    alert('Chưa chọn nhóm bệnh.');
    return;
  }
  if (input && !input.value.trim()) {
    input.value = `Hãy sàng lọc y khoa cho ảnh này với nhóm bệnh ${selected.label} bằng modality ${modality}.`;
  }

  if (pendingAttachments.length === 0) {
    const pacsActiveImage = document.getElementById('pacsActiveImage');
    const src = pacsActiveImage ? (pacsActiveImage.getAttribute('src') || '') : '';
    if (src && !src.endsWith('/')) {
      const rel = src.startsWith('/output/') ? src.slice('/output/'.length) : decodeURIComponent(src.replace(/^\//, ''));
      pendingAttachments.push({
        path: rel,
        kind: 'image',
        name: rel.split('/').pop()
      });
    }
  }

  if (pendingAttachments.length > 0) {
    sendMessage();
  } else {
    alert('Vui lòng tải phim chụp lên trước khi yêu cầu phân tích AI.');
  }
}

async function sendMessage() {
  if (isAnalyzing) return;
  const input = document.getElementById('messageInput');
  const text = input ? input.value.trim() : '';
  if (!text && !pendingAttachments.length) return;

  const c = conversations[activeConvIndex];
  if (!c.id) {
    try {
      const r = await fetch('/api/conversations', { method: 'POST' });
      const d = await r.json();
      c.id = d.conversation_id;
    } catch (e) {}
  }

  if (text && c.title === t('new_chat')) {
    const fl = text.split('\n')[0].trim();
    if (fl && fl.length > 2) c.title = fl.substring(0, 28);
  }

  for (let i = 0; i < pendingAttachments.length; i++) {
    const att = pendingAttachments[i];
    const msgText = i === 0 && text ? text : '';
    const fd = new URLSearchParams();
    fd.append('sender', 'user');
    fd.append('text', msgText);
    fd.append('attachment_path', att.path);
    fd.append('attachment_kind', att.kind);
    if (c.id) {
      await fetch(`/api/conversations/${c.id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: fd
      }).catch(() => {});
    }
    c.messages.push({ sender: 'user', text: msgText, attachment_path: att.path, attachment_kind: att.kind });
  }

  if (text && !pendingAttachments.length) {
    const fd = new URLSearchParams();
    fd.append('sender', 'user');
    fd.append('text', text);
    if (c.id) {
      await fetch(`/api/conversations/${c.id}/messages`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: fd
      }).catch(() => {});
    }
    c.messages.push({ sender: 'user', text: text });
  }

  if (input) {
    input.value = '';
    input.style.height = 'auto';
  }

  renderMessages();
  const atts = [...pendingAttachments];
  pendingAttachments = [];
  renderPreviews();

  const statusEl = document.getElementById('medicalStatus');
  const origStatusText = statusEl ? statusEl.textContent : '';
  if (statusEl) statusEl.textContent = t('medical_analyzing');

  isAnalyzing = true;
  showAILoadingIndicator();

  try {
    if (atts.length > 0) {
      const target = selectedTarget();
      if (!target) throw new Error('Chưa chọn nhóm bệnh.');
      const modality = document.getElementById('modalitySelect')?.value || '';

      // Thu thập thông tin ca bệnh từ form
      const patientCode = (document.getElementById('patientCodeInput')?.value || '').trim();
      const clinicalNote = (document.getElementById('clinicalNoteInput')?.value || '').trim();
      const patientAge = (document.getElementById('patientAgeInput')?.value || '').trim();
      const patientSex = (document.getElementById('patientSexInput')?.value || '').trim();

      // Phân tích từng ảnh riêng biệt, mỗi ảnh có kết quả và tọa độ riêng
      for (let i = 0; i < atts.length; i++) {
        const att = atts[i];
        hideAILoadingIndicator();
        showAILoadingIndicator();
        let prompt = text || `Phân tích ảnh: ${att.name}`;
        // Thêm thông tin lâm sàng vào prompt nếu có
        if (clinicalNote) {
          prompt = prompt + ' | Lâm sàng: ' + clinicalNote;
        }
        const fd = new URLSearchParams();
        fd.append('image_path', att.path);
        fd.append('user_prompt', prompt);
        fd.append('target_key', target.key);
        fd.append('modality', modality);
        if (patientCode) fd.append('patient_code', patientCode);
        if (patientAge) fd.append('patient_age', patientAge);
        if (patientSex) fd.append('patient_sex', patientSex);
        if (clinicalNote) fd.append('clinical_note', clinicalNote);
        if (c.id) fd.append('conversation_id', c.id);

        try {
          const r = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: fd
          });
          let d;
          try {
            d = await r.json();
          } catch (e) {
            d = {};
          }
          if (!r.ok || !d.ok) throw new Error(d.detail || `Lỗi máy chủ (${r.status}).`);

          // Cập nhật model 3D đúng loại ung thư AI phát hiện (để tab 3D hiển thị đúng ung thư, không phải dropdown)
          const suspected = d.metadata && d.metadata.suspected_cancer;
          if (suspected && Array.isArray(window.CANCER_TARGETS)) {
            const match = window.CANCER_TARGETS.find(t => t.key === suspected);
            if (match) {
              last3DTargetKey = suspected;
              const sel = document.getElementById('targetSelect');
              if (sel) sel.value = suspected;
            }
          }

          // Thêm nhãn ảnh nếu có nhiều ảnh để phân biệt
          let replyText = d.reply_text;
          if (atts.length > 1) {
            replyText = `📷 **Ảnh ${i + 1}/${atts.length}: ${att.name}**\n\n${replyText}`;
          }
          const metaJson = d.metadata ? JSON.stringify(d.metadata) : null;
          addBubble({
            sender: 'assistant',
            text: replyText,
            attachment_path: d.attachment_path,
            attachment_kind: d.attachment_kind,
            metadata_json: metaJson,
            image_name: att.name
          }, true);

          if (c.id) {
            const fd2 = new URLSearchParams();
            fd2.append('sender', 'assistant');
            fd2.append('text', replyText);
            fd2.append('attachment_path', d.attachment_path || '');
            fd2.append('attachment_kind', d.attachment_kind || '');
            fd2.append('metadata_json', metaJson || '');
            fd2.append('image_name', att.name || '');
            await fetch(`/api/conversations/${c.id}/messages`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
              body: fd2
            }).catch(() => {});
          }

          c.messages.push({
            sender: 'assistant',
            text: replyText,
            attachment_path: d.attachment_path,
            attachment_kind: d.attachment_kind,
            metadata_json: metaJson,
            image_name: att.name
          });
        } catch (e) {
          hideAILoadingIndicator();
          const errorText = `❌ Phân tích ảnh "${att.name}" lỗi: ${e.message || 'Không thể kết nối tới máy chủ.'}`;
          addBubble({ sender: 'assistant', text: errorText, attachment_path: att.path, attachment_kind: att.kind }, true);
          if (c.id) {
            const fd2 = new URLSearchParams();
            fd2.append('sender', 'assistant');
            fd2.append('text', errorText);
            fd2.append('attachment_path', att.path);
            fd2.append('attachment_kind', att.kind);
            await fetch(`/api/conversations/${c.id}/messages`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
              body: fd2
            }).catch(() => {});
          }
          c.messages.push({ sender: 'assistant', text: errorText, attachment_path: att.path, attachment_kind: att.kind });
        }
      }

      // Show 2D/3D tabs and init 3D viewer if target is available
      showViewerTabs();
      init3DViewerForTarget(last3DTargetKey || target.key);
    } else {
      hideAILoadingIndicator();
      addBubble({ sender: 'assistant', text: t('system_reply_text') }, true);
      if (c.id) {
        const fd2 = new URLSearchParams();
        fd2.append('sender', 'assistant');
        fd2.append('text', t('system_reply_text'));
        await fetch(`/api/conversations/${c.id}/messages`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: fd2
        }).catch(() => {});
      }
      c.messages.push({ sender: 'assistant', text: t('system_reply_text') });
    }
  } finally {
    isAnalyzing = false;
    hideAILoadingIndicator();
    if (statusEl) statusEl.textContent = origStatusText;
    renderHistory();
    syncPACSView();
  }
}

function autoResize(el) {
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function fillPrompt(txt) {
  const input = document.getElementById('messageInput');
  if (input) {
    input.value = txt;
    autoResize(input);
  }
}

function toggleVoice() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    alert('Trình duyệt không hỗ trợ nhận diện giọng nói.');
    return;
  }
  if (isRecording) {
    if (recognition) recognition.stop();
    return;
  }
  isRecording = true;
  const recPanel = document.getElementById('recordingPanel');
  if (recPanel) recPanel.classList.add('active');
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.lang = currentLang === 'vi' ? 'vi-VN' : 'en-US';
  recognition.interimResults = true;
  recognition.onresult = e => {
    const txt = Array.from(e.results).map(r => r[0].transcript).join('');
    const input = document.getElementById('messageInput');
    if (input) input.value = txt;
  };
  recognition.onerror = () => { stopVoice(); };
  recognition.onend = () => {
    stopVoice();
    const input = document.getElementById('messageInput');
    const txt = input ? input.value.trim() : '';
    if (txt) sendMessage();
  };
  recognition.start();
}

function stopVoice() {
  isRecording = false;
  const recPanel = document.getElementById('recordingPanel');
  if (recPanel) recPanel.classList.remove('active');
  if (recognition) {
    try { recognition.stop(); } catch (e) {}
    recognition = null;
  }
}

function openCamera() {
  const m = document.getElementById('plusMenu');
  if (m) m.style.display = 'none';
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = 'image/*';
  inp.capture = 'environment';
  inp.onchange = async () => {
    if (!inp.files[0]) return;
    const fd = new FormData();
    fd.append('file', inp.files[0]);
    try {
      const r = await fetch('/api/upload', { method: 'POST', body: fd });
      const d = await r.json();
      if (d.ok) {
        addAttachment(d.stored_path, 'camera', '📷 ' + inp.files[0].name);
        const ed = document.getElementById('emptyDropzone');
        if (ed) ed.style.display = 'none';
        const pacsActiveImage = document.getElementById('pacsActiveImage');
        if (pacsActiveImage) pacsActiveImage.src = attachmentUrl(d.stored_path);
        const pic = document.getElementById('pacsImageContainer');
        if (pic) pic.style.display = 'flex';
        const ppc = document.getElementById('pacsPatientCode');
        if (ppc) ppc.textContent = "CAMERA_CAPTURE";
        const pd = document.getElementById('pacsDate');
        if (pd) pd.textContent = new Date().toLocaleDateString('vi-VN');
      }
    } catch (e) {
      alert('Không thể lưu ảnh camera: ' + e);
    }
  };
  inp.click();
}

function previewImage(src) {
  const fullImg = document.getElementById('previewFullImg');
  const modal = document.getElementById('imagePreviewModal');
  if (fullImg) fullImg.src = src;
  if (modal) {
    modal.style.display = 'flex';
    modal.classList.add('active');
  }
}

function closeImagePreview() {
  const modal = document.getElementById('imagePreviewModal');
  if (modal) {
    modal.classList.remove('active');
    modal.style.display = 'none';
  }
}

// Drag & Drop Setup
const pacsScreen = document.getElementById('pacsScreen');
if (pacsScreen) {
  pacsScreen.addEventListener('dragenter', e => { e.preventDefault(); pacsScreen.classList.add('dragover'); });
  pacsScreen.addEventListener('dragover', e => { e.preventDefault(); pacsScreen.classList.add('dragover'); });
  pacsScreen.addEventListener('dragleave', () => { pacsScreen.classList.remove('dragover'); });
  pacsScreen.addEventListener('drop', async e => {
    e.preventDefault();
    pacsScreen.classList.remove('dragover');
    for (const f of e.dataTransfer.files) {
      const fd = new FormData();
      fd.append('file', f);
      try {
        const r = await fetch('/api/upload', { method: 'POST', body: fd });
        const d = await r.json();
        if (d.ok) {
          addAttachment(d.stored_path, 'image', f.name);
          if (d.detected_target) {
            const ts = document.getElementById('targetSelect');
            if (ts) {
              ts.value = d.detected_target;
              ts.dispatchEvent(new Event('change'));
            }
          }
          // Tự động điền modality nếu backend trả về detected_modality
          if (d.detected_modality) {
            setTimeout(() => {
              const ms = document.getElementById('modalitySelect');
              if (ms) {
                const detected = d.detected_modality.toLowerCase();
                const options = Array.from(ms.options);
                const modalityMap = {
                  'ct': 'ct',
                  'mri': 'mri',
                  'x-quang': 'x-quang',
                  'x-quang ngực': 'x-quang',
                  'siêu âm': 'siêu âm',
                  'nội soi': 'nội soi',
                };
                let keyword = null;
                for (const [key, val] of Object.entries(modalityMap)) {
                  if (detected.includes(key)) {
                    keyword = val;
                    break;
                  }
                }
                if (keyword) {
                  const match = options.find(o => o.value.toLowerCase().includes(keyword));
                  if (match) {
                    ms.value = match.value;
                    ms.dispatchEvent(new Event('change'));
                  }
                }
              }
            }, 100);
          }
          const ed = document.getElementById('emptyDropzone');
          if (ed) ed.style.display = 'none';
          const pacsActiveImage = document.getElementById('pacsActiveImage');
          if (pacsActiveImage) pacsActiveImage.src = attachmentUrl(d.stored_path);
          const pic = document.getElementById('pacsImageContainer');
          if (pic) pic.style.display = 'flex';
          const ppc = document.getElementById('pacsPatientCode');
          if (ppc) ppc.textContent = "DRAG_DROP";
          const pd = document.getElementById('pacsDate');
          if (pd) pd.textContent = new Date().toLocaleDateString('vi-VN');
        }
      } catch (err) {}
    }
  });
}

document.addEventListener('dragover', e => { e.preventDefault(); });
document.addEventListener('drop', e => { e.preventDefault(); });

// Comparison Logic
let compareSelection = [];
async function openCompare(preId) {
  closeAllModals();
  const m = document.getElementById('compareModal');
  if (m) m.classList.add('active');
  const pv = document.getElementById('comparePickView');
  const rv = document.getElementById('compareResultView');
  if (pv) pv.style.display = 'block';
  if (rv) rv.style.display = 'none';
  compareSelection = preId ? [preId] : [];
  try {
    const r = await fetch('/api/cases');
    const d = await r.json();
    renderCasePickList(d.cases || []);
  } catch (e) {
    renderCasePickList([]);
  }
}

function renderCasePickList(cases) {
  const l = document.getElementById('casePickList');
  const goBtn = document.getElementById('compareGoBtn');
  if (!l) return;
  if (!cases.length) {
    l.innerHTML = '<p style="font-size:13px;color:var(--text-muted);padding:12px">Chưa có ca bệnh nào. Hãy phân tích ảnh trước.</p>';
    if (goBtn) goBtn.disabled = true;
    return;
  }
  l.innerHTML = '';
  cases.forEach(c => {
    const d = document.createElement('div');
    d.className = 'case-pick-item' + (compareSelection.includes(c.case_id) ? ' selected' : '');
    const nDet = (c.detections || []).length;
    d.innerHTML = `
      <img src="${attachmentUrl(c.processed_image_path || c.image_path)}" loading="lazy">
      <div>
        <b>#${c.case_id} ${escHtml(c.patient_code)}</b>
        <div style="font-size:11px;color:var(--text-muted)">${escHtml(c.created_at || '')} • ${nDet} nhãn model</div>
      </div>
      <span class="risk-badge risk-${(c.risk_level || 'uncertain').toLowerCase()}" style="margin-left:auto">${c.clinical_assessment_available ? escHtml(c.risk_level || '') : 'Chưa đánh giá'}</span>
    `;
    d.onclick = () => {
      const i = compareSelection.indexOf(c.case_id);
      if (i >= 0) compareSelection.splice(i, 1);
      else {
        if (compareSelection.length >= 2) compareSelection.shift();
        compareSelection.push(c.case_id);
      }
      renderCasePickList(cases);
    };
    l.appendChild(d);
  });
  if (goBtn) goBtn.disabled = compareSelection.length !== 2;
}

function compareColHtml(c) {
  if (!c) return '<div class="compare-col"><p>Dữ liệu trống</p></div>';
  const risk = (c.risk_level || 'uncertain').toLowerCase();
  const row = (k, v) => `<div class="compare-row"><span>${k}</span><b>${v}</b></div>`;
  return `
    <div class="compare-col">
      <h3>#${c.case_id} ${escHtml(c.patient_code)} <span class="risk-badge risk-${risk}">${c.clinical_assessment_available ? escHtml(risk) : 'Chưa đánh giá'}</span></h3>
      <img src="${attachmentUrl(c.processed_image_path || c.image_path)}" onclick="previewImage(this.src)" loading="lazy">
      <div style="margin-top:8px">
        ${row('Nhãn dự đoán', (c.detections || []).length)}
        ${row('Nhận định ác tính', c.clinical_assessment_available ? (c.suspected_malignant ? 'Có' : 'Không') : 'Chưa đánh giá')}
        ${row('Model', escHtml(c.model_name || '-'))}
      </div>
      <a class="result-btn" href="/api/cases/${c.case_id}/pdf" download style="margin-top:8px;display:block">⬇ Xuất PDF</a>
    </div>
  `;
}

async function renderCompare() {
  if (compareSelection.length !== 2) return;
  try {
    const results = await Promise.all(compareSelection.map(id => fetch('/api/cases/' + id).then(r => r.json())));
    const [a, b] = results.map(r => r.case);
    const view = document.getElementById('compareResultView');
    if (view) {
      view.innerHTML = `
        <div class="compare-grid">${compareColHtml(a)}${compareColHtml(b)}</div>
        <div style="text-align:center;margin-top:12px">
          <button class="result-btn" onclick="openCompare()" style="display:inline-block;min-width:160px">← Chọn lại</button>
        </div>
      `;
      const pv = document.getElementById('comparePickView');
      if (pv) pv.style.display = 'none';
      view.style.display = 'block';
    }
  } catch (e) {
    alert('Không thể so sánh ca bệnh: ' + e);
  }
}

// Global modal background click handler
document.addEventListener('click', e => {
  const settingsModal = document.getElementById('settingsModal');
  const compareModal = document.getElementById('compareModal');
  const previewModal = document.getElementById('imagePreviewModal');
  if (settingsModal && e.target === settingsModal) closeSettings();
  if (compareModal && e.target === compareModal) compareModal.classList.remove('active');
  if (previewModal && e.target === previewModal) closeImagePreview();
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeAllModals();
});

// App Initialization
(async function initApp() {
  const msgInput = document.getElementById('messageInput');
  if (msgInput) {
    msgInput.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  try {
    const r = await fetch('/api/settings');
    const d = await r.json();
    if (d.ok) {
      currentLang = d.language || 'vi';
    }
  } catch (e) {}

  localStorage.setItem('lang', currentLang);

  const ls = document.getElementById('langSetting');
  if (ls) ls.value = currentLang;

  initTargets();
  applyLanguage(currentLang);

  // Khởi tạo trạng thái thanh bên & khung chat từ lưu trữ
  const savedSidebar = localStorage.getItem('sidebarCollapsed');
  if (savedSidebar === '1' || (savedSidebar === null && window.innerWidth <= 900)) {
    toggleSidebar(false);
  }

  const savedCustomWidth = localStorage.getItem('chatCustomWidth');
  const savedWidthMode = localStorage.getItem('chatWidthMode') || 'normal';
  if (savedWidthMode === 'custom' && savedCustomWidth) {
    setChatWidth('custom', parseInt(savedCustomWidth, 10));
  } else {
    setChatWidth(savedWidthMode);
  }

  if (localStorage.getItem('chatCollapsed') === '1') {
    toggleChatCollapse(true);
  }

  initChatResizer();
  initKeyboardShortcuts();
  initPacsInteractions();

  await loadConversations();

  try {
    const r2 = await fetch('/api/status');
    const d2 = await r2.json();
    if (d2.ok) {
      const ms = document.getElementById('medicalStatus');
      if (ms) {
        const capabilities = [
          d2.classification_ready ? 'phân loại ✓' : 'phân loại ✗',
          d2.localization_ready ? 'định vị ✓' : 'định vị ✗',
          d2.volume_ready ? '3D ✓' : '3D ✗',
          d2.clinical_validation_ready ? 'kiểm định lâm sàng ✓' : 'kiểm định lâm sàng ✗',
        ];
        ms.textContent = 'Model: ' + capabilities.join(' • ');
        ms.title = d2.model_message || '';
      }
      const mb = document.getElementById('modeBadge');
      if (mb) mb.textContent = 'Y khoa • ' + (d2.analyzed_cancers || []).length + ' bệnh';
      
      // Show demo mode warning if applicable
      if (d2.demo_mode) {
        const warningDiv = document.getElementById('demoModeWarning');
        if (warningDiv) {
          warningDiv.textContent = d2.demo_warning || '⚠️ DEMO MODE: Hệ thống đang chạy mô phỏng.';
          warningDiv.style.display = 'block';
        } else {
          // Create warning element if it doesn't exist
          const warning = document.createElement('div');
          warning.id = 'demoModeWarning';
          warning.className = 'alert alert-warning mt-2';
          warning.style.backgroundColor = '#fff3cd';
          warning.style.border = '1px solid #ffc107';
          warning.style.borderRadius = '4px';
          warning.style.padding = '10px';
          warning.style.color = '#856404';
          warning.textContent = d2.demo_warning || '⚠️ DEMO MODE: Hệ thống đang chạy mô phỏng. Kết quả không phản ánh thực tế và không dùng cho chẩn đoán y khoa.';
          // Insert after modeBadge
          if (mb && mb.parentNode) {
            mb.parentNode.insertBefore(warning, mb.nextSibling);
          }
        }
      }
    }
  } catch (e) {}
})();

// ── 3D Anatomy Viewer Integration ─────────────────────────────────────────

let last3DTargetKey = null;

function showViewerTabs() {
  const tabs = document.getElementById('viewerTabs');
  if (tabs) tabs.style.display = 'flex';
}

function switchViewerTab(tab) {
  const tab2D = document.getElementById('tab2D');
  const tab3D = document.getElementById('tab3D');
  const panel2D = document.getElementById('panel2D');
  const panel3D = document.getElementById('panel3D');
  const fpsControl = document.getElementById('fpsControl');

  if (tab === '3d') {
    if (tab2D) tab2D.classList.remove('active');
    if (tab3D) tab3D.classList.add('active');
    if (panel2D) panel2D.classList.remove('active');
    if (panel3D) panel3D.classList.add('active');
    // Show FPS control
    if (fpsControl) fpsControl.style.display = 'flex';
    // Init 3D viewer (Three.js đã load sẵn từ HTML)
    const targetKey = last3DTargetKey
      || (document.getElementById('targetSelect') && document.getElementById('targetSelect').value)
      || 'brain';
    if (typeof init3DViewer === 'function' && targetKey) {
      init3DViewer('anatomy3dContainer', targetKey);
      last3DTargetKey = targetKey;
    }
  } else {
    if (tab3D) tab3D.classList.remove('active');
    if (tab2D) tab2D.classList.add('active');
    if (panel3D) panel3D.classList.remove('active');
    if (panel2D) panel2D.classList.add('active');
    // Hide FPS control
    if (fpsControl) fpsControl.style.display = 'none';
  }
}

function init3DViewerForTarget(targetKey) {
  last3DTargetKey = targetKey;
  // Don't auto-init 3D viewer; wait for user to click the 3D tab
  // The switchViewerTab function will init it when needed
}

// ── FPS Control UI ──────────────────────────────────────────────────
function set3DFPSUI(fps) {
  if (typeof set3DFPS === 'function') {
    set3DFPS(fps);
  }
  // Update UI buttons
  const btn30 = document.getElementById('fps30');
  const btn60 = document.getElementById('fps60');
  if (fps === 30) {
    if (btn30) btn30.classList.add('active');
    if (btn60) btn60.classList.remove('active');
  } else {
    if (btn30) btn30.classList.remove('active');
    if (btn60) btn60.classList.add('active');
  }
  // Save preference
  localStorage.setItem('3dViewerFPS', String(fps));
}

// Load saved FPS preference on startup
(function () {
  const saved = localStorage.getItem('3dViewerFPS');
  if (saved === '60' && typeof set3DFPS === 'function') {
    set3DFPS(60);
  }
})();
