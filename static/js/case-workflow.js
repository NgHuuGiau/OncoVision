const workflowById = id => document.getElementById(id);
const workflowEsc = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);

async function workflowRequest(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.ok === false) throw new Error(data.detail || 'Không thể xử lý yêu cầu.');
  return data;
}

const workflowStatusLabel = status => ({ approved: 'Đã duyệt', submitted: 'Chờ admin duyệt', pending: 'Chờ duyệt' }[status] || 'Chờ duyệt');

async function openCaseAssignment() {
  const modal = workflowById('caseAssignmentModal');
  if (!modal) return;
  modal.classList.add('active');
  const list = workflowById('assignmentCaseList');
  list.innerHTML = '<p class="workflow-muted">Đang tải danh sách…</p>';
  try {
    const [caseData, clinicianData, viewerData] = await Promise.all([
      workflowRequest('/api/cases'),
      workflowRequest('/api/clinicians'),
      workflowRequest('/api/viewers'),
    ]);
    if (!caseData.cases.length) {
      list.innerHTML = '<p class="workflow-muted">Chưa có ca phân tích. Tải ảnh lên và phân tích trước.</p>';
      return;
    }
    const options = clinicianData.clinicians.map(user => `<option value="${workflowEsc(user.username)}">${workflowEsc(user.username)}</option>`).join('');
    const viewerOptions = (viewerData.viewers || []).map(user => `<option value="${workflowEsc(user.username)}">${workflowEsc(user.username)}</option>`).join('');
    list.innerHTML = caseData.cases.map(item => `
      <div class="assignment-row">
        <div><strong>#${item.case_id} · ${workflowEsc(item.patient_code)}</strong><div class="workflow-case-meta">${workflowEsc(item.created_at)} · ${workflowEsc(workflowStatusLabel(item.review_status))}${item.patient_username ? ` · TK: ${workflowEsc(item.patient_username)}` : ''}</div></div>
        <label>Nhân viên phụ trách<select data-case-assignee="${item.case_id}" ${options ? '' : 'disabled'}>${options}</select></label>
        <button type="button" class="workflow-primary" data-assign-case="${item.case_id}" ${options ? '' : 'disabled'}>Giao ca</button>
        ${item.review_status === 'submitted' ? `<button type="button" class="workflow-primary" data-approve-case="${item.case_id}">Duyệt & phát hành</button>` : ''}
        ${item.review_status === 'approved' ? `<label>TK người dùng${viewerOptions ? `<select data-case-patient="${item.case_id}">${viewerOptions}</select>` : '<span class="workflow-muted">Chưa có TK người dùng</span>'}</label>
        ${viewerOptions ? `<button type="button" class="workflow-primary" data-link-patient="${item.case_id}">Gán TK</button>` : ''}` : ''}
      </div>`).join('');
    caseData.cases.forEach(item => {
      const select = list.querySelector(`[data-case-assignee="${item.case_id}"]`);
      if (select && item.assigned_to) select.value = item.assigned_to;
      const patientSelect = list.querySelector(`[data-case-patient="${item.case_id}"]`);
      if (patientSelect && item.patient_username) patientSelect.value = item.patient_username;
    });
    list.querySelectorAll('[data-assign-case]').forEach(button => button.addEventListener('click', async () => {
      const caseId = button.dataset.assignCase;
      const username = list.querySelector(`[data-case-assignee="${caseId}"]`).value;
      button.disabled = true;
      try {
        await workflowRequest(`/api/cases/${caseId}/assign`, { method: 'POST', body: new URLSearchParams({ username }) });
        button.textContent = 'Đã giao';
        button.classList.add('workflow-success');
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
      }
    }));
    list.querySelectorAll('[data-approve-case]').forEach(button => button.addEventListener('click', async () => {
      const caseId = button.dataset.approveCase;
      button.disabled = true;
      try {
        const data = await workflowRequest(`/api/cases/${caseId}/approve`, { method: 'POST' });
        button.textContent = 'Đã duyệt';
        button.classList.add('workflow-success');
        alert(`Đã duyệt. Mã tra cứu: ${data.public_code}`);
        await openCaseAssignment();
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
      }
    }));
    list.querySelectorAll('[data-link-patient]').forEach(button => button.addEventListener('click', async () => {
      const caseId = button.dataset.linkPatient;
      const username = list.querySelector(`[data-case-patient="${caseId}"]`).value;
      button.disabled = true;
      try {
        await workflowRequest(`/api/cases/${caseId}/link-patient`, { method: 'POST', body: new URLSearchParams({ username }) });
        button.textContent = 'Đã gán';
        button.classList.add('workflow-success');
      } catch (error) {
        alert(error.message);
      } finally {
        button.disabled = false;
      }
    }));
  } catch (error) {
    list.innerHTML = `<p class="workflow-error">${workflowEsc(error.message)}</p>`;
  }
}

function closeCaseAssignment() {
  workflowById('caseAssignmentModal')?.classList.remove('active');
}

async function loadWorkflowCases() {
  const list = workflowById('workflowCaseList');
  if (!list) return;
  try {
    const data = await workflowRequest('/api/cases');
    if (!data.cases.length) {
      list.innerHTML = '<p class="workflow-muted">Chưa có ca được phân công cho bạn.</p>';
      return;
    }
    list.innerHTML = data.cases.map(item => `
      <button type="button" class="workflow-case-item" data-review-case="${item.case_id}">
        <strong>#${item.case_id} · ${workflowEsc(item.patient_code)}</strong>
        <span class="workflow-case-meta">${workflowEsc(item.created_at)} · ${workflowEsc(item.risk_level)}</span>
        <span class="workflow-status">${workflowEsc(workflowStatusLabel(item.review_status))}</span>
      </button>`).join('');
    list.querySelectorAll('[data-review-case]').forEach(button => button.addEventListener('click', () => loadWorkflowCase(button.dataset.reviewCase)));
  } catch (error) {
    list.innerHTML = `<p class="workflow-error">${workflowEsc(error.message)}</p>`;
  }
}

async function loadWorkflowCase(caseId) {
  const detail = workflowById('workflowDetail');
  detail.innerHTML = '<p class="workflow-muted">Đang tải ca…</p>';
  try {
    const data = await workflowRequest(`/api/cases/${caseId}`);
    const item = data.case;
    detail.innerHTML = `
      <h2>Ca #${item.case_id} · ${workflowEsc(item.patient_code)}</h2>
      <p class="workflow-status">Trạng thái: ${workflowEsc(workflowStatusLabel(item.review_status))}${item.reviewed_at ? ` · ${workflowEsc(item.reviewed_at)}` : ''}</p>
      <img class="workflow-review-image" src="/api/cases/${item.case_id}/image" alt="Ảnh đã xử lý của ca bệnh">
      <p class="workflow-case-meta">${(item.detections || []).length} vùng phát hiện (tọa độ bbox) · Độ tin cậy TB ${((item.average_confidence || 0) * 100).toFixed(1)}% · Model ${workflowEsc(item.model_name || '-')}</p>
      <form id="workflowReviewForm" class="workflow-form">
        <label>Mức nguy cơ<select name="risk_level"><option value="uncertain">Chưa xác định</option><option value="low">Thấp</option><option value="medium">Trung bình</option><option value="high">Cao</option></select></label>
        <label class="workflow-check"><input type="checkbox" name="suspected_malignant" value="true"> Nhận định nghi ngờ ác tính</label>
        <label>Khuyến nghị đã rà soát<textarea name="recommendation" maxlength="5000" required></textarea></label>
        <button type="submit" class="workflow-primary">Lưu và gửi admin duyệt</button>
      </form>
      <p id="workflowReviewMessage" aria-live="polite"></p>`;
    const form = workflowById('workflowReviewForm');
    form.elements.risk_level.value = item.risk_level || 'uncertain';
    form.elements.suspected_malignant.checked = Boolean(item.suspected_malignant);
    form.elements.recommendation.value = item.recommendation || '';
    form.addEventListener('submit', async event => {
      event.preventDefault();
      const button = form.querySelector('button[type="submit"]');
      const message = workflowById('workflowReviewMessage');
      button.disabled = true;
      message.className = 'workflow-muted';
      message.textContent = 'Đang gửi admin duyệt…';
      const payload = new URLSearchParams(new FormData(form));
      payload.set('suspected_malignant', String(form.elements.suspected_malignant.checked));
      try {
        await workflowRequest(`/api/cases/${caseId}/review`, { method: 'POST', body: payload });
        message.className = 'workflow-success';
        message.textContent = 'Đã gửi admin duyệt. Người dùng sẽ xem được sau khi admin duyệt và gán tài khoản.';
        await loadWorkflowCases();
      } catch (error) {
        message.className = 'workflow-error';
        message.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    });
  } catch (error) {
    detail.innerHTML = `<p class="workflow-error">${workflowEsc(error.message)}</p>`;
  }
}

async function searchPublicCase(event) {
  event.preventDefault();
  const code = workflowById('publicCaseCode').value.trim().toUpperCase();
  const errorBox = workflowById('publicCaseError');
  const result = workflowById('publicCaseResult');
  errorBox.hidden = true;
  result.hidden = true;
  try {
    const data = await workflowRequest(`/api/public/cases/${encodeURIComponent(code)}`);
    const item = data.case;
    const risk = { high: 'Cao', medium: 'Trung bình', low: 'Thấp', uncertain: 'Chưa xác định' }[item.risk_level] || 'Chưa xác định';
    result.innerHTML = `
      <h2>Kết quả đã được nhân viên y tế duyệt</h2>
      <p>Mã bệnh nhân: <strong>${workflowEsc(item.patient_code)}</strong></p>
      <p class="public-report-risk">Mức nguy cơ: ${workflowEsc(risk)}</p>
      <p>${item.suspected_malignant ? 'Có' : 'Không'} ghi nhận nghi ngờ ác tính trong kết quả đã duyệt.</p>
      <h3>Khuyến nghị</h3><p class="workflow-recommendation">${workflowEsc(item.recommendation)}</p>
      <p class="workflow-case-meta">Ngày phân tích: ${workflowEsc(item.created_at)} · Ngày duyệt: ${workflowEsc(item.reviewed_at)}</p>
      <a class="workflow-primary" href="/api/public/cases/${encodeURIComponent(code)}/pdf">Tải kết quả PDF</a>
      <p class="workflow-disclaimer">Kết quả AI chỉ mang tính hỗ trợ sàng lọc; vui lòng trao đổi với bác sĩ để được đánh giá và chẩn đoán chính thức.</p>`;
    result.hidden = false;
  } catch (error) {
    errorBox.textContent = error.message || 'Không tìm thấy kết quả đã duyệt.';
    errorBox.hidden = false;
  }
  return false;
}

async function loadMyCases() {
  const list = workflowById('myCaseList');
  if (!list) return;
  try {
    const data = await workflowRequest('/api/cases');
    if (!data.cases.length) {
      list.innerHTML = '<p class="workflow-muted">Chưa có kết quả nào được gán cho tài khoản của bạn.</p>';
      return;
    }
    const risk = { high: 'Cao', medium: 'Trung bình', low: 'Thấp', uncertain: 'Chưa xác định' };
    list.innerHTML = data.cases.map(item => `
      <div class="workflow-case-item">
        <strong>#${item.case_id} · ${workflowEsc(item.patient_code)}</strong>
        <span class="workflow-case-meta">Nguy cơ: ${workflowEsc(risk[item.risk_level] || 'Chưa xác định')} · ${item.suspected_malignant ? 'Có nghi ngờ ác tính' : 'Không nghi ngờ ác tính'}</span>
        <p class="workflow-recommendation">${workflowEsc(item.recommendation || '')}</p>
        <p class="workflow-case-meta">Ngày phân tích: ${workflowEsc(item.created_at)} · Ngày duyệt: ${workflowEsc(item.reviewed_at || '-')}</p>
        <a class="workflow-primary" href="/api/cases/${item.case_id}/pdf">Tải kết quả PDF</a>
      </div>`).join('');
  } catch (error) {
    list.innerHTML = `<p class="workflow-error">${workflowEsc(error.message)}</p>`;
  }
}

if (window.USER_ROLE === 'admin') workflowById('caseAssignmentModal') && document.addEventListener('click', event => {
  if (event.target === workflowById('caseAssignmentModal')) closeCaseAssignment();
});
if (workflowById('workflowCaseList')) loadWorkflowCases();
if (workflowById('myCaseList')) loadMyCases();
workflowById('publicCaseSearch')?.addEventListener('submit', searchPublicCase);
