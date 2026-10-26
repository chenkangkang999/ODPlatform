(() => {
  'use strict';
  const API_BASE = '/football/api/';
  const CLASS_INFO = {
    ball: {name: '足球', color: '#FDE047'},
    goalkeeper: {name: '门将', color: '#C084FC'},
    player: {name: '球员', color: '#38BDF8'},
    referee: {name: '裁判', color: '#FB923C'}
  };
  const PAGES = {
    overview: ['你的足球观察室', '选择一个应用，开始你的比赛观察。', '应用大厅'],
    detect: ['检测工作台', '上传素材，使用可信 YOLO 权重进行本地检测，保留完整目标与处理记录。', '检测工作台'],
    var: ['VAR 辅助复核台', '围绕越位、冲撞、进球与出牌整理真实证据，再形成可追溯的人工复核意见。', 'VAR 复核台'],
    monitor: ['运行与数据监控', '真实任务、证据与下载清单，跟进42项能力的数据和模型准备情况。', '运行监控'],
    analysis: ['结果与数据分析', '让每一个数字都能回到原始检测结果，统一图表、报告和智能解读的口径。', '结果与数据分析'],
    models: ['模型与算法接口', '查看实际使用的模型与来源，清楚区分模型评估、当前任务和未知信息。', '模型与接口'],
    assistant: ['基于事实的智能解读', '围绕当前任务生成报告、诊断检测局限，或继续提问。', 'LLM 智能助手'],
    history: ['任务历史', '保留检测与导入记录，随时返回同一份结果继续分析。', '任务历史'],
    settings: ['系统配置与调用用量', '模型与 API 配置由服务端管理，浏览器不保存密钥。', '系统配置与用量']
  };
  const STATUS_NAMES = {queued: '等待处理', running: '正在检测', completed: '已完成', failed: '处理失败', cancelled: '已取消'};
  const state = {status: null, models: [], jobs: [], jobsLoaded: false, currentJob: null,
    uploading: false, llmBusy: false, pollTimer: null, toastTimer: null,
    chatHistory: [], historyLoadedFor: null, assistantEpoch: 0, requestedJob: null};
  const $ = id => document.getElementById(id);
  const number = value => typeof value === 'number' && Number.isFinite(value);
  const count = value => number(value) ? value.toLocaleString('zh-CN', {maximumFractionDigits: 0}) : '未提供';
  const percent = value => number(value) ? `${(value * 100).toFixed(1)}%` : '未提供';
  const dateText = value => {
    if (!value) return '未提供';
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? '未提供' : date.toLocaleString('zh-CN', {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false});
  };
  const className = value => CLASS_INFO[value]?.name || String(value || '未知类别');
  const element = (tag, classValue, text) => {
    const node = document.createElement(tag);
    if (classValue) node.className = classValue;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  };

  function toast(message, error = false) {
    const node = $('toast');
    node.textContent = String(message);
    node.classList.toggle('error', error);
    node.hidden = false;
    clearTimeout(state.toastTimer);
    state.toastTimer = setTimeout(() => {node.hidden = true;}, error ? 7000 : 4000);
  }

  function csrfToken() {
    const cookie = document.cookie.split(';').map(x => x.trim()).find(x => x.startsWith('csrftoken='));
    return cookie ? decodeURIComponent(cookie.slice('csrftoken='.length)) : document.querySelector('[name="csrfmiddlewaretoken"]')?.value || '';
  }

  async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    const method = options.method || 'GET';
    if (method !== 'GET') headers.set('X-CSRFToken', csrfToken());
    let body = options.body;
    if (body && !(body instanceof FormData)) {
      headers.set('Content-Type', 'application/json');
      body = JSON.stringify(body);
    }
    const response = await fetch(API_BASE + path, {credentials: 'same-origin', ...options, method, headers, body});
    let data;
    try {data = await response.json();}
    catch (_) {throw new Error(`服务响应暂不可用（HTTP ${response.status}），请检查服务状态。`);}
    if (!response.ok) {
      const error = new Error(typeof data.error === 'string' ? data.error : `请求未完成（HTTP ${response.status}）`);
      error.code = data.code;
      throw error;
    }
    return data;
  }

  function safeArtifactUrl(value) {
    if (typeof value !== 'string' || !value) return null;
    try {
      const url = new URL(value, window.location.origin);
      return /^https?:$/.test(url.protocol) && url.origin === window.location.origin ? url.href : null;
    } catch (_) {return null;}
  }

  function artifactLink(label, value) {
    const url = safeArtifactUrl(value);
    if (!url) return null;
    const link = element('a', '', label);
    link.href = url;
    link.target = '_blank';
    link.rel = 'noopener';
    return link;
  }

  function setDownloads(container, urls, keys) {
    container.replaceChildren();
    for (const [key, label] of keys) {
      const link = artifactLink(label, urls?.[key]);
      if (link) container.append(link);
    }
  }

  function statusBadge(status) {
    const cls = status === 'completed' ? 'tag-green' : status === 'failed' ? 'tag-red' : ['queued', 'running'].includes(status) ? 'tag-blue' : '';
    return element('span', `tag ${cls}`, STATUS_NAMES[status] || status || '未知');
  }

  function empty(container, title, description = '') {
    const wrapper = element('div', 'empty-state compact');
    wrapper.append(element('h3', '', title));
    if (description) wrapper.append(element('p', '', description));
    container.replaceChildren(wrapper);
  }

  function renderTable(container, headers, rows, emptyText = '暂无记录') {
    if (!rows.length) {empty(container, emptyText); return;}
    const wrap = element('div', 'table-wrap');
    const table = element('table');
    const thead = element('thead');
    const tr = element('tr');
    for (const header of headers) tr.append(element('th', '', header));
    thead.append(tr);
    const tbody = element('tbody');
    for (const values of rows) {
      const row = element('tr');
      for (const value of values) {
        const cell = element('td');
        if (value instanceof Node) cell.append(value);
        else cell.textContent = value === null || value === undefined ? '未提供' : String(value);
        row.append(cell);
      }
      tbody.append(row);
    }
    table.append(thead, tbody);
    wrap.append(table);
    container.replaceChildren(wrap);
  }

  function currentReady() {return state.currentJob?.status === 'completed' && Boolean(state.currentJob.result);}
  function llmReady() {return state.status?.llm?.configured === true;}
  function modelReady(model) {return model.ready !== false;}

  function updateButtons() {
    const processing = ['queued', 'running'].includes(state.currentJob?.status);
    $('start-detection').disabled = state.uploading || processing || !$('upload-file').files.length || !state.models.some(modelReady) || !$('model-select').value;
    $('start-detection').textContent = state.uploading ? '正在上传…' : processing ? '任务处理中…' : '开始检测与分析 →';
    const demoEnabled = state.status?.demo_available === true && state.models.some(modelReady) && !state.uploading && !processing;
    $('try-demo').disabled = !demoEnabled;
    $('try-demo-home').disabled = !demoEnabled;
    const aiEnabled = currentReady() && llmReady() && !state.llmBusy;
    $('generate-report').disabled = !aiEnabled;
    $('send-question').disabled = !aiEnabled;
    $('chat-question').disabled = !aiEnabled;
    $('local-summary').disabled = !currentReady() || state.llmBusy;
  }

  function renderReadiness() {
    const data = [
      [Boolean(state.status), 'Web 与任务服务', state.status ? '已连接' : '连接中'],
      [state.models.some(modelReady), '可信模型', state.models.some(modelReady) ? '已就绪' : '待配置'],
      [llmReady(), 'LLM 接口', llmReady() ? '密钥已配置' : '可先使用本地解读']
    ];
    const list = $('readiness-list');
    list.replaceChildren();
    for (const [ready, label, description] of data) {
      const li = element('li');
      li.append(element('span', `ready-symbol ${ready ? '' : 'pending'}`, ready ? '✓' : '·'), element('span', '', label));
      const detail = element('span', 'muted', description);
      detail.style.marginLeft = 'auto';
      detail.style.fontSize = '10px';
      li.append(detail);
      list.append(li);
    }
  }

  function renderModelOptions() {
    const select = $('model-select');
    const previous = select.value;
    select.replaceChildren();
    if (!state.models.length) {
      const option = element('option', '', '尚未配置可信足球模型');
      option.value = '';
      select.append(option);
    } else {
      for (const model of state.models) {
        const option = element('option', '', `${model.name || model.id}${modelReady(model) ? '' : '（未就绪）'}`);
        option.value = model.id;
        option.disabled = !modelReady(model);
        select.append(option);
      }
      if (state.models.some(m => m.id === previous && modelReady(m))) select.value = previous;
      else select.value = state.models.find(modelReady)?.id || '';
    }
    $('model-select-note').textContent = state.models.some(modelReady) ? '使用服务端的可信权重；本页面不训练或改写算法模型。' : '请先配置模型路径；也可以导入同学已生成的结果 JSON。';
    updateButtons();
  }

  function renderStatus() {
    const ready = Boolean(state.status);
    $('connection-dot').className = `connection-dot ${ready ? 'ready' : 'error'}`;
    $('connection-label').textContent = ready ? '本地服务已连接' : '服务暂不可用';
    $('overview-models').textContent = state.status ? String(state.models.filter(modelReady).length) : '—';
    $('overview-model-note').textContent = state.models.length ? '来源于服务端可信模型配置' : '待算法同学提供模型或配置路径';
    $('overview-llm').textContent = state.status ? llmReady() ? '已配置' : '待配置' : '检查中';
    const llm = state.status?.llm || {};
    $('llm-status-note').textContent = llmReady() ? `已配置 ${llm.model || '兼容模型'}，仅在你点击后发起 API 请求。` : '尚未配置大模型密钥。可先查看本地统计解读，检测与分析功能不受影响。';
    setKeyValues($('system-status'), [['Web 服务', ready ? '已连接' : '未连接'], ['可用模型', state.status ? String(state.models.filter(modelReady).length) : '未知'], ['结果来源', '当前会话的实际任务'], ['推理方式', '本地可信模型 / 导入结果']]);
    setKeyValues($('llm-config-status'), [['配置状态', llmReady() ? '密钥已配置（不展示）' : '尚未配置'], ['模型名称', llm.model || '未提供'], ['接口主机', llm.base_host || '未提供'], ['原始媒体上传', '默认不发送给大模型']]);
    const limits = state.status?.limits || {};
    const imageMB = limits.image_mb || limits.max_image_mb || (limits.image_bytes ? Math.floor(limits.image_bytes / 1048576) : null);
    const videoMB = limits.video_mb || limits.max_video_mb || (limits.video_bytes ? Math.floor(limits.video_bytes / 1048576) : null);
    $('upload-limit-note').textContent = imageMB && videoMB ? `图片不超过 ${imageMB} MB；视频不超过 ${videoMB} MB` : '支持常见图片与视频格式，以服务端限制为准';
    $('demo-note').textContent = state.status?.demo_available === true ? '已配置本机联调示例；点击后创建属于当前会话的新任务，并真实调用 YOLO。它不代表最终同学权重的成果。' : '本机联调示例尚未配置。示例也会真实调用 YOLO，不代表最终同学权重的成果。';
    renderReadiness();
    renderModelOptions();
    renderModels();
  }

  function setKeyValues(container, rows) {
    container.replaceChildren();
    for (const [key, value] of rows) container.append(element('dt', '', key), element('dd', '', value ?? '未提供'));
  }

  async function refreshStatus() {
    const data = await api('status/');
    state.status = data;
    if (Array.isArray(data.models)) state.models = data.models;
    else {
      const result = await api('models/');
      if (!Array.isArray(result.models)) throw new Error('模型接口格式不符合约定。');
      state.models = result.models;
    }
    renderStatus();
  }

  async function refreshJobs() {
    const data = await api('jobs/');
    if (!Array.isArray(data.jobs)) throw new Error('任务接口格式不符合约定。');
    state.jobs = data.jobs;
    state.jobsLoaded = true;
    renderJobs();
  }

  function jobRows(jobs) {
    return jobs.map(job => {
      const filename = element('span', 'table-filename', job.input_name || '导入结果');
      filename.title = job.input_name || '导入结果';
      const button = element('button', 'job-open-link', '打开结果 →');
      button.type = 'button';
      button.addEventListener('click', () => selectJob(job.id, job.status === 'completed' ? 'analysis' : 'detect').catch(error => toast(error.message, true)));
      return [filename, job.kind === 'video' ? '视频' : job.kind === 'image' ? '图片' : '导入结果', statusBadge(job.status), dateText(job.created_at), button];
    });
  }

  function renderJobs() {
    $('overview-total').textContent = state.jobsLoaded ? String(state.jobs.length) : '—';
    $('overview-completed').textContent = state.jobsLoaded ? String(state.jobs.filter(j => j.status === 'completed').length) : '—';
    if (state.jobs.length) {
      $('recent-jobs').className = '';
      renderTable($('recent-jobs'), ['素材名称', '类型', '状态', '创建时间', ''], jobRows(state.jobs.slice(0, 5)));
      $('history-table').className = '';
      renderTable($('history-table'), ['素材名称', '类型', '状态', '创建时间', ''], jobRows(state.jobs));
    } else if (state.jobsLoaded) {
      empty($('recent-jobs'), '从第一份素材开始', '上传图片、视频或算法结果 JSON，创建第一条真实任务。');
      empty($('history-table'), '还没有任务记录', '上传素材，或导入同学已经生成的结果。');
    }
  }

  function resultOf(job) {return job?.status === 'completed' && job.result && typeof job.result === 'object' ? job.result : null;}
  function previewOf(result) {return result?.preview || {boxes: result?.boxes || [], frame_index: result?.frame_index, timestamp: result?.timestamp, image_size: result?.image_size};}

  function setCurrentJob(job) {
    const switched = state.currentJob?.id !== job.id;
    state.currentJob = job;
    state.requestedJob = job.id;
    try {sessionStorage.setItem('football_current_job', job.id);} catch (_) {}
    if (switched) {
      state.assistantEpoch += 1;
      state.chatHistory = [];
      state.historyLoadedFor = null;
      resetChat();
    }
    const i = state.jobs.findIndex(item => item.id === job.id);
    if (i >= 0) state.jobs[i] = {...state.jobs[i], ...job};
    else state.jobs.unshift(job);
    state.jobsLoaded = true;
    renderJobs();
    renderJob();
    if (switched && currentReady()) loadAssistantHistory().catch(() => {});
  }

  async function selectJob(id, route = null) {
    if (!id) return;
    state.requestedJob = id;
    const data = await api(`jobs/${encodeURIComponent(id)}/`);
    if (state.requestedJob !== id) return;
    if (!data.job) throw new Error('任务详情格式不符合约定。');
    setCurrentJob(data.job);
    if (route) window.location.hash = route;
    startPolling();
  }

  function startPolling() {
    clearTimeout(state.pollTimer);
    const id = state.currentJob?.id;
    if (!id || !['queued', 'running'].includes(state.currentJob.status)) return;
    state.pollTimer = setTimeout(async () => {
      try {
        const data = await api(`jobs/${encodeURIComponent(id)}/`);
        if (state.currentJob?.id !== id) return;
        setCurrentJob(data.job);
        if (data.job.status === 'completed') {
          toast('检测与统计已完成，可查看真实结果。');
          refreshJobs().catch(() => {});
          loadAssistantHistory().catch(() => {});
        }
        startPolling();
      } catch (error) {toast(error.message, true);}
    }, 1400);
  }

  function renderJob() {
    const job = state.currentJob;
    const result = resultOf(job);
    $('page-detect').classList.toggle('has-result', Boolean(result));
    $('job-empty').hidden = Boolean(job);
    $('job-status-content').hidden = !job;
    if (!job) {updateButtons(); return;}
    $('current-task-pill').textContent = job.input_name || `任务 ${String(job.id).slice(0, 8)}`;
    $('current-task-pill').href = result ? '#analysis' : '#detect';
    const badge = statusBadge(job.status);
    $('job-status').className = badge.className;
    $('job-status').textContent = badge.textContent;
    $('job-name').textContent = job.input_name || '导入的算法结果';
    $('job-facts').replaceChildren(element('span', '', `ID · ${String(job.id).slice(0, 8)}`), element('span', '', dateText(job.created_at)), element('span', '', job.kind === 'video' ? '视频任务' : job.kind === 'image' ? '图片任务' : '结果导入'));
    const progress = number(job.progress) ? Math.max(0, Math.min(100, job.progress)) : 0;
    $('job-progress').value = progress;
    $('job-progress-value').textContent = `${Math.round(progress)}%`;
    $('job-progress-label').textContent = STATUS_NAMES[job.status] || job.status;
    $('job-error').hidden = !job.error;
    $('job-error').textContent = job.error || '';
    $('cancel-job').hidden = !['queued', 'running'].includes(job.status);
    $('go-analysis').hidden = !result;
    $('assistant-context').replaceChildren(element('span', '', '当前依据'), element('b', '', `${job.input_name || '导入结果'} · ${STATUS_NAMES[job.status] || job.status}`));
    renderMedia(job, result);
    renderBoxes(result);
    renderAnalysis(job, result);
    updateButtons();
  }

  function mediaNode(url, kind, alternative) {
    const safe = safeArtifactUrl(url);
    if (!safe) return element('div', 'empty-state compact', alternative);
    const node = document.createElement(kind === 'video' ? 'video' : 'img');
    node.src = safe;
    if (kind === 'video') {
      node.controls = true;
      node.preload = 'metadata';
      node.playsInline = true;
      node.setAttribute('aria-label', alternative);
    } else {node.alt = alternative; node.loading = 'lazy';}
    return node;
  }

  function renderMedia(job, result) {
    const urls = job.artifact_urls || {};
    const hasMedia = Boolean(result) && ['image', 'video'].includes(job.kind);
    $('media-empty').hidden = hasMedia;
    $('media-comparison').hidden = !hasMedia;
    $('original-media').replaceChildren();
    $('annotated-media').replaceChildren();
    $('video-preview-media').replaceChildren();
    $('video-preview').hidden = !(job.kind === 'video' && result && safeArtifactUrl(urls.annotated));
    if (hasMedia) {
      $('original-media').append(mediaNode(urls.original, job.kind, '原始素材暂不可预览'));
      const annotatedKind = job.kind === 'video' && urls.video ? 'video' : 'image';
      $('annotated-media').append(mediaNode(annotatedKind === 'video' ? urls.video : urls.annotated, annotatedKind, '标注素材暂不可预览'));
      if (job.kind === 'video' && urls.annotated) $('video-preview-media').append(mediaNode(urls.annotated, 'image', '视频标注预览帧'));
    } else if (result) empty($('media-empty'), '本任务由算法结果导入', '没有上传原始媒体，可查看完整检测框和统计结果。');
    $('video-help').hidden = job.kind !== 'video' || !result;
    setDownloads($('media-downloads'), urls, [['original', '打开原始素材'], ['annotated', '打开标注图'], ['video', '下载标注视频'], ['result', '结果 JSON']]);
    const preview = previewOf(result);
    const info = [];
    if (Number.isInteger(preview.frame_index)) info.push(`源帧 ${preview.frame_index}`);
    if (number(preview.timestamp)) info.push(`${preview.timestamp.toFixed(2)} 秒`);
    $('preview-frame-label').textContent = info.length ? info.join(' · ') : '原图坐标';
    renderBallFocus(job, result);
  }

  function renderBallFocus(job, result) {
    const panel = $('ball-focus');
    const canvas = $('ball-focus-crop');
    panel.hidden = !result;
    canvas.hidden = true;
    if (!result) return;
    const boxes = Array.isArray(previewOf(result).boxes) ? previewOf(result).boxes : [];
    const balls = boxes.filter(box => box.cls === 'ball').sort((a, b) => b.conf - a.conf);
    $('ball-focus-title').textContent = balls.length ? `足球焦点 · 本预览帧 ${balls.length} 个候选` : '足球焦点 · 本预览帧未检出';
    $('ball-focus-description').textContent = balls.length ? `最高置信度 ${percent(balls[0].conf)} · 亮黄色圆环定位，局部放大辅助查看。` : '没有达到当前阈值的足球候选；不代表画面中一定没有足球。';
    const urls = job.artifact_urls || {};
    const source = safeArtifactUrl(urls.preview || (job.kind === 'image' ? urls.original : null));
    if (!balls.length || !source) return;
    const image = new Image();
    const taskId = job.id;
    image.onload = () => {
      if (state.currentJob?.id !== taskId) return;
      const size = previewOf(result).image_size;
      if (!size || image.naturalWidth !== size.width || image.naturalHeight !== size.height) return;
      const box = balls[0];
      const side = Math.min(image.naturalWidth, image.naturalHeight, Math.max(32, (box.x2 - box.x1) * 5, (box.y2 - box.y1) * 5));
      const x = Math.max(0, Math.min((box.x1 + box.x2 - side) / 2, image.naturalWidth - side));
      const y = Math.max(0, Math.min((box.y1 + box.y2 - side) / 2, image.naturalHeight - side));
      const context = canvas.getContext('2d');
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.imageSmoothingEnabled = false;
      context.drawImage(image, x, y, side, side, 0, 0, canvas.width, canvas.height);
      canvas.hidden = false;
    };
    image.src = source;
  }

  function renderBoxes(result) {
    if (!result) {empty($('detection-table'), '还没有检测结果'); $('detection-count').textContent = '未提供结果'; return;}
    const boxes = Array.isArray(previewOf(result).boxes) ? previewOf(result).boxes : [];
    $('detection-count').textContent = `${boxes.length} 个预览帧检测框`;
    const rows = boxes.map((box, i) => {
      const values = [box.x1, box.y1, box.x2, box.y2];
      const coordinates = values.every(number) ? values.map(v => v.toFixed(1)).join(' , ') : '未提供';
      const team = box.team === 'A' ? 'A 队' : box.team === 'B' ? 'B 队' : box.team === 'unknown' ? '未知' : '未提供';
      return [String(i + 1), className(box.cls || box.label), percent(box.conf), coordinates, team];
    });
    renderTable($('detection-table'), ['序号', '类别', '模型置信度', '原图坐标 x1, y1, x2, y2', '球队属性'], rows, '预览帧没有检测到符合当前阈值的目标');
  }

  function svgNode(tag, attrs = {}, text) {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function plotBars(container, counts) {
    const svg = svgNode('svg', {viewBox: '0 0 560 240', role: 'presentation'});
    const keys = Object.keys(CLASS_INFO);
    const values = keys.map(key => number(counts[key]) ? Math.max(0, counts[key]) : 0);
    const max = Math.max(...values, 1);
    for (let i = 0; i <= 4; i++) {
      const y = 190 - i * 40;
      svg.append(svgNode('line', {x1: 45, y1: y, x2: 540, y2: y, stroke: '#355044'}));
      svg.append(svgNode('text', {x: 35, y: y + 4, fill: '#9baabd', 'font-size': 10, 'text-anchor': 'end'}, Math.round(max * i / 4)));
    }
    keys.forEach((key, i) => {
      const x = 80 + i * 123;
      const height = 160 * values[i] / max;
      svg.append(svgNode('rect', {x, y: 190 - height, width: 58, height, fill: CLASS_INFO[key].color, rx: 5}));
      svg.append(svgNode('text', {x: x + 29, y: 178 - height, fill: '#6b83a2', 'font-size': 11, 'text-anchor': 'middle'}, values[i]));
      svg.append(svgNode('text', {x: x + 29, y: 216, fill: '#90a0b5', 'font-size': 11, 'text-anchor': 'middle'}, CLASS_INFO[key].name));
    });
    container.replaceChildren(svg);
  }

  function plotTimeline(container, frames) {
    if (!Array.isArray(frames) || !frames.length) {
      empty(container, '当前结果未提供逐帧摘要', '可下载逐帧 JSONL / CSV 查看；不会补造变化曲线。');
      return;
    }
    const data = frames.slice(0, 1000);
    const keys = Object.keys(CLASS_INFO);
    const countsAt = frame => frame.counts || frame.per_class || frame;
    const maximum = Math.max(1, ...data.flatMap(frame => keys.map(k => number(countsAt(frame)[k]) ? countsAt(frame)[k] : 0)));
    const svg = svgNode('svg', {viewBox: '0 0 560 240', role: 'presentation'});
    for (let i = 0; i <= 4; i++) {
      const y = 190 - i * 40;
      svg.append(svgNode('line', {x1: 45, y1: y, x2: 540, y2: y, stroke: '#355044'}));
      svg.append(svgNode('text', {x: 35, y: y + 4, fill: '#9baabd', 'font-size': 10, 'text-anchor': 'end'}, Math.round(maximum * i / 4)));
    }
    const firstIndex = number(data[0].frame_index) ? data[0].frame_index : 0;
    const lastIndex = number(data[data.length - 1].frame_index) ? data[data.length - 1].frame_index : data.length - 1;
    for (const key of keys) {
      const points = data.map((frame, i) => {
        const index = number(frame.frame_index) ? frame.frame_index : i;
        const x = 45 + (lastIndex === firstIndex ? .5 : (index - firstIndex) / (lastIndex - firstIndex)) * 495;
        const value = number(countsAt(frame)[key]) ? countsAt(frame)[key] : 0;
        return [x, 190 - value / maximum * 160];
      });
      svg.append(svgNode('polyline', {points: points.map(p => p.join(',')).join(' '), fill: 'none', stroke: CLASS_INFO[key].color, 'stroke-width': 2, opacity: .85}));
      if (points.length === 1) svg.append(svgNode('circle', {cx: points[0][0], cy: points[0][1], r: 3, fill: CLASS_INFO[key].color}));
    }
    svg.append(svgNode('text', {x: 45, y: 216, fill: '#94a3b7', 'font-size': 10}, firstIndex));
    svg.append(svgNode('text', {x: 540, y: 216, fill: '#94a3b7', 'font-size': 10, 'text-anchor': 'end'}, lastIndex));
    container.replaceChildren(svg);
  }

  function plotPositions(container, positions) {
    const points = Array.isArray(positions) ? positions.filter(point => CLASS_INFO[point.cls] && number(point.x) && number(point.y) && point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1).slice(0, 2000) : [];
    if (!points.length) {
      empty(container, '当前结果没有可用位置数据', '原图尺寸或坐标未提供时，不填入示例点。');
      return;
    }
    const svg = svgNode('svg', {viewBox: '0 0 600 260', role: 'presentation'});
    svg.append(svgNode('rect', {x: 50, y: 15, width: 520, height: 205, fill: '#123429', stroke: '#355244', rx: 4}));
    for (let i = 0; i <= 4; i++) {
      const x = 50 + i * 130;
      const y = 15 + i * 51.25;
      svg.append(svgNode('line', {x1: x, y1: 15, x2: x, y2: 220, stroke: '#2c493d'}));
      svg.append(svgNode('line', {x1: 50, y1: y, x2: 570, y2: y, stroke: '#2c493d'}));
      svg.append(svgNode('text', {x, y: 240, fill: '#9aa9bd', 'font-size': 10, 'text-anchor': 'middle'}, (i / 4).toFixed(2)));
      svg.append(svgNode('text', {x: 36, y: y + 4, fill: '#9aa9bd', 'font-size': 10, 'text-anchor': 'end'}, (i / 4).toFixed(2)));
    }
    for (const point of points) svg.append(svgNode('circle', {cx: 50 + point.x * 520, cy: 15 + point.y * 205, r: point.cls === 'ball' ? 3.3 : 2.5, fill: CLASS_INFO[point.cls].color, opacity: .5}));
    svg.append(svgNode('text', {x: 310, y: 258, fill: '#91a1b6', 'font-size': 9, 'text-anchor': 'middle'}, '原图 x / 宽度（y 方向向下）'));
    container.replaceChildren(svg);
  }

  function renderAnalysis(job, result) {
    const analysis = result?.analysis;
    $('analysis-empty').hidden = Boolean(analysis);
    $('analysis-content').hidden = !analysis;
    if (!analysis) return;
    $('analysis-job-name').textContent = job.input_name || '导入的算法结果';
    $('analysis-frames').textContent = count(analysis.processed_frames);
    $('analysis-total').textContent = count(analysis.total_detections);
    const ballFrames = analysis.ball_frames ?? analysis.frames_with_ball_detection;
    const fraction = analysis.ball_frame_fraction ?? analysis.frames_with_ball_detection_fraction;
    $('analysis-ball-frames').textContent = count(ballFrames);
    $('analysis-ball-fraction').textContent = percent(fraction);
    const counts = analysis.per_class || analysis.cumulative_detections_by_class || {};
    plotBars($('class-chart'), counts);
    $('class-legend').replaceChildren();
    for (const info of Object.values(CLASS_INFO)) {
      const span = element('span');
      const dot = element('i', 'legend-dot');
      dot.style.backgroundColor = info.color;
      span.append(dot, document.createTextNode(info.name));
      $('class-legend').append(span);
    }
    plotTimeline($('timeline-chart'), analysis.counts_series || analysis.frame_counts || analysis.frames);
    plotPositions($('position-chart'), analysis.positions);
    $('position-note').textContent = `只表示目标出现在画面的哪里，不是球场坐标或球员跑动热力图。${analysis.position_sampled ? '此图使用服务端保留的位置样本，不代表全部目标。' : ''}`;
    const confidence = analysis.confidence_by_class || {};
    renderTable($('class-summary-table'), ['目标类别', '累计框数', '平均每帧可见框', '平均置信度'], Object.keys(CLASS_INFO).map(key => {
      const value = counts[key];
      const meanValue = analysis.mean_per_frame?.[key] ?? (number(value) && analysis.processed_frames > 0 ? value / analysis.processed_frames : null);
      const conf = number(confidence[key]) ? confidence[key] : confidence[key]?.mean;
      return [CLASS_INFO[key].name, count(value), number(meanValue) ? meanValue.toFixed(2) : '未提供', percent(conf)];
    }));
    const limitations = Array.isArray(analysis.limitations) ? analysis.limitations : [];
    $('analysis-limitations').replaceChildren(element('b', '', '结果解释边界'));
    const text = element('span', '', limitations.length ? limitations.map(item => typeof item === 'string' ? item : JSON.stringify(item)).join('；') : '置信度不等于准确率。没有人工真值、跟踪与场地校准时，不计算准确率、控球率或真实跑动距离。');
    $('analysis-limitations').append(text);
    setDownloads($('analysis-downloads'), job.artifact_urls, [['result', '结果 JSON'], ['counts', '统计 CSV'], ['frames', '逐帧 JSONL'], ['report', '下载报告']]);
  }

  function renderModels() {
    const container = $('model-cards');
    container.replaceChildren();
    if (!state.models.length) {
      const panel = element('article', 'panel');
      empty(panel, '还没有配置足球模型', '在服务端配置可信权重路径后刷新。已有算法结果时，可直接导入 JSON 使用分析与解读。');
      container.append(panel);
      return;
    }
    for (const model of state.models) {
      const card = element('article', 'panel model-card');
      const top = element('div', 'model-card-top');
      top.append(element('h3', '', model.name || model.id), element('span', `tag ${modelReady(model) ? 'tag-green' : 'tag-amber'}`, modelReady(model) ? '可调用' : '未就绪'));
      const meta = model.metadata || {};
      const details = element('dl', 'key-values');
      const classes = meta.class_names || meta.names;
      setKeyValues(details, [['模型版本', model.version || meta.model_version || '未提供'], ['运行 ID', meta.run_id || '未提供'], ['权重 SHA256', model.sha256 || meta.model_sha256 || '未提供'], ['类别', Array.isArray(classes) ? classes.map(className).join(' / ') : '运行时核验'], ['数据指纹', meta.data_fingerprint || meta.content_fingerprint || '未提供']]);
      card.append(top, details);
      const metrics = meta.evaluation?.metrics || meta.metrics || model.metrics;
      if (metrics && (number(metrics.map50) || number(metrics.map50_95))) {
        const values = element('div', 'model-eval');
        for (const [name, value] of [['mAP50', metrics.map50], ['mAP50–95', metrics.map50_95]]) {
          const metric = element('div');
          metric.append(element('span', '', name), element('b', '', percent(value)));
          values.append(metric);
        }
        card.append(values, element('p', 'stat-note', '以上是提供的模型评估资料，不是本次上传素材的准确率。请结合评估划分、样本量和来源说明。'));
      } else card.append(element('p', 'small-note', '尚未提供模型评估指标；页面不会用置信度代替准确率。'));
      if (meta.development_only) card.append(element('div', 'notice notice-neutral', '来源声明：开发集实验模型，未证明跨比赛泛化。'));
      container.append(card);
    }
  }

  function resetChat() {
    const wrapper = element('div', 'chat-intro');
    wrapper.append(element('span', 'assistant-star', '✦'), element('h3', '', '从当前任务结果开始解读'), element('p', '', '切换任务后上下文同步切换。可生成检测报告，或询问小目标识别的局限。'));
    $('chat-messages').replaceChildren(wrapper);
    $('chat-error').hidden = true;
  }

  function appendMessage(role, text, source = 'llm', usage = null, model = '') {
    $('chat-messages').querySelector('.chat-intro')?.remove();
    const node = element('article', `message message-${role} ${source === 'local' ? 'message-local' : ''}`);
    const label = role === 'user' ? '你' : source === 'local' ? '本地统计解读 · 未调用大模型' : `AI 智能解读${model ? ' · ' + model : ''}`;
    node.append(element('div', 'message-label', label), element('div', 'message-body', typeof text === 'string' ? text : JSON.stringify(text)));
    if (usage && role !== 'user') {
      const total = usage.total_tokens;
      node.append(element('div', 'message-usage', number(total) ? `服务商返回用量：${total} tokens` : '服务商未返回完整用量，不能当作零消耗。'));
    }
    $('chat-messages').append(node);
    $('chat-messages').scrollTop = $('chat-messages').scrollHeight;
    return node;
  }

  async function loadAssistantHistory() {
    const job = state.currentJob;
    if (!job || !currentReady() || state.historyLoadedFor === job.id) return;
    const id = job.id;
    const data = await api(`assistant/history/?job_id=${encodeURIComponent(id)}`);
    if (state.currentJob?.id !== id) return;
    state.historyLoadedFor = id;
    const records = Array.isArray(data.records) ? data.records : [];
    if (!records.length) return;
    resetChat();
    state.chatHistory = [];
    for (const record of records.slice(0, 12).reverse()) {
      if (record.question) appendMessage('user', record.question);
      if (record.answer) appendMessage('assistant', record.answer, record.source || 'llm', record.usage, record.model);
      if (record.source !== 'local' && record.answer) {
        if (record.question) state.chatHistory.push({role: 'user', content: record.question});
        state.chatHistory.push({role: 'assistant', content: record.answer});
      }
    }
  }

  async function askAssistant(mode, question) {
    if (!currentReady()) {toast('请先选择一个已完成任务。', true); return;}
    if (!llmReady()) {toast('尚未配置大模型密钥，可先使用本地统计解读。', true); return;}
    if (state.llmBusy) return;
    const id = state.currentJob.id;
    const epoch = state.assistantEpoch;
    state.llmBusy = true;
    updateButtons();
    $('chat-error').hidden = true;
    appendMessage('user', question);
    const waiting = appendMessage('assistant', '正在根据当前任务的真实结果生成解读…');
    waiting.classList.add('dots-loading');
    try {
      const result = await api('assistant/', {method: 'POST', body: {job_id: id, mode, question, history: state.chatHistory.slice(-16)}});
      if (state.assistantEpoch !== epoch || state.currentJob?.id !== id) return;
      waiting.remove();
      appendMessage('assistant', result.answer, result.source || 'llm', result.usage, result.model);
      if (result.source === 'llm') state.chatHistory.push({role: 'user', content: question}, {role: 'assistant', content: result.answer});
      state.chatHistory = state.chatHistory.slice(-16);
      $('chat-question').value = '';
    } catch (error) {
      if (state.assistantEpoch !== epoch) return;
      waiting.remove();
      $('chat-error').hidden = false;
      $('chat-error').textContent = error.message;
    } finally {state.llmBusy = false; updateButtons();}
  }

  async function localSummary() {
    if (!currentReady() || state.llmBusy) return;
    const id = state.currentJob.id;
    const epoch = state.assistantEpoch;
    state.llmBusy = true;
    updateButtons();
    $('chat-error').hidden = true;
    try {
      const result = await api('local-summary/', {method: 'POST', body: {job_id: id}});
      if (state.assistantEpoch !== epoch || state.currentJob?.id !== id) return;
      appendMessage('assistant', result.answer, 'local');
    } catch (error) {
      if (state.assistantEpoch !== epoch || state.currentJob?.id !== id) return;
      $('chat-error').hidden = false;
      $('chat-error').textContent = error.message;
    }
    finally {state.llmBusy = false; updateButtons();}
  }

  async function refreshUsage() {
    const data = await api('usage/');
    const records = Array.isArray(data.records) ? data.records : [];
    const rows = records.map(record => {
      const usage = record.usage || record;
      const input = usage.input_tokens ?? usage.prompt_tokens;
      const output = usage.output_tokens ?? usage.completion_tokens;
      const total = usage.total_tokens;
      const cost = record.estimated_cost;
      return [dateText(record.created_at || record.timestamp), record.model || '未知模型', count(input), count(output), count(total), number(cost) ? `${cost.toFixed(6)} ${record.currency || 'USD'}` : '待核实'];
    });
    renderTable($('usage-table'), ['时间', '模型', '输入 token', '输出 token', '总 token', '费用估算'], rows, '还没有大模型调用用量记录');
  }

  function showPage() {
    const key = window.location.hash.slice(1).split('?')[0] || 'overview';
    const page = PAGES[key] ? key : 'overview';
    document.body.dataset.page = page;
    document.querySelectorAll('.page[data-page]').forEach(node => {node.hidden = node.dataset.page !== page;});
    document.querySelectorAll('[data-nav]').forEach(node => {
      const selected = node.dataset.nav === page;
      node.classList.toggle('active', selected);
      if (selected) node.setAttribute('aria-current', 'page'); else node.removeAttribute('aria-current');
    });
    $('page-title').textContent = PAGES[page][0];
    $('page-subtitle').textContent = PAGES[page][1];
    $('breadcrumb-page').textContent = PAGES[page][2];
    document.title = `${PAGES[page][2]} · 足球视觉分析平台`;
    if (page === 'assistant') loadAssistantHistory().catch(error => toast(error.message, true));
    if (page === 'settings') refreshUsage().catch(error => toast(error.message, true));
    if (page === 'history') refreshJobs().catch(error => toast(error.message, true));
  }

  async function refreshAll() {
    const outcomes = await Promise.allSettled([refreshStatus(), refreshJobs()]);
    const failures = outcomes.filter(item => item.status === 'rejected');
    $('service-error').hidden = !failures.length;
    $('service-error').textContent = failures.length ? `暂时无法读取完整服务状态：${failures[0].reason.message}。可以点击右上角刷新。` : '';
    if (failures.length) {$('connection-dot').className = 'connection-dot error'; $('connection-label').textContent = '连接需要检查';}
  }

  async function startDemo() {
    if ($('try-demo').disabled) return;
    state.uploading = true;
    updateButtons();
    $('upload-form-error').hidden = true;
    window.location.hash = 'detect';
    try {
      const result = await api('demo/', {method: 'POST', body: {
        model_id: $('model-select').value,
        conf: Number($('confidence').value),
        imgsz: Number($('image-size').value),
        stride: Number($('frame-stride').value)
      }});
      setCurrentJob(result.job);
      startPolling();
      toast('本机联调示例任务已创建，正在真实推理。');
    } catch (error) {$('upload-form-error').hidden = false; $('upload-form-error').textContent = error.message;}
    finally {state.uploading = false; updateButtons();}
  }

  function bindEvents() {
    window.addEventListener('hashchange', showPage);
    $('refresh-all').addEventListener('click', async () => {
      $('refresh-all').disabled = true;
      try {await refreshAll(); if (state.currentJob) await selectJob(state.currentJob.id);}
      catch (error) {toast(error.message, true);}
      finally {$('refresh-all').disabled = false;}
    });
    $('refresh-jobs').addEventListener('click', () => refreshJobs().catch(error => toast(error.message, true)));
    $('refresh-usage').addEventListener('click', () => refreshUsage().catch(error => toast(error.message, true)));
    $('confidence').addEventListener('input', () => {$('confidence-value').textContent = Number($('confidence').value).toFixed(2);});
    $('model-select').addEventListener('change', updateButtons);
    $('try-demo').addEventListener('click', startDemo);
    $('try-demo-home').addEventListener('click', startDemo);
    const fileChanged = () => {
      const file = $('upload-file').files[0];
      $('selected-file').textContent = file ? `${file.name} · ${(file.size / 1048576).toFixed(2)} MB` : '尚未选择文件';
      updateButtons();
    };
    $('upload-file').addEventListener('change', fileChanged);
    for (const name of ['dragenter', 'dragover']) $('dropzone').addEventListener(name, event => {event.preventDefault(); $('dropzone').classList.add('dragover');});
    for (const name of ['dragleave', 'drop']) $('dropzone').addEventListener(name, event => {event.preventDefault(); $('dropzone').classList.remove('dragover');});
    $('dropzone').addEventListener('drop', event => {if (event.dataTransfer.files.length) {$('upload-file').files = event.dataTransfer.files; fileChanged();}});
    $('upload-form').addEventListener('submit', async event => {
      event.preventDefault();
      if ($('start-detection').disabled) return;
      const form = new FormData();
      form.append('file', $('upload-file').files[0]);
      form.append('model_id', $('model-select').value);
      form.append('conf', $('confidence').value);
      form.append('imgsz', $('image-size').value);
      form.append('stride', $('frame-stride').value);
      state.uploading = true;
      updateButtons();
      $('upload-form-error').hidden = true;
      try {
        const data = await api('jobs/', {method: 'POST', body: form});
        setCurrentJob(data.job);
        startPolling();
        toast('任务已创建，正在处理素材。');
      } catch (error) {$('upload-form-error').hidden = false; $('upload-form-error').textContent = error.message;}
      finally {state.uploading = false; updateButtons();}
    });
    $('import-form').addEventListener('submit', async event => {
      event.preventDefault();
      const file = $('import-file').files[0];
      if (!file) return;
      const form = new FormData();
      form.append('file', file);
      $('import-button').disabled = true;
      try {
        const data = await api('import/', {method: 'POST', body: form});
        setCurrentJob(data.job);
        window.location.hash = 'analysis';
        toast('算法结果已导入，统计已按统一口径计算。');
      } catch (error) {toast(error.message, true);}
      finally {$('import-button').disabled = false;}
    });
    $('cancel-job').addEventListener('click', async () => {
      if (!state.currentJob) return;
      $('cancel-job').disabled = true;
      try {
        const data = await api(`jobs/${encodeURIComponent(state.currentJob.id)}/cancel/`, {method: 'POST', body: {}});
        setCurrentJob(data.job);
        startPolling();
      } catch (error) {toast(error.message, true);}
      finally {$('cancel-job').disabled = false;}
    });
    $('generate-report').addEventListener('click', () => {
      const mode = $('assistant-mode').value;
      if (mode === 'chat') {$('chat-question').focus(); return;}
      const question = mode === 'diagnosis' ? '请结合当前任务结果，分析小足球检测的潜在局限与可核验的改进方向。' : '请基于当前任务的真实检测和统计结果，生成一份清晰的检测分析报告。';
      askAssistant(mode, question);
    });
    $('assistant-mode').addEventListener('change', () => {$('generate-report').textContent = $('assistant-mode').value === 'chat' ? '在下方输入问题' : '生成 AI 解读';});
    $('local-summary').addEventListener('click', localSummary);
    $('chat-form').addEventListener('submit', event => {
      event.preventDefault();
      const question = $('chat-question').value.trim();
      if (question) askAssistant('chat', question);
    });
    $('clear-chat').addEventListener('click', () => {state.chatHistory = []; resetChat();});
  }

  async function main() {
    bindEvents();
    showPage();
    await refreshAll();
    let previous = null;
    try {previous = sessionStorage.getItem('football_current_job');} catch (_) {}
    if (previous) {
      try {await selectJob(previous);}
      catch (_) {try {sessionStorage.removeItem('football_current_job');} catch (_) {}}
    }
  }
  main().catch(error => {toast(error.message, true);});
})();

