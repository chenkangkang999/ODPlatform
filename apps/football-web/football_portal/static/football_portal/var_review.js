/* Evidence workbench. Browser observations remain separate from model decisions. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const root = '/football/api/var/';
  const state = {current: null, lines: [], drawing: null, resources: null, catalog: null, loaded: false, busy: false};
  const labels = {offside: '越位', foul: '冲撞 / 犯规', goal: '进球 / 门线', card: '出牌 / 受牌'};
  const statuses = {pending: '待复核', needs_evidence: '需补充证据', reviewed: '已人工复核'};

  const node = (tag, text, css) => {const item = document.createElement(tag); if (text !== undefined) item.textContent = String(text); if (css) item.className = css; return item;};
  function message(text, failed = true) {const item = $('var-error'); item.hidden = !text; item.textContent = text; item.className = failed ? 'notice notice-error' : 'notice var-status-message';}
  function csrf() {const part = document.cookie.split(';').map(x => x.trim()).find(x => x.startsWith('csrftoken=')); return part ? decodeURIComponent(part.slice(10)) : document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';}
  async function api(tail, options = {}) {
    const method = options.method || 'GET', headers = new Headers(options.headers || {});
    if (method !== 'GET') headers.set('X-CSRFToken', csrf());
    let body = options.body;
    if (body && !(body instanceof FormData)) {headers.set('Content-Type', 'application/json'); body = JSON.stringify(body);}
    const response = await fetch(root + tail, {...options, method, headers, body, credentials: 'same-origin'});
    let data; try {data = await response.json();} catch (_) {throw new Error(`复核服务响应无效（${response.status}）`);}
    if (!response.ok) throw new Error(data.error || '复核请求未完成');
    return data;
  }
  function privateUrl(value) {try {const url = new URL(value, location.origin); return url.origin === location.origin && url.pathname.startsWith(root) ? url.href : null;} catch (_) {return null;}}
  function publicLink(text, value) {const link = node('a', text); try {const url = new URL(value); if (url.protocol !== 'https:') return node('span', '链接待核验'); link.href = url.href;} catch (_) {return node('span', '链接待核验');} link.target = '_blank'; link.rel = 'noopener noreferrer'; return link;}
  async function action(fn) {if (state.busy) return; state.busy = true; message(''); try {await fn();} catch (error) {message(error.message);} finally {state.busy = false;}}
  async function refreshLists(select = true) {
    const [data, jobsResponse, catalogResponse] = await Promise.all([api('cases/'), fetch('/football/api/jobs/', {credentials: 'same-origin'}), fetch('/football/api/events/', {credentials: 'same-origin'})]);
    if (!catalogResponse.ok) throw new Error('事件目录暂不可用');
    state.catalog = await catalogResponse.json();
    Object.assign(labels, data.event_types);
    const previousType = $('var-event-type').value, typeSelect = $('var-event-type'); typeSelect.replaceChildren();
    for (const group of state.catalog.groups) {
      const options = node('optgroup'); options.label = group;
      for (const event of state.catalog.events.filter(item => item.group === group)) {const option = node('option', event.name); option.value = event.code; options.append(option);}
      typeSelect.append(options);
    }
    if ((data.cases || []).some(item => item.event_type === 'card')) {const option = node('option', labels.card); option.value = 'card'; typeSelect.append(option);}
    if ([...typeSelect.options].some(option => option.value === previousType)) typeSelect.value = previousType;
    const jobs = await jobsResponse.json();
    const cases = Array.isArray(data.cases) ? data.cases : [];
    let chosen = state.current?.id;
    if (!chosen) {try {chosen = sessionStorage.getItem('football_var_case');} catch (_) {}}
    $('var-case-select').replaceChildren(node('option', cases.length ? '选择复核案例' : '暂无案例'));
    $('var-case-select').firstChild.value = '';
    for (const item of cases) {const option = node('option', `${labels[item.event_type] || '事件'} · ${item.title}`); option.value = item.id; $('var-case-select').append(option);}
    $('var-existing-job').replaceChildren(node('option', '选择已上传的图片 / 视频')); $('var-existing-job').firstChild.value = '';
    for (const item of jobs.jobs || []) {if (!item.artifact_urls?.original || !['image', 'video'].includes(item.kind)) continue; const option = node('option', item.input_name); option.value = item.id; $('var-existing-job').append(option);}
    const selected = cases.find(item => item.id === chosen) || cases[0];
    if (selected) {$('var-case-select').value = selected.id; if (select) await loadCase(selected.id);}
  }
  function guide(type) {
    const event = state.catalog?.events.find(item => item.code === type), list = node('ul');
    for (const item of event?.evidence_needed || ['核对出牌动作、牌色、受牌身份与官方记录；普通黄牌记录不自动属于VAR介入范围。']) list.append(node('li', item));
    const content = [node('h3', event ? `${event.name}：先检查证据` : '保留可核验的事件证据'), list];
    if (event?.detector_role) content.push(node('p', event.detector_role));
    if (event?.rule_url) content.push(publicLink('查看官方规则 ↗', event.rule_url));
    $('var-checklist').replaceChildren(...content);
  }
  function mediaTime() {return state.current?.media_kind === 'video' && Number.isFinite($('var-video').currentTime) ? $('var-video').currentTime : 0;}
  function sourceMedia() {return state.current?.media_kind === 'video' ? $('var-video') : $('var-image');}
  function naturalSize() {const media = sourceMedia(); return state.current?.media_kind === 'video' ? [media.videoWidth, media.videoHeight] : [media.naturalWidth, media.naturalHeight];}
  function drawLines(preview = null) {
    const canvas = $('var-reference-layer'), context = canvas.getContext('2d'); context.clearRect(0, 0, canvas.width, canvas.height);
    context.strokeStyle = '#FDE047'; context.lineWidth = Math.max(2, canvas.width / 450); context.setLineDash([canvas.width / 110, canvas.width / 200]);
    for (const line of [...state.lines, ...(preview ? [preview] : [])]) {context.beginPath(); context.moveTo(line.x1 * canvas.width, line.y1 * canvas.height); context.lineTo(line.x2 * canvas.width, line.y2 * canvas.height); context.stroke();}
  }
  function fitStage() {
    if (!state.current) return; const [width, height] = naturalSize(); if (!width || !height) return;
    const stage = $('var-media-stage'), ratio = Math.min(stage.clientWidth / width, stage.clientHeight / height), canvas = $('var-reference-layer'), media = sourceMedia();
    media.style.width = `${width * ratio}px`; media.style.height = `${height * ratio}px`;
    canvas.width = width; canvas.height = height; canvas.style.width = `${width * ratio}px`; canvas.style.height = `${height * ratio}px`;
    canvas.style.left = `${(stage.clientWidth - width * ratio) / 2}px`; canvas.style.top = `${(stage.clientHeight - height * ratio) / 2}px`; drawLines();
  }
  function updateLabels() {$('var-case-title').textContent = state.current.title; $('var-case-badge').textContent = statuses[state.current.status] || state.current.status; $('var-time').textContent = `${mediaTime().toFixed(3)} s`;}
  async function loadCase(id) {const data = await api(`cases/${encodeURIComponent(id)}/`); setCase(data.case);}
  function setCase(value) {
    $('var-video').pause(); state.current = value; state.lines = []; state.drawing = null;
    try {sessionStorage.setItem('football_var_case', value.id);} catch (_) {}
    $('var-workspace').hidden = false; $('var-initial').hidden = true; $('var-case-select').value = value.id;
    $('var-observations').value = value.observations || ''; $('var-verdict').value = value.verdict || ''; $('var-review-status').value = value.status;
    $('var-event-type').value = value.event_type; $('var-frame-note').value = ''; $('var-draw').checked = false; $('var-media-stage').classList.remove('drawing');
    const video = value.media_kind === 'video', url = privateUrl(value.original_url);
    $('var-video').hidden = !video; $('var-image').hidden = video;
    if (!url) throw new Error('原始素材地址无效');
    if (video) {$('var-video').src = url; $('var-video').playbackRate = Number($('var-rate').value);} else {$('var-video').removeAttribute('src'); $('var-image').src = url;}
    for (const id of ['var-back', 'var-play', 'var-forward', 'var-rate']) $(id).disabled = !video;
    guide(value.event_type); updateLabels(); renderFrames();
    $('var-export').replaceChildren();
    for (const [extension, label] of [['json', '证据 JSON'], ['md', '复核报告']]) {const link = node('a', label); link.href = root + `cases/${value.id}/report.${extension}`; link.target = '_blank'; link.rel = 'noopener'; $('var-export').append(link);}
    requestAnimationFrame(fitStage);
  }
  function renderFrames() {
    const container = $('var-keyframes'), frames = state.current?.keyframes || []; container.replaceChildren(); $('var-frame-count').textContent = `${frames.length} 张`;
    if (!frames.length) {container.append(node('p', '尚未保存关键帧。截图保存原始画面，辅助线单独记录。', 'var-resource-empty')); return;}
    for (const frame of frames) {
      const card = node('article', undefined, 'var-frame-card'), image = node('img'), url = privateUrl(frame.image_url); if (url) image.src = url; image.alt = '人工保存的关键帧'; image.loading = 'lazy';
      const copy = node('div', undefined, 'var-frame-copy'), actions = node('div', undefined, 'var-frame-actions'); copy.append(node('b', `${Number(frame.media_time).toFixed(3)} s`), node('p', frame.notes || '未填写说明'));
      const locate = node('button', '定位'); locate.type = 'button'; locate.addEventListener('click', () => {if (state.current.media_kind === 'video') {$('var-video').pause(); $('var-video').currentTime = frame.media_time;} state.lines = frame.lines || []; drawLines(); $('var-frame-note').value = frame.notes || '';});
      const open = node('a', '原帧'); if (url) open.href = url; open.target = '_blank'; open.rel = 'noopener';
      const remove = node('button', '删除', 'var-frame-delete'); remove.type = 'button'; remove.addEventListener('click', () => action(async () => {const result = await api(`cases/${state.current.id}/keyframes/${frame.id}/`, {method: 'DELETE'}); state.current = result.case; $('var-review-status').value = state.current.status; renderFrames(); updateLabels();}));
      actions.append(locate, open, remove); copy.append(actions); card.append(image, copy); container.append(card);
    }
  }
  async function capture() {
    if (!state.current) throw new Error('请先创建复核案例'); const media = sourceMedia(), [width, height] = naturalSize(); if (!width || !height) throw new Error('素材尚未完成加载');
    if (state.current.media_kind === 'video') media.pause();
    const canvas = document.createElement('canvas'); canvas.width = width; canvas.height = height; canvas.getContext('2d').drawImage(media, 0, 0, width, height);
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/png')); if (!blob) throw new Error('关键帧无法编码');
    const form = new FormData(); form.append('file', blob, 'keyframe.png'); form.append('media_time', String(mediaTime())); form.append('notes', $('var-frame-note').value); form.append('lines', JSON.stringify(state.lines));
    const result = await api(`cases/${state.current.id}/keyframes/`, {method: 'POST', body: form}); state.current = result.case; renderFrames(); updateLabels(); message('已保存原始关键帧及独立参考线记录。', false);
  }
  async function resources() {if (!state.resources) state.resources = await api('resources/'); renderResources();}
  function renderResources() {
    const filter = $('var-resource-filter').value;
    const groups = [['matches', 'var-match-resources'], ['datasets', 'var-dataset-resources'], ['rules', 'var-rule-resources']];
    for (const [key, id] of groups) {
      const container = $(id); container.replaceChildren();
      const entries = (state.resources?.[key] || []).filter(item => filter === 'all' || (item.event_types || []).includes(filter));
      if (!entries.length) {container.append(node('p', '当前筛选没有条目。', 'var-resource-empty')); continue;}
      for (const item of entries) {
        const card = node('article', undefined, 'var-resource-card'); const date = item.match_date ? `比赛 ${item.match_date}` : item.published_date ? `发布 ${item.published_date}` : `核验 ${item.verified_on || state.resources.as_of}`;
        card.append(node('span', `${(item.event_types || []).map(value => labels[value] || value).join(' · ')} / ${date}`, 'var-resource-meta'), node('h4', item.title));
        for (const value of [item.media_type, item.locator, item.scale, item.labels, item.access, item.notes]) if (value) card.append(node('p', Array.isArray(value) ? value.join('；') : value));
        if (key === 'datasets') card.append(node('p', item.local_download_note || '尚未取得该来源的完整数据包；已下载的实际文件见运行监控。'));
        const rights = item.rights || item.license; if (rights) card.append(node('p', rights, 'var-resource-rights'));
        const links = node('div', undefined, 'var-resource-links'); links.append(publicLink(key === 'matches' ? '查看官方图片 / 视频入口 ↗' : '查看主来源 ↗', item.source_url));
        if (item.download_url && item.download_url !== item.source_url) links.append(publicLink('下载 / 申请入口 ↗', item.download_url)); card.append(links); container.append(card);
      }
    }
  }
  function setTab(name) {document.querySelectorAll('[data-var-tab]').forEach(button => button.setAttribute('aria-selected', String(button.dataset.varTab === name))); $('var-review-tab').hidden = name !== 'review'; $('var-resources-tab').hidden = name !== 'resources'; if (name === 'resources') resources().catch(error => message(error.message));}
  document.querySelectorAll('[data-var-tab]').forEach(button => button.addEventListener('click', () => setTab(button.dataset.varTab)));
  $('var-resource-filter').addEventListener('change', renderResources);
  $('var-create-form').addEventListener('submit', event => {event.preventDefault(); action(async () => {const file = $('var-file').files[0]; if (!file) throw new Error('请选择比赛素材'); const form = new FormData(); form.append('file', file); form.append('title', $('var-title').value || file.name); form.append('event_type', $('var-event-type').value); const result = await api('cases/', {method: 'POST', body: form}); setCase(result.case); await refreshLists(false);});});
  $('var-use-job').addEventListener('click', () => action(async () => {const id = $('var-existing-job').value; if (!id) throw new Error('请选择已有素材'); const result = await api('cases/', {method: 'POST', body: {job_id: id, title: $('var-title').value || '比赛事件复核', event_type: $('var-event-type').value}}); setCase(result.case); await refreshLists(false);}));
  $('var-case-select').addEventListener('change', () => {if ($('var-case-select').value) action(() => loadCase($('var-case-select').value));});
  $('var-refresh').addEventListener('click', () => action(() => refreshLists()));
  $('var-save-form').addEventListener('submit', event => {event.preventDefault(); action(async () => {const result = await api(`cases/${state.current.id}/`, {method: 'PATCH', body: {observations: $('var-observations').value, verdict: $('var-verdict').value, status: $('var-review-status').value}}); state.current = result.case; updateLabels(); message('人工复核记录已保存；未生成自动判罚。', false); await refreshLists(false);});});
  $('var-capture').addEventListener('click', () => action(capture));
  $('var-play').addEventListener('click', () => {const media = $('var-video'); if (media.paused) media.play().catch(error => message(error.message)); else media.pause();});
  for (const [id, amount] of [['var-back', -.04], ['var-forward', .04]]) $(id).addEventListener('click', () => {const media = $('var-video'); media.pause(); media.currentTime = Math.min(Number.isFinite(media.duration) ? media.duration : Infinity, Math.max(0, media.currentTime + amount));});
  $('var-rate').addEventListener('change', () => {$('var-video').playbackRate = Number($('var-rate').value);});
  $('var-video').addEventListener('timeupdate', () => {$('var-time').textContent = `${mediaTime().toFixed(3)} s`;});
  $('var-video').addEventListener('loadedmetadata', () => {fitStage(); const duration = $('var-video').duration, id = state.current?.id; if (id && Number.isFinite(duration) && duration > 0 && duration <= 21600) api(`cases/${id}/`, {method: 'PATCH', body: {duration_seconds: duration}}).then(data => {if (state.current?.id === id) state.current.duration_seconds = data.case.duration_seconds;}).catch(error => message(error.message));});
  $('var-video').addEventListener('error', () => {if (state.current?.media_kind === 'video') message('浏览器无法解码此视频，请使用可播放的MP4/H.264或WebM素材。');});
  $('var-image').addEventListener('load', fitStage);
  $('var-draw').addEventListener('change', () => {$('var-media-stage').classList.toggle('drawing', $('var-draw').checked);});
  $('var-clear-lines').addEventListener('click', () => {state.lines = []; drawLines();});
  const layer = $('var-reference-layer');
  function point(event) {const rectangle = layer.getBoundingClientRect(); return {x: Math.max(0, Math.min(1, (event.clientX - rectangle.left) / rectangle.width)), y: Math.max(0, Math.min(1, (event.clientY - rectangle.top) / rectangle.height))};}
  layer.addEventListener('pointerdown', event => {if (!$('var-draw').checked || !state.current) return; const start = point(event); state.drawing = {x1: start.x, y1: start.y, x2: start.x, y2: start.y}; layer.setPointerCapture(event.pointerId);});
  layer.addEventListener('pointermove', event => {if (!state.drawing) return; const end = point(event); state.drawing.x2 = end.x; state.drawing.y2 = end.y; drawLines(state.drawing);});
  layer.addEventListener('pointerup', event => {if (!state.drawing) return; const end = point(event); state.drawing.x2 = end.x; state.drawing.y2 = end.y; if (state.lines.length < 12 && Math.hypot(state.drawing.x2 - state.drawing.x1, state.drawing.y2 - state.drawing.y1) > .003) state.lines.push(state.drawing); state.drawing = null; if (layer.hasPointerCapture(event.pointerId)) layer.releasePointerCapture(event.pointerId); drawLines();});
  layer.addEventListener('pointercancel', () => {state.drawing = null; drawLines();});
  window.addEventListener('resize', fitStage);
  if ('ResizeObserver' in window) new ResizeObserver(fitStage).observe($('var-media-stage'));
  async function activate() {if (location.hash.split('?')[0] !== '#var') {$('var-video').pause(); return;} try {if (new URLSearchParams(location.hash.split('?')[1] || '').get('tab') === 'resources') setTab('resources'); if (!state.loaded) {await refreshLists(); state.loaded = true;} requestAnimationFrame(fitStage);} catch (error) {message(error.message);}}
  window.addEventListener('hashchange', activate); activate();
})();
