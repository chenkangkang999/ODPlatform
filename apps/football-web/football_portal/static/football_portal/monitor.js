/* Operations are real session records. Capability targets are never model scores. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const state = {data: null, timer: null, busy: false};
  const labels = {queued: '排队中', running: '处理中', completed: '已完成', failed: '失败', cancelled: '已取消'};
  const dataLabels = {visual_samples: '有视觉样本', annotations_only: '仅结构化标注', pending: '待取得数据'};
  const number = value => Number(value || 0).toLocaleString('zh-CN');
  const size = bytes => bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(2)} GiB` : `${(bytes / 1024 ** 2).toFixed(1)} MiB`;
  const node = (tag, text, cls) => {const value = document.createElement(tag); if (text !== undefined) value.textContent = text; if (cls) value.className = cls; return value;};
  function renderCoverage() {
    if (!state.data) return;
    const query = $('monitor-search').value.trim().toLowerCase(), selected = $('monitor-data-filter').value;
    const items = state.data.event_coverage.filter(item => (!query || `${item.name} ${item.code} ${(item.subtypes || []).join(' ')}`.toLowerCase().includes(query)) && (selected === 'all' || item.data_state === selected));
    $('monitor-coverage-count').textContent = `${items.length} / ${state.data.event_coverage.length} 项`;
    const host = $('monitor-coverage'); host.replaceChildren();
    for (const item of items) {
      const card = node('article', undefined, 'capability-card');
      const top = node('div', undefined, 'capability-title'); top.append(node('h3', item.name), node('span', dataLabels[item.data_state], `data-badge ${item.data_state}`));
      card.append(top, node('p', item.group, 'capability-group'));
      if (item.subtypes?.length) card.append(node('p', item.subtypes.join(' / '), 'capability-subtypes'));
      const counts = node('div', undefined, 'capability-counts');
      counts.append(node('span', `视频 ${number(item.video_files)}`), node('span', `图片 ${number(item.image_files)}`), node('span', `标注文件 ${number(item.annotation_files)}`));
      card.append(counts, node('p', item.automated_recognition ? '已登记自动模型 · 仍需按证据验收' : '自动模型待接入 · 可人工记录', 'capability-model'));
      const details = node('details'); details.append(node('summary', '需要哪些证据？'));
      const list = node('ul'); for (const text of item.evidence_needed || []) list.append(node('li', text)); details.append(list); card.append(details); host.append(card);
    }
    if (!items.length) host.append(node('p', '没有符合筛选条件的事件。', 'small-note'));
  }
  function render(data) {
    state.data = data;
    const counts = data.jobs.counts, files = data.datasets.counts;
    $('monitor-peek-queue').textContent = number((counts.queued || 0) + (counts.running || 0));
    $('monitor-peek-data').textContent = number(data.datasets.files.length);
    $('monitor-peek-evidence').textContent = number(data.reviews.total);
    $('monitor-peek-size').textContent = size(data.datasets.total_bytes);
    $('monitor-peek-status').textContent = 'Django 服务在线';
    $('monitor-total').textContent = number(data.jobs.total);
    $('monitor-active').textContent = number((counts.queued || 0) + (counts.running || 0));
    $('monitor-failed').textContent = number(counts.failed);
    $('monitor-local-bytes').textContent = size(data.datasets.total_bytes);
    $('monitor-timestamp').textContent = `更新于 ${new Date(data.generated_at).toLocaleTimeString('zh-CN')} · 每${data.service.refresh_seconds}秒刷新`;
    const alerts = $('monitor-alerts'); alerts.replaceChildren();
    for (const alert of data.alerts) alerts.append(node('p', alert.message, `monitor-alert ${alert.level === 'warning' ? 'warning' : 'info'}`));
    const jobs = $('monitor-jobs'); jobs.replaceChildren();
    for (const job of data.jobs.recent) {
      const row = node('a', undefined, 'monitor-job'); row.href = '#history';
      const identity = node('div'); identity.append(node('b', job.input_name), node('small', job.error || new Date(job.updated_at).toLocaleString('zh-CN')));
      const badge = node('span', `${labels[job.status] || '未知'}${job.status === 'running' ? ` ${job.progress}%` : ''}`, 'tag'); row.append(identity, badge); jobs.append(row);
    }
    if (!data.jobs.recent.length) jobs.append(node('p', '当前会话还没有检测任务。', 'small-note'));
    const facts = $('monitor-evidence-facts'); facts.replaceChildren();
    for (const [key, value] of [['人工案例', data.reviews.total], ['已人工复核', data.reviews.counts.reviewed || 0], ['关键帧', data.reviews.keyframes], ['已记录LLM调用', data.assistant.real_calls]]) facts.append(node('dt', key), node('dd', number(value)));
    const summary = data.coverage_summary;
    $('monitor-model-ready').textContent = `${summary.trained_event_models} / ${summary.targets}`;
    $('monitor-coverage-summary').textContent = `${summary.visual_sample_events}项有相关视觉样本，${summary.annotation_only_events}项仅有结构化标注；样本关联不等于已完成分类标注或具备自动识别能力。`;
    $('monitor-file-summary').textContent = `视频 ${number(files.video)} · 图片 ${number(files.image)} · 标注 ${number(files.annotations)} · 元数据/许可 ${number((files.metadata || 0) + (files.license || 0))}`;
    $('monitor-data-directory').textContent = data.datasets.directory;
    const fileList = $('monitor-files'); fileList.replaceChildren();
    for (const file of data.datasets.files) {
      const row = node('li'); row.append(node('span', file.path), node('small', `${file.kind} · ${size(file.bytes)}`)); fileList.append(row);
    }
    if (!data.datasets.files.length) fileList.append(node('li', '下载尚未登记完成，请稍后刷新。'));
    const blocked = $('monitor-blocked'); blocked.replaceChildren();
    for (const item of data.datasets.blocked_sources || []) blocked.append(node('li', typeof item === 'string' ? item : `${item.name || item.source_id || '数据来源'}：${item.reason || item.access || '需按来源要求申请'}`));
    for (const item of data.datasets.download_failures || []) blocked.append(node('li', `下载未完成 · ${item.path}：${item.error}`));
    if (!blocked.childElementCount) blocked.append(node('li', '尚无登记项；完整数据需求仍见42项能力清单。'));
    renderCoverage();
  }
  async function refresh() {
    if (state.busy) return;
    state.busy = true;
    try {
      const response = await fetch('/football/api/monitor/', {credentials: 'same-origin', cache: 'no-store'});
      if (!response.ok) throw new Error(`监控接口返回 ${response.status}`);
      render(await response.json()); $('monitor-error').hidden = true;
    } catch (error) {
      $('monitor-error').textContent = `状态更新失败：${error.message}`; $('monitor-error').hidden = false;
      $('monitor-peek-status').textContent = '状态暂不可用';
    } finally {state.busy = false;}
  }
  function activate() {
    clearInterval(state.timer); state.timer = null;
    const page = location.hash.slice(1).split('?')[0] || 'overview';
    if (!document.hidden && ['overview', 'monitor'].includes(page)) {
      refresh();
      const interval = Number(document.body.dataset.monitorIntervalMs);
      if (Number.isFinite(interval) && interval >= 1000) state.timer = setInterval(refresh, interval);
    }
  }
  $('monitor-search').addEventListener('input', renderCoverage);
  $('monitor-data-filter').addEventListener('change', renderCoverage);
  $('monitor-refresh').addEventListener('click', refresh);
  window.addEventListener('hashchange', activate); document.addEventListener('visibilitychange', activate);
  activate();
})();
