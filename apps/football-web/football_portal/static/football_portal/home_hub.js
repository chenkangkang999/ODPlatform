/* Home-only interactions: calm scene playback and the app launcher. */
(() => {
  'use strict';

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const carousel = document.getElementById('scene-carousel');
  const scenes = carousel ? Array.from(carousel.querySelectorAll('[data-scene]')) : [];
  const previous = document.getElementById('scene-prev');
  const next = document.getElementById('scene-next');
  const dots = Array.from(document.querySelectorAll('[data-scene-dot]'));
  const live = document.getElementById('scene-live');
  const sceneWindow = carousel?.closest('.scene-window');
  const autoplay = document.getElementById('scene-autoplay');
  const configuredDelay = Number(carousel?.dataset.intervalMs);
  const autoplayDelay = Number.isFinite(configuredDelay) && configuredDelay >= 1000 ? configuredDelay : null;
  const intervalLabel = autoplayDelay === null ? '' : `${autoplayDelay / 1000}秒`;
  let currentScene = 0;
  let scrollFrame = null;
  let drag = null;
  let suppressClickUntil = 0;
  let autoplayTimer = null;
  let userPaused = false;
  let pointerInside = false;
  let focusInside = false;
  let touching = false;

  function activeHome() {
    return document.body.dataset.page === 'overview' && !document.hidden;
  }

  function playbackBlocked() {
    return autoplayDelay === null || !activeHome() || reducedMotion.matches || userPaused || pointerInside || focusInside ||
      touching || Boolean(drag);
  }

  function updatePlaybackButton() {
    if (!autoplay) return;
    const staticMode = reducedMotion.matches || autoplayDelay === null;
    autoplay.disabled = staticMode;
    autoplay.setAttribute('aria-pressed', String(userPaused || staticMode));
    autoplay.textContent = staticMode ? '静态浏览' : userPaused ? '播放轮播' : '暂停轮播';
    autoplay.setAttribute('aria-label', staticMode ? '已停播，仍可手动切换' :
      userPaused ? `播放场景轮播，每${intervalLabel}切换` : '暂停场景轮播');
    autoplay.title = reducedMotion.matches ? '系统已启用减少动态效果，仍可手动切换' :
      autoplayDelay === null ? '自动轮播间隔配置不可用，仍可手动切换' :
      `每${intervalLabel}切换；鼠标悬停、键盘聚焦、拖动或离开页面时暂停`;
    if (live) live.setAttribute('aria-live', playbackBlocked() ? 'polite' : 'off');
  }

  function syncAutoplay() {
    if (autoplayTimer !== null) window.clearTimeout(autoplayTimer);
    autoplayTimer = null;
    updatePlaybackButton();
    if (!carousel || scenes.length < 2 || playbackBlocked()) return;
    autoplayTimer = window.setTimeout(() => {
      autoplayTimer = null;
      if (!playbackBlocked()) goToScene((nearestScene() + 1) % scenes.length);
    }, autoplayDelay);
  }

  function markScene(index) {
    currentScene = index;
    dots.forEach(dot => {
      const selected = Number(dot.dataset.sceneDot) === index;
      dot.classList.toggle('active', selected);
      dot.setAttribute('aria-pressed', String(selected));
      if (selected) dot.setAttribute('aria-current', 'true');
      else dot.removeAttribute('aria-current');
    });
    if (previous) previous.disabled = index <= 0;
    if (next) next.disabled = index >= scenes.length - 1;
    if (live) {
      const heading = scenes[index]?.querySelector('[data-scene-title], h2, h3');
      const name = heading?.textContent.trim() || '足球场景';
      live.textContent = `${name}，第 ${index + 1} 个场景，共 ${scenes.length} 个。`;
    }
  }

  function nearestScene() {
    if (!carousel || !scenes.length || !carousel.clientWidth) return currentScene;
    const edge = carousel.getBoundingClientRect().left + carousel.clientLeft;
    let nearest = 0;
    let distance = Infinity;
    scenes.forEach((scene, index) => {
      const offset = Math.abs(scene.getBoundingClientRect().left - edge);
      if (offset < distance) {distance = offset; nearest = index;}
    });
    return nearest;
  }

  function goToScene(index) {
    if (!carousel || !scenes.length) return;
    index = Math.max(0, Math.min(scenes.length - 1, index));
    const edge = carousel.getBoundingClientRect().left + carousel.clientLeft;
    const left = carousel.scrollLeft + scenes[index].getBoundingClientRect().left - edge;
    carousel.scrollTo({left, behavior: reducedMotion.matches ? 'auto' : 'smooth'});
    markScene(index);
    syncAutoplay();
  }

  function queueSceneSync() {
    if (scrollFrame !== null) return;
    scrollFrame = requestAnimationFrame(() => {
      scrollFrame = null;
      if (carousel?.clientWidth) markScene(nearestScene());
    });
  }

  function alignSceneAfterResize() {
    if (!carousel?.clientWidth || !scenes[currentScene]) return;
    if (scrollFrame !== null) cancelAnimationFrame(scrollFrame);
    scrollFrame = null;
    const edge = carousel.getBoundingClientRect().left + carousel.clientLeft;
    const left = carousel.scrollLeft + scenes[currentScene].getBoundingClientRect().left - edge;
    // A breakpoint changes card width: align immediately rather than leaving half a card.
    const behavior = carousel.style.scrollBehavior;
    carousel.style.scrollBehavior = 'auto';
    carousel.scrollLeft = left;
    carousel.style.scrollBehavior = behavior;
    markScene(currentScene);
    syncAutoplay();
  }

  function endDrag(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const wasDragging = drag.active;
    if (wasDragging) {
      suppressClickUntil = performance.now() + 400;
      carousel.style.scrollSnapType = drag.snapType;
      carousel.style.userSelect = drag.userSelect;
      carousel.classList.remove('is-dragging');
      if (carousel.hasPointerCapture(event.pointerId)) carousel.releasePointerCapture(event.pointerId);
    }
    drag = null;
    if (wasDragging) goToScene(nearestScene());
    else syncAutoplay();
  }

  if (carousel && scenes.length) {
    markScene(0);
    autoplay?.addEventListener('click', () => {userPaused = !userPaused; syncAutoplay();});
    previous?.addEventListener('click', () => goToScene(currentScene - 1));
    next?.addEventListener('click', () => goToScene(currentScene + 1));
    dots.forEach(dot => dot.addEventListener('click', () => goToScene(Number(dot.dataset.sceneDot))));
    carousel.addEventListener('scroll', queueSceneSync, {passive: true});
    carousel.addEventListener('keydown', event => {
      // Links/buttons inside a scene keep their own keyboard behavior.
      if (event.target !== carousel) return;
      if (event.key === 'ArrowRight' || event.key === 'ArrowLeft') {
        event.preventDefault();
        goToScene(currentScene + (event.key === 'ArrowRight' ? 1 : -1));
      }
    });
    carousel.addEventListener('dragstart', event => event.preventDefault());
    carousel.addEventListener('pointerdown', event => {
      // Touch uses the browser's native inertial horizontal scrolling.
      if (event.pointerType === 'touch' || event.button !== 0 ||
          event.target.closest('button, input, select, textarea, [contenteditable="true"]')) return;
      drag = {pointerId: event.pointerId, x: event.clientX, y: event.clientY,
        left: carousel.scrollLeft, active: false, snapType: carousel.style.scrollSnapType,
        userSelect: carousel.style.userSelect};
      syncAutoplay();
    });
    carousel.addEventListener('pointermove', event => {
      if (!drag || event.pointerId !== drag.pointerId) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (!drag.active && Math.abs(dx) > 8 && Math.abs(dx) > Math.abs(dy) * 1.2) {
        drag.active = true;
        carousel.style.scrollSnapType = 'none';
        carousel.style.userSelect = 'none';
        carousel.classList.add('is-dragging');
        carousel.setPointerCapture(event.pointerId);
      }
      if (drag.active) {
        event.preventDefault();
        carousel.scrollLeft = drag.left - dx;
      }
    });
    window.addEventListener('pointerup', endDrag);
    window.addEventListener('pointercancel', endDrag);
    carousel.addEventListener('click', event => {
      if (performance.now() < suppressClickUntil) {
        event.preventDefault();
        event.stopPropagation();
      }
    }, true);
    window.addEventListener('resize', alignSceneAfterResize, {passive: true});
    if ('ResizeObserver' in window) new ResizeObserver(alignSceneAfterResize).observe(carousel);
    sceneWindow?.addEventListener('pointerenter', event => {
      if (event.pointerType !== 'touch') {pointerInside = true; syncAutoplay();}
    });
    sceneWindow?.addEventListener('pointerleave', event => {
      if (event.pointerType !== 'touch') {pointerInside = false; syncAutoplay();}
    });
    sceneWindow?.addEventListener('focusin', event => {
      focusInside = event.target !== autoplay;
      syncAutoplay();
    });
    sceneWindow?.addEventListener('focusout', () => {
      // Wait for focus to settle before deciding whether it left the controls.
      queueMicrotask(() => {
        focusInside = sceneWindow.contains(document.activeElement) && document.activeElement !== autoplay;
        syncAutoplay();
      });
    });
    carousel.addEventListener('touchstart', () => {touching = true; syncAutoplay();}, {passive: true});
    const endTouch = () => {touching = false; syncAutoplay();};
    carousel.addEventListener('touchend', endTouch, {passive: true});
    carousel.addEventListener('touchcancel', endTouch, {passive: true});
    document.addEventListener('visibilitychange', syncAutoplay);
    window.addEventListener('hashchange', syncAutoplay);
    new MutationObserver(syncAutoplay).observe(document.body, {attributes: true, attributeFilter: ['data-page']});
    reducedMotion.addEventListener('change', syncAutoplay);
    syncAutoplay();
  }

  const switcher = document.getElementById('app-switcher');
  const navigation = document.getElementById('main-nav');
  const launcher = navigation?.closest('.nav-track');

  function closeLauncher(restoreFocus = false) {
    document.body.classList.remove('launcher-open');
    switcher?.setAttribute('aria-expanded', 'false');
    if (restoreFocus) switcher?.focus();
  }

  if (switcher && navigation && launcher) {
    switcher.setAttribute('aria-expanded', 'false');
    switcher.addEventListener('click', () => {
      if (document.body.classList.contains('launcher-open')) {closeLauncher(true); return;}
      document.body.classList.add('launcher-open');
      switcher.setAttribute('aria-expanded', 'true');
      requestAnimationFrame(() => navigation.querySelector('a[href], button')?.focus());
    });
    navigation.addEventListener('click', event => {
      if (event.target.closest('[data-nav]')) closeLauncher();
    });
    document.addEventListener('pointerdown', event => {
      if (document.body.classList.contains('launcher-open') &&
          !launcher.contains(event.target) && !switcher.contains(event.target)) closeLauncher();
    });
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && document.body.classList.contains('launcher-open')) {
        event.preventDefault();
        closeLauncher(true);
      }
    });
    window.addEventListener('hashchange', () => {closeLauncher(); queueSceneSync();});
  }

  // The retired kit picker must not recolor the new monitoring workspace from a saved preference.
  delete document.documentElement.dataset.clubTheme;
})();
