(() => {
  const root = document.documentElement;
  const systemTheme = window.matchMedia('(prefers-color-scheme: light)');

  function applyTheme() {
    const saved = localStorage.getItem('theme') || 'system';
    root.dataset.theme = saved === 'system' ? (systemTheme.matches ? 'light' : 'dark') : saved;
    root.classList.toggle('light', root.dataset.theme === 'light');
    document.body?.classList.toggle('light', root.dataset.theme === 'light');
  }

  window.toggleAuthTheme = () => {
    const next = root.dataset.theme === 'light' ? 'dark' : 'light';
    localStorage.setItem('theme', next);
    root.dataset.theme = next;
    root.classList.toggle('light', next === 'light');
    document.body?.classList.toggle('light', next === 'light');
  };

  applyTheme();
  systemTheme.addEventListener('change', () => {
    if ((localStorage.getItem('theme') || 'system') === 'system') applyTheme();
  });
})();
