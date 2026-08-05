(() => {
  const state = {
    root: '',
    entryMap: new Map(),
    childrenByParent: new Map(),
    expanded: new Set(),
    selected: new Set(),
    sort: 'size',
    direction: 'desc',
    filter: '',
    browsePath: '/',
    browseParent: null,
    scanning: false,
    scanId: null,
    scanGeneration: 0,
  };

  const $ = (selector) => document.querySelector(selector);
  const elements = {};

  function escapeHtml(value) {
    return String(value).replace(/[&<>'"]/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;',
    })[char]);
  }

  function formatBytes(value) {
    if (!Number.isFinite(value)) return '—';
    if (value === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
    const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1);
    const number = value / (1024 ** index);
    return `${number.toFixed(index === 0 || number >= 100 ? 0 : number >= 10 ? 1 : 2)} ${units[index]}`;
  }

  function formatDate(timestamp) {
    if (!Number.isFinite(timestamp)) return '—';
    return new Intl.DateTimeFormat(undefined, {
      year: 'numeric', month: 'short', day: '2-digit',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
    }).format(new Date(timestamp * 1000));
  }

  function formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return 'Estimating time remaining…';
    if (seconds < 2) return 'Less than a few seconds remaining';
    if (seconds < 60) return `About ${Math.ceil(seconds)} seconds remaining`;
    if (seconds < 3600) return `About ${Math.ceil(seconds / 60)} minutes remaining`;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.ceil((seconds % 3600) / 60);
    return `About ${hours}h ${minutes}m remaining`;
  }

  const wait = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

  async function jsonRequest(url, options = {}) {
    const response = await fetch(url, options);
    const reply = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(reply.detail || response.statusText || 'Request failed');
    return reply;
  }

  function showError(message) {
    elements.errorBanner.textContent = message || '';
    elements.errorBanner.hidden = !message;
  }

  async function browse(path) {
    elements.directoryError.hidden = true;
    elements.directoryList.innerHTML = '<div class="directory-loading">Loading directories…</div>';
    try {
      const result = await jsonRequest(`/api/disk-manager/browse?path=${encodeURIComponent(path)}`);
      state.browsePath = result.path;
      state.browseParent = result.parent;
      elements.directoryPath.value = result.path;
      elements.directoryUp.disabled = !result.parent;
      if (!result.directories.length) {
        elements.directoryList.innerHTML = '<div class="directory-empty">No subdirectories</div>';
      } else {
        elements.directoryList.innerHTML = result.directories.map((directory) => `
          <button type="button" class="directory-item" data-path="${escapeHtml(directory.path)}">
            <span class="folder-icon">▰</span><span>${escapeHtml(directory.name)}</span>
          </button>`).join('');
      }
    } catch (error) {
      elements.directoryError.textContent = error.message;
      elements.directoryError.hidden = false;
      elements.directoryList.innerHTML = '';
    }
  }

  function openDirectoryDialog() {
    const initial = state.root || state.browsePath || '/';
    elements.directoryDialog.showModal();
    browse(initial);
  }

  async function scan(path) {
    const previousScanId = state.scanId;
    if (previousScanId) {
      fetch(`/api/disk-manager/scan/${previousScanId}/cancel`, { method: 'POST' }).catch(() => {});
    }
    const generation = ++state.scanGeneration;
    state.scanning = true;
    state.scanId = null;
    state.entryMap = new Map();
    state.childrenByParent = new Map();
    state.expanded = new Set();
    state.filter = '';
    elements.filter.value = '';
    state.selected.clear();
    showError('');
    elements.summaryStatus.textContent = 'Scanning…';
    elements.summaryStatus.classList.add('working');
    elements.scanProgress.hidden = false;
    elements.progressTitle.textContent = 'Scanning directory…';
    elements.progressPercent.textContent = '0%';
    elements.progressBar.style.width = '0%';
    elements.progressEta.textContent = 'Estimating time remaining…';
    elements.progressItems.textContent = '0 items discovered';
    elements.progressSize.textContent = '0 B found';
    elements.progressRate.textContent = 'Calculating scan rate…';
    elements.progressCurrent.textContent = '.';
    elements.rescan.disabled = true;
    elements.filter.disabled = true;
    elements.rows.innerHTML = '<tr class="empty-row"><td colspan="5"><span class="spinner"></span> Searching for files and directories…</td></tr>';
    try {
      const started = await jsonRequest('/api/disk-manager/scan', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ path }),
      });
      if (generation !== state.scanGeneration) return;
      state.scanId = started.scan_id;
      state.root = started.root;
      state.browsePath = started.root;
      elements.summaryPath.textContent = started.root;
      elements.summaryPath.title = started.root;

      let cursor = 0;
      let lastRender = 0;
      while (generation === state.scanGeneration) {
        const result = await jsonRequest(`/api/disk-manager/scan/${started.scan_id}?since=${cursor}`);
        if (generation !== state.scanGeneration) return;
        cursor = result.next_cursor;
        result.changes.forEach((entry) => {
          const isNew = !state.entryMap.has(entry.path);
          state.entryMap.set(entry.path, entry);
          if (isNew) {
            const slash = entry.path.lastIndexOf('/');
            const parent = slash < 0 ? '' : entry.path.slice(0, slash);
            if (!state.childrenByParent.has(parent)) state.childrenByParent.set(parent, []);
            state.childrenByParent.get(parent).push(entry.path);
          }
        });
        elements.summaryCount.textContent = `${result.file_count.toLocaleString()} files · ${result.directory_count.toLocaleString()} directories`;
        elements.summarySize.textContent = formatBytes(result.complete ? result.total_size : result.scanned_size);
        updateScanProgress(result);
        const now = performance.now();
        if (result.complete || now - lastRender >= 500) {
          render();
          renderTreemap();
          lastRender = now;
        }
        if (result.complete) {
          const skipped = result.errors.length + (result.skipped_mounts?.length || 0);
          elements.summaryStatus.textContent = result.status === 'cancelled'
            ? 'Cancelled'
            : result.status === 'failed'
              ? 'Scan failed'
              : skipped ? `Complete · ${skipped} skipped` : 'Complete';
          if (result.errors.length) {
            showError(`${result.errors.length} item${result.errors.length === 1 ? '' : 's'} could not be read. ${result.errors[0]}`);
          }
          break;
        }
        await wait(result.changes.length >= 5000 ? 0 : 300);
      }
    } catch (error) {
      if (generation !== state.scanGeneration) return;
      state.entryMap = new Map();
      state.childrenByParent = new Map();
      elements.summaryStatus.textContent = 'Scan failed';
      elements.rows.innerHTML = '<tr class="empty-row"><td colspan="5">The directory could not be scanned.</td></tr>';
      showError(error.message);
    } finally {
      if (generation !== state.scanGeneration) return;
      state.scanning = false;
      state.scanId = null;
      elements.summaryStatus.classList.remove('working');
      elements.rescan.disabled = !state.root;
      elements.filter.disabled = !state.root;
      updateSelectionControls();
    }
  }

  function updateScanProgress(result) {
    const percent = Math.max(0, Math.min(100, result.progress_pct || 0));
    const itemCount = result.file_count + result.directory_count;
    elements.progressBar.style.width = `${percent}%`;
    elements.progressPercent.textContent = `${Math.round(percent)}% estimated`;
    elements.progressItems.textContent = `${itemCount.toLocaleString()} items discovered`;
    elements.progressSize.textContent = `${formatBytes(result.scanned_size)} found`;
    elements.progressRate.textContent = result.rate > 0 ? `${Math.round(result.rate).toLocaleString()} items/sec` : 'Calculating scan rate…';
    elements.progressCurrent.textContent = result.current_path || '.';
    elements.progressCurrent.title = result.current_path || '.';
    if (result.complete) {
      elements.progressTitle.textContent = result.status === 'complete' ? 'Scan complete' : `Scan ${result.status}`;
      elements.progressPercent.textContent = result.status === 'complete' ? '100%' : `${Math.round(percent)}%`;
      elements.progressEta.textContent = result.status === 'complete' ? 'Complete' : '';
    } else {
      elements.progressTitle.textContent = 'Scanning directory…';
      elements.progressEta.textContent = formatDuration(result.eta_seconds);
    }
  }

  function treeEntries() {
    const needle = state.filter.trim().toLocaleLowerCase();
    const included = new Set();
    let children = state.childrenByParent;
    if (needle) {
      state.entryMap.forEach((entry) => {
        if (!entry.path.toLocaleLowerCase().includes(needle)) return;
        included.add(entry.path);
        let parent = entry.path;
        while (parent.includes('/')) {
          parent = parent.slice(0, parent.lastIndexOf('/'));
          included.add(parent);
        }
      });
      children = new Map();
      included.forEach((path) => {
        if (!state.entryMap.has(path)) return;
        const slash = path.lastIndexOf('/');
        const parent = slash < 0 ? '' : path.slice(0, slash);
        if (!children.has(parent)) children.set(parent, []);
        children.get(parent).push(path);
      });
    }
    const multiplier = state.direction === 'asc' ? 1 : -1;
    const sortEntries = (entries) => entries.sort((left, right) => {
      const primary = Number(left[state.sort]) - Number(right[state.sort]);
      if (primary) return primary * multiplier;
      return left.path.localeCompare(right.path, undefined, { numeric: true }) * multiplier;
    });
    const flattened = [];
    const visit = (parent, depth) => {
      const siblings = (children.get(parent) || [])
        .map((path) => state.entryMap.get(path))
        .filter(Boolean);
      sortEntries(siblings).forEach((entry) => {
        flattened.push({ entry, depth });
        if (entry.type === 'directory' && (needle || state.expanded.has(entry.path))) {
          visit(entry.path, depth + 1);
        }
      });
    };
    visit('', 0);
    return flattened;
  }

  function render() {
    const nodes = treeEntries();
    elements.visibleCount.textContent = `${nodes.length.toLocaleString()} of ${state.entryMap.size.toLocaleString()}`;
    if (!nodes.length) {
      const text = state.entryMap.size ? 'No items match the filter.' : 'This directory is empty.';
      elements.rows.innerHTML = `<tr class="empty-row"><td colspan="5">${text}</td></tr>`;
    } else {
      elements.rows.innerHTML = nodes.map(({ entry, depth }) => {
        const checked = state.selected.has(entry.path) ? ' checked' : '';
        const icon = entry.type === 'directory' ? '▰' : entry.type === 'mount' ? '◆' : entry.type === 'symlink' ? '↗' : '▤';
        const hasChildren = state.childrenByParent.has(entry.path);
        const toggle = entry.type === 'directory' && hasChildren
          ? `<button class="tree-toggle" type="button" aria-label="${state.expanded.has(entry.path) ? 'Collapse' : 'Expand'} ${escapeHtml(entry.name)}">${state.expanded.has(entry.path) ? '▾' : '▸'}</button>`
          : '<span class="tree-toggle-spacer"></span>';
        const folderSize = state.scanning && entry.type === 'directory' && entry.size === 0 ? 'Scanning…' : formatBytes(entry.size);
        return `<tr data-path="${escapeHtml(entry.path)}">
          <td class="check-cell"><input type="checkbox" class="row-check" aria-label="Select ${escapeHtml(entry.path)}"${checked}></td>
          <td><div class="item-name" style="--tree-depth:${depth}">${toggle}<span class="type-icon ${entry.type}">${icon}</span><span><strong>${escapeHtml(entry.name)}</strong><small class="mono">${escapeHtml(entry.path)}</small></span></div></td>
          <td class="number-cell" data-value="${entry.size}">${folderSize}</td>
          <td class="date-cell" data-value="${entry.modified}">${formatDate(entry.modified)}</td>
          <td><span class="type-badge">${escapeHtml(entry.type)}</span></td>
        </tr>`;
      }).join('');
    }
    document.querySelectorAll('.sort-button').forEach((button) => {
      const active = button.dataset.sort === state.sort;
      button.classList.toggle('active', active);
      button.querySelector('.sort-arrow').textContent = active ? (state.direction === 'asc' ? '↑' : '↓') : '';
    });
    updateSelectionControls(nodes);
  }

  function updateSelectionControls(nodes = treeEntries()) {
    const visiblePaths = nodes.map(({ entry }) => entry.path);
    const selectedVisible = visiblePaths.filter((path) => state.selected.has(path)).length;
    elements.selectAll.disabled = !visiblePaths.length || state.scanning;
    elements.selectAll.checked = visiblePaths.length > 0 && selectedVisible === visiblePaths.length;
    elements.selectAll.indeterminate = selectedVisible > 0 && selectedVisible < visiblePaths.length;
    elements.deleteSelected.disabled = !state.selected.size || state.scanning;
    elements.deleteSelected.textContent = state.selected.size
      ? `Permanently delete (${state.selected.size.toLocaleString()})`
      : 'Permanently delete';
  }

  function renderTreemap() {
    const topLevel = (state.childrenByParent.get('') || [])
      .map((path) => state.entryMap.get(path))
      .filter(Boolean)
      .sort((left, right) => right.size - left.size)
      .slice(0, 24);
    elements.treemapSection.hidden = !topLevel.length;
    if (!topLevel.length) return;
    const max = Math.max(topLevel[0].size, 1);
    elements.treemap.innerHTML = topLevel.map((entry, index) => {
      const weight = Math.max(1, Math.round(4 + 30 * Math.sqrt(entry.size / max)));
      return `<button class="treemap-tile color-${index % 8}" data-path="${escapeHtml(entry.path)}" style="flex-grow:${weight}" title="${escapeHtml(entry.path)} · ${formatBytes(entry.size)}">
        <strong>${escapeHtml(entry.name)}</strong><span>${formatBytes(entry.size)}</span>
      </button>`;
    }).join('');
  }

  function openDeleteDialog() {
    const paths = [...state.selected];
    elements.deleteSummary.textContent = `${paths.length.toLocaleString()} selected item${paths.length === 1 ? '' : 's'} will be permanently removed.`;
    const preview = paths.slice(0, 12).map(escapeHtml).join('<br>');
    elements.deletePreview.innerHTML = preview + (paths.length > 12 ? `<br>…and ${paths.length - 12} more` : '');
    elements.deleteConfirm.value = '';
    elements.deleteConfirmButton.disabled = true;
    elements.deleteDialog.showModal();
    setTimeout(() => elements.deleteConfirm.focus(), 0);
  }

  async function permanentlyDelete() {
    const paths = [...state.selected];
    elements.deleteDialog.close();
    elements.deleteSelected.disabled = true;
    elements.deleteSelected.textContent = 'Deleting…';
    showError('');
    try {
      const result = await jsonRequest('/api/disk-manager/delete', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ root: state.root, paths }),
      });
      state.selected.clear();
      if (result.errors.length) {
        showError(`${result.deleted.length} deleted; ${result.errors.length} failed. ${result.errors[0].path}: ${result.errors[0].error}`);
      }
      await scan(state.root);
    } catch (error) {
      showError(error.message);
      updateSelectionControls();
    }
  }

  function bindEvents() {
    elements.chooseDirectory.addEventListener('click', openDirectoryDialog);
    elements.rescan.addEventListener('click', () => scan(state.root));
    elements.directoryGo.addEventListener('click', () => browse(elements.directoryPath.value));
    elements.directoryPath.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') { event.preventDefault(); browse(elements.directoryPath.value); }
    });
    elements.directoryUp.addEventListener('click', () => state.browseParent && browse(state.browseParent));
    elements.directoryList.addEventListener('click', (event) => {
      const button = event.target.closest('.directory-item');
      if (button) browse(button.dataset.path);
    });
    elements.directoryOk.addEventListener('click', (event) => {
      event.preventDefault();
      elements.directoryDialog.close();
      scan(elements.directoryPath.value.trim());
    });
    elements.filter.addEventListener('input', () => {
      state.filter = elements.filter.value;
      render();
    });
    document.querySelectorAll('.sort-button').forEach((button) => button.addEventListener('click', () => {
      if (state.sort === button.dataset.sort) state.direction = state.direction === 'asc' ? 'desc' : 'asc';
      else { state.sort = button.dataset.sort; state.direction = 'desc'; }
      render();
    }));
    elements.rows.addEventListener('change', (event) => {
      if (!event.target.classList.contains('row-check')) return;
      const path = event.target.closest('tr').dataset.path;
      if (event.target.checked) state.selected.add(path); else state.selected.delete(path);
      updateSelectionControls();
    });
    elements.rows.addEventListener('click', (event) => {
      const toggle = event.target.closest('.tree-toggle');
      if (!toggle) return;
      const path = toggle.closest('tr').dataset.path;
      if (state.expanded.has(path)) state.expanded.delete(path); else state.expanded.add(path);
      render();
    });
    elements.selectAll.addEventListener('change', () => {
      treeEntries().forEach(({ entry }) => {
        if (elements.selectAll.checked) state.selected.add(entry.path); else state.selected.delete(entry.path);
      });
      render();
    });
    elements.deleteSelected.addEventListener('click', openDeleteDialog);
    elements.deleteConfirm.addEventListener('input', () => {
      elements.deleteConfirmButton.disabled = elements.deleteConfirm.value !== 'DELETE';
    });
    elements.deleteConfirmButton.addEventListener('click', (event) => {
      event.preventDefault();
      if (elements.deleteConfirm.value === 'DELETE') permanentlyDelete();
    });
    elements.treemap.addEventListener('click', (event) => {
      const tile = event.target.closest('.treemap-tile');
      if (!tile) return;
      elements.filter.value = tile.dataset.path;
      state.filter = tile.dataset.path;
      render();
      elements.rows.closest('.disk-table-panel').scrollIntoView({ behavior: 'smooth' });
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    Object.assign(elements, {
      chooseDirectory: $('#choose-directory'), rescan: $('#rescan'),
      summaryPath: $('#summary-path'), summaryCount: $('#summary-count'), summarySize: $('#summary-size'), summaryStatus: $('#summary-status'),
      scanProgress: $('#scan-progress'), progressTitle: $('#progress-title'), progressPercent: $('#progress-percent'),
      progressEta: $('#progress-eta'), progressBar: $('#progress-bar'), progressItems: $('#progress-items'),
      progressSize: $('#progress-size'), progressRate: $('#progress-rate'), progressCurrent: $('#progress-current'),
      treemapSection: $('#treemap-section'), treemap: $('#treemap'),
      visibleCount: $('#visible-count'), filter: $('#filter'), deleteSelected: $('#delete-selected'),
      errorBanner: $('#error-banner'), selectAll: $('#select-all'), rows: $('#disk-rows'),
      directoryDialog: $('#directory-dialog'), directoryPath: $('#directory-path'), directoryGo: $('#directory-go'),
      directoryUp: $('#directory-up'), directoryList: $('#directory-list'), directoryError: $('#directory-error'), directoryOk: $('#directory-ok'),
      deleteDialog: $('#delete-dialog'), deleteSummary: $('#delete-summary'), deletePreview: $('#delete-preview'),
      deleteConfirm: $('#delete-confirm'), deleteConfirmButton: $('#delete-confirm-button'),
    });
    bindEvents();
    window.addEventListener('beforeunload', () => {
      if (state.scanId) navigator.sendBeacon(`/api/disk-manager/scan/${state.scanId}/cancel`);
    });
    openDirectoryDialog();
  });
})();
