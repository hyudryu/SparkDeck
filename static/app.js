function app() {
  return {
    tab: 'containers',
    state: null,
    connected: false,
    showCreate: false,
    creating: false,

    form: {
      model: '',
      port: null,
      gpu_mem: 0.9,
      extra: '',
    },

    infer: {
      target: '',
      model: '',
      system: '',
      prompt: '',
      temperature: 0.7,
      max_tokens: 512,
    },
    lastSubmitted: '',

    pullForm: { image: 'nvcr.io/nvidia/vllm:26.03.post1-py3', busy: false, lines: [] },

    settingsForm: {},
    settingsSaved: false,

    logsModal: { open: false, name: '', text: '' },

    // ─── lifecycle ───────────────────────────────────────────
    async init() {
      await this.refresh();
      // fast loop for stats
      setInterval(() => this.refreshStats(), 1500);
      // slower full refresh
      setInterval(() => this.refresh(), 2500);
    },

    async refresh() {
      try {
        const r = await fetch('/api/state');
        if (!r.ok) throw new Error(r.statusText);
        const s = await r.json();
        this.state = s;
        this.connected = true;
        // sync settings form on first load
        if (Object.keys(this.settingsForm).length === 0 && s.settings) {
          this.settingsForm = { ...s.settings };
        }
        // default infer model if blank
        if (!this.infer.model && s.containers.length) {
          const r0 = s.containers.find(c => c.status === 'running');
          if (r0?.model) this.infer.model = r0.model;
        }
      } catch (e) {
        this.connected = false;
      }
    },

    async refreshStats() {
      // Lightweight stats refresh between full refreshes
      try {
        const r = await fetch('/api/stats');
        if (!r.ok) return;
        const stats = await r.json();
        if (this.state) this.state.stats = stats;
        this.connected = true;
      } catch {
        this.connected = false;
      }
    },

    // ─── helpers ─────────────────────────────────────────────
    gpu() { return this.state?.stats?.gpus?.[0] || null; },
    pendingCount() {
      return (this.state?.queue || [])
        .filter(j => ['pending', 'dispatching', 'running'].includes(j.status)).length;
    },

    fmtPct(v) {
      if (v == null || isNaN(v)) return '—';
      return `${Number(v).toFixed(0)}%`;
    },
    fmtTemp(v) {
      if (v == null || isNaN(v)) return '—';
      return `${Number(v).toFixed(0)}°C`;
    },
    fmtWatts(d, l) {
      if (d == null || isNaN(d)) return '—';
      const draw = `${Number(d).toFixed(1)} W`;
      if (l != null && !isNaN(l)) return `${draw} / ${Number(l).toFixed(0)} W`;
      return draw;
    },
    fmtBytes(b) {
      if (b == null) return '—';
      const u = ['B', 'KB', 'MB', 'GB', 'TB'];
      let i = 0; let n = Number(b);
      while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
      return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
    },
    relTime(ts) {
      if (!ts) return '';
      const d = Math.floor(Date.now() / 1000 - ts);
      if (d < 5) return 'just now';
      if (d < 60) return `${d}s ago`;
      if (d < 3600) return `${Math.floor(d / 60)}m ago`;
      if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
      return `${Math.floor(d / 86400)}d ago`;
    },

    sortedQueue() {
      const order = { running: 0, dispatching: 1, pending: 2, error: 3, done: 4, canceled: 5 };
      return [...(this.state?.queue || [])].sort((a, b) => {
        const oa = order[a.status] ?? 9, ob = order[b.status] ?? 9;
        if (oa !== ob) return oa - ob;
        return b.requested_at - a.requested_at;
      });
    },

    // ─── containers ──────────────────────────────────────────
    async createContainer() {
      if (!this.form.model) return;
      this.creating = true;
      try {
        const body = {
          model: this.form.model,
          port: this.form.port || null,
          gpu_memory_utilization: this.form.gpu_mem || null,
          extra_args: this.form.extra ? this.form.extra.trim().split(/\s+/) : [],
        };
        const r = await fetch('/api/containers', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        this.showCreate = false;
        this.form = { model: '', port: null, gpu_mem: 0.9, extra: '' };
        await this.refresh();
      } catch (e) {
        alert('Failed to create container: ' + e.message);
      } finally {
        this.creating = false;
      }
    },

    async startContainer(name) {
      await fetch(`/api/containers/${encodeURIComponent(name)}/start`, { method: 'POST' });
      await this.refresh();
    },
    async stopContainer(name) {
      await fetch(`/api/containers/${encodeURIComponent(name)}/stop`, { method: 'POST' });
      await this.refresh();
    },
    async removeContainer(name) {
      if (!confirm(`Remove container "${name}"? This cannot be undone.`)) return;
      const r = await fetch(`/api/containers/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (!r.ok) alert('Remove failed: ' + (await r.text()));
      await this.refresh();
    },

    async openLogs(name) {
      this.logsModal.open = true;
      this.logsModal.name = name;
      this.logsModal.text = 'Loading…';
      await this.refreshLogs();
    },
    async refreshLogs() {
      const name = this.logsModal.name;
      if (!name) return;
      try {
        const r = await fetch(`/api/containers/${encodeURIComponent(name)}/logs?tail=400`);
        const j = await r.json();
        this.logsModal.text = j.logs || '(no logs)';
      } catch (e) {
        this.logsModal.text = 'Failed to fetch logs: ' + e.message;
      }
    },

    // ─── images ──────────────────────────────────────────────
    async pullImage() {
      if (!this.pullForm.image) return;
      this.pullForm.busy = true;
      this.pullForm.lines = [`pulling ${this.pullForm.image}…`];
      try {
        const r = await fetch('/api/images/pull', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ image: this.pullForm.image }),
        });
        if (!r.ok) throw new Error(await r.text());
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const parts = buf.split('\n\n');
          buf = parts.pop();
          for (const p of parts) {
            const line = p.replace(/^data: /, '');
            try {
              const obj = JSON.parse(line);
              if (obj.error) this.pullForm.lines.push(`ERROR: ${obj.error}`);
              else if (obj.status) {
                let msg = obj.status;
                if (obj.id) msg = `${obj.id}: ${msg}`;
                if (obj.progress) msg += ` ${obj.progress}`;
                this.pullForm.lines.push(msg);
              } else if (obj.done) this.pullForm.lines.push('done.');
            } catch {}
          }
        }
        await this.refresh();
      } catch (e) {
        this.pullForm.lines.push(`error: ${e.message}`);
      } finally {
        this.pullForm.busy = false;
      }
    },

    async removeImage(id) {
      if (!confirm(`Remove image ${id}?`)) return;
      const r = await fetch(`/api/images/${encodeURIComponent(id)}`, { method: 'DELETE' });
      if (!r.ok) alert('Remove failed: ' + (await r.text()));
      await this.refresh();
    },

    // ─── inference ───────────────────────────────────────────
    async sendInference() {
      const messages = [];
      if (this.infer.system) messages.push({ role: 'system', content: this.infer.system });
      messages.push({ role: 'user', content: this.infer.prompt });

      const body = {
        model: this.infer.model,
        messages,
        params: {
          temperature: this.infer.temperature,
          max_tokens: this.infer.max_tokens,
        },
      };
      if (this.infer.target.startsWith('container:')) {
        body.container = this.infer.target.slice('container:'.length);
      }

      try {
        const r = await fetch('/api/inference', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        const j = await r.json();
        this.lastSubmitted = j.id;
        this.tab = 'queue';
        await this.refresh();
      } catch (e) {
        alert('Submit failed: ' + e.message);
      }
    },

    // ─── queue ───────────────────────────────────────────────
    async forceRun(id) {
      await fetch(`/api/queue/${id}/run`, { method: 'POST' });
      await this.refresh();
    },
    async cancelJob(id) {
      await fetch(`/api/queue/${id}`, { method: 'DELETE' });
      await this.refresh();
    },
    async clearFinished() {
      await fetch('/api/queue/clear', { method: 'POST' });
      await this.refresh();
    },

    // ─── settings ────────────────────────────────────────────
    async saveSettings() {
      try {
        const r = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(this.settingsForm),
        });
        if (!r.ok) throw new Error(await r.text());
        this.settingsSaved = true;
        setTimeout(() => { this.settingsSaved = false; }, 2500);
        await this.refresh();
      } catch (e) {
        alert('Save failed: ' + e.message);
      }
    },
  };
}
