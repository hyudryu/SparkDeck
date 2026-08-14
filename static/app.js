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
      engine: 'vllm', // 'vllm' | 'sglang'
      deployment_mode: 'single', // 'single' | 'sharded' | 'replicated'
      node_ids: ['local'],
      gpu_mem_mode: 'fraction', // 'fraction' | 'gb'
      gpu_mem: 0.9,
      gpu_mem_gb: 8,
      extra: '',
      image: '',
      pasteCmd: '',
      // SGLang-specific fields
      sg_tp_size: 1,
      sg_context_length: 32768,
      sg_max_running_requests: null,
      sg_mem_fraction: 0.92,
      sg_image: '',
    },
    recipeSaved: false,
    recipeLaunching: {},
    containerStopping: {},
    dockerSettingsOpen: {},
    dockerSettingsSaved: {},
    dockerSettingsSaving: {},
    dockerSettingsForm: {},
    clusterForm: {
      name: '', agent_url: '', pairing_code: '', fabric_ip: '', fabric_interface: '',
    },
    clusterPairing: false,
    clusterBusy: {},
    deploymentSettingsOpen: {},
    deploymentSettingsSaved: {},
    deploymentSettingsSaving: {},
    deploymentSettingsForm: {},
    deploymentSettingsBaseline: {},
    deploymentPricingSaving: {},
    deploymentPricingSaved: {},
    deploymentPricingBaseline: {},
    deploymentIdCopied: {},
    KV_CACHE_DTYPE_OPTIONS: [
      'auto', 'bfloat16', 'float16', 'fp8', 'fp8_ds_mla', 'fp8_e4m3',
      'fp8_e5m2', 'fp8_inc', 'fp8_per_token_head', 'int4_per_token_head',
      'int8_per_token_head', 'nvfp4', 'nvfp4_ds_mla', 'turboquant_3bit_nc',
      'turboquant_4bit_nc', 'turboquant_k3v4_nc', 'turboquant_k8v4',
    ],
    aliasEditorOpen: {},
    aliasEditorValue: {},
    aliasEditorSaving: {},
    aliasEditorSaved: {},

    chat: {
      model: '',
      system: '',
      input: '',
      temperature: 0.3,
      max_tokens: 32000,
      busy: false,
      abort: null, // AbortController for the in-flight stream
      messages: [], // assistant messages also track output/thinking/total rates
    },

    pullForm: { image: 'nvcr.io/nvidia/vllm:26.03.post1-py3', busy: false, lines: [] },

    ollamaForm: { base_url: '' },
    ollamaPull: { name: '', busy: false, lines: [] },
    ollamaSelected: '',

    // Llama Server (GGUF) per-model state (keyed by model id).
    unslothBusy: {},            // model_id -> bool (launch in flight)
    unslothStopping: {},        // model_id -> bool (stop in flight)
    unslothSettingsOpen: {},    // model_id -> bool (drawer open)
    unslothSettingsSaved: {},   // model_id -> bool (transient "Saved." tick)
    unslothForm: {},            // model_id -> working copy of launch settings
    unslothVariants: {},        // model_id -> downloaded GGUF quant variants
    unslothVariantsBusy: {},    // model_id -> bool (variant fetch in flight)
    UNSLOTH_KV_OPTIONS: ['bf16', 'f16', 'f32', 'q8_0', 'q5_1', 'q5_0', 'q4_1', 'q4_0', 'iq4_nl'],

    // Saved SparkRun targets. The reference remains the source of truth;
    // SparkRun resolves its current model/configuration when it is saved.
    sparkEditor: { open: false, id: null },
    sparkForm: {
      reference: '',
    },
    sparkSaving: false,
    sparkSaveError: '',
    sparkSaved: false,
    sparkYamlError: '',
    sparkRun: {
      open: false, recipeId: null, recipeName: '', lines: [],
      status: '', busy: false, runId: null, abort: null,
    },
    sparkRunHistory: {
      open: false, runId: null, recipeName: '', lines: [], status: '', autoScroll: true,
    },
    sparkLaunchRun: {
      open: false, id: null, name: '', reference: '', optionsText: '',
      resolvedCommand: '', recipeDefaults: [], recipeEnv: [], recipeMods: [],
      overrides: { parallel_streams: null, tensor_parallel: null, gpu_memory_utilization: null,
                   max_model_len: null, port: null, served_model_name: '' },
    },
    sparkRecipeRuns: {
      open: false, recipeId: null, recipeName: '', runs: [],
    },
    sparkDownload: {
      open: false, recipe: null, copied: false,
    },

    ab: {
      prompt: '',
      system: '',
      temperature: 0.3,
      top_p: 0.1,
      seed: null,
      max_tokens: 32000,
      modelA: '',
      modelB: '',
      busy: false,
      warning: '',
      panelA: { text: '', reasoning: '', error: '', status: '', tokens: 0, elapsedMs: 0, streamMs: 0, busy: false, abort: null },
      panelB: { text: '', reasoning: '', error: '', status: '', tokens: 0, elapsedMs: 0, streamMs: 0, busy: false, abort: null },
    },

    settingsForm: {},
    settingsSaved: false,

    // Usage tab sub-navigation.
    usageSubTab: 'cost',           // 'cost' | 'analysis'
    usageSortKey: 'input',         // model | input | cached | output | requests | speed | time | cost
    usageSortDirection: 'desc',
    usageAliasEditing: {},
    usageAliasValue: {},
    usageMergeValue: {},
    usageAliasSaving: {},
    usageAliasSaved: {},
    analysisDateStart: '',         // YYYY-MM-DD or '' for default range
    analysisDateEnd: '',           // YYYY-MM-DD or '' for default range
    analysisChartMode: 'hour',     // 'hour' | 'day'
    analysisHourly: [],            // fetched hourly data
    analysisDaily: [],             // fetched daily data
    analysisLoading: false,

    logsModal: {
      open: false, name: '', text: '', deploymentId: null, members: [], autoScroll: true,
      llama: false, llamaModel: null, logId: null, logOffset: 0,
    },
    _logsTailInterval: null,
    _logsRefreshInFlight: false,
    _sparkHistoryTailInterval: null,
    _sparkHistoryRefreshInFlight: false,

    serverLogs: { lines: [], autoScroll: true, _interval: null, _loading: false },

    disk: null,
    diskLoading: false,
    diskLastAt: null,
    temperatureHistoryByNode: {},
    liveRequestRates: {},
    fanMaxSpeed: false,
    topbarStatsCollapsed: false,
    statsNodeId: 'local',
    fanSettingsOpen: false,
    fanSettingsLiveMode: '',
    fanSettingsExpectedMode: '',
    fanSettingsMode: '',
    fanSettingsDraft: {},
    fanSettingsDrafts: {},
    fanSettingsLoading: false,
    fanSettingsSaving: false,
    fanSettingsSaved: false,
    fanSettingsError: '',

    // Lifetime token stats (persisted server-side, cleared only via reset).
    tokenModel: '',              // model selected in the topbar token card
    _tokenCostCache: {},         // cached token cost per model (30s TTL)

    // Flagship pricing (persisted in settings, edited from Usage tab).
    flagshipPricing: {},
    flagshipPricingDirty: false, // true while there are unsaved local edits
    pricingSaving: false,
    pricingSaved: false,

    // ─── lifecycle ───────────────────────────────────────────
    async init() {
      // On mobile, start stats collapsed so the topbar doesn't eat the whole screen.
      this.topbarStatsCollapsed = window.innerWidth <= 720;
      await this.refresh();
      await this.refreshDisk();
      await this.refreshTemperatureHistory();
      await this.refreshLiveRequestRates();
      // 1 Hz stats polling, paused when the tab isn't visible so we don't
      // wake the GPU compositor for invisible repaints.
      setInterval(() => {
        if (!document.hidden) this.refreshStats();
      }, 1000);
      // Token rates are deliberately sampled at a stable two-second cadence;
      // SSE chunk frequency must not control how often the widget repaints.
      setInterval(() => {
        if (!document.hidden) this.refreshLiveRequestRates();
      }, 2 * 1000);
      setInterval(() => {
        if (!document.hidden) this.refresh();
      }, 2500);
      // Disk space changes slowly — poll it every 5 minutes instead.
      setInterval(() => {
        if (!document.hidden) this.refreshDisk();
      }, 5 * 60 * 1000);
      // Temperature history changes only once per 30-second server sample.
      setInterval(() => {
        if (!document.hidden) this.refreshTemperatureHistory();
      }, 30 * 1000);
      // Refresh immediately on tab regaining focus.
      document.addEventListener('visibilitychange', () => {
        if (!document.hidden) {
          this.refreshStats();
          this.refresh();
          this.refreshTemperatureHistory();
          this.refreshLiveRequestRates();
          if (!this.diskLastAt || Date.now() - this.diskLastAt > 5 * 60 * 1000) {
            this.refreshDisk();
          }
        }
      });
      // Start/stop server log polling when on the Logs tab
      this.$watch('tab', (v) => {
        if (v === 'logs') {
          this._startLogPolling();
        } else {
          this._stopLogPolling();
        }
      });
      this.$watch('logsModal.open', (open) => {
        if (open) this._startLogsTail();
        else this._stopLogsTail();
      });
      this.$watch('sparkRunHistory.open', (open) => {
        if (open) this._startSparkHistoryTail();
        else this._stopSparkHistoryTail();
      });
      this.$watch('statsNodeId', () => this.refreshTemperatureHistory());
      // Load analysis data when switching to the Analysis sub-tab
      this.$watch('usageSubTab', (v) => {
        if (v === 'analysis') this.loadAnalysisData();
      });
    },

    async refresh() {
      try {
        const r = await fetch(`/api/state?t=${Date.now()}`);
        if (!r.ok) throw new Error(r.statusText);
        const s = await r.json();
        this.state = s;
        if (!(s.nodes || []).some(n => n.id === this.statsNodeId && n.online)) {
          this.statsNodeId = (s.nodes || []).find(n => n.local && n.online)?.id
            || (s.nodes || []).find(n => n.online)?.id
            || 'local';
        }
        this._syncFanSettings(s.stats?.fan);
        this._tokenCostCache = {}; // invalidate cost cache on refresh
        this.connected = true;
        // sync settings form on first load
        if (Object.keys(this.settingsForm).length === 0 && s.settings) {
          this.settingsForm = { ...s.settings };
        }
        // sync flagship pricing from settings
        this._syncFlagshipPricing(s.settings);
        // default chat model if blank
        if (!this.chat.model && s.containers.length) {
          const r0 = s.containers.find(c => c.status === 'running');
          if (r0?.model) this.chat.model = r0.model;
        }
        // Seed (and keep in sync) the per-model Unsloth settings forms.
        // A draft only exists once the user opens a drawer; until then we
        // mirror the latest saved values so the form isn't stale.
        for (const m of (s.unsloth?.models || [])) {
          if (!m?.id) continue;
          if (!this.unslothForm[m.id] || !this.unslothSettingsOpen[m.id]) {
            // Closed drawer (or never opened) — keep the form aligned to saved values.
            this.unslothForm[m.id] = { ...this.unslothSettingsFor(m.id) };
          }
          // Ensure a reactive busy flag exists for every model so the
          // Launch/Stop buttons update correctly.
          if (this.unslothBusy[m.id] == null) {
            this.unslothBusy[m.id] = false;
          }
          if (this.unslothStopping[m.id] == null) {
            this.unslothStopping[m.id] = false;
          }
        }
        // Keep closed Docker load-settings drawers aligned with the command
        // Docker is actually storing. Open drawers retain the user's draft.
        for (const c of (s.containers || [])) {
          if (!c?.name || !c.load_settings) continue;
          // Alpine's boolean-attribute binding is most reliable when dynamic
          // object keys are initialized explicitly instead of left undefined.
          if (this.containerStopping[c.name] == null) {
            this.containerStopping[c.name] = false;
          }
          if (this.dockerSettingsSaving[c.name] == null) {
            this.dockerSettingsSaving[c.name] = false;
          }
          if (!this.dockerSettingsForm[c.name] || !this.dockerSettingsOpen[c.name]) {
            this.dockerSettingsForm[c.name] = { ...c.load_settings };
          }
        }
        // Closed cluster editors track the saved backend values. Open
        // editors retain their local draft while state polling continues.
        for (const deployment of (s.deployments || [])) {
          if (!deployment?.id) continue;
          if (this.deploymentSettingsSaving[deployment.id] == null) {
            this.deploymentSettingsSaving[deployment.id] = false;
          }
          if (!this.deploymentSettingsForm[deployment.id] || !this.deploymentSettingsOpen[deployment.id]) {
            const draft = this.deploymentSettingsDraft(deployment);
            this.deploymentSettingsForm[deployment.id] = draft;
            this.deploymentSettingsBaseline[deployment.id] = this.deploymentSettingsSnapshot(
              deployment, draft
            );
            this.deploymentPricingBaseline[deployment.id] = this.deploymentPricingSnapshot(
              deployment, draft
            );
          }
          if (this.deploymentPricingSaving[deployment.id] == null) {
            this.deploymentPricingSaving[deployment.id] = false;
          }
        }
        // Dynamic Alpine object keys must be initialized explicitly. Leaving
        // a per-model saving flag undefined can make the boolean disabled
        // binding sticky in some browsers after a state refresh.
        for (const model of Object.keys(s.token_stats || {})) {
          if (this.usageAliasSaving[model] == null) this.usageAliasSaving[model] = false;
          if (this.usageAliasEditing[model] == null) this.usageAliasEditing[model] = false;
          if (this.usageAliasSaved[model] == null) this.usageAliasSaved[model] = false;
        }
        // Auto-switch the token card to the newly loaded model.
        this._syncTokenModel();
      } catch (e) {
        this.connected = false;
      }
    },

    async refreshStats() {
      // Lightweight stats refresh between full refreshes
      try {
        // Bust caches so browser doesn't serve a stale stats snapshot.
        const r = await fetch(`/api/stats?t=${Date.now()}`);
        if (!r.ok) return;
        const stats = await r.json();
        if (this.state) this.state.stats = stats;
        // Sync the max-speed toggle with the daemon's reported state.
        if (stats.fan && typeof stats.fan.max_speed === 'boolean') {
          this.fanMaxSpeed = stats.fan.max_speed;
        }
        this._syncFanSettings(stats.fan);
        this.connected = true;
        return stats;
      } catch {
        this.connected = false;
      }
    },

    async refreshDisk() {
      try {
        this.diskLoading = true;
        const r = await fetch(`/api/disk?t=${Date.now()}`);
        if (!r.ok) return;
        this.disk = await r.json();
        this.diskLastAt = Date.now();
      } catch {
        // Leave existing value in place; the main stats poller handles connectivity.
      } finally {
        this.diskLoading = false;
      }
    },

    async toggleFanMaxSpeed() {
      try {
        const r = await fetch('/api/fan/max-speed', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ enabled: this.fanMaxSpeed }),
        });
        if (!r.ok) throw new Error(r.statusText);
        const j = await r.json();
        this.fanMaxSpeed = Boolean(j.enabled);
      } catch (e) {
        console.error('fan max-speed toggle failed:', e);
        this.fanMaxSpeed = !this.fanMaxSpeed;
      }
    },

    async refreshTemperatureHistory() {
      const nodeId = this.selectedStatsNode()?.id || this.statsNodeId || 'local';
      try {
        const r = await fetch(`/api/temperature-history?node_id=${encodeURIComponent(nodeId)}&t=${Date.now()}`);
        if (!r.ok) return;
        const history = await r.json();
        this.temperatureHistoryByNode = {
          ...this.temperatureHistoryByNode,
          [nodeId]: history,
        };
      } catch {
        // Keep the last successful two-hour window visible during a brief
        // node or network interruption.
      }
    },

    async refreshLiveRequestRates() {
      try {
        const r = await fetch(`/api/active-request-rates?t=${Date.now()}`);
        if (!r.ok) return;
        this.liveRequestRates = await r.json();
      } catch {
        // Retain the last sample until the next fixed-cadence poll.
      }
    },

    openDiskManager() {
      window.open('/disk-manager', '_blank', 'noopener');
    },

    _syncFanSettings(fan, force = false) {
      if (!fan || !fan.mode) return;
      this.fanSettingsLiveMode = fan.mode;
      if (!force && this.fanSettingsOpen) return;
      const active = fan.active_settings && typeof fan.active_settings === 'object'
        ? JSON.parse(JSON.stringify(fan.active_settings)) : {};
      this.fanSettingsExpectedMode = fan.mode;
      this.fanSettingsMode = fan.mode;
      this.fanSettingsDrafts[fan.mode] = active;
      this.fanSettingsDraft = this.fanSettingsDrafts[fan.mode];
    },

    async toggleFanSettings() {
      this.fanSettingsOpen = !this.fanSettingsOpen;
      this.fanSettingsError = '';
      this.fanSettingsSaved = false;
      if (this.fanSettingsOpen) {
        this._syncFanSettings(this.state?.stats?.fan, true);
        await this.loadFanSettings();
      }
    },

    async loadFanSettings() {
      this.fanSettingsLoading = true;
      try {
        const r = await fetch(`/api/fan/settings?t=${Date.now()}`);
        const reply = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(reply.detail || r.statusText);
        this.fanSettingsLiveMode = reply.mode;
        this.fanSettingsExpectedMode = reply.mode;
        this.fanSettingsMode = reply.mode;
        this.fanSettingsDrafts = JSON.parse(JSON.stringify(reply.settings || {}));
        this.fanSettingsDraft = this.fanSettingsDrafts[this.fanSettingsMode] || {};
      } catch (e) {
        this.fanSettingsError = e.message || String(e);
      } finally {
        this.fanSettingsLoading = false;
      }
    },

    selectFanSettingsMode(mode) {
      if (!this.fanSettingsDrafts[mode] || this.fanSettingsSaving) return;
      this.fanSettingsMode = mode;
      this.fanSettingsDraft = this.fanSettingsDrafts[mode];
      this.fanSettingsError = '';
      this.fanSettingsSaved = false;
    },

    addFanCurvePoint() {
      const points = this.fanSettingsDraft.curve_points;
      if (!Array.isArray(points) || points.length < 2) return;
      const right = points[points.length - 1];
      const left = points[points.length - 2];
      const temp = (Number(left[0]) + Number(right[0])) / 2;
      const duty = (Number(left[1]) + Number(right[1])) / 2;
      if (!Number.isFinite(temp) || temp <= Number(left[0]) || temp >= Number(right[0])) {
        this.fanSettingsError = 'Make room between the final two temperatures before adding a point.';
        return;
      }
      points.splice(points.length - 1, 0, [temp, duty]);
      this.fanSettingsError = '';
    },

    removeFanCurvePoint(index) {
      const points = this.fanSettingsDraft.curve_points;
      if (Array.isArray(points) && points.length > 2) points.splice(index, 1);
    },

    resetFanCurve() {
      this.fanSettingsDraft.curve_points = [
        [40, 0], [60, 30], [75, 60], [90, 100],
      ];
      this.fanSettingsError = '';
    },

    fanCurveX(temp) {
      const lo = Number(this.fanSettingsDraft.curve_min_temp ?? 30);
      const hi = Number(this.fanSettingsDraft.curve_max_temp ?? 100);
      const span = Math.max(1, hi - lo);
      return 30 + Math.max(0, Math.min(1, (Number(temp) - lo) / span)) * 295;
    },

    fanCurveY(duty) {
      return 135 - Math.max(0, Math.min(100, Number(duty))) * 1.2;
    },

    fanCurvePolyline() {
      return (this.fanSettingsDraft.curve_points || [])
        .map(point => `${this.fanCurveX(point[0])},${this.fanCurveY(point[1])}`)
        .join(' ');
    },

    async saveFanSettings() {
      this.fanSettingsSaving = true;
      this.fanSettingsSaved = false;
      this.fanSettingsError = '';
      try {
        const r = await fetch('/api/fan/settings', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({
            mode: this.fanSettingsMode,
            expected_mode: this.fanSettingsExpectedMode,
            active_settings: this.fanSettingsDraft,
          }),
        });
        const reply = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(reply.detail || r.statusText);

        // FanController normally reloads within one second. Keep the save in
        // progress until its broadcast confirms the values it actually loaded.
        let applied = false;
        for (let attempt = 0; attempt < 20; attempt++) {
          await new Promise(resolve => setTimeout(resolve, 300));
          const stats = await this.refreshStats();
          const fan = stats?.fan;
          if (fan?.mode !== reply.mode) {
            continue;
          }
          if (JSON.stringify(fan.active_settings) === JSON.stringify(reply.active_settings)) {
            this._syncFanSettings(fan, true);
            await this.loadFanSettings();
            applied = true;
            break;
          }
        }
        if (!applied) {
          throw new Error('Settings were written, but FanController has not confirmed them yet.');
        }
        this.fanSettingsSaved = true;
        setTimeout(() => { this.fanSettingsSaved = false; }, 2500);
      } catch (e) {
        this.fanSettingsError = e.message || String(e);
      } finally {
        this.fanSettingsSaving = false;
      }
    },

    // ─── helpers ─────────────────────────────────────────────
    renderMarkdown(text) {
      if (!text) return '';
      try {
        const html = window.marked
          ? window.marked.parse(text, { breaks: true, gfm: true })
          : text;
        return window.DOMPurify ? window.DOMPurify.sanitize(html) : html;
      } catch {
        return this.escapeHtml(text);
      }
    },

    escapeHtml(s) {
      return String(s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
    },

    statsNodes() { return this.state?.nodes || []; },
    selectedStatsNode() {
      return this.statsNodes().find(n => n.id === this.statsNodeId && n.online)
        || this.statsNodes().find(n => n.local && n.online)
        || this.statsNodes().find(n => n.online)
        || null;
    },
    nodeStats() {
      const node = this.selectedStatsNode();
      return node?.local ? (this.state?.stats || {}) : (node?.stats || {});
    },
    nodeDisk() {
      const node = this.selectedStatsNode();
      return node?.local ? this.disk : (node?.disk || null);
    },
    statsGpu() { return this.nodeStats()?.gpus?.[0] || null; },
    temperatureHistory() {
      const nodeId = this.selectedStatsNode()?.id || this.statsNodeId || 'local';
      return this.temperatureHistoryByNode[nodeId] || {
        window_seconds: 2 * 60 * 60,
        samples: [],
      };
    },
    temperatureChart(kind) {
      const history = this.temperatureHistory();
      const key = kind === 'gpu' ? 'gpu_temp_c' : 'cpu_temp_c';
      const samples = Array.isArray(history.samples) ? history.samples : [];
      const values = samples
        .map(sample => sample?.[key])
        .filter(value => typeof value === 'number' && Number.isFinite(value));
      const latest = values.length ? values[values.length - 1] : null;
      const windowSeconds = Number(history.window_seconds) || 2 * 60 * 60;
      const now = Date.now() / 1000;
      const windowStart = now - windowSeconds;
      const left = 38;
      const right = 354;
      const top = 10;
      const bottom = 94;

      let minTemp = values.length ? Math.min(...values) : 30;
      let maxTemp = values.length ? Math.max(...values) : 90;
      minTemp = Math.max(0, Math.floor((minTemp - 5) / 10) * 10);
      maxTemp = Math.min(120, Math.ceil((maxTemp + 5) / 10) * 10);
      if (maxTemp - minTemp < 20) {
        const middle = (maxTemp + minTemp) / 2;
        minTemp = Math.max(0, Math.floor((middle - 10) / 10) * 10);
        maxTemp = minTemp + 20;
        if (maxTemp > 120) {
          maxTemp = 120;
          minTemp = 100;
        }
      }

      const x = ts => left + Math.max(0, Math.min(1, (ts - windowStart) / windowSeconds)) * (right - left);
      const y = value => bottom - ((value - minTemp) / (maxTemp - minTemp)) * (bottom - top);
      let path = '';
      let drawing = false;
      let previousTs = null;
      const expectedInterval = Number(history.sample_interval_seconds) || 30;
      let lastPoint = null;
      for (const sample of samples) {
        const ts = Number(sample?.ts);
        const rawValue = sample?.[key];
        const value = typeof rawValue === 'number' ? rawValue : NaN;
        if (!Number.isFinite(ts) || !Number.isFinite(value) || ts < windowStart) {
          drawing = false;
          previousTs = null;
          continue;
        }
        const px = x(ts);
        const py = y(value);
        const gap = previousTs != null && ts - previousTs > expectedInterval * 2.5;
        path += `${!drawing || gap ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)} `;
        drawing = true;
        previousTs = ts;
        lastPoint = { x: px, y: py, value };
      }

      const middleTemp = (minTemp + maxTemp) / 2;
      return {
        path: path.trim(),
        latest,
        lastPoint,
        ticks: [maxTemp, middleTemp, minTemp].map(value => ({
          value,
          y: y(value),
          label: `${Math.round(value)}°`,
        })),
      };
    },
    gpu() { return this.state?.stats?.gpus?.[0] || null; },
    gpuTotalGb() {
      const totalMib = this.state?.stats?.gpus?.[0]?.mem_total_mib;
      return totalMib ? Math.floor(totalMib / 1024) : 24;
    },
    gpuVramPct() {
      const g = this.gpu();
      if (!g || !g.mem_total_mib) return 0;
      return Math.round((g.mem_used_mib / g.mem_total_mib) * 100);
    },
    gpuVramColor() {
      const pct = this.gpuVramPct();
      if (pct < 70) return 'var(--accent-blue)';
      if (pct < 90) return 'var(--accent-amber)';
      return 'var(--accent-red)';
    },
    estimateVram(model, extraArgs, cacheTypeKv) {
      // Estimate VRAM from model name + quant settings, mirroring the
      // server-side _estimate_params_and_quant. Returns GB string or null.
      if (!model) return null;
      // Parse param count from model name (e.g. "7B", "72B", "3.5B").
      const pm = model.match(/(\d+(?:\.\d+)?)\s*b\b/i);
      if (!pm) return null;
      const params = parseFloat(pm[1]);
      // Determine bytes/param from quant settings.
      let bpp = 2.0;  // bf16 default
      if (extraArgs) {
        const joined = extraArgs.join(' ');
        if (/--quantization\s+(?:awq|gptq|sqqp)/i.test(joined)) bpp = 0.5;
        else if (/--quantization\s+(?:fp8|e4m3)/i.test(joined)) bpp = 1.0;
        else if (/--dtype\s+(?:int8)/i.test(joined)) bpp = 1.0;
        else if (/--dtype\s+(?:float8|fp8)/i.test(joined)) bpp = 1.0;
      }
      // GGUF quant variant from the model id or filename (e.g. "Q4_0", "q8_0").
      const qm = model.match(/\bQ(\d+)\w*\b/i);
      if (qm) {
        const q = parseInt(qm[1], 10);
        const qBpp = q <= 3 ? 0.375 : q <= 4 ? 0.5 : q <= 5 ? 0.625 : q <= 6 ? 0.75 : q <= 8 ? 1.0 : 2.0;
        bpp = Math.min(bpp, qBpp);
      }
      // KV cache type also signals precision (for unsloth/llama.cpp).
      if (cacheTypeKv) {
        const ck = String(cacheTypeKv).toLowerCase();
        if (ck.startsWith('q4')) bpp = Math.min(bpp, 0.5);
        else if (ck.startsWith('q5')) bpp = Math.min(bpp, 0.625);
        else if (ck.startsWith('q6')) bpp = Math.min(bpp, 0.75);
        else if (ck.startsWith('q8')) bpp = Math.min(bpp, 1.0);
      }
      // 1.2 overhead for KV cache + activation memory.
      return (params * 1e9 * bpp * 1.2 / (1024 ** 3)).toFixed(1);
    },

    maxConcurrentLocalModels() {
      return this.state?.settings?.max_concurrent_models || 1;
    },
    pendingCount() {
      return (this.state?.queue || [])
        .filter(j => ['pending', 'dispatching', 'running'].includes(j.status)).length;
    },

    // ─── lifetime token stats ──────────────────────────────
    // Token-stats key of the model currently holding the GPU (includes the
    // quant/dtype variant tag, e.g. "Qwen/Qwen3-8B [awq]"). Computed
    // server-side; falls back to local state when absent.
    loadedModel() {
      const s = this.state;
      if (!s) return null;
      if (s.summary?.loaded_stats_key) return s.summary.loaded_stats_key;
      const unsloth = s.summary?.unsloth_loaded;
      if (unsloth) return unsloth;
      const running = (s.containers || []).find(c => c.status === 'running' && c.model);
      return (running?.stats_key || running?.model) || null;
    },
    // Always sync the token card to the currently loaded model so it
    // auto-selects on page load and whenever the GPU model switches.
    _syncTokenModel() {
      const lm = this.loadedModel();
      if (lm) this.tokenModel = lm;
    },
    tokenModelOptions() {
      const models = Object.keys(this.state?.session_token_stats || {});
      const lm = this.loadedModel();
      const others = models.filter(m => m !== lm).sort();
      // The model holding the GPU is always first, even before it has
      // completed a request and therefore has no persisted counters yet.
      return lm ? [lm, ...others] : others;
    },
    tokenModelLabel(key) {
      // Stats are keyed by the active quant (e.g. "model [Q8_0]") so runs
      // do not get mixed. Display the actual loaded llama-server model name
      // instead of exposing that implementation key in the selector.
      const loadedLlama = this.state?.unsloth?.loaded_model;
      return loadedLlama && key === this.loadedModel() ? loadedLlama : key;
    },
    tokenStatsFor(model) {
      if (!model) return null;
      // The topbar widget shows session-scoped counters so the reset
      // button only clears the current session, not the lifetime stats
      // (which live in state.token_stats and are shown in the Usage tab).
      return (this.state?.session_token_stats || {})[model] || null;
    },
    fmtTokens(n) {
      if (n == null || isNaN(n)) return '—';
      const v = Number(n);
      if (v < 1000) return String(v);
      if (v < 1_000_000) return `${(v / 1000).toFixed(v < 10_000 ? 1 : 0)}k`;
      if (v >= 1_000_000_000) return `${(v / 1_000_000_000).toFixed(v < 10_000_000_000 ? 1 : 0)}B`;
      return `${(v / 1_000_000).toFixed(v < 10_000_000 ? 1 : 0)}M`;
    },
    fmtDuration(seconds) {
      const value = Number(seconds);
      if (!isFinite(value) || value <= 0) return '—';
      if (value < 60) return `${value.toFixed(1)} s`;
      if (value < 3600) return `${(value / 60).toFixed(1)} m`;
      if (value < 86400) return `${(value / 3600).toFixed(1)} h`;
      return `${(value / 86400).toFixed(1)} d`;
    },
    // Format a cost value as USD (e.g. "$1.23" or "$0.00").
    fmtCost(v) {
      if (v == null || isNaN(v)) return '';
      return `$${Number(v).toFixed(2)}`;
    },
    // Token cost display for the topbar widget (session-scoped).
    async tokenCostDisplay(model) {
      if (!model) return '';
      const cost = this._tokenCostCache?.[model];
      if (cost && cost._fetchedAt && Date.now() - cost._fetchedAt < 30000) {
        return cost.total_cost > 0 ? this.fmtCost(cost.total_cost) : '';
      }
      try {
        const r = await fetch(`/api/token-cost/${encodeURIComponent(model)}?session=true`);
        if (!r.ok) return '';
        const cost = await r.json();
        if (!this._tokenCostCache) this._tokenCostCache = {};
        cost._fetchedAt = Date.now();
        this._tokenCostCache[model] = cost;
        return cost.total_cost > 0 ? this.fmtCost(cost.total_cost) : '';
      } catch {
        return '';
      }
    },
    // Pricing display for unsloth model cards.
    unslothPricingDisplay(id) {
      const s = this.unslothSettingsFor(id);
      const inp = s?.input_cost_per_1m;
      const out = s?.output_cost_per_1m;
      const cache = s?.cache_cost_per_1m;
      if (inp || out || cache) {
        const i = inp ? `$${Number(inp).toFixed(2)}` : '—';
        const o = out ? `$${Number(out).toFixed(2)}` : '—';
        const c = cache ? `$${Number(cache).toFixed(2)}` : '—';
        return `↓ ${i} · ↻ ${c} · ↑ ${o} / 1M`;
      }
      return '—';
    },
    // Average decode speed (output tokens / generation time, prefill excluded).
    avgSpeed(model) {
      const r = this.tokenStatsFor(model);
      if (!r || !r.gen_time_s) return null;
      const v = r.gen_tokens / r.gen_time_s;
      return isFinite(v) ? v : null;
    },
    // Live in-flight stream stats for a model, fed by the fixed 2-second poll:
    // {connections, tok_s} where tok/s is the aggregate across all active
    // streams of that model. Null when nothing is streaming right now.
    liveReqs(model) {
      if (!model) return null;
      const a = this.liveRequestRates?.[model];
      return a && a.connections > 0 ? a : null;
    },
    // While streams are active this is the live aggregate rate; otherwise
    // the lifetime average.
    fmtSpeed(model) {
      const live = this.liveReqs(model);
      if (live) return `${live.output_tok_s.toFixed(1)} tok/s`;
      const v = this.avgSpeed(model);
      return v != null ? `${v.toFixed(1)} tok/s` : '—';
    },
    fmtMessageRate(tokens, elapsedMs) {
      if (!tokens || !elapsedMs) return '—';
      return `${(tokens / (elapsedMs / 1000)).toFixed(1)} tok/s`;
    },
    async resetTokenStats() {
      if (!confirm('Reset lifetime token counters for all models?')) return;
      try {
        const r = await fetch('/api/token-stats/reset', { method: 'POST' });
        if (!r.ok) throw new Error(r.statusText);
        this._tokenCostCache = {};
        await this.refresh();
      } catch (e) {
        alert('Reset failed: ' + e.message);
      }
    },
    async resetSessionTokenStats() {
      // Only resets the current session's counters (shown in the topbar).
      // The lifetime stats in the Usage tab are untouched.
      try {
        const r = await fetch('/api/token-stats/session-reset', { method: 'POST' });
        if (!r.ok) throw new Error(r.statusText);
        this._tokenCostCache = {};
        await this.refresh();
      } catch (e) {
        alert('Reset failed: ' + e.message);
      }
    },

    // ─── Usage → Analysis charts ────────────────────────────
    async loadAnalysisData() {
      this.analysisLoading = true;
      this.analysisHourly = [];
      this.analysisDaily = [];
      const params = new URLSearchParams();
      if (this.analysisDateStart) params.set('start', this.analysisDateStart);
      if (this.analysisDateEnd) params.set('end', this.analysisDateEnd);
      const qs = params.toString();
      try {
        const [dh, dd] = await Promise.all([
          fetch(`/api/token-stats/hourly${qs ? '?' + qs : ''}`),
          fetch(`/api/token-stats/daily${qs ? '?' + qs : ''}`),
        ]);
        if (dh.ok) this.analysisHourly = await dh.json();
        if (dd.ok) this.analysisDaily = await dd.json();
      } catch (e) {
        // silently ignore — the empty-state message will show
      } finally {
        this.analysisLoading = false;
      }
    },
    // Build a YYYY-MM-DD string for a Date object.
    _fmtDate(d) {
      return d.toISOString().slice(0, 10);
    },
    // GitHub-style activity grid: 7 rows (Sun–Sat) × N week-columns.
    renderActivityGrid() {
      const daily = this.analysisDaily || [];
      if (!daily.length) return '<div class="empty">No data for this period.</div>';
      // Map date → data
      const byDate = {};
      let maxTotal = 0;
      for (const d of daily) {
        byDate[d.date] = d;
        const t = (d.input || 0) + (d.output || 0);
        if (t > maxTotal) maxTotal = t;
      }
      // Determine the date range: use the data's range, padded to full weeks.
      const dates = daily.map(d => d.date).sort();
      let start = new Date(dates[0]);
      let end = new Date(dates[dates.length - 1]);
      // Pad start back to Sunday
      start = new Date(start);
      start.setDate(start.getDate() - start.getDay());
      // Pad end forward to Saturday
      end = new Date(end);
      end.setDate(end.getDate() + (6 - end.getDay()));
      // Calculate number of weeks
      const msPerDay = 24 * 60 * 60 * 1000;
      const numWeeks = Math.ceil((end - start) / msPerDay / 7);
      // Day labels
      const dayLabels = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
      // Build cells: 7 rows × numWeeks columns
      let html = '<div class="activity-grid-wrap">';
      html += '<div class="activity-grid" style="grid-template-columns: repeat(' + numWeeks + ', 1fr);">';
      for (let dow = 0; dow < 7; dow++) {
        for (let w = 0; w < numWeeks; w++) {
          const cellDate = new Date(start);
          cellDate.setDate(start.getDate() + w * 7 + dow);
          const dateStr = this._fmtDate(cellDate);
          const d = byDate[dateStr];
          const total = d ? (d.input || 0) + (d.output || 0) : 0;
          let bg, opacity;
          if (total === 0) {
            bg = 'var(--bg-2)'; opacity = '';
          } else {
            const ratio = maxTotal > 0 ? total / maxTotal : 0;
            const level = Math.max(0.2, ratio);
            bg = 'rgba(111, 207, 151, ' + level.toFixed(2) + ')';
            opacity = '';
          }
          const title = d
            ? dateStr + ': ' + this.fmtTokens(d.input) + ' in, ' + this.fmtTokens(d.output) + ' out, ' + (d.requests || 0) + ' reqs'
            : dateStr + ': no data';
          html += '<div class="activity-cell' + (total > 0 ? ' has-data' : '') + '" style="background:' + bg + ';' + opacity + '" title="' + title + '"></div>';
        }
      }
      html += '</div>';
      // Day labels on the left
      html += '<div class="activity-day-labels">';
      for (let dow = 0; dow < 7; dow++) {
        html += '<div class="activity-day-label">' + dayLabels[dow] + '</div>';
      }
      html += '</div>';
      html += '</div>';
      return html;
    },
    // Horizontal bar chart: input (blue) + output (green) per data point.
    renderBarChart() {
      let data = [];
      let labels = [];
      if (this.analysisChartMode === 'hour') {
        // Show 24 hours for the selected date (or today).
        const targetDate = this.analysisDateStart || this._fmtDate(new Date());
        const hourly = this.analysisHourly || [];
        const byHour = {};
        for (const h of hourly) {
          // h.hour looks like "2024-01-15T10"
          if (h.hour.startsWith(targetDate)) {
            byHour[parseInt(h.hour.slice(-2), 10)] = h;
          }
        }
        for (let hr = 0; hr < 24; hr++) {
          const d = byHour[hr];
          data.push({ input: d?.input || 0, output: d?.output || 0 });
          labels.push(hr.toString().padStart(2, '0') + ':00');
        }
      } else {
        // Per-day mode
        const daily = this.analysisDaily || [];
        for (const d of daily) {
          data.push({ input: d.input || 0, output: d.output || 0 });
          labels.push(d.date);
        }
      }
      if (!data.length) return '<div class="empty">No data for this period.</div>';
      const maxVal = Math.max(...data.map(d => (d.input || 0) + (d.output || 0)));
      if (maxVal === 0) return '<div class="empty">No data for this period.</div>';
      let html = '<div class="bar-chart">';
      for (let i = 0; i < data.length; i++) {
        const d = data[i];
        const total = d.input + d.output;
        const inputW = (d.input / maxVal) * 100;
        const outputW = (d.output / maxVal) * 100;
        html += '<div class="bar-row">';
        html += '<div class="bar-label" title="' + labels[i] + '">' + labels[i] + '</div>';
        html += '<div class="bar-track">';
        html += '<div class="bar-input" style="width:' + inputW.toFixed(1) + '%"></div>';
        html += '<div class="bar-output" style="left:' + inputW.toFixed(1) + '%; width:' + outputW.toFixed(1) + '%"></div>';
        html += '</div>';
        html += '<div class="bar-value">' + this.fmtTokens(total) + '</div>';
        html += '</div>';
      }
      html += '</div>';
      return html;
    },
    // Ensure the flagship pricing object is seeded from settings.
    // We use deep-clone so Alpine reactivity picks up mutations.
    _syncFlagshipPricing(settings) {
      const fp = settings?.flagship_pricing;
      if (!fp) return;
      const keys = Object.keys(fp);
      if (keys.length === 0) return;
      // Only seed on first load or if models changed.
      const existing = Object.keys(this.flagshipPricing);
      if (existing.length === 0 || keys.join(',') !== existing.join(',')) {
        // Deep-clone so each entry is reactive.
        const cloned = {};
        for (const k of keys) {
          cloned[k] = { input: fp[k].input ?? 0, output: fp[k].output ?? 0, enabled: !!fp[k].enabled };
        }
        this.flagshipPricing = cloned;
      } else if (!this.flagshipPricingDirty) {
        // Update in-place for reactive changes — but only when the user
        // hasn't made local edits. Otherwise we'd clobber unsaved input
        // (e.g. a half-typed price) every poll cycle.
        for (const k of keys) {
          if (this.flagshipPricing[k]) {
            this.flagshipPricing[k].input = fp[k].input ?? 0;
            this.flagshipPricing[k].output = fp[k].output ?? 0;
            this.flagshipPricing[k].enabled = !!fp[k].enabled;
          }
        }
      }
    },
    flagshipModelList() {
      return Object.keys(this.flagshipPricing);
    },
    flagshipPricingFor(name) {
      return this.flagshipPricing[name] || { input: 0, output: 0, enabled: false };
    },
    async saveFlagshipPricing() {
      this.pricingSaving = true;
      this.pricingSaved = false;
      const fp = {};
      for (const name of this.flagshipModelList()) {
        const p = this.flagshipPricing[name];
        fp[name] = { input: p.input, output: p.output, enabled: p.enabled };
      }
      try {
        const r = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ flagship_pricing: fp }),
        });
        if (!r.ok) throw new Error(r.statusText);
        this.pricingSaved = true;
        setTimeout(() => { this.pricingSaved = false; }, 2500);
        this.flagshipPricingDirty = false;
        await this.refresh();
      } catch (e) {
        alert('Save failed: ' + e.message);
      } finally {
        this.pricingSaving = false;
      }
    },
    addFlagshipModel() {
      const name = prompt('Enter flagship model name:');
      if (!name || !name.trim()) return;
      const k = name.trim();
      if (this.flagshipPricing[k]) {
        alert('"' + k + '" already exists.');
        return;
      }
      this.flagshipPricing[k] = { input: 0, output: 0, enabled: true };
      this.saveFlagshipPricing();
    },
    removeFlagshipModel(name) {
      if (!confirm('Remove "' + name + '" from flagship pricing?')) return;
      delete this.flagshipPricing[name];
      this.saveFlagshipPricing();
    },

    // ─── aggregate usage ────────────────────────────────────
    usageAlias(model) {
      return this.state?.usage_aliases?.[model] || '';
    },
    usageMergeGroup(model) {
      return this.state?.usage_merge_groups?.[model] || '';
    },
    usageMergeGroups() {
      return [...new Set(Object.values(this.state?.usage_merge_groups || {}).filter(Boolean))]
        .sort((a, b) => a.localeCompare(b));
    },
    usageAverageSpeed(row) {
      if (row?.speed?.tok_s == null) return null;
      const value = Number(row?.speed?.tok_s);
      return Number.isFinite(value) ? value : null;
    },
    usageCostValue(row) {
      const value = Number(row?.total_cost);
      return Number.isFinite(value) ? value : 0;
    },
    usageCostDisplay(row) {
      const value = this.usageCostValue(row);
      return value > 0 ? this.fmtCost(value) : '—';
    },
    usageSortValue(row, key) {
      const stats = row.stats || {};
      if (key === 'model') return String(row.label || row.key).toLocaleLowerCase();
      if (key === 'speed') return this.usageAverageSpeed(row);
      if (key === 'time') return Number(stats.gen_time_s || 0);
      if (key === 'cost') return this.usageCostValue(row);
      return Number(stats[key] || 0);
    },
    usageRows() {
      const rows = (this.state?.usage_rows || []).map(row => ({...row}));
      const key = this.usageSortKey;
      const direction = this.usageSortDirection === 'asc' ? 1 : -1;
      return rows.sort((left, right) => {
        const a = this.usageSortValue(left, key);
        const b = this.usageSortValue(right, key);
        const aMissing = a == null || (typeof a === 'number' && !Number.isFinite(a));
        const bMissing = b == null || (typeof b === 'number' && !Number.isFinite(b));
        if (aMissing !== bMissing) return aMissing ? 1 : -1;
        let comparison = 0;
        if (typeof a === 'string' || typeof b === 'string') {
          comparison = String(a).localeCompare(String(b), undefined, {numeric: true, sensitivity: 'base'});
        } else {
          comparison = Number(a) - Number(b);
        }
        if (comparison === 0) comparison = left.key.localeCompare(right.key);
        return comparison * direction;
      });
    },
    setUsageSort(key) {
      if (this.usageSortKey === key) {
        this.usageSortDirection = this.usageSortDirection === 'asc' ? 'desc' : 'asc';
      } else {
        this.usageSortKey = key;
        this.usageSortDirection = key === 'model' ? 'asc' : 'desc';
      }
    },
    usageSortIndicator(key) {
      if (this.usageSortKey !== key) return '↕';
      return this.usageSortDirection === 'asc' ? '↑' : '↓';
    },
    usageSortAria(key) {
      if (this.usageSortKey !== key) return 'none';
      return this.usageSortDirection === 'asc' ? 'ascending' : 'descending';
    },
    startUsageAliasEdit(model) {
      this.usageAliasValue[model] = this.usageAlias(model);
      this.usageMergeValue[model] = this.usageMergeGroup(model);
      this.usageAliasSaving[model] = false;
      this.usageAliasEditing[model] = true;
    },
    cancelUsageAliasEdit(model) {
      this.usageAliasEditing[model] = false;
    },
    async saveUsageAlias(model) {
      if (this.usageAliasSaving[model]) return;
      this.usageAliasSaving[model] = true;
      this.usageAliasSaved[model] = false;
      try {
        const response = await fetch('/api/token-stats/alias', {
          method: 'PUT',
          headers: {'content-type': 'application/json'},
          body: JSON.stringify({
            model,
            alias: this.usageAliasValue[model]?.trim() || null,
            merge_group: this.usageMergeValue[model]?.trim() || null,
          }),
        });
        const reply = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(reply.detail || response.statusText);
        this.usageAliasEditing[model] = false;
        this.usageAliasSaved[model] = true;
        setTimeout(() => { this.usageAliasSaved[model] = false; }, 2500);
        await this.refresh();
      } catch (error) {
        alert('Rename failed: ' + error.message);
      } finally {
        this.usageAliasSaving[model] = false;
      }
    },
    totalInputTokens() {
      return Object.values(this.state?.token_stats || {}).reduce((s, r) => s + (r.input || 0), 0);
    },
    totalCachedTokens() {
      return Object.values(this.state?.token_stats || {}).reduce((s, r) => s + (r.cached || 0), 0);
    },
    totalOutputTokens() {
      return Object.values(this.state?.token_stats || {}).reduce((s, r) => s + (r.output || 0), 0);
    },
    totalRequests() {
      return Object.values(this.state?.token_stats || {}).reduce((s, r) => s + (r.requests || 0), 0);
    },
    // Total cost across all raw models, using the server's authoritative
    // deployment/standalone/built-in pricing resolution.
    totalCost() {
      const total = Object.values(this.state?.token_costs || {})
        .reduce((sum, cost) => sum + Number(cost?.total_cost || 0), 0);
      return total > 0 ? this.fmtCost(total) : '—';
    },
    // Opportunity cost for a flagship model: total tokens × per-1M rate.
    oppCostInput(name) {
      const p = this.flagshipPricingFor(name);
      if (!p.enabled || !p.input) return '—';
      return this.fmtCost((this.totalInputTokens() / 1_000_000) * p.input);
    },
    oppCostOutput(name) {
      const p = this.flagshipPricingFor(name);
      if (!p.enabled || !p.output) return '—';
      return this.fmtCost((this.totalOutputTokens() / 1_000_000) * p.output);
    },
    oppCostTotal(name) {
      const p = this.flagshipPricingFor(name);
      if (!p.enabled || (!p.input && !p.output)) return '—';
      const total = (this.totalInputTokens() / 1_000_000) * (p.input || 0)
                  + (this.totalOutputTokens() / 1_000_000) * (p.output || 0);
      return this.fmtCost(total);
    },
    // Async cost cell for the usage table (per-model).
    async tokenCostCell(model) {
      try {
        const r = await fetch(`/api/token-cost/${encodeURIComponent(model)}`);
        if (!r.ok) return '—';
        const cost = await r.json();
        return cost.total_cost > 0 ? this.fmtCost(cost.total_cost) : '—';
      } catch {
        return '—';
      }
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
    fmtClock(mhz, maxMhz) {
      if (mhz == null || isNaN(mhz)) return '—';
      const unit = mhz >= 1000 ? 'GHz' : 'MHz';
      const val = mhz >= 1000 ? (mhz / 1000).toFixed(2) : Number(mhz).toFixed(0);
      if (maxMhz != null && !isNaN(maxMhz)) {
        const maxUnit = maxMhz >= 1000 ? 'GHz' : 'MHz';
        const maxVal = maxMhz >= 1000 ? (maxMhz / 1000).toFixed(2) : Number(maxMhz).toFixed(0);
        return `${val} ${unit} / ${maxVal} ${maxUnit}`;
      }
      return `${val} ${unit}`;
    },
    // Signed delta of current clock vs base clock. Positive = overclocked,
    // negative = underclocked. Falls back to the raw "current / max" display
    // when no base clock is known (e.g. nvidia-smi reports [N/A] on GB10, or
    // /sys base_frequency is absent).
    fmtClockDelta(mhz, baseMhz, maxMhz) {
      if (mhz == null || isNaN(mhz)) return '—';
      if (baseMhz == null || isNaN(baseMhz) || !baseMhz) {
        return this.fmtClock(mhz, maxMhz);  // graceful fallback to raw
      }
      const delta = Number(mhz) - Number(baseMhz);
      const sign = delta > 0 ? '+' : (delta < 0 ? '−' : '±');  // explicit sign; − is the minus glyph
      const absD = Math.abs(delta);
      const unit = absD >= 1000 ? 'GHz' : 'MHz';
      const val = absD >= 1000 ? (absD / 1000).toFixed(2) : absD.toFixed(0);
      return `${sign}${val} ${unit}`;
    },
    // Bar fill for the clock delta: negative (underclock) maps to 0–50%,
    // positive (overclock) maps to 50–100%, with base at the midpoint.
    // Falls back to current/max scaling when no base is known.
    clockPct(mhz, base, max) {
      if (mhz == null || isNaN(mhz)) return 0;
      if (base == null || isNaN(base) || !base) {
        // Fallback: scale against max (old behavior).
        if (!max) return 0;
        return Math.round(Math.max(0, Math.min(100, (Number(mhz) / max) * 100)));
      }
      const d = Number(mhz) - Number(base);
      // Scale so ±50% of base reaches the ends of the bar.
      const scaled = 50 + (d / Number(base)) * 50;
      return Math.round(Math.max(0, Math.min(100, scaled)));
    },
    clockColor(temp) {
      if (temp == null || isNaN(temp)) return 'var(--text-muted)';
      const t = Number(temp);
      if (t < 70) return 'var(--accent-green)';
      if (t < 85) return 'var(--accent-amber)';
      return 'var(--accent-red)';
    },
    // Map a temperature (°C) to 0–100% across a fixed visual range (30–95°C).
    // Rounded to whole percent so trivial fluctuations don't cause repaints.
    tempPct(t) {
      if (t == null || isNaN(t)) return 0;
      return Math.round(Math.max(0, Math.min(100, ((Number(t) - 30) / 65) * 100)));
    },
    tempColor(t) {
      if (t == null || isNaN(t)) return 'var(--text-muted)';
      const v = Number(t);
      if (v < 60) return 'var(--accent-green)';
      if (v < 80) return 'var(--accent-amber)';
      return 'var(--accent-red)';
    },
    // Power as fraction of limit (or 150W default cap for GB10).
    powerPct(d, l) {
      if (d == null || isNaN(d)) return 0;
      const max = l && !isNaN(l) ? Number(l) : 150;
      return Math.round(Math.max(0, Math.min(100, (Number(d) / max) * 100)));
    },
    powerColor(d, l) {
      const max = l && !isNaN(l) ? Number(l) : 150;
      const r = (Number(d) || 0) / max;
      if (r < 0.5) return 'var(--accent-green)';
      if (r < 0.85) return 'var(--accent-amber)';
      return 'var(--accent-red)';
    },

    async copyCode(el) {
      try {
        await navigator.clipboard.writeText(el.innerText);
        const btn = el.parentElement.querySelector('.copy-btn');
        if (btn) {
          btn.textContent = 'Copied';
          btn.classList.add('copied');
          setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 1200);
        }
      } catch (e) {
        alert('Copy failed: ' + e.message);
      }
    },
    copyLabel() { return 'Copy'; },

    fmtBytes(b) {
      if (b == null) return '—';
      const u = ['B', 'KB', 'MB', 'GB', 'TB'];
      let i = 0; let n = Number(b);
      while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
      return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
    },
    // "12s" / "3m05s" elapsed since a unix timestamp (launch progress).
    elapsedSince(ts) {
      if (!ts) return '';
      let s = Math.max(0, Math.floor(Date.now() / 1000 - Number(ts)));
      const m = Math.floor(s / 60);
      s %= 60;
      return m ? `${m}m${String(s).padStart(2, '0')}s` : `${s}s`;
    },
    fmtGb1(b) {
      if (b == null || isNaN(b)) return '—';
      return `${(Number(b) / 1024 / 1024 / 1024).toFixed(1)} GB`;
    },
    fmtBps(b) {
      if (b == null || isNaN(b)) return '—';
      const u = ['B/s', 'KB/s', 'MB/s', 'GB/s'];
      let i = 0; let n = Number(b);
      while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
      return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
    },
    netPct(bps) {
      // Scale 0–100% relative to 1 Gbit/s as a sensible full-bar ceiling.
      if (bps == null || isNaN(bps)) return 0;
      return Math.round(Math.max(0, Math.min(100, (Number(bps) * 8) / 1_000_000_000 * 100)));
    },
    // Memory bandwidth as a fraction of Grace LPDDR5X peak (~800 GB/s).
    bwPct(bps) {
      if (bps == null || isNaN(bps)) return 0;
      return Math.round(Math.max(0, Math.min(100, (Number(bps) / 800_000_000_000) * 100)));
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

    // vLLM images that aren't currently used by any (running or stopped) container.
    availableImages() {
      const containers = this.state?.containers || [];
      const usedTags = new Set(containers.map(c => c.image).filter(Boolean));
      const usedIds = new Set(containers.map(c => c.id).filter(Boolean));
      return (this.state?.images || []).filter(img => {
        if (!img.is_vllm) return false;
        if (img.tags?.some(t => usedTags.has(t))) return false;
        if (usedIds.has(img.id)) return false;
        return true;
      });
    },

    launchFromImage(tag) {
      this.tab = 'containers';
      this.showCreate = true;
      this.form.image = tag;
      // Scroll the form into view; a tick later so x-show transition has begun.
      setTimeout(() => {
        document.querySelector('.create-card')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);
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
    // Shell-aware whitespace splitter: respects single/double quotes, leaves
    // their content intact (strips the surrounding quote chars).
    shellSplit(str) {
      if (!str) return [];
      const out = [];
      let cur = '';
      let q = null;       // '"', "'", or null
      let inWord = false;
      for (let i = 0; i < str.length; i++) {
        const ch = str[i];
        if (q) {
          if (ch === q) { q = null; }
          else cur += ch;
          inWord = true;
        } else if (ch === '"' || ch === "'") {
          q = ch;
          inWord = true;
        } else if (/\s/.test(ch)) {
          if (inWord) { out.push(cur); cur = ''; inWord = false; }
        } else if (ch === '\\' && i + 1 < str.length) {
          // backslash-escape (incl. line continuation \<newline>)
          const next = str[i + 1];
          if (next === '\n' || next === '\r') { i++; continue; }
          cur += next; i++; inWord = true;
        } else {
          cur += ch; inWord = true;
        }
      }
      if (inWord) out.push(cur);
      return out;
    },

    shellJoin(args) {
      return (args || []).map(value => {
        const text = String(value);
        if (/^[a-zA-Z0-9_@%+=:,./-]+$/.test(text)) return text;
        return "'" + text.replaceAll("'", "'\\''") + "'";
      }).join(' ');
    },

    parsePasteCmd() {
      const txt = (this.form.pasteCmd || '').trim();
      if (!txt) return;
      const tokens = this.shellSplit(txt.replace(/\\\n/g, ' '));
      if (!tokens.length) return;

      // Detect SGLang command pattern: "python -m sglang.launch_server"
      let isSglang = false;
      let i = 0;
      // Skip optional leading commands
      while (i < tokens.length) {
        if (tokens[i] === 'vllm' || tokens[i] === 'serve') {
          i++;
        } else if (tokens[i] === 'python' || tokens[i] === 'python3' || tokens[i] === '-m') {
          i++;
        } else if (/^vllm\.entrypoints/.test(tokens[i])) {
          i++;
        } else if (tokens[i] === 'sglang' || tokens[i] === 'launch_server') {
          i++;
        } else if (tokens[i] === 'sglang.launch_server') {
          isSglang = true;
          i++;
          break;
        } else {
          break;
        }
      }
      if (isSglang) {
        this.form.engine = 'sglang';
      } else {
        this.form.engine = 'vllm';
      }

      // Next token (not starting with -) is the model id.
      while (i < tokens.length && tokens[i].startsWith('-')) i++;
      const remaining = tokens.slice(i);
      if (!remaining.length) return;
      const modelTok = remaining[0];
      const flagTokens = remaining.slice(1);

      // Strip --host/--port/--gpu-memory-utilization/--mem-fraction-static since we manage those.
      const cleaned = [];
      let parsedGpu = null;
      let parsedMemFrac = null;
      let parsedTpSize = null;
      let parsedCtxLen = null;
      let parsedMaxRunning = null;

      for (let j = 0; j < flagTokens.length; j++) {
        const t = flagTokens[j];
        if (t === '--host' || t === '--port') {
          j++; continue;
        }
        if (t === '--gpu-memory-utilization' || t === '--gpu_memory_utilization') {
          parsedGpu = parseFloat(flagTokens[j + 1]);
          j++; continue;
        }
        if (t === '--mem-fraction-static') {
          parsedMemFrac = parseFloat(flagTokens[j + 1]);
          j++; continue;
        }
        if (t === '--tp-size') {
          parsedTpSize = parseInt(flagTokens[j + 1], 10);
          j++; continue;
        }
        if (t === '--context-length') {
          parsedCtxLen = parseInt(flagTokens[j + 1], 10);
          j++; continue;
        }
        if (t === '--max-running-requests') {
          parsedMaxRunning = parseInt(flagTokens[j + 1], 10);
          j++; continue;
        }
        cleaned.push(t);
      }

      this.form.model = modelTok;
      this.form.extra = cleaned.join(' ');

      if (isSglang) {
        if (parsedTpSize != null && !isNaN(parsedTpSize)) this.form.sg_tp_size = parsedTpSize;
        if (parsedCtxLen != null && !isNaN(parsedCtxLen)) this.form.sg_context_length = parsedCtxLen;
        if (parsedMaxRunning != null && !isNaN(parsedMaxRunning)) this.form.sg_max_running_requests = parsedMaxRunning;
        if (parsedMemFrac != null && !isNaN(parsedMemFrac)) this.form.sg_mem_fraction = parsedMemFrac;
      } else if (parsedGpu != null && !isNaN(parsedGpu)) {
        this.form.gpu_mem_mode = 'fraction';
        this.form.gpu_mem = parsedGpu;
      }
    },

    onlineClusterNodes() {
      return (this.state?.nodes || []).filter(n => n.online && n.enabled !== false);
    },

    deploymentModeChanged() {
      if (this.form.deployment_mode === 'single') {
        const current = this.form.node_ids.find(id => this.onlineClusterNodes().some(n => n.id === id));
        this.form.node_ids = [current || 'local'];
      } else {
        this.form.node_ids = this.onlineClusterNodes().map(n => n.id);
      }
    },

    toggleLaunchNode(node) {
      if (!node.online || node.enabled === false) return;
      if (this.form.deployment_mode === 'single') {
        this.form.node_ids = [node.id];
        return;
      }
      if (node.id === 'local' && this.form.deployment_mode === 'sharded') return;
      const selected = this.form.node_ids.includes(node.id);
      this.form.node_ids = selected
        ? this.form.node_ids.filter(id => id !== node.id)
        : [...this.form.node_ids, node.id];
    },

    launchNodeSelected(id) {
      return (this.form.node_ids || []).includes(id);
    },

    launchSelectionValid() {
      const ids = this.form.node_ids || [];
      if (!ids.length) return false;
      if (this.form.deployment_mode !== 'single' && ids.length < 2) return false;
      return ids.every(id => this.onlineClusterNodes().some(n => n.id === id));
    },

    deploymentApiUrls(deployment) {
      const members = deployment?.mode === 'sharded'
        ? (deployment.members || []).filter(member => member.rank === 0)
        : (deployment?.members || []);
      return members.map(member => {
        const node = (this.state?.nodes || []).find(value => value.id === member.node_id);
        if (!node || node.local) {
          return `${location.protocol}//${location.hostname}:${deployment.api_port}`;
        }
        try {
          const agent = new URL(node.agent_url);
          return `${agent.protocol}//${agent.hostname}:${deployment.api_port}`;
        } catch {
          return `${node.hostname || node.name}:${deployment.api_port}`;
        }
      });
    },

    deploymentIsLaunching(deployment) {
      return ['launching', 'starting'].includes(deployment?.status);
    },

    deploymentProgress(deployment) {
      const members = deployment?.members || [];
      if (!members.length) return null;
      let hasKnownProgress = false;
      let total = 0;
      for (const member of members) {
        const phase = member?.phase || {};
        if (phase.phase === 'ready') {
          total += 1;
          hasKnownProgress = true;
        } else if (phase.progress != null && Number.isFinite(Number(phase.progress))) {
          total += Math.max(0, Math.min(1, Number(phase.progress)));
          hasKnownProgress = true;
        }
      }
      return hasKnownProgress ? total / members.length : null;
    },

    deploymentProgressMessage(deployment) {
      const members = deployment?.members || [];
      if (!members.length) return 'Preparing cluster launch…';
      const ready = members.filter(member => member?.phase?.phase === 'ready').length;
      const active = members.find(member => member?.phase?.phase !== 'ready') || members[0];
      const detail = active?.phase?.message || active?.status || 'Starting…';
      const rank = active?.rank != null ? `rank ${active.rank}` : (active?.node_name || 'member');
      return ready > 0
        ? `${ready}/${members.length} ranks ready · ${rank}: ${detail}`
        : `${rank}: ${detail}`;
    },

    async pairClusterNode() {
      if (!this.clusterForm.agent_url || !this.clusterForm.pairing_code) return;
      this.clusterPairing = true;
      try {
        const r = await fetch('/api/nodes', {
          method: 'POST', headers: {'content-type': 'application/json'},
          body: JSON.stringify(this.clusterForm),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        this.clusterForm = {name:'', agent_url:'', pairing_code:'', fabric_ip:'', fabric_interface:''};
        await this.refresh();
      } catch (e) {
        alert('Pairing failed: ' + e.message);
      } finally {
        this.clusterPairing = false;
      }
    },

    async refreshClusterNode(id) {
      this.clusterBusy[id] = true;
      try {
        await fetch(`/api/nodes/${encodeURIComponent(id)}/refresh`, {method:'POST'});
        await this.refresh();
      } finally {
        this.clusterBusy[id] = false;
      }
    },

    async toggleClusterNode(node) {
      if (node.local) return;
      await fetch(`/api/nodes/${encodeURIComponent(node.id)}`, {
        method:'PATCH', headers:{'content-type':'application/json'},
        body:JSON.stringify({enabled: node.enabled === false}),
      });
      await this.refresh();
    },

    async removeClusterNode(node) {
      if (node.local || !confirm(`Remove ${node.name} from this cluster?`)) return;
      const r = await fetch(`/api/nodes/${encodeURIComponent(node.id)}`, {method:'DELETE'});
      if (!r.ok) alert('Remove failed: ' + await r.text());
      await this.refresh();
    },

    async deploymentAction(deployment, action) {
      const verb = action === 'remove' ? 'Remove' : (action === 'stop' ? 'Stop' : 'Start');
      if ((action === 'remove' || action === 'stop') && !confirm(`${verb} ${deployment.name} on every node?`)) return;
      this.clusterBusy[deployment.id] = true;
      try {
        const r = await fetch(`/api/deployments/${deployment.id}/${action}`, {method:'POST'});
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        await this.refresh();
      } catch (e) {
        alert(`${verb} failed: ${e.message}`);
      } finally {
        this.clusterBusy[deployment.id] = false;
      }
    },

    async copyDeploymentId(deployment) {
      try {
        await navigator.clipboard.writeText(deployment.id);
        this.deploymentIdCopied[deployment.id] = true;
        setTimeout(() => { this.deploymentIdCopied[deployment.id] = false; }, 2000);
      } catch (error) {
        alert('Copy failed: ' + error.message);
      }
    },

    aliasEditorKey(kind, item) {
      return `${kind}:${kind === 'deployment' ? item.id : item.name}`;
    },

    toggleAliasEditor(kind, item) {
      const key = this.aliasEditorKey(kind, item);
      if (!this.aliasEditorOpen[key]) {
        this.aliasEditorValue[key] = kind === 'deployment' ? (item.name || '') : (item.alias || '');
      }
      this.aliasEditorOpen[key] = !this.aliasEditorOpen[key];
    },

    async saveAlias(kind, item) {
      const key = this.aliasEditorKey(kind, item);
      if (this.aliasEditorSaving[key]) return;
      const identifier = kind === 'deployment' ? item.id : item.name;
      const path = kind === 'deployment'
        ? `/api/deployments/${encodeURIComponent(identifier)}/alias`
        : `/api/containers/${encodeURIComponent(identifier)}/alias`;
      this.aliasEditorSaving[key] = true;
      this.aliasEditorSaved[key] = false;
      try {
        const response = await fetch(path, {
          method: 'PUT',
          headers: {'content-type': 'application/json'},
          body: JSON.stringify({alias: this.aliasEditorValue[key]?.trim() || null}),
        });
        const reply = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(reply.detail || response.statusText);
        this.aliasEditorOpen[key] = false;
        this.aliasEditorSaved[key] = true;
        setTimeout(() => { this.aliasEditorSaved[key] = false; }, 2500);
        await this.refresh();
      } catch (error) {
        alert('Rename failed: ' + error.message);
      } finally {
        this.aliasEditorSaving[key] = false;
      }
    },

    deploymentSettingsDraft(deployment) {
      const settings = deployment?.launch_settings || {};
      const controls = deployment?.launch_controls || {};
      const gpuGb = settings.gpu_memory_gb;
      return {
        deployment_name: settings.deployment_name || deployment?.name || '',
        model: settings.model || deployment?.model || '',
        engine: settings.engine || deployment?.engine || 'vllm',
        image: settings.image || '',
        extra: this.shellJoin(settings.extra_args || []),
        gpu_mem_mode: gpuGb != null ? 'gb' : 'fraction',
        gpu_memory_utilization: settings.gpu_memory_utilization ?? 0.9,
        gpu_memory_gb: gpuGb ?? null,
        context_window: controls.context_window ?? null,
        max_concurrency: controls.max_concurrency ?? null,
        kv_cache_dtype: controls.kv_cache_dtype || '',
        thinking_mode: controls.thinking_mode || 'default',
        dspark_num_speculative_tokens: controls.dspark_num_speculative_tokens ?? null,
        max_cudagraph_capture_size: controls.max_cudagraph_capture_size ?? null,
        max_num_batched_tokens: controls.max_num_batched_tokens ?? null,
        input_cost_per_1m: settings.input_cost_per_1m ?? null,
        cache_cost_per_1m: settings.cache_cost_per_1m ?? null,
        output_cost_per_1m: settings.output_cost_per_1m ?? null,
        sg_tp_size: settings.sg_tp_size ?? 1,
        sg_context_length: settings.sg_context_length ?? 32768,
        sg_max_running_requests: settings.sg_max_running_requests ?? null,
        sg_mem_fraction: settings.sg_mem_fraction ?? 0.92,
        sg_image: settings.sg_image || '',
        deployment_mode: settings.deployment_mode || deployment?.mode || 'single',
        node_ids: [...(settings.node_ids || deployment?.node_ids || ['local'])],
        port: settings.port ?? deployment?.api_port ?? null,
      };
    },

    deploymentSettingsFor(deployment) {
      if (!this.deploymentSettingsForm[deployment.id]) {
        this.deploymentSettingsForm[deployment.id] = this.deploymentSettingsDraft(deployment);
      }
      return this.deploymentSettingsForm[deployment.id];
    },

    toggleDeploymentSettings(deployment) {
      if (!this.deploymentSettingsOpen[deployment.id]) {
        const draft = this.deploymentSettingsDraft(deployment);
        this.deploymentSettingsForm[deployment.id] = draft;
        this.deploymentSettingsBaseline[deployment.id] = this.deploymentSettingsSnapshot(
          deployment, draft
        );
        this.deploymentPricingBaseline[deployment.id] = this.deploymentPricingSnapshot(
          deployment, draft
        );
      }
      this.deploymentSettingsOpen[deployment.id] = !this.deploymentSettingsOpen[deployment.id];
    },

    deploymentSettingsNodeSelected(deployment, nodeId) {
      return this.deploymentSettingsFor(deployment).node_ids.includes(nodeId);
    },

    toggleDeploymentSettingsNode(deployment, node) {
      if (!node.online || deployment.status !== 'stopped') return;
      const form = this.deploymentSettingsFor(deployment);
      if (form.deployment_mode === 'sharded' && node.id === 'local' && form.node_ids.includes('local')) return;
      if (form.node_ids.includes(node.id)) {
        form.node_ids = form.node_ids.filter(id => id !== node.id);
      } else if (form.deployment_mode === 'single') {
        form.node_ids = [node.id];
      } else {
        form.node_ids = [...form.node_ids, node.id];
      }
      if (form.deployment_mode === 'sharded' && form.node_ids.includes('local')) {
        form.node_ids = ['local', ...form.node_ids.filter(id => id !== 'local')];
      }
    },

    deploymentSettingsModeChanged(deployment) {
      const form = this.deploymentSettingsFor(deployment);
      if (form.deployment_mode === 'single') {
        form.node_ids = form.node_ids.slice(0, 1);
      } else if (form.deployment_mode === 'sharded' && form.node_ids.includes('local')) {
        form.node_ids = ['local', ...form.node_ids.filter(id => id !== 'local')];
      }
    },

    deploymentSettingsValid(deployment) {
      const form = this.deploymentSettingsFor(deployment);
      if (!form.model || !form.node_ids.length) return false;
      return form.deployment_mode === 'single' || form.node_ids.length >= 2;
    },

    deploymentSettingsPayload(deployment, suppliedForm = null) {
      const form = suppliedForm || this.deploymentSettingsFor(deployment);
      return {
        deployment_name: form.deployment_name?.trim() || form.model.trim(),
        model: form.model.trim(),
        engine: form.engine,
        image: form.engine === 'vllm' ? (form.image?.trim() || null) : null,
        extra_args: this.shellSplit(form.extra),
        deployment_mode: form.deployment_mode,
        node_ids: [...form.node_ids],
        port: form.port || null,
        gpu_memory_utilization: form.engine === 'vllm' && form.gpu_mem_mode === 'fraction'
          ? (form.gpu_memory_utilization || null) : null,
        gpu_memory_gb: form.engine === 'vllm' && form.gpu_mem_mode === 'gb'
          ? (form.gpu_memory_gb || null) : null,
        sg_tp_size: form.engine === 'sglang' ? (form.sg_tp_size || 1) : null,
        sg_context_length: form.engine === 'sglang' ? (form.context_window || 32768) : null,
        sg_max_running_requests: form.engine === 'sglang' ? (form.max_concurrency || null) : null,
        sg_mem_fraction: form.engine === 'sglang' ? (form.sg_mem_fraction || 0.92) : null,
        sg_image: form.engine === 'sglang' ? (form.sg_image?.trim() || null) : null,
        launch_controls: {
          context_window: form.context_window || null,
          max_concurrency: form.max_concurrency || null,
          kv_cache_dtype: form.kv_cache_dtype || null,
          thinking_mode: form.thinking_mode || 'default',
          dspark_num_speculative_tokens: form.engine === 'vllm'
            ? (form.dspark_num_speculative_tokens || null) : null,
          max_cudagraph_capture_size: form.engine === 'vllm'
            ? (form.max_cudagraph_capture_size || null) : null,
          max_num_batched_tokens: form.engine === 'vllm'
            ? (form.max_num_batched_tokens || null) : null,
        },
      };
    },

    deploymentSettingsSnapshot(deployment, suppliedForm = null) {
      return JSON.stringify(this.deploymentSettingsPayload(deployment, suppliedForm));
    },

    nullablePrice(value) {
      return value === '' || value == null ? null : Number(value);
    },

    deploymentPricingPayload(deployment, suppliedForm = null) {
      const form = suppliedForm || this.deploymentSettingsFor(deployment);
      return {
        input_cost_per_1m: this.nullablePrice(form.input_cost_per_1m),
        cache_cost_per_1m: this.nullablePrice(form.cache_cost_per_1m),
        output_cost_per_1m: this.nullablePrice(form.output_cost_per_1m),
      };
    },

    deploymentPricingSnapshot(deployment, suppliedForm = null) {
      return JSON.stringify(this.deploymentPricingPayload(deployment, suppliedForm));
    },

    deploymentPricingChanged(deployment) {
      return this.deploymentPricingSnapshot(deployment)
        !== this.deploymentPricingBaseline[deployment.id];
    },

    async saveDeploymentPricing(deployment) {
      if (this.deploymentPricingSaving[deployment.id]
          || !this.deploymentPricingChanged(deployment)) return;
      const form = this.deploymentSettingsFor(deployment);
      this.deploymentPricingSaving[deployment.id] = true;
      this.deploymentPricingSaved[deployment.id] = false;
      try {
        const response = await fetch(`/api/deployments/${deployment.id}/pricing`, {
          method: 'PUT',
          headers: {'content-type': 'application/json'},
          body: JSON.stringify(this.deploymentPricingPayload(deployment, form)),
        });
        const reply = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(reply.detail || response.statusText);
        this.deploymentPricingBaseline[deployment.id] = this.deploymentPricingSnapshot(
          deployment, form
        );
        this.deploymentPricingSaved[deployment.id] = true;
        setTimeout(() => { this.deploymentPricingSaved[deployment.id] = false; }, 2500);
        await this.refresh();
      } catch (error) {
        alert('Save pricing failed: ' + error.message);
      } finally {
        this.deploymentPricingSaving[deployment.id] = false;
      }
    },

    deploymentSettingsChanged(deployment) {
      const baseline = this.deploymentSettingsBaseline[deployment.id];
      if (baseline == null) return false;
      return this.deploymentSettingsSnapshot(deployment) !== baseline;
    },

    async saveDeploymentSettings(deployment) {
      if (deployment.status !== 'stopped' || this.deploymentSettingsSaving[deployment.id]
          || !this.deploymentSettingsChanged(deployment)) return;
      const form = this.deploymentSettingsFor(deployment);
      const body = this.deploymentSettingsPayload(deployment, form);
      this.deploymentSettingsSaving[deployment.id] = true;
      this.deploymentSettingsSaved[deployment.id] = false;
      try {
        const response = await fetch(`/api/deployments/${deployment.id}/settings`, {
          method: 'PUT',
          headers: {'content-type': 'application/json'},
          body: JSON.stringify(body),
        });
        const reply = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(reply.detail || response.statusText);
        this.deploymentSettingsBaseline[deployment.id] = this.deploymentSettingsSnapshot(
          deployment, form
        );
        this.deploymentSettingsSaved[deployment.id] = true;
        setTimeout(() => { this.deploymentSettingsSaved[deployment.id] = false; }, 2500);
        await this.refresh();
      } catch (error) {
        alert('Save launch settings failed: ' + error.message);
      } finally {
        this.deploymentSettingsSaving[deployment.id] = false;
      }
    },

    async openDeploymentLogs(deployment) {
      this.logsModal = {
        open: true,
        name: `Cluster · ${deployment.name}`,
        text: 'Loading cluster logs…',
        deploymentId: deployment.id,
        members: [],
        autoScroll: true,
        llama: false,
        llamaModel: null,
        logId: null,
        logOffset: 0,
      };
      this._startLogsTail();
    },

    async refreshDeploymentLogs() {
      const deploymentId = this.logsModal.deploymentId;
      if (!deploymentId) return;
      try {
        const r = await fetch(`/api/deployments/${deploymentId}/logs`);
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        const data = await r.json();
        if (!this.logsModal.open || this.logsModal.deploymentId !== deploymentId) return;
        this.logsModal.members = data.members || [];
        this.logsModal.text = this.logsModal.members.length ? '' : '(no cluster members)';
        this._scrollLogsToBottom();
      } catch (e) {
        if (!this.logsModal.open || this.logsModal.deploymentId !== deploymentId) return;
        this.logsModal.members = [];
        this.logsModal.text = 'Failed to fetch cluster logs: ' + e.message;
      }
    },

    async createContainer() {
      if (!this.form.model) return;
      this.creating = true;
      try {
        const engine = this.form.engine;
        const body = {
          model: this.form.model,
          port: this.form.port || null,
          engine: engine,
          extra_args: this.shellSplit(this.form.extra),
          image: engine === 'vllm' ? (this.form.image?.trim() || null) : null,
          deployment_mode: this.form.deployment_mode || 'single',
          node_ids: this.form.node_ids?.length ? this.form.node_ids : ['local'],
        };
        // vLLM-specific fields
        if (engine === 'vllm') {
          body.gpu_memory_utilization = this.form.gpu_mem_mode === 'fraction' ? (this.form.gpu_mem || null) : null;
          body.gpu_memory_gb = this.form.gpu_mem_mode === 'gb' ? (this.form.gpu_mem_gb || null) : null;
        }
        // SGLang-specific fields
        if (engine === 'sglang') {
          body.sg_tp_size = this.form.sg_tp_size || 1;
          body.sg_context_length = this.form.sg_context_length || 32768;
          body.sg_max_running_requests = this.form.sg_max_running_requests || null;
          body.sg_mem_fraction = this.form.sg_mem_fraction || 0.92;
          body.sg_image = this.form.sg_image?.trim() || null;
        }
        const r = await fetch('/api/containers', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        this.showCreate = false;
        this.form = {
          model: '', port: null, engine: 'vllm', deployment_mode: 'single', node_ids: ['local'],
          gpu_mem_mode: 'fraction', gpu_mem: 0.9, gpu_mem_gb: Math.floor(this.gpuTotalGb() / 2),
          extra: '', image: '', pasteCmd: '',
          sg_tp_size: 1, sg_context_length: 32768, sg_max_running_requests: null,
          sg_mem_fraction: 0.92, sg_image: '',
        };
        await this.refresh();
      } catch (e) {
        alert('Failed to create container: ' + e.message);
      } finally {
        this.creating = false;
      }
    },

    async saveRecipe() {
      if (!this.form.model) return;
      const engine = this.form.engine;
      const extra = this.shellSplit(this.form.extra);
      const templated = extra.filter(a => /\{[a-zA-Z_][a-zA-Z0-9_]*\}/.test(a));
      if (templated.length) {
        if (!confirm(`Extra args contain unresolved placeholders: ${templated.join(', ')}. Save anyway?`)) return;
      }
      try {
        const body = {
          model: this.form.model,
          name: this.form.model,
          engine: engine,
          extra_args: extra,
          deployment_mode: this.form.deployment_mode || 'single',
          node_ids: this.form.node_ids?.length ? this.form.node_ids : ['local'],
        };
        if (engine === 'vllm') {
          body.image = this.form.image?.trim() || null;
          body.gpu_memory_utilization = this.form.gpu_mem_mode === 'fraction' ? (this.form.gpu_mem || null) : null;
          body.gpu_memory_gb = this.form.gpu_mem_mode === 'gb' ? (this.form.gpu_mem_gb || null) : null;
        }
        if (engine === 'sglang') {
          body.sg_tp_size = this.form.sg_tp_size || 1;
          body.sg_context_length = this.form.sg_context_length || 32768;
          body.sg_max_running_requests = this.form.sg_max_running_requests || null;
          body.sg_mem_fraction = this.form.sg_mem_fraction || 0.92;
          body.sg_image = this.form.sg_image?.trim() || null;
        }
        const r = await fetch('/api/recipes', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        this.recipeSaved = true;
        setTimeout(() => { this.recipeSaved = false; }, 2500);
        await this.refresh();
      } catch (e) {
        alert('Save recipe failed: ' + e.message);
      }
    },

    async launchRecipe(rid) {
      if (this.recipeLaunching[rid]) return;
      this.recipeLaunching[rid] = true;
      try {
        const r = await fetch(`/api/recipes/${rid}/launch`, { method: 'POST' });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        await this.refresh();
      } catch (e) {
        alert('Launch failed: ' + e.message);
      } finally {
        this.recipeLaunching[rid] = false;
      }
    },

    async deleteRecipe(rid) {
      if (!confirm('Delete this recipe? (containers using it stay untouched)')) return;
      const resp = await fetch(`/api/recipes/${rid}`, { method: 'DELETE' });
      if (!resp.ok) {
        alert('Delete failed: ' + (await resp.text()));
      } else {
        this.state.recipes = (this.state.recipes || []).filter(r => r.id !== rid);
      }
      this.refresh();
    },

    editRecipe(r) {
      this.tab = 'containers';
      this.showCreate = true;
      this.form.model = r.model || '';
      this.form.deployment_mode = r.deployment_mode || 'single';
      this.form.node_ids = [...(r.node_ids || ['local'])];
      // Set engine type
      this.form.engine = r.engine || 'vllm';
      // Set engine-specific fields
      if (this.form.engine === 'sglang') {
        this.form.sg_tp_size = r.sg_tp_size || 1;
        this.form.sg_context_length = r.sg_context_length || 32768;
        this.form.sg_max_running_requests = r.sg_max_running_requests;
        this.form.sg_mem_fraction = r.sg_mem_fraction || 0.92;
        this.form.sg_image = r.sg_image || '';
        this.form.image = ''; // clear vLLM image field
      } else {
        this.form.image = r.image || '';
        this.form.sg_image = ''; // clear SGLang image field
      }
      if (this.form.engine === 'vllm') {
        if (r.gpu_memory_gb != null) {
          this.form.gpu_mem_mode = 'gb';
          this.form.gpu_mem_gb = r.gpu_memory_gb;
        } else {
          this.form.gpu_mem_mode = 'fraction';
          this.form.gpu_mem = r.gpu_memory_utilization ?? 0.9;
        }
      }
      this.form.extra = (r.extra_args || []).map(a => /\s|["']/.test(a) ? `'${a.replace(/'/g, "'\\''")}'` : a).join(' ');
      setTimeout(() => {
        document.querySelector('.create-card')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 50);
    },

    async startContainer(name) {
      try {
        const r = await fetch(`/api/containers/${encodeURIComponent(name)}/start`, { method: 'POST' });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        await this.refresh();
      } catch (e) {
        alert('Start failed: ' + e.message);
      }
    },
    canStopContainer(c) {
      // Health can report ready while Docker's cached status is briefly stale.
      // The server-side stop operation is idempotent, so keep Stop available
      // for any live/ready card unless an action is already in flight.
      return ['running', 'restarting', 'paused'].includes(c?.status)
        || c?.phase?.phase === 'ready';
    },
    async stopContainer(name) {
      if (this.containerStopping[name]) return;
      this.containerStopping[name] = true;
      const ac = new AbortController();
      // The server allows up to 30 seconds for Docker shutdown and cleanup;
      // leave room for the HTTP response so a successful stop is not aborted.
      const timer = setTimeout(() => ac.abort(), 45000);
      try {
        const r = await fetch(`/api/containers/${encodeURIComponent(name)}/stop`, {
          method: 'POST',
          signal: ac.signal,
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        await this.refresh();
      } catch (e) {
        if (e.name === 'AbortError') {
          // Stop request timed out — clear the flag so the user can retry or remove
          return;
        }
        alert('Stop failed: ' + e.message);
      } finally {
        clearTimeout(timer);
        this.containerStopping[name] = false;
      }
    },
    async removeContainer(name) {
      if (!confirm(`Remove container "${name}"? This cannot be undone.`)) return;
      const resp = await fetch(`/api/containers/${encodeURIComponent(name)}`, { method: 'DELETE' });
      if (!resp.ok) {
        alert('Remove failed: ' + (await resp.text()));
      } else {
        this.state.containers = (this.state.containers || []).filter(c => c.name !== name);
      }
      this.refresh();
    },

    dockerSettingsFor(c) {
      if (!this.dockerSettingsForm[c.name]) {
        this.dockerSettingsForm[c.name] = { ...(c.load_settings || {}) };
      }
      return this.dockerSettingsForm[c.name];
    },

    toggleDockerSettings(c) {
      if (!this.dockerSettingsOpen[c.name]) {
        this.dockerSettingsForm[c.name] = { ...(c.load_settings || {}) };
      }
      this.dockerSettingsOpen[c.name] = !this.dockerSettingsOpen[c.name];
    },

    onDockerSettingsToggle(c, open) {
      if (open && !this.dockerSettingsOpen[c.name]) {
        this.dockerSettingsForm[c.name] = { ...(c.load_settings || {}) };
      }
      this.dockerSettingsOpen[c.name] = open;
    },

    async saveDockerSettings(c) {
      if (this.dockerSettingsSaving[c.name]) return;
      const action = c.status === 'running' ? 'restart' : 'recreate';
      if (!confirm(`Save load settings and ${action} "${c.name}"?`)) return;
      this.dockerSettingsSaving[c.name] = true;
      this.dockerSettingsSaved[c.name] = false;
      try {
        const r = await fetch(`/api/containers/${encodeURIComponent(c.name)}/settings`, {
          method: 'PUT',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ settings: this.dockerSettingsFor(c) }),
        });
        const reply = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(reply.detail || r.statusText);
        this.dockerSettingsOpen[c.name] = false;
        this.dockerSettingsSaved[c.name] = true;
        setTimeout(() => { this.dockerSettingsSaved[c.name] = false; }, 2500);
        await this.refresh();
      } catch (e) {
        alert('Save settings failed: ' + e.message);
      } finally {
        this.dockerSettingsSaving[c.name] = false;
      }
    },

    async copyToRecipe(name) {
      try {
        const r = await fetch(`/api/containers/${encodeURIComponent(name)}/to-recipe`, { method: 'POST' });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        await this.refresh();
      } catch (e) {
        alert('Copy to recipe failed: ' + e.message);
      }
    },

    async openLogs(name) {
      this.logsModal = {
        open: true, name, text: 'Loading…', deploymentId: null, members: [], autoScroll: true,
        llama: false, logId: null, logOffset: 0,
        llamaModel: null,
      };
      this._startLogsTail();
    },
    async openUnslothLogs(model) {
      this.logsModal = {
        open: true,
        name: `llama-server · ${model}`,
        text: 'Loading full llama-server log…',
        deploymentId: null,
        members: [],
        autoScroll: true,
        llama: true,
        llamaModel: model,
        logId: null,
        logOffset: 0,
      };
      this._startLogsTail();
    },
    async refreshUnslothLogs() {
      const model = this.logsModal.llamaModel;
      const offset = Number(this.logsModal.logOffset || 0);
      try {
        const params = new URLSearchParams({
          model_path: model || '',
          since: String(offset),
          limit_bytes: '1048576',
        });
        const r = await fetch(`/api/unsloth/logs?${params}`);
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
        const data = await r.json();
        if (!this.logsModal.open || !this.logsModal.llama || this.logsModal.llamaModel !== model) return;
        if (!data.log_id) {
          this.logsModal.text = '(no llama-server log available)';
          return;
        }
        if (data.model) this.logsModal.name = `llama-server · ${data.model}`;
        if (this.logsModal.logId !== data.log_id || data.offset === 0 && offset > 0) {
          this.logsModal.logId = data.log_id;
          this.logsModal.text = '';
          this.logsModal.logOffset = 0;
        }
        if (data.text) this.logsModal.text += data.text;
        this.logsModal.logOffset = data.next_offset || 0;
        this._scrollLogsToBottom();
      } catch (e) {
        if (!this.logsModal.open || !this.logsModal.llama || this.logsModal.llamaModel !== model) return;
        this.logsModal.text = 'Failed to fetch llama-server logs: ' + e.message;
      }
    },
    async refreshLogs() {
      if (!this.logsModal.open || this._logsRefreshInFlight) return;
      this._logsRefreshInFlight = true;
      try {
        if (this.logsModal.deploymentId) return await this.refreshDeploymentLogs();
        if (this.logsModal.llama) return await this.refreshUnslothLogs();
        const name = this.logsModal.name;
        if (!name) return;
        try {
          const r = await fetch(`/api/containers/${encodeURIComponent(name)}/logs?tail=400`);
          if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
          const j = await r.json();
          if (!this.logsModal.open || this.logsModal.name !== name || this.logsModal.deploymentId) return;
          this.logsModal.text = j.logs || '(no logs)';
          this._scrollLogsToBottom();
        } catch (e) {
          if (!this.logsModal.open || this.logsModal.name !== name || this.logsModal.deploymentId) return;
          this.logsModal.text = 'Failed to fetch logs: ' + e.message;
        }
      } finally {
        this._logsRefreshInFlight = false;
      }
    },
    _startLogsTail() {
      this._stopLogsTail();
      this.refreshLogs();
      this._logsTailInterval = setInterval(() => {
        if (this.logsModal.open && !document.hidden) this.refreshLogs();
      }, 1000);
    },
    _stopLogsTail() {
      if (this._logsTailInterval) {
        clearInterval(this._logsTailInterval);
        this._logsTailInterval = null;
      }
    },
    _scrollLogsToBottom() {
      if (!this.logsModal.autoScroll) return;
      setTimeout(() => {
        document.querySelectorAll('#logs-modal .logbox').forEach((element) => {
          element.scrollTop = element.scrollHeight;
        });
      }, 0);
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
            } catch { }
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

    // ─── chat (multi-turn) ───────────────────────────────────
    clearChat() {
      this.chat.messages = [];
      this.chat.input = '';
    },

    scrollChat() {
      setTimeout(() => {
        const el = this.$refs?.chatWindow;
        if (el) el.scrollTop = el.scrollHeight;
      }, 0);
    },

    stopChat() {
      if (this.chat.abort) {
        try { this.chat.abort.abort(); } catch { }
      }
    },

    runningManaged() {
      return (this.state?.containers || []).filter(c => c.managed && c.status === 'running');
    },

    async forceKillRunning() {
      const running = this.runningManaged();
      if (!running.length) {
        alert('No running managed containers to kill.');
        return;
      }
      const names = running.map(c => c.name).join('\n  • ');
      if (!confirm(
        `Force-stop ${running.length} running vLLM container(s)?\n\n  • ${names}\n\n` +
        `This frees GPU memory immediately. Any in-flight generation is killed. ` +
        `The next request will pay the cold-load cost.`
      )) return;
      for (const c of running) {
        try {
          await fetch(`/api/containers/${encodeURIComponent(c.name)}/stop`, { method: 'POST' });
        } catch (e) {
          console.error('force-kill failed for', c.name, e);
        }
      }
      // Also abort any in-flight client streams so the UI doesn't hang waiting.
      this.stopChat();
      this.stopABBoth();
      await this.refresh();
    },

    async sendChat() {
      const text = (this.chat.input || '').trim();
      if (!text || !this.chat.model || this.chat.busy) return;

      this.chat.messages.push({ role: 'user', content: text });
      this.chat.input = '';
      const assistantIdx = this.chat.messages.push({
        role: 'assistant', content: '', reasoning: '', status: 'connecting…',
        tokens: 0, thinkingTokens: 0, totalTokens: 0,
        elapsedMs: 0, streamMs: 0, thinkingMs: 0,
        generationMs: 0, ttftMs: 0,
      }) - 1;
      this.chat.busy = true;
      const ac = new AbortController();
      this.chat.abort = ac;
      const started = performance.now();
      let firstTokenAt = null;
      let firstThinkingAt = null;
      let firstOutputAt = null;
      this.scrollChat();

      const sys = (this.chat.system || '').trim() || 'You are a helpful assistant.';
      const payload = {
        model: this.chat.model,
        messages: [
          { role: 'system', content: sys },
          ...this.chat.messages
            .slice(0, assistantIdx)
            .map(m => ({ role: m.role, content: m.content })),
        ],
        stream: true,
        temperature: this.chat.temperature,
        max_tokens: this.chat.max_tokens,
      };

      try {
        const r = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload),
          signal: ac.signal,
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 400)}`);
        this.chat.messages[assistantIdx].status = 'streaming';
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
            for (const raw of p.split('\n')) {
              const line = raw.replace(/^data: /, '').trim();
              if (!line || line === '[DONE]') continue;
              let obj;
              try {
                obj = JSON.parse(line);
              } catch {
                continue;
              }
              if (obj.error) {
                const detail = typeof obj.error === 'string'
                  ? obj.error
                  : (obj.error.message || JSON.stringify(obj.error));
                throw new Error(detail);
              }
              {
                const choice = obj.choices?.[0];
                const delta = choice?.delta;
                const content = delta?.content;
                const reasoning = delta?.reasoning_content || delta?.reasoning;
                const tokenIds = Array.isArray(choice?.token_ids) ? choice.token_ids : null;
                const tokenCount = tokenIds?.length || ((reasoning || content) ? 1 : 0);
                if (tokenCount > 0) {
                  const now = performance.now();
                  if (firstTokenAt === null) {
                    firstTokenAt = now;
                    this.chat.messages[assistantIdx].ttftMs = firstTokenAt - started;
                  }
                  const msg = this.chat.messages[assistantIdx];
                  // Count exact token ids once across hidden reasoning and
                  // visible output. This is the primary decode-rate counter.
                  msg.totalTokens += tokenCount;
                  msg.generationMs = now - firstTokenAt;
                }
                if (reasoning) {
                  const now = performance.now();
                  if (firstThinkingAt === null) firstThinkingAt = now;
                  const msg = this.chat.messages[assistantIdx];
                  msg.reasoning += reasoning;
                  msg.thinkingTokens += tokenCount;
                  msg.thinkingMs = now - firstThinkingAt;
                  msg.elapsedMs = now - started;
                  this.scrollChat();
                }
                if (content) {
                  const now = performance.now();
                  if (firstOutputAt === null) firstOutputAt = now;
                  const msg = this.chat.messages[assistantIdx];
                  msg.content += content;
                  // Combined reasoning+content chunks are unusual; count
                  // their token ids once, against the reasoning phase.
                  if (!reasoning) msg.tokens += tokenCount;
                  msg.elapsedMs = now - started;
                  msg.streamMs = now - firstOutputAt;
                  this.scrollChat();
                }
                // The final usage chunk is authoritative and repairs totals
                // for older upstreams that cannot return token_ids.
                const usageTokens = Number(obj.usage?.completion_tokens);
                if (Number.isFinite(usageTokens) && usageTokens >= 0) {
                  this.chat.messages[assistantIdx].totalTokens = usageTokens;
                }
              }
            }
          }
        }
        this.chat.messages[assistantIdx].status = 'done';
      } catch (e) {
        if (e.name === 'AbortError') {
          this.chat.messages[assistantIdx].status = 'stopped';
        } else {
          this.chat.messages[assistantIdx].error = e.message;
          this.chat.messages[assistantIdx].status = 'error';
        }
      } finally {
        this.chat.messages[assistantIdx].elapsedMs = performance.now() - started;
        this.chat.busy = false;
        this.chat.abort = null;
        this.scrollChat();
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

    // ─── ollama ──────────────────────────────────────────────
    async saveOllamaUrl() {
      const url = (this.ollamaForm.base_url || '').trim();
      if (!url) return;
      try {
        const r = await fetch('/api/settings', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ ollama_base_url: url }),
        });
        if (!r.ok) throw new Error(await r.text());
        this.ollamaForm.base_url = '';
        await this.refresh();
      } catch (e) {
        alert('Save failed: ' + e.message);
      }
    },

    async pullOllama() {
      const name = (this.ollamaPull.name || '').trim();
      if (!name) return;
      this.ollamaPull.busy = true;
      this.ollamaPull.lines = [`pulling ${name}…`];
      try {
        const r = await fetch('/api/ollama/pull', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ name }),
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
              if (obj.error) this.ollamaPull.lines.push(`ERROR: ${obj.error}`);
              else if (obj.done) this.ollamaPull.lines.push('done.');
              else if (obj.status) {
                let msg = obj.status;
                if (obj.completed && obj.total) {
                  const pct = ((obj.completed / obj.total) * 100).toFixed(1);
                  msg += ` ${pct}%`;
                }
                this.ollamaPull.lines.push(msg);
              }
            } catch { }
          }
        }
        await this.refresh();
      } catch (e) {
        this.ollamaPull.lines.push(`error: ${e.message}`);
      } finally {
        this.ollamaPull.busy = false;
      }
    },

    async deleteOllama(name) {
      if (!name) return;
      if (!confirm(`Delete Ollama model ${name}?`)) return;
      try {
        const r = await fetch(`/api/ollama/models/${encodeURIComponent(name)}`, {
          method: 'DELETE',
        });
        if (!r.ok) throw new Error(await r.text());
        if (this.ollamaSelected === name) this.ollamaSelected = '';
        await this.refresh();
      } catch (e) {
        alert('Delete failed: ' + e.message);
      }
    },

    // ─── unsloth ────────────────────────────────────────────
    unslothModels() {
      // Exclude entries with no id (defensive — the list filter already drops them server-side).
      return (this.state?.unsloth?.models || []).filter(m => m && m.id);
    },

    isUnslothLoaded(id) {
      return id && this.state?.unsloth?.loaded_model === id;
    },

    // Effective settings for a model: saved values (from server) over defaults.
    unslothSettingsFor(id) {
      const saved = this.state?.unsloth?.settings?.[id] || {};
      return {
        cache_type_kv: 'bf16',
        max_seq_length: 0,
        gguf_variant: 'Q8_0',
        parallel: 4,
        kv_unified: false,
        load_in_4bit: false,
        tensor_parallel: false,
        tensor_parallel_size: 2,
        split_mode: 'tensor',
        trust_remote_code: false,
        mtp_enabled: false,
        dspark_enabled: false,
        mtp_predict_tokens: 3,
        speculative_type: 'auto',
        spec_draft_n_max: null,
        input_cost_per_1m: 0.0,
        output_cost_per_1m: 0.0,
        cache_cost_per_1m: 0.0,
        ...saved,
      };
    },

    // Working copy of settings for the form. Initialized lazily when a drawer
    // opens so edits don't mutate the canonical state object.
    unslothFormFor(id) {
      if (!this.unslothForm[id]) {
        // Deep-ish copy: spec_draft_n_max may be null.
        this.unslothForm[id] = { ...this.unslothSettingsFor(id) };
      }
      return this.unslothForm[id];
    },

    toggleUnslothSettings(id) {
      // Opening (re)seeds the form from the latest saved settings so a
      // stale draft doesn't overwrite newer saves from another tab.
      if (!this.unslothSettingsOpen[id]) {
        this.unslothForm[id] = { ...this.unslothSettingsFor(id) };
        this.fetchUnslothVariants(id);
      }
      this.unslothSettingsOpen[id] = !this.unslothSettingsOpen[id];
    },

    // Native <details> toggles (clicking the summary) bypass
    // toggleUnslothSettings, so seed/fetch here too.
    onUnslothSettingsToggle(id, open) {
      if (open && !this.unslothSettingsOpen[id]) {
        this.unslothForm[id] = { ...this.unslothSettingsFor(id) };
        this.fetchUnslothVariants(id);
      }
      this.unslothSettingsOpen[id] = open;
    },

    // Downloaded GGUF quant variants for the dropdown, fetched from the
    // controller (which scans the local HF cache). Refetched each
    // time the settings drawer opens so fresh downloads show up.
    async fetchUnslothVariants(id) {
      if (!id || this.unslothVariantsBusy[id]) return;
      this.unslothVariantsBusy[id] = true;
      try {
        const r = await fetch('/api/unsloth/gguf-variants?model_path=' + encodeURIComponent(id));
        if (!r.ok) throw new Error(r.statusText);
        const data = await r.json();
        this.unslothVariants[id] = data.variants || [];
      } catch (e) {
        // Leave any previously fetched list in place; the dropdown falls
        // back to just the current setting.
        console.warn('gguf-variants fetch failed for', id, e);
      } finally {
        this.unslothVariantsBusy[id] = false;
      }
    },

    // Options for the variant dropdown. Always includes the form's current
    // value so a saved setting is never silently dropped, even if the fetch
    // failed or the file was deleted.
    unslothVariantOptions(id) {
      const list = [...(this.unslothVariants[id] || [])];
      const cur = this.unslothFormFor(id).gguf_variant;
      if (cur && !list.some(v => v.quant === cur)) {
        list.unshift({ quant: cur, size_bytes: null });
      }
      return list;
    },

    async saveUnslothSettings(id) {
      const form = this.unslothForm[id] || this.unslothSettingsFor(id);
      try {
        const r = await fetch('/api/unsloth/settings', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model_path: id, settings: form }),
        });
        if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
        this.unslothSettingsSaved[id] = true;
        setTimeout(() => { this.unslothSettingsSaved[id] = false; }, 2500);
        await this.refresh();
      } catch (e) {
        alert('Save settings failed: ' + e.message);
      }
    },

    async loadUnsloth(id) {
      if (!id || this.isUnslothLoaded(id)) return;
      this.unslothBusy[id] = true;
      try {
        // Launch with the current form values; the server persists them on
        // success so the next load defaults to the last launched settings.
        const settings = { ...this.unslothSettingsFor(id), ...(this.unslothForm[id] || {}) };
        const r = await fetch('/api/unsloth/load', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model_path: id, settings }),
        });
        if (!r.ok) {
          const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
          throw new Error(detail);
        }
        await this.refresh();
      } catch (e) {
        alert(`Load failed: ${e.message}\n\n(Load can take several minutes for large models; the request blocks until the model is ready.)`);
      } finally {
        this.unslothBusy[id] = false;
      }
    },

    async unloadUnsloth(id) {
      if (!id || !this.isUnslothLoaded(id)) return;
      this.unslothStopping[id] = true;
      try {
        const r = await fetch('/api/unsloth/unload', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model_path: id }),
        });
        if (!r.ok) {
          const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
          throw new Error(detail);
        }
        await this.refresh();
      } catch (e) {
        alert('Unload failed: ' + e.message);
      } finally {
        this.unslothStopping[id] = false;
      }
    },

    async cancelUnslothLoad() {
      try {
        const r = await fetch('/api/unsloth/load/cancel', { method: 'POST' });
        if (!r.ok) throw new Error(r.statusText);
        await this.refresh();
      } catch (e) {
        alert('Cancel failed: ' + e.message);
      }
    },

    // ─── saved SparkRun targets ──────────────────────────────
    openSparkLaunchEditor() {
      this.sparkEditor.id = null;
      this.sparkForm = { reference: '' };
      this.sparkSaveError = '';
      this.sparkEditor.open = true;
    },

    async saveSparkLaunch() {
      const reference = this.sparkForm.reference.trim();
      if (!reference || this.sparkSaving) return;
      this.sparkSaving = true;
      this.sparkSaveError = '';
      try {
        const r = await fetch('/api/spark-launches', {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ reference }),
        });
        const reply = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(reply.detail || r.statusText);
        this.sparkSaved = true;
        setTimeout(() => { this.sparkSaved = false; }, 2500);
        this.sparkEditor.open = false;
        await this.refresh();
      } catch (e) {
        this.sparkSaveError = e.message || 'SparkRun could not be saved';
      } finally {
        this.sparkSaving = false;
      }
    },

    async refreshSparkLaunch(id) {
      try {
        const r = await fetch(`/api/spark-launches/${id}/refresh`, { method: 'POST' });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
        const launch = await r.json();
        if (launch.resolve_error) alert('Details could not be refreshed: ' + launch.resolve_error);
        await this.refresh();
      } catch (e) {
        alert('Refresh failed: ' + e.message);
      }
    },

    async deleteSparkLaunch(id) {
      if (!confirm('Delete this saved SparkRun?')) return;
      try {
        const r = await fetch(`/api/spark-launches/${id}`, { method: 'DELETE' });
        if (!r.ok) throw new Error(await r.text());
        await this.refresh();
      } catch (e) {
        alert('Delete failed: ' + e.message);
      }
    },

    async copySparkReference(launch) {
      try {
        await navigator.clipboard.writeText(`sparkrun run ${launch.reference}`);
      } catch (e) {
        alert('Copy failed: ' + e.message);
      }
    },

    sparkResolvedCommand(command, defaults) {
      let resolved = String(command || '');
      for (const [key, value] of Object.entries(defaults || {})) {
        const rendered = typeof value === 'string' ? value : JSON.stringify(value);
        resolved = resolved.split(`{${key}}`).join(rendered);
      }
      // SparkRun command templates use doubled braces for literal JSON.
      return resolved.replaceAll('{{', '{').replaceAll('}}', '}');
    },

    openSparkLaunchRun(launch) {
      const metadata = launch.metadata || {};
      const defaults = metadata.defaults || {};
      this.sparkLaunchRun = {
        open: true, id: launch.id, name: launch.name || launch.reference, reference: launch.reference,
        optionsText: '',
        resolvedCommand: this.sparkResolvedCommand(metadata.command, defaults),
        recipeDefaults: Object.entries(defaults),
        recipeEnv: Object.entries(metadata.env || {}),
        recipeMods: metadata.mods || [],
        overrides: {
          parallel_streams: defaults.max_num_seqs ?? null,
          tensor_parallel: defaults.tensor_parallel ?? null,
          gpu_memory_utilization: defaults.gpu_memory_utilization ?? null,
          max_model_len: defaults.max_model_len ?? null,
          port: defaults.port ?? null,
          served_model_name: defaults.served_model_name || '',
        },
      };
    },

    async runSparkLaunch() {
      const launch = this.sparkLaunchRun;
      launch.open = false;
      this.sparkRun = {
        open: true, recipeId: launch.id, recipeName: launch.name, lines: [],
        status: 'starting', busy: true, runId: null, abort: null,
      };
      const overrides = { ...launch.overrides,
        options: launch.optionsText.split('\n').map(v => v.trim()).filter(Boolean) };
      this.scrollSparkRun();
      try {
        const r = await fetch(`/api/spark-launches/${launch.id}/run`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ solo: true, overrides }),
        });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
        const reader = r.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const parts = buf.split('\n\n');
          buf = parts.pop();
          for (const part of parts) {
            try {
              const event = JSON.parse(part.replace(/^data: /, '').trim());
              if (event.run_id) this.sparkRun.runId = event.run_id;
              if (event.line) this.sparkRun.lines.push(event.line);
              if (event.cmd) this.sparkRun.lines.push('$ ' + event.cmd);
              if (event.status) this.sparkRun.status = event.status;
              if (event.error) { this.sparkRun.lines.push('ERROR: ' + event.error); this.sparkRun.status = 'error'; }
              this.scrollSparkRun();
            } catch { }
          }
        }
        if (!['error', 'canceled'].includes(this.sparkRun.status)) this.sparkRun.status = 'done';
      } catch (e) {
        this.sparkRun.lines.push('ERROR: ' + e.message);
        this.sparkRun.status = 'error';
      } finally {
        this.sparkRun.busy = false;
        await this.refresh();
      }
    },

    scrollSparkRun() {
      setTimeout(() => {
        const el = this.$refs?.sparkRunLog;
        if (el) el.scrollTop = el.scrollHeight;
      }, 0);
    },

    async stopSparkRun() {
      if (!this.sparkRun.runId) return;
      try { await fetch(`/api/spark-runs/${this.sparkRun.runId}/cancel`, { method: 'POST' }); } catch { }
    },

    async viewSparkRunLog(run) {
      if (!run) return;
      this.sparkRunHistory = {
        open: true, runId: run.id, recipeName: run.recipe_name || '',
        lines: [], status: run.status || '', autoScroll: true,
      };
      this._startSparkHistoryTail();
    },

    async refreshSparkRunHistory() {
      const runId = this.sparkRunHistory.runId;
      if (!this.sparkRunHistory.open || !runId || this._sparkHistoryRefreshInFlight) return;
      this._sparkHistoryRefreshInFlight = true;
      try {
        const r = await fetch(`/api/spark-runs/${runId}/logs`);
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
        const data = await r.json();
        if (!this.sparkRunHistory.open || this.sparkRunHistory.runId !== runId) return;
        this.sparkRunHistory.lines = data.lines || [];
        const run = (this.state?.spark_runs || []).find((item) => item.id === runId);
        if (run?.status) this.sparkRunHistory.status = run.status;
        this._scrollSparkHistoryToBottom();
      } catch (e) {
        if (this.sparkRunHistory.open && this.sparkRunHistory.runId === runId) {
          this.sparkRunHistory.lines = [`Failed to load logs: ${e.message}`];
        }
      } finally {
        this._sparkHistoryRefreshInFlight = false;
      }
    },

    _startSparkHistoryTail() {
      this._stopSparkHistoryTail();
      this.refreshSparkRunHistory();
      this._sparkHistoryTailInterval = setInterval(() => {
        if (this.sparkRunHistory.open && !document.hidden) this.refreshSparkRunHistory();
      }, 1000);
    },

    _stopSparkHistoryTail() {
      if (this._sparkHistoryTailInterval) {
        clearInterval(this._sparkHistoryTailInterval);
        this._sparkHistoryTailInterval = null;
      }
    },

    _scrollSparkHistoryToBottom() {
      if (!this.sparkRunHistory.autoScroll) return;
      setTimeout(() => {
        const element = this.$refs?.sparkRunHistoryLog;
        if (element) element.scrollTop = element.scrollHeight;
      }, 0);
    },

    openSparkRecipeRuns(launch) {
      if (!launch) return;
      this.sparkRecipeRuns = { open: true, recipeId: launch.id, recipeName: launch.name || '',
        runs: (this.state?.spark_runs || []).filter(r => r.recipe_id === launch.id) };
    },

    /* Removed YAML-recipe editor implementation. Saved SparkRuns use only
       runnable references and the endpoints above. */
    /*
    // Build a YAML string from the structured form fields. Kept simple on
    // purpose: round-trips through the backend validator, and the raw-YAML
    // textarea remains authoritative — users can hand-edit anything we omit.
    sparkEmptyYaml() {
      return `model: ""
runtime: vllm
container: ""
command: "{container} --model {model}"
defaults: {}
min_nodes: 1
max_nodes: 1
metadata:
  description: ""
  author: ""
  tags: []
`;
    },

    openSparkEditor(recipe) {
      this.sparkEditor.id = recipe?.id || null;
      if (recipe) {
        this.sparkForm.name = recipe.name || '';
        this.sparkForm.yaml = recipe.yaml || this.sparkEmptyYaml();
        const m = recipe.metadata || {};
        this.sparkForm.model = m.model || '';
        this.sparkForm.runtime = m.runtime || 'vllm';
        this.sparkForm.container = m.container || '';
        this.sparkForm.min_nodes = m.min_nodes || 1;
        this.sparkForm.max_nodes = m.max_nodes || 1;
        this.sparkForm.description = m.description || '';
        this.sparkForm.author = m.author || '';
        this.sparkForm.recipe_version = m.recipe_version || '';
        this.sparkForm.cluster_only = m.cluster_only != null ? m.cluster_only : null;
        this.sparkForm.spark_arena_id = m.spark_arena_id || '';
      } else {
        this.sparkForm = {
          name: '', model: '', runtime: 'vllm', container: '',
          min_nodes: 1, max_nodes: 1, description: '', author: '',
          recipe_version: '', cluster_only: null, spark_arena_id: '', yaml: this.sparkEmptyYaml(),
        };
      }
      this.sparkYamlError = '';
      this.sparkEditor.open = true;
      this.sparkValidateYaml();
    },

    // Regenerate YAML from the structured form fields. We merge edits into
    // the existing YAML so that pasted Spark Arena recipes keep their extra
    // fields (mods, env, recipe_version, cluster_only, etc.).
    sparkSyncFormToYaml() {
      const f = this.sparkForm;
      let y = this.sparkForm.yaml || this.sparkEmptyYaml();

      function yamlScalar(s) {
        if (s == null || s === '') return '""';
        const str = String(s);
        // Quote pure numbers and YAML reserved words so they round-trip as strings.
        if (/^(true|false|null|~|yes|no|on|off)$/i.test(str)) return `"${str}"`;
        if (/^\d+(\.\d+)?$/.test(str)) return `"${str}"`;
        // Plain scalar is safe for these unquoted characters.
        if (/^[A-Za-z0-9_.~\/:@+-]+$/.test(str)) return str;
        // Double-quoted scalar with basic YAML escapes.
        return '"' + str.replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\n/g, '\\n') + '"';
      }

      function setTopLevel(key, val) {
        const re = new RegExp(`^(${key}\\s*:\\s*).*$`, 'm');
        if (re.test(y)) {
          y = y.replace(re, `$1${val}`);
        } else {
          y = y.trimEnd() + `\n${key}: ${val}\n`;
        }
      }

      function setMetadataField(key, val) {
        // Prefer an existing metadata.key line (standard sparkrun location).
        const indentedRe = new RegExp(`^([ \\t]+${key}\\s*:\\s*).*$`, 'm');
        if (indentedRe.test(y)) {
          y = y.replace(indentedRe, `$1${val}`);
          return;
        }
        // Spark Arena recipes sometimes place description/author at the top level.
        const topRe = new RegExp(`^(${key}\\s*:\\s*).*$`, 'm');
        if (topRe.test(y)) {
          y = y.replace(topRe, `$1${val}`);
          return;
        }
        // Otherwise add the field under the metadata block.
        const blockRe = /^metadata:\s*$/m;
        if (blockRe.test(y)) {
          y = y.replace(blockRe, `metadata:\n  ${key}: ${val}`);
        } else {
          y = y.trimEnd() + `\nmetadata:\n  ${key}: ${val}\n`;
        }
      }

      setTopLevel('model', yamlScalar(f.model));
      if (f.runtime) setTopLevel('runtime', yamlScalar(f.runtime));
      setTopLevel('container', yamlScalar(f.container));
      setTopLevel('min_nodes', f.min_nodes || 1);
      setTopLevel('max_nodes', f.max_nodes || 1);
      if (f.recipe_version) setTopLevel('recipe_version', yamlScalar(f.recipe_version));
      if (f.cluster_only != null) setTopLevel('cluster_only', f.cluster_only ? 'true' : 'false');
      setMetadataField('description', yamlScalar(f.description || ''));
      setMetadataField('author', yamlScalar(f.author || ''));
      if (f.spark_arena_id?.trim()) {
        setMetadataField('spark_arena_id', yamlScalar(f.spark_arena_id.trim()));
      } else {
        // Remove the field from both locations when cleared.
        y = y.replace(/^[ \t]*spark_arena_id\s*:\s*.*\n?/gm, '');
        y = y.replace(/^[ \t]+spark_arena_id\s*:\s*.*\n?/gm, '');
      }

      this.sparkForm.yaml = y;
      this.sparkValidateYaml();
    },

    async sparkValidateYaml() {
      try {
        const r = await fetch('/api/spark-recipes/validate', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ yaml: this.sparkForm.yaml }),
        });
        const j = await r.json();
        if (j.ok === false) {
          this.sparkYamlError = (j.error || 'parse error').slice(0, 200);
        } else {
          this.sparkYamlError = '';
          // Backfill form fields from the parsed YAML so they stay in sync
          // when the user edits YAML directly.
          this.sparkForm.model = j.model || this.sparkForm.model;
          this.sparkForm.runtime = j.runtime || this.sparkForm.runtime;
          this.sparkForm.container = j.container || this.sparkForm.container;
          this.sparkForm.min_nodes = j.min_nodes || this.sparkForm.min_nodes;
          this.sparkForm.max_nodes = j.max_nodes || this.sparkForm.max_nodes;
          this.sparkForm.description = j.description || this.sparkForm.description;
          this.sparkForm.author = j.author || this.sparkForm.author;
          this.sparkForm.recipe_version = j.recipe_version || this.sparkForm.recipe_version;
          this.sparkForm.cluster_only = j.cluster_only != null ? j.cluster_only : this.sparkForm.cluster_only;
          this.sparkForm.spark_arena_id = j.spark_arena_id || this.sparkForm.spark_arena_id;
          // If the recipe carries its own name and the user hasn't typed one, use it.
          if (!this.sparkForm.name && j.name) this.sparkForm.name = j.name;
        }
      } catch (e) {
        // Offline / transient — don't block editing.
        this.sparkYamlError = '';
      }
    },

    async saveSparkRecipe() {
      if (!this.sparkForm.name || !this.sparkForm.yaml.trim()) return;
      // Final validation gate.
      if (this.sparkYamlError) {
        if (!confirm('YAML has a parse error. Save anyway? (It will be stored but flagged.)')) return;
      }
      try {
        const body = { name: this.sparkForm.name, yaml: this.sparkForm.yaml };
        const method = this.sparkEditor.id ? 'PUT' : 'POST';
        const url = this.sparkEditor.id
          ? `/api/spark-recipes/${this.sparkEditor.id}`
          : '/api/spark-recipes';
        const r = await fetch(url, {
          method, headers: { 'content-type': 'application/json' }, body: JSON.stringify(body),
        });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
        this.sparkSaved = true;
        setTimeout(() => { this.sparkSaved = false; }, 2500);
        this.sparkEditor.open = false;
        await this.refresh();
      } catch (e) {
        alert('Save failed: ' + e.message);
      }
    },

    async deleteSparkRecipe(id) {
      if (!confirm('Delete this Spark Run recipe?')) return;
      try {
        const r = await fetch(`/api/spark-recipes/${id}`, { method: 'DELETE' });
        if (!r.ok) throw new Error(await r.text());
        await this.refresh();
      } catch (e) {
        alert('Delete failed: ' + e.message);
      }
    },

    async copySparkYaml(r) {
      try {
        await navigator.clipboard.writeText(r.yaml || '');
      } catch (e) {
        alert('Copy failed: ' + e.message);
      }
    },

    openSparkDownload(r) {
      this.sparkDownload = { open: true, recipe: r, copied: false };
    },

    sparkDownloadUrl() {
      const r = this.sparkDownload.recipe;
      if (!r) return '';
      const arenaId = r.metadata?.spark_arena_id;
      if (arenaId) {
        return `https://spark-arena.com/api/recipes/${arenaId}/raw`;
      }
      return `${window.location.origin}/api/spark-recipes/${r.id}/raw`;
    },

    async copySparkDownloadUrl() {
      const url = this.sparkDownloadUrl();
      if (!url) return;
      try {
        await navigator.clipboard.writeText(url);
        this.sparkDownload.copied = true;
        setTimeout(() => { this.sparkDownload.copied = false; }, 2000);
      } catch (e) {
        alert('Copy failed: ' + e.message);
      }
    },

    scrollSparkRun() {
      setTimeout(() => {
        const el = this.$refs?.sparkRunLog;
        if (el) el.scrollTop = el.scrollHeight;
      }, 0);
    },

    async runSparkRecipe(id) {
      const recipe = (this.state?.spark_recipes || []).find(r => r.id === id);
      if (!recipe) return;
      this.sparkRun = {
        open: true, recipeId: id, recipeName: recipe.name, lines: [],
        status: 'starting', busy: true, runId: null, abort: null,
      };
      this.scrollSparkRun();
      try {
        const r = await fetch(`/api/spark-recipes/${id}/run`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ solo: true }),
        });
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
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
            const line = p.replace(/^data: /, '').trim();
            if (!line) continue;
            try {
              const obj = JSON.parse(line);
              if (obj.line) {
                this.sparkRun.lines.push(obj.line);
                this.scrollSparkRun();
              } else if (obj.status) {
                this.sparkRun.status = obj.status;
                if (obj.cmd) this.sparkRun.lines.push('$ ' + obj.cmd);
                this.scrollSparkRun();
              } else if (obj.run_id) {
                this.sparkRun.runId = obj.run_id;
              } else if (obj.error) {
                this.sparkRun.lines.push('ERROR: ' + obj.error);
                this.sparkRun.status = 'error';
                this.scrollSparkRun();
              } else if (obj.done) {
                // finalize below
              }
            } catch { }
          }
        }
        if (this.sparkRun.status !== 'error' && this.sparkRun.status !== 'canceled') {
          this.sparkRun.status = 'done';
        }
      } catch (e) {
        this.sparkRun.lines.push('ERROR: ' + e.message);
        this.sparkRun.status = 'error';
      } finally {
        this.sparkRun.busy = false;
        await this.refresh();
      }
    },

    async stopSparkRun() {
      if (!this.sparkRun.runId) return;
      try {
        await fetch(`/api/spark-runs/${this.sparkRun.runId}/cancel`, { method: 'POST' });
      } catch (e) {
        // The stream will reflect cancellation; non-fatal.
      }
    },

    async viewSparkRunLog(run) {
      if (!run) return;
      this.sparkRunHistory = {
        open: true, runId: run.id, recipeName: run.recipe_name || '',
        lines: [], status: run.status || '',
      };
      try {
        const r = await fetch(`/api/spark-runs/${run.id}/logs`);
        if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
        const j = await r.json();
        this.sparkRunHistory.lines = j.lines || [];
      } catch (e) {
        this.sparkRunHistory.lines = [`Failed to load logs: ${e.message}`];
      }
    },

    openSparkRecipeRuns(recipe) {
      if (!recipe) return;
      this.sparkRecipeRuns = {
        open: true,
        recipeId: recipe.id,
        recipeName: recipe.name || '',
        runs: (this.state?.spark_runs || []).filter(r => r.recipe_id === recipe.id),
      };
    },

    */
    // ─── A/B compare ─────────────────────────────────────────
    isLocal(model) {
      return !!model && !model.startsWith('CLOUD ');
    },

    localModels() {
      const out = new Set();
      const loadedLlama = this.state?.unsloth?.loaded_model;
      if (loadedLlama) out.add(loadedLlama);
      for (const c of (this.state?.containers || [])) {
        if (c.model) out.add(c.model);
      }
      for (const model of Object.keys(this.state?.sparkrun_targets || {})) {
        out.add(model);
      }
      return Array.from(out).sort((a, b) => {
        if (a === loadedLlama) return -1;
        if (b === loadedLlama) return 1;
        return a.localeCompare(b);
      });
    },

    cloudModels() {
      return (this.state?.ollama?.models || [])
        .map(m => `CLOUD ${m.name}`)
        .sort();
    },

    runAB() {
      this.ab.warning = '';
      if (!this.ab.prompt || !this.ab.modelA || !this.ab.modelB) return;
      if (this.isLocal(this.ab.modelA) && this.isLocal(this.ab.modelB) && this.maxConcurrentLocalModels() < 2) {
        this.ab.warning = 'Pick at most one local model, or increase Settings → Max concurrent models.';
        return;
      }
      // Reset panels
      for (const k of ['panelA', 'panelB']) {
        this.ab[k] = { text: '', reasoning: '', error: '', status: 'connecting…', tokens: 0, elapsedMs: 0, streamMs: 0, busy: true, abort: null };
      }
      this.ab.busy = true;
      // Fire both in parallel; track completion to flip ab.busy off.
      Promise.allSettled([
        this.streamAB(this.ab.modelA, this.ab.panelA),
        this.streamAB(this.ab.modelB, this.ab.panelB),
      ]).then(() => { this.ab.busy = false; });
    },

    stopAB(panel) {
      if (panel?.abort) {
        try { panel.abort.abort(); } catch { }
      }
    },

    stopABBoth() {
      this.stopAB(this.ab.panelA);
      this.stopAB(this.ab.panelB);
    },

    async streamAB(model, panel) {
      const messages = [];
      const sys = (this.ab.system || '').trim() || 'You are a helpful assistant.';
      messages.push({ role: 'system', content: sys });
      messages.push({ role: 'user', content: this.ab.prompt });
      const started = performance.now();
      let firstTokenAt = null;
      const ac = new AbortController();
      panel.abort = ac;
      try {
        const payload = {
          model,
          messages,
          stream: true,
          temperature: this.ab.temperature,
          top_p: this.ab.top_p,
          max_tokens: this.ab.max_tokens,
        };
        if (this.ab.seed !== null && this.ab.seed !== undefined && this.ab.seed !== '') {
          payload.seed = Number(this.ab.seed);
        }
        const r = await fetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify(payload),
          signal: ac.signal,
        });
        if (!r.ok) {
          const text = await r.text();
          throw new Error(`HTTP ${r.status}: ${text.slice(0, 400)}`);
        }
        panel.status = 'streaming';
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
            for (const raw of p.split('\n')) {
              const line = raw.replace(/^data: /, '').trim();
              if (!line || line === '[DONE]') continue;
              let obj;
              try {
                obj = JSON.parse(line);
              } catch {
                continue;
              }
              if (obj.error) {
                const detail = typeof obj.error === 'string'
                  ? obj.error
                  : (obj.error.message || JSON.stringify(obj.error));
                throw new Error(detail);
              }
              {
                const choice = obj.choices?.[0];
                const delta = choice?.delta;
                const content = delta?.content;
                const reasoning = delta?.reasoning_content || delta?.reasoning;
                const tokenIds = Array.isArray(choice?.token_ids) ? choice.token_ids : null;
                const tokenCount = tokenIds?.length || ((reasoning || content) ? 1 : 0);
                if ((reasoning || content) && firstTokenAt === null) {
                  firstTokenAt = performance.now();
                }
                if (reasoning) {
                  panel.reasoning += reasoning;
                }
                if (content) {
                  panel.text += content;
                }
                if (reasoning || content) {
                  panel.tokens += tokenCount;
                  const now = performance.now();
                  panel.elapsedMs = now - started;
                  panel.streamMs = now - firstTokenAt;
                }
                const usageTokens = Number(obj.usage?.completion_tokens);
                if (Number.isFinite(usageTokens)) panel.tokens = usageTokens;
              }
            }
          }
        }
        panel.status = 'done';
        panel.elapsedMs = performance.now() - started;
      } catch (e) {
        if (e.name === 'AbortError') {
          panel.status = 'stopped';
        } else {
          panel.error = e.message;
          panel.status = 'error';
        }
        panel.elapsedMs = performance.now() - started;
      } finally {
        panel.busy = false;
        panel.abort = null;
      }
    },

    // ─── server logs ─────────────────────────────────────────
    async refreshServerLogs() {
      if (this.serverLogs._loading) return;
      this.serverLogs._loading = true;
      try {
        const r = await fetch('/api/server-logs?tail=500');
        if (!r.ok) return;
        const j = await r.json();
        this.serverLogs.lines = j.logs || [];
        if (this.serverLogs.autoScroll) {
          setTimeout(() => {
            const el = this.$refs?.serverLogBox;
            if (el) el.scrollTop = el.scrollHeight;
          }, 0);
        }
      } catch { }
      finally { this.serverLogs._loading = false; }
    },

    _startLogPolling() {
      this._stopLogPolling();
      this.refreshServerLogs();
      this.serverLogs._interval = setInterval(() => {
        if (this.tab === 'logs' && !document.hidden) {
          this.refreshServerLogs();
        }
      }, 1000);
    },

    _stopLogPolling() {
      if (this.serverLogs._interval) {
        clearInterval(this.serverLogs._interval);
        this.serverLogs._interval = null;
      }
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
        this.settingsForm = await r.json();
        this.settingsSaved = true;
        setTimeout(() => { this.settingsSaved = false; }, 2500);
        await this.refresh();
      } catch (e) {
        alert('Save failed: ' + e.message);
      }
    },
  };
}
