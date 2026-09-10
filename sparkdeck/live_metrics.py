"""Rolling live-inference history for the History panel.

SparkDeck already tracks what every in-flight inference request is doing right
now: :meth:`Manager.active_request_groups` classifies each live session as
outputting, thinking, or still prompt processing, and the proxy records the
engine-measured prompt-processing rate when a prefill completes.  The dashboard
renders that as an instantaneous snapshot.

The History panel needs the same numbers as a trailing timeline: one graph per
serving unit (deployment, pair, or sharded group), one bucket every five
seconds for the last hour, each bucket carrying output, thinking, and prompt
processing tokens per second plus the session states behind them.

Three properties drive this design:

* **Buckets must be complete, not instantaneous.**  A five-second point sampled
  once would miss every request that started and finished between samples, so the
  collector ticks once a second and accumulates token counts into the open
  bucket.  Deltas come from the manager's cumulative per-request counters
  (``total_tokens``, ``pp_tokens``) rather than its five-second trailing deques,
  which only ever describe the last few seconds.
* **Concurrency is time-averaged.**  A bucket reports the mean live-session
  count over its samples, so the colour of a segment describes what the serving
  unit actually carried rather than one instant inside it.
* **Prompt processing must be estimated honestly.**  Only a completed prefill
  has an engine-measured rate.  While a prefill is still running the collector
  aggregates the in-flight prompts' only real evidence -- how many prompt tokens
  they hold and how long they have held them -- into one group-level estimate,
  and reports ``null`` (rendered as "pending") when no prompt token count is
  known yet.  No rate is ever invented from an empty measurement.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import deque
from typing import Any, Callable, Mapping

# Trailing window the panel draws, independent of how often it is sampled.
KEEP_SECONDS = 3_600.0
# Sampling cadence, in seconds, chosen by the user between these bounds.  The
# bucket span follows the cadence, so a longer interval means a coarser graph
# over the same trailing hour rather than gaps in it.
MIN_SAMPLE_SECONDS = 1
MAX_SAMPLE_SECONDS = 30
DEFAULT_HISTORY_SAMPLE_SECONDS = 5
DEFAULT_HISTORY_ENABLED = True
# A serving unit that has been silent this long is dropped from the panel.
RETAIN_ACTIVE_SECONDS = 900.0
# Bound on simultaneously tracked serving units; least recently active go first.
MAX_SERIES = 32
# A prefill younger than this has too little evidence to estimate a rate from.
MIN_PREFILL_SECONDS = 0.25

_GROUP_FIELDS = ("group_id", "model", "deployment_id", "instance_id", "node_names")


def clamp_sample_seconds(value: Any) -> int:
    """Bound a requested sampling interval to the supported range."""
    try:
        seconds = int(round(float(value)))
    except (TypeError, ValueError):
        return DEFAULT_HISTORY_SAMPLE_SECONDS
    if seconds < MIN_SAMPLE_SECONDS:
        return MIN_SAMPLE_SECONDS
    return min(seconds, MAX_SAMPLE_SECONDS)


def buckets_for(keep_seconds: float, bucket_seconds: float) -> int:
    """How many buckets cover the trailing window at this cadence."""
    if bucket_seconds <= 0:
        return 1
    return max(2, int(math.ceil(keep_seconds / bucket_seconds)))


class _Series:
    """One serving unit's trailing buckets plus its open bucket state."""

    __slots__ = (
        "meta", "buckets", "open_started_at", "samples",
        "output_tokens", "thinking_tokens", "prompt_tokens", "prompt_seconds",
        "previous", "counted_prompt",
        "pending", "pending_prompt_tokens", "pending_prompt_seconds",
        "concurrency_sum", "concurrency_peak",
        "state_high",
        "output_peak", "thinking_peak", "prefill_peak", "live_prefill_rate",
        "last_state", "last_at",
    )

    def __init__(self, meta: dict[str, Any], now: float) -> None:
        self.meta = meta
        # Sized for the fastest cadence the user can choose, so any trailing hour
        # fits no matter which interval produced it.
        self.buckets: deque[dict[str, Any]] = deque(
            maxlen=buckets_for(KEEP_SECONDS, MIN_SAMPLE_SECONDS),
        )
        self.open_started_at = now
        self.previous: dict[Any, tuple[str, int]] = {}
        self.counted_prompt: dict[Any, int] = {}
        self.pending: dict[str, int] = {"output": 0, "thinking": 0}
        self.pending_prompt_tokens = 0
        self.pending_prompt_seconds = 0.0
        self.last_state: tuple[int, int, int] | None = None
        self.last_at = now
        self.reset_open()

    def flush_pending(self) -> None:
        """Fold counters captured from finished requests into the open bucket."""
        self.output_tokens += self.pending["output"]
        self.thinking_tokens += self.pending["thinking"]
        # A measured prefill reports tokens/second for the serving unit, so
        # concurrent prefills add their tokens while only the longest wall time
        # is divided out.
        self.prompt_tokens += self.pending_prompt_tokens
        self.prompt_seconds = max(self.prompt_seconds, self.pending_prompt_seconds)
        self.pending = {"output": 0, "thinking": 0}
        self.pending_prompt_tokens = 0
        self.pending_prompt_seconds = 0.0

    def reset_open(self) -> None:
        self.samples = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        self.prompt_tokens = 0
        self.prompt_seconds = 0.0
        self.concurrency_sum = 0.0
        self.concurrency_peak = 0
        # Per-state counts held at the busiest single observation in the bucket:
        # every session that was live at some point is still represented, and
        # the bucket reports the peak the panel would have shown live.
        self.state_high = (0, 0, 0)
        self.output_peak = 0.0
        self.thinking_peak = 0.0
        self.prefill_peak = 0.0
        self.live_prefill_rate = 0.0
        # ``previous`` deliberately survives a bucket rollover, so it is not
        # reset here: the tokens the first sample of a new bucket sees were
        # emitted before that bucket opened and belong to the one just closed.


class LiveHistory:
    """Records trailing per-serving-unit throughput history from the manager.

    The object is a read-only observer: the manager keeps owning request
    tracking, and this collector only reads the same records the dashboard
    reads.  :meth:`tick` is called by :meth:`sampler` at the configured cadence
    and is also safe to call directly from tests.

    Recording can be switched off, in which case every hook and tick returns
    immediately so a disabled History panel costs nothing at all.
    """

    def __init__(
        self,
        manager: Any,
        *,
        interval_provider: Callable[[], Any] | None = None,
        enabled_provider: Callable[[], Any] | None = None,
        keep_seconds: float = KEEP_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.manager = manager
        self.clock = clock
        self.keep_seconds = float(keep_seconds)
        # Both settings are read live on every tick: the operator can change the
        # cadence or switch recording off without restarting the controller.
        self._interval_provider = interval_provider
        self._enabled_provider = enabled_provider
        self._series: dict[str, _Series] = {}
        self._live: dict[str, int] = {}
        self._task: asyncio.Task | None = None

    # ----- settings ----------------------------------------------------
    @property
    def enabled(self) -> bool:
        if self._enabled_provider is None:
            return True
        try:
            return bool(self._enabled_provider())
        except Exception:
            # A failed settings read must not silently stop recording.
            return True

    @property
    def interval_seconds(self) -> int:
        """Sampling cadence, which is also the span of one bucket."""
        if self._interval_provider is None:
            return DEFAULT_HISTORY_SAMPLE_SECONDS
        try:
            return clamp_sample_seconds(self._interval_provider())
        except Exception:
            return DEFAULT_HISTORY_SAMPLE_SECONDS

    def forget(self) -> None:
        """Drop every recorded bucket, e.g. once recording is switched off."""
        self._series.clear()
        self._live.clear()

    # ----- event hooks -------------------------------------------------
    def start(
        self, group: Mapping[str, Any] | None = None,
        rec: Mapping[str, Any] | None = None,
    ) -> None:
        """Count a newly admitted inference request.

        ``rec`` is accepted and ignored so every lifecycle hook shares one
        signature: only :meth:`end` needs the request record, and the manager
        forwards the record it already has to whichever event it reports.
        """
        if not self.enabled:
            return
        key = _series_key(group) if group else ""
        if key:
            self._live[key] = self._live.get(key, 0) + 1

    def end(
        self, group: Mapping[str, Any] | None = None,
        rec: Mapping[str, Any] | None = None,
    ) -> None:
        """Release a request counted by :meth:`start`.

        A request that finishes between two samples still emitted tokens, and
        its record disappears from the manager before the next tick can read it.
        Its final cumulative counters are captured here and folded into the
        bucket that was open when it ended.
        """
        if not self.enabled:
            return
        key = _series_key(group) if group else ""
        if not key:
            return
        remaining = self._live.get(key, 0) - 1
        if remaining > 0:
            self._live[key] = remaining
        else:
            self._live.pop(key, None)
        if rec is None:
            return
        series = self._series.get(key)
        if series is None:
            return
        # Requests waiting for a FIFO slot are not running work.
        if rec.get("paused"):
            return
        state = _session_state(rec)
        series.flush_pending()
        tokens = int(rec.get("total_tokens") or 0)
        previous = series.previous.pop(id(rec), None)
        # A session that ends while still prefilling has no token total of its
        # own to attribute; only output and thinking are accumulated here.
        if previous is not None and state in series.pending:
            series.pending[state] += max(0, tokens - previous[1])
        prompt_tokens = int(rec.get("pp_tokens") or 0)
        prompt_seconds = float(rec.get("pp_time_s") or 0.0)
        if prompt_tokens > 0 and prompt_seconds > 0:
            series.pending_prompt_tokens += prompt_tokens
            series.pending_prompt_seconds = max(
                series.pending_prompt_seconds, prompt_seconds,
            )
        # Fold the captured counters in right away: nothing samples again until
        # the next tick, and the evidence belongs to the bucket open now.
        series.flush_pending()

    def live_keys(self) -> tuple[str, ...]:
        """Serving units with at least one live request right now."""
        return tuple(self._live)

    # ----- sampling ----------------------------------------------------
    def tick(self, now: float | None = None) -> None:
        """Fold the current instant into every open series and its bucket."""
        if not self.enabled:
            return
        now = self.clock() if now is None else now
        observed: dict[str, dict[str, Any]] = {}
        admission = self._admission()
        live_records: dict[str, set[Any]] = {}
        for rec in list(getattr(self.manager, "_active_reqs", {}).values()):
            if rec.get("paused"):
                # A replay-paused stream released its slot and is waiting in the
                # FIFO; the dashboard does not report it as running either.
                continue
            group = rec.get("group") or self.manager._request_group(rec.get("key"))
            key = _series_key(group)
            if not key:
                continue
            entry = observed.get(key) or self._new_entry(observed, key, group)
            entry["records"].append((id(rec), rec))
            live_records.setdefault(key, set()).add(id(rec))
            state = _session_state(rec)
            entry["states"][_STATE_FIELDS[state]] += 1
            if state == "prefill":
                started = rec.get("started_at")
                held = max(0.0, now - float(started)) if started is not None else 0.0
                entry["prefill_oldest"] = max(entry["prefill_oldest"], held)
            if int(rec.get("pp_tokens") or 0) > 0 and float(rec.get("pp_time_s") or 0) > 0:
                # Aggregating per record id lets the sample fold add a measurement
                # exactly once: the record keeps reporting the same prompt token
                # count on every later sample.
                entry["prompt_records"][id(rec)] = (
                    int(rec["pp_tokens"]), float(rec["pp_time_s"]),
                )
        # Admission owns the authoritative running count from slot grant to
        # release, so it also covers work with no tracked token timestamps.  The
        # dashboard raises its own count the same way rather than adding, and so
        # does this collector: the states are informational, the total is not
        # allowed to overcount the same sessions twice.
        for key, running in self._admission_running(admission).items():
            entry = observed.get(key)
            if entry is None:
                meta = self._admission_meta(admission, key)
                if meta is None:
                    continue
                entry = self._new_entry(observed, key, meta)
            tracked = sum(entry["states"])
            entry["states"][0] += max(0, running - tracked)

        for key, entry in observed.items():
            series = self._series.get(key)
            if series is None:
                series = self._series[key] = _Series(entry["meta"], now)
            else:
                series.meta.update(entry["meta"])
            series.last_at = now
            # Drop baselines for requests that are no longer tracked, so a
            # recycled record id can never inherit another request's counters.
            series.previous = {
                rid: value for rid, value in series.previous.items()
                if rid in live_records.get(key, ())
            }
            self._accumulate(series, entry, now)
        for key, series in self._series.items():
            if key not in observed:
                # A serving unit with no live request still needs its clock
                # advanced so a zero-traffic stretch is drawn as zero rather
                # than silently skipped.
                series.previous = {}
                self._accumulate(series, None, now)
        self._retire(now)

    def _accumulate(
        self, series: _Series, entry: dict[str, Any] | None, now: float,
    ) -> None:
        # Counters from requests that finished since the last sample belong to
        # the bucket that was open when they ended, so fold them in first.
        series.flush_pending()
        # This sample belongs to the bucket that is open now.  One bucket spans
        # exactly one sampling interval, so the chosen cadence is the graph's
        # time resolution.  The bucket is closed at the very end, after the
        # sample has been folded in, so no measurement is attributed to a bucket
        # that follows it.
        bucket_seconds = float(self.interval_seconds)
        close_at = (
            series.open_started_at + bucket_seconds
            if now - series.open_started_at >= bucket_seconds
            else None
        )
        series.samples += 1
        if entry is not None:
            self._fold_sample(series, entry, now)

        if close_at is not None:
            self._close_bucket(series, close_at)
            # ``previous`` deliberately survives the rollover: the tokens the
            # first sample of the new bucket sees were emitted before it opened
            # and still belong to the bucket that just closed.
            series.open_started_at = now
            series.reset_open()

    def _fold_sample(self, series: _Series, entry: dict[str, Any], now: float) -> None:
        deltas = {"output": 0, "thinking": 0}
        current: dict[Any, tuple[str, int]] = {}
        for rid, rec in entry["records"]:
            state = _session_state(rec)
            tokens = int(rec.get("total_tokens") or 0)
            current[rid] = (state, tokens)
            previous = series.previous.get(rid)
            # ``total_tokens`` is cumulative for the whole request, so a bucket
            # receives exactly the tokens emitted while it was open -- including
            # the final ones from a request that ends mid-bucket, which the end
            # hook captures separately.  A session whose state changed since the
            # last sample (prefill to outputting, or thinking to outputting) has
            # already emitted tokens, so its growth is attributed to the state it
            # is in now rather than dropped.  A request first seen mid-stream has
            # no baseline to compare against, so it contributes nothing here
            # instead of double counting its earlier tokens.
            if previous is None or state not in deltas:
                continue
            deltas[state] += max(0, tokens - previous[1])
        series.previous = current
        series.output_tokens += deltas["output"]
        series.thinking_tokens += deltas["thinking"]
        # Measured prompt processing is recorded once, at the moment the engine
        # reports it, and is never recomputed from a growing elapsed time.  The
        # measured rate is an aggregate tokens/second for the serving unit, so
        # concurrent prefills add their tokens while only the longest wall time
        # is divided out.
        for rid, (tokens, seconds) in entry["prompt_records"].items():
            counted = series.counted_prompt.get(rid)
            if counted is not None and counted >= tokens:
                continue
            series.counted_prompt[rid] = tokens
            series.prompt_tokens += tokens
            series.prompt_seconds = max(series.prompt_seconds, seconds)

        output_states, thinking_states, prefill_states = entry["states"]
        concurrency = output_states + thinking_states + prefill_states
        series.concurrency_sum += concurrency
        series.concurrency_peak = max(series.concurrency_peak, concurrency)
        series.last_state = (output_states, thinking_states, prefill_states)
        # Monotonic high-water marks.  A session can start and finish inside one
        # bucket, and ``state_high`` keeps the busiest session mix the panel
        # would have shown live, while the token totals carry the evidence of
        # work that finished between two samples.
        series.state_high = (
            max(series.state_high[0], output_states),
            max(series.state_high[1], thinking_states),
            max(series.state_high[2], prefill_states),
        )

        # The peak is the highest rate this bucket had reached by the end of any
        # one sample, which is what the panel would have shown while watching
        # live rather than the bucket-wide average.
        elapsed = max(0.001, now - series.open_started_at)
        series.output_peak = max(series.output_peak, series.output_tokens / elapsed)
        series.thinking_peak = max(series.thinking_peak, series.thinking_tokens / elapsed)
        if prefill_states:
            estimate = self._estimate_prefill_rate(entry, series)
            series.live_prefill_rate = estimate
            if estimate:
                series.prefill_peak = max(series.prefill_peak, estimate)

    def _estimate_prefill_rate(
        self, entry: dict[str, Any], series: _Series,
    ) -> float:
        """Best available prompt-processing rate for the held prefills.

        Aggregate the in-flight prompts' prompt token counts over the longest
        prefill currently held -- the same relationship the engine measures when
        a prefill completes.  When no prompt token count is known yet, fall back
        to this serving unit's most recently measured rate so the panel shows a
        labelled estimate instead of a blank while the first prompt runs.
        """
        tokens = 0
        for _, rec in entry["records"]:
            if _session_state(rec) != "prefill":
                continue
            prompt_tokens = int(rec.get("pp_tokens") or 0)
            if prompt_tokens > 0:
                tokens += prompt_tokens
        held = float(entry["prefill_oldest"])
        if tokens > 0 and held >= MIN_PREFILL_SECONDS:
            return tokens / held
        return self._recent_measured_prefill(series)

    @staticmethod
    def _recent_measured_prefill(series: _Series) -> float:
        for bucket in reversed(series.buckets):
            value = bucket.get("prefill_tok_s")
            if bucket.get("prefill_measured") and value:
                return float(value)
        return 0.0

    def _close_bucket(self, series: _Series, closed_at: float) -> None:
        if series.samples == 0:
            return
        elapsed = max(0.001, closed_at - series.open_started_at)
        measured = series.prompt_seconds > 0 and series.prompt_tokens > 0
        series.buckets.append({
            "at": round(closed_at, 3),
            "output_tok_s": round(series.output_tokens / elapsed, 2),
            "thinking_tok_s": round(series.thinking_tokens / elapsed, 2),
            "prefill_tok_s": (
                round(series.prompt_tokens / series.prompt_seconds, 2) if measured
                else (round(series.live_prefill_rate, 2) if series.live_prefill_rate else None)
            ),
            "prefill_measured": measured,
            "concurrent": round(series.concurrency_sum / series.samples, 2),
            "concurrent_peak": series.concurrency_peak,
            "output_sessions": series.state_high[0],
            "thinking_sessions": series.state_high[1],
            "prefill_sessions": series.state_high[2],
            "output_active": series.output_tokens > 0,
            "thinking_active": series.thinking_tokens > 0,
            "output_peak_tok_s": round(series.output_peak, 2),
            "thinking_peak_tok_s": round(series.thinking_peak, 2),
            "prefill_peak_tok_s": round(series.prefill_peak, 2),
        })

    def _admission(self) -> dict[str, Mapping[str, Any]]:
        """One admission snapshot per tick, shared by every lookup."""
        provider = getattr(self.manager, "inference_admission", None)
        if not callable(provider):
            return {}
        try:
            return provider() or {}
        except Exception:
            return {}

    @staticmethod
    def _new_entry(
        observed: dict[str, dict[str, Any]], key: str, group: Mapping[str, Any],
    ) -> dict[str, Any]:
        entry = observed[key] = {
            "meta": _clean_meta(group), "states": [0, 0, 0],
            "prompt_records": {}, "prefill_oldest": 0.0,
            "records": [],
        }
        return entry

    @staticmethod
    def _admission_running(
        admission: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, int]:
        """Authoritative running counts per serving unit, when limited."""
        running: dict[str, int] = {}
        for snapshot in admission.values():
            key = _series_key(snapshot)
            if key:
                running[key] = running.get(key, 0) + int(snapshot.get("running") or 0)
        return running

    @staticmethod
    def _admission_meta(
        admission: Mapping[str, Mapping[str, Any]], key: str,
    ) -> dict[str, Any] | None:
        for snapshot in admission.values():
            if _series_key(snapshot) == key:
                return _clean_meta(snapshot)
        return None

    def _retire(self, now: float) -> None:
        for key in [
            key for key, series in self._series.items()
            if key not in self._live
            and now - float(series.last_at) > RETAIN_ACTIVE_SECONDS
        ]:
            self._series.pop(key, None)
        while len(self._series) > MAX_SERIES:
            oldest = min(self._series, key=lambda key: self._series[key].last_at)
            self._series.pop(oldest, None)

    async def sampler(self) -> None:
        """Advance the timeline at the configured cadence until shutdown.

        The interval is re-read every tick, so changing it takes effect on the
        next sample.  While recording is switched off the loop idles on the
        slowest cadence instead of spinning, and keeps nothing in memory.
        """
        while True:
            if self.enabled:
                self.tick()
                delay = float(self.interval_seconds)
            else:
                if self._series or self._live:
                    self.forget()
                delay = float(MAX_SAMPLE_SECONDS)
            await asyncio.sleep(delay)

    # ----- reads -------------------------------------------------------
    def series(self, now: float | None = None) -> list[dict[str, Any]]:
        """Return the trailing timeline for every tracked serving unit."""
        now = self.clock() if now is None else now
        cutoff = now - self.keep_seconds
        result: list[dict[str, Any]] = []
        for key, series in self._series.items():
            buckets = [bucket for bucket in series.buckets if bucket["at"] >= cutoff]
            # A serving unit is worth a graph once it has any bucket behind it,
            # whether or not it has a session live at this instant.
            if not buckets:
                continue
            entry = dict(series.meta)
            entry["key"] = key
            entry["live_sessions"] = self._live.get(key, 0)
            entry["state"] = _state_record(series.last_state)
            entry["last_at"] = round(float(series.last_at), 3)
            entry["bucket_seconds"] = float(self.interval_seconds)
            entry["buckets"] = buckets
            result.append(entry)
        result.sort(
            key=lambda entry: (
                int(entry["live_sessions"]) <= 0,
                -float(entry["last_at"]),
                str(entry.get("model") or ""),
            )
        )
        return result

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = self.clock() if now is None else now
        interval = self.interval_seconds
        enabled = self.enabled
        return {
            "generated_at": round(now, 3),
            "enabled": enabled,
            "bucket_seconds": float(interval),
            "range_seconds": self.keep_seconds,
            "sample_seconds": float(interval),
            "series": self.series(now) if enabled else [],
        }

    # ----- lifecycle ---------------------------------------------------
    def ensure_sampler(self) -> None:
        """Start the background sampler if it is not already running."""
        if self._task is not None and not self._task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._task = loop.create_task(self.sampler())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


_STATE_FIELDS = {"output": 0, "thinking": 1, "prefill": 2}


def _series_key(group: Mapping[str, Any]) -> str:
    value = group.get("group_id") or group.get("deployment_id") or group.get("model")
    return str(value) if value else ""


def _clean_meta(group: Mapping[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for field in _GROUP_FIELDS:
        if field not in group:
            continue
        value = group[field]
        if field == "node_names":
            meta[field] = [str(name) for name in value or ()]
        else:
            meta[field] = value
    meta.setdefault("model", str(group.get("model") or ""))
    meta.setdefault("group_id", _series_key(group))
    return meta


def _session_state(rec: Mapping[str, Any]) -> str:
    """Mirror the dashboard's live-session classification.

    Visible output wins over reasoning, and either token kind wins over prompt
    processing, so a chunked-prefill request that already emitted a token counts
    as outputting rather than prefilling.
    """
    if rec.get("output"):
        return "output"
    if rec.get("thinking"):
        return "thinking"
    return "prefill"


def _state_record(states: tuple[int, int, int] | None) -> dict[str, int]:
    output, thinking, prefill = states or (0, 0, 0)
    return {
        "output_sessions": output,
        "thinking_sessions": thinking,
        "prefill_sessions": prefill,
    }
