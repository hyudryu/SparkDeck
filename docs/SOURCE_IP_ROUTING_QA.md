# Source IP routing override QA

Run this checklist against an isolated controller with disposable storage and mocked
engines. Never point mutation or inference tests at a production controller: even
a short inference test competes with existing requests. Use documentation-only IPs
such as `192.0.2.10` and `192.0.2.20` in fixtures.

## Repeatable setup and cleanup

- [ ] Allocate a temporary data directory and an unused loopback port per run.
- [ ] Seed two deployments serving the same model, or one grouped deployment with
  Group 1 on Node 1 + Node 2 and Group 2 on Node 3 + Node 4.
- [ ] Record every test-owned resource and use deterministic rule keys.
- [ ] Run setup twice; it must update the same rules without adding duplicates.
- [ ] On success, failure, or cancellation, stop only test-owned processes and
  remove only the temporary directory created by this run.
- [ ] Run the complete suite twice and confirm identical outcomes, with no
  surviving test resources or changes to production deployments.

## Rule management

- [ ] Save two source IPs targeting different groups of the same model.
- [ ] Repeat an identical save; list still contains one rule for that IP/model.
- [ ] Disable and re-enable each rule; settings survive reload and restart.
- [ ] Change a target while retaining the same IP/model key; the rule is updated.
- [ ] Remove a rule twice; the first delete succeeds and the second returns
  not found. Cleanup treats not found as already removed.
- [ ] Canonical IPv6 forms match consistently; invalid IPs, CIDRs, hostnames,
  zone identifiers, invalid booleans, and malformed targets are rejected.
- [ ] A failed save preserves the previous rule both in memory and on disk.
- [ ] In the disposable fixture, corrupt the saved JSON, version, or an enabled
  row, then restart. Listing rules and inference return an explicit unavailable
  error; requests never fall back to ordinary routing. Restore the fixture file
  and restart to verify recovery. A first run without a rules file works normally.
- [ ] An unavailable or deleted target remains visible and can still be disabled
  or removed; it is never silently replaced by another target.

## Request dispatch

- [ ] Exercise both `/v1/chat/completions` and `/v1/completions`, with streaming
  enabled and disabled, through the controller's public router.
- [ ] Each matching enabled IP/model reaches only its configured group despite
  load differences or a prefix-affinity hint for another group.
- [ ] Duplicate served names resolve through the matching override to the
  intended deployment, retaining normal admission and request accounting.
- [ ] Unmatched IPs, unmatched models, and disabled rules use ordinary routing.
- [ ] A stopped/offline target, missing coordinator, changed node membership,
  removed deployment, or changed served model returns unavailable without
  dispatch to another group.
- [ ] Upstream failures before and after the first streamed token never cause
  an override request to be replayed on a different group.
- [ ] Disconnects and cancellations release admission and active-request state.
- [ ] Spoofed forwarding headers and private fields in public request bodies
  cannot change the matched source IP or selected target.
- [ ] Authenticated worker forwarding retains the original caller identity.

## UI and regression checks

- [ ] Target labels distinguish deployment names, group numbers, and node names.
- [ ] A deployment with explicit served names also offers its alias; an existing
  alias rule stays selectable and can be toggled.
- [ ] In a degraded replicated deployment, only healthy replicas are selectable.
  Stopped, offline, and unknown replicas cannot be enabled through either the UI
  or API, while a surviving replica remains usable.
- [ ] Keyboard users can save, toggle, and remove rules; pending requests prevent
  duplicate actions and errors preserve the form.
- [ ] At desktop and mobile widths, long model names and IPv6 addresses wrap,
  and all controls remain reachable.
- [ ] Request-routing errors do not hide existing usage statistics.
- [ ] Existing accounting-only model routing rules still behave as before.

For each failure, record the commit, fixture topology, rule payload, request
endpoint, expected and actual target, response status, and cleanup result.
Do not include production prompts, credentials, or observed caller addresses.
