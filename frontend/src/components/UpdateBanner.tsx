import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { DownloadCloud, X } from 'lucide-react'
import { api } from '../api/client'

const DISMISSED_REVISION_KEY = 'sparkdeck.update-banner.dismissed-revision'
const RECHECK_INTERVAL_MS = 10 * 60 * 1000
const ROLLOUT_RECHECK_INTERVAL_MS = 10 * 1000
const UPDATE_CHECK_EVENT = 'sparkdeck:update-check'

/** Ask the banner to re-check now; hide additionally drops the notice at once. */
export function requestUpdateCheck(options: { hide?: boolean } = {}) {
  window.dispatchEvent(new CustomEvent(UPDATE_CHECK_EVENT, { detail: options }))
}

export function dismissedRevision(): string | null {
  return localStorage.getItem(DISMISSED_REVISION_KEY)
}

export function UpdateBanner({ controllerAvailable = true }: { controllerAvailable?: boolean }) {
  const [targetRevision, setTargetRevision] = useState<string>()
  const [dismissed, setDismissed] = useState(() => dismissedRevision())

  useEffect(() => {
    if (!controllerAvailable) return
    let disposed = false
    let requestController: AbortController | undefined
    let latestRequest = 0
    let timer: number | undefined
    let inFlight = false
    let hideRequested = false
    // Once a rollout is seen, keep the fast cadence through controller
    // restarts and transient outages until an overview reports it settled.
    let watchingRollout = false

    const schedule = (delay: number) => {
      if (timer !== undefined) window.clearTimeout(timer)
      timer = window.setTimeout(run, delay)
    }

    const run = () => {
      timer = undefined
      if (hideRequested) {
        // An update just started: drop the notice before the authoritative
        // overview has a chance to respond.
        hideRequested = false
        setTargetRevision(undefined)
      }
      requestController?.abort()
      const request = new AbortController()
      requestController = request
      const requestId = ++latestRequest
      inFlight = true
      const isFresh = () => !disposed && requestId === latestRequest && !request.signal.aborted
      api.updates.overview(request.signal)
        .then((overview) => {
          if (!isFresh()) return
          inFlight = false
          // Only advertise an update the cluster can actually install;
          // "up to date", running, and blocked states stay quiet.
          const jobActive = Boolean(overview.job?.active)
          setTargetRevision(
            overview.up_to_date === false && overview.can_update && !jobActive
              ? overview.target?.revision
              : undefined,
          )
          // Keep watching a running rollout so a failed or partial update
          // re-advertises its revision without a trip back to Settings.
          watchingRollout = jobActive
          schedule(jobActive ? ROLLOUT_RECHECK_INTERVAL_MS : RECHECK_INTERVAL_MS)
        })
        .catch(() => {
          if (!isFresh()) return
          inFlight = false
          // A failed update check is never worth a banner, but a rollout's
          // controller restart must not drop the watching cadence.
          setTargetRevision(undefined)
          schedule(watchingRollout ? ROLLOUT_RECHECK_INTERVAL_MS : RECHECK_INTERVAL_MS)
        })
    }

    // Focus and visibility events fire together when returning to a tab and
    // their overview request probes every cluster node, so coalesce them.
    const requestCheck = (hide = false) => {
      if (hide) {
        latestRequest += 1
        hideRequested = true
        setTargetRevision(undefined)
      } else if (inFlight) {
        return
      }
      schedule(0)
    }

    const onUpdateCheck = (event: Event) => {
      requestCheck(Boolean((event as CustomEvent<{ hide?: boolean }>).detail?.hide))
    }
    const recheckWhenVisible = () => {
      if (document.visibilityState === 'visible') requestCheck()
    }

    run()
    window.addEventListener(UPDATE_CHECK_EVENT, onUpdateCheck)
    window.addEventListener('focus', recheckWhenVisible)
    document.addEventListener('visibilitychange', recheckWhenVisible)
    return () => {
      disposed = true
      latestRequest += 1
      if (timer !== undefined) window.clearTimeout(timer)
      requestController?.abort()
      window.removeEventListener(UPDATE_CHECK_EVENT, onUpdateCheck)
      window.removeEventListener('focus', recheckWhenVisible)
      document.removeEventListener('visibilitychange', recheckWhenVisible)
    }
  }, [controllerAvailable])

  if (!targetRevision || dismissed === targetRevision) return null

  const dismiss = () => {
    // Remember which revision was dismissed so a newer update re-shows
    // the banner while the dismissed one stays quiet.
    localStorage.setItem(DISMISSED_REVISION_KEY, targetRevision)
    setDismissed(targetRevision)
  }

  return (
    <div className="update-banner" role="status">
      <Link to="/settings#software-update" className="update-banner-link" aria-label={`SparkDeck update to ${targetRevision.slice(0, 8)} is available. Open the software update settings.`}>
        <DownloadCloud size={15} aria-hidden="true" />
        <span>
          <strong>Update available</strong>
          {' '}— a new version of SparkDeck is ready to install. Open Settings to update.
        </span>
      </Link>
      <button className="update-banner-dismiss" onClick={dismiss} aria-label="Dismiss update notification">
        <X size={15} aria-hidden="true" />
      </button>
    </div>
  )
}
