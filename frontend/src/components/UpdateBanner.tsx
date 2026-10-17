import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { DownloadCloud, X } from 'lucide-react'
import { api } from '../api/client'

const DISMISSED_REVISION_KEY = 'sparkdeck.update-banner.dismissed-revision'
const RECHECK_INTERVAL_MS = 10 * 60 * 1000
const UPDATE_CHECK_EVENT = 'sparkdeck:update-check'

/** Ask the banner to re-check now instead of waiting for its periodic poll. */
export function requestUpdateCheck() {
  window.dispatchEvent(new Event(UPDATE_CHECK_EVENT))
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
    let controller: AbortController | undefined
    let latestRequest = 0
    const check = () => {
      controller?.abort()
      const requestController = new AbortController()
      const requestId = ++latestRequest
      controller = requestController
      const isFresh = () => !disposed && requestId === latestRequest && !requestController.signal.aborted
      api.updates.overview(requestController.signal)
        .then((overview) => {
          if (!isFresh()) return
          // Only advertise an update the cluster can actually install;
          // "up to date", running, and blocked states stay quiet.
          setTargetRevision(
            overview.up_to_date === false && overview.can_update && !overview.job?.active
              ? overview.target?.revision
              : undefined,
          )
        })
        .catch(() => {
          // A failed update check is never worth a banner.
          if (isFresh()) setTargetRevision(undefined)
        })
    }
    check()
    const interval = window.setInterval(check, RECHECK_INTERVAL_MS)
    // Update work changes what the banner should advertise. Re-check when a
    // page reports update progress, and when the user returns to this tab
    // (covers updates finished outside the app, e.g. from a terminal).
    const recheckWhenVisible = () => {
      if (document.visibilityState === 'visible') check()
    }
    window.addEventListener(UPDATE_CHECK_EVENT, check)
    window.addEventListener('focus', check)
    document.addEventListener('visibilitychange', recheckWhenVisible)
    return () => {
      disposed = true
      latestRequest += 1
      controller?.abort()
      window.clearInterval(interval)
      window.removeEventListener(UPDATE_CHECK_EVENT, check)
      window.removeEventListener('focus', check)
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
