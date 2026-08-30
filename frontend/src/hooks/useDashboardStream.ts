import { useEffect, useState } from 'react'
import type { RefObject } from 'react'
import { api, deploymentFromWire, syncStatusFromWire } from '../api/client'
import type { WireDeployment } from '../api/client'
import type {
  AdmissionStats,
  Deployment,
  NodeInventoryItem,
  SyncStatus,
  SystemStats,
} from '../api/types'

const RECONNECT_DELAY_MS = 5_000

export type DashboardStreamSource = 'stats' | 'admission' | 'deployments' | 'sync' | 'nodes'

export interface DashboardStreamResources {
  stats: { apply: (value: SystemStats) => void }
  admission: { apply: (value: Record<string, AdmissionStats>) => void }
  deployments: { apply: (value: Deployment[]) => void }
  sync: { apply: (value: SyncStatus) => void }
  nodes: { apply: (value: NodeInventoryItem[]) => void }
}

interface DashboardSnapshot {
  type: 'snapshot'
  stats?: SystemStats | null
  admission?: Record<string, AdmissionStats> | null
  deployments?: { items: WireDeployment[] } | null
  community_sync?: Parameters<typeof syncStatusFromWire>[0] | null
  nodes?: { items: NodeInventoryItem[] } | null
}

// The resources arrive by ref because they are created below the stream hook
// in DashboardPage: the hook must run first so its `live` flag can pause the
// polling fallback, while snapshots only need the resources once connected.
export function useDashboardStream(resourcesRef: RefObject<DashboardStreamResources | null>) {
  const [live, setLive] = useState(false)
  const [failed, setFailed] = useState<ReadonlySet<DashboardStreamSource>>(new Set())

  useEffect(() => {
    if (typeof WebSocket === 'undefined') return undefined
    let socket: WebSocket | undefined
    let reconnectTimer: number | undefined
    let closed = false
    const roleController = new AbortController()

    const markFailed = (source: DashboardStreamSource, isFailed: boolean) => {
      setFailed((current) => {
        if (current.has(source) === isFailed) return current
        const next = new Set(current)
        if (isFailed) next.add(source)
        else next.delete(source)
        return next
      })
    }

    const connect = () => {
      let next: WebSocket
      try {
        const scheme = window.location.protocol === 'https:' ? 'wss' : 'ws'
        next = new WebSocket(`${scheme}://${window.location.host}/api/ws/dashboard`)
      } catch {
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS)
        return
      }
      socket = next
      socket.onopen = () => setLive(true)
      socket.onmessage = (event) => {
        try {
          const snapshot = JSON.parse(String(event.data)) as DashboardSnapshot
          if (snapshot?.type !== 'snapshot') return
          const resources = resourcesRef.current
          if (!resources) return
          // A null value means that source failed server-side; keep polling
          // it via REST while the socket stays live for the other sources.
          if (snapshot.stats !== undefined) {
            markFailed('stats', snapshot.stats === null)
            if (snapshot.stats) resources.stats.apply(snapshot.stats)
          }
          if (snapshot.admission !== undefined) {
            markFailed('admission', snapshot.admission === null)
            if (snapshot.admission) resources.admission.apply(snapshot.admission)
          }
          if (snapshot.deployments !== undefined) {
            markFailed('deployments', snapshot.deployments === null)
            if (snapshot.deployments) {
              resources.deployments.apply(snapshot.deployments.items.map(deploymentFromWire))
            }
          }
          if (snapshot.community_sync !== undefined) {
            markFailed('sync', snapshot.community_sync === null)
            if (snapshot.community_sync) resources.sync.apply(syncStatusFromWire(snapshot.community_sync))
          }
          if (snapshot.nodes !== undefined) {
            markFailed('nodes', snapshot.nodes === null)
            if (snapshot.nodes) resources.nodes.apply(snapshot.nodes.items)
          }
        } catch {
          // Ignore malformed frames; the next snapshot replaces the state.
        }
      }
      socket.onerror = () => socket?.close()
      socket.onclose = () => {
        if (closed) return
        setLive(false)
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS)
      }
    }

    // Joined workers deliberately reject this local WebSocket and serve the
    // dashboard through controller-forwarded REST polling instead. Check the
    // unforwarded onboarding status first so that expected fallback does not
    // produce a failed-handshake error in the browser console every 5 seconds.
    void api.onboarding.get(roleController.signal).then((status) => {
      if (!closed && status.role !== 'worker') connect()
    }).catch(() => {
      // Preserve the existing stream-first behavior if role discovery fails.
      if (!closed) connect()
    })
    return () => {
      closed = true
      roleController.abort()
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [resourcesRef])

  return { live, failed }
}
