import { useEffect, useState } from 'react'
import type { RefObject } from 'react'
import { deploymentFromWire, syncStatusFromWire } from '../api/client'
import type { WireDeployment } from '../api/client'
import type {
  AdmissionStats,
  Deployment,
  NodeInventoryItem,
  SyncStatus,
  SystemStats,
} from '../api/types'

const RECONNECT_DELAY_MS = 5_000

export interface DashboardStreamResources {
  stats: { setData: (value: SystemStats) => void }
  admission: { setData: (value: Record<string, AdmissionStats>) => void }
  deployments: { setData: (value: Deployment[]) => void }
  sync: { setData: (value: SyncStatus) => void }
  nodes: { setData: (value: NodeInventoryItem[]) => void }
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

  useEffect(() => {
    if (typeof WebSocket === 'undefined') return undefined
    let socket: WebSocket | undefined
    let reconnectTimer: number | undefined
    let closed = false

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
          if (snapshot.stats) resources.stats.setData(snapshot.stats)
          if (snapshot.admission) resources.admission.setData(snapshot.admission)
          if (snapshot.deployments) {
            resources.deployments.setData(snapshot.deployments.items.map(deploymentFromWire))
          }
          if (snapshot.community_sync) resources.sync.setData(syncStatusFromWire(snapshot.community_sync))
          if (snapshot.nodes) resources.nodes.setData(snapshot.nodes.items)
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

    connect()
    return () => {
      closed = true
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer)
      socket?.close()
    }
  }, [resourcesRef])

  return { live }
}
