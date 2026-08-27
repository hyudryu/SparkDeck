import { useCallback, useEffect, useState } from 'react'

export function useResource<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  dependencies: unknown[] = [],
  enabled = true,
) {
  const [data, setData] = useState<T>()
  const [error, setError] = useState<string>()
  const [loading, setLoading] = useState(true)
  const [version, setVersion] = useState(0)

  const reload = useCallback(() => setVersion((value) => value + 1), [])

  useEffect(() => {
    const controller = new AbortController()
    if (!enabled) {
      setData(undefined)
      setError(undefined)
      setLoading(false)
      return () => controller.abort()
    }
    setLoading(true)
    setError(undefined)
    loader(controller.signal)
      .then(setData)
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Something went wrong')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false)
      })
    return () => controller.abort()
    // The caller provides the values that should reload this resource.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...dependencies, enabled, version])

  return { data, error, loading, reload, setData }
}
