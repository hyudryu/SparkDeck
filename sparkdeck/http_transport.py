"""httpx transport that does not leak connection-pool slots on cancellation.

httpcore (1.0.9, the latest release) can strand a pooled connection when the
task sending a request is cancelled at the wrong moment:

* after the TCP connect finished but before the HTTP/1.1 connection moved from
  ``NEW`` to ``ACTIVE`` (anyio's lock acquire yields there), or
* while a queued request was being handed a freshly created connection.

The pool then drops the cancelled request but keeps the connection. A ``NEW``
or ``CONNECTING`` connection is neither *available* (only ``IDLE`` connections
are reused) nor *idle* (so it is never evicted), yet it still counts towards
``max_connections``. Each stranded connection permanently consumes one slot and,
in the ``NEW`` case, holds an open socket that never carries a request. Once
every slot is stranded, each new request waits out its pool timeout and fails
with an empty ``httpx.PoolTimeout``.

SparkDeck cancels in-flight agent requests routinely (client disconnects,
readiness probes racing a cancel event), so the controller's shared client
slowly filled with stranded connections until it could no longer reach any
node. :class:`CancellationSafeTransport` closes such orphans as soon as a
request fails or is cancelled, and wakes any request queued for the freed slot.
"""

from __future__ import annotations

import logging

import anyio
import httpx

logger = logging.getLogger(__name__)


def _is_orphan(connection, owned: set[int]) -> bool:
    """True when no request owns ``connection`` and it can never serve one."""
    if id(connection) in owned:
        return False
    inner = getattr(connection, "_connection", None)
    if inner is None:
        # A connection the pool created for a request that has since gone away
        # before it dialed. A failed connect is reported closed and is already
        # removed by the pool itself.
        return getattr(connection, "_connect_failed", True) is False
    state = getattr(getattr(inner, "_state", None), "name", None)
    # Only HTTP/1.1 connections sit in NEW without a request; IDLE, ACTIVE and
    # CLOSED connections are managed correctly by the pool.
    return state == "NEW" and type(inner).__name__ == "AsyncHTTP11Connection"


def _take_orphaned_connections(pool) -> tuple[list, list]:
    """Detach orphaned connections from ``pool``.

    Returns ``(orphans, closing)``: the orphans removed from the pool, plus any
    connections the pool wants closed after it re-assigned queued requests to
    the freed slots.
    """
    connections = getattr(pool, "_connections", None)
    requests = getattr(pool, "_requests", None)
    if connections is None or requests is None:
        return [], []
    lock = getattr(pool, "_optional_thread_lock", None)
    if lock is None:
        return [], []
    with lock:
        owned = {
            id(request.connection)
            for request in requests
            if getattr(request, "connection", None) is not None
        }
        orphans = [
            connection for connection in list(connections)
            if _is_orphan(connection, owned)
        ]
        if not orphans:
            return [], []
        for connection in orphans:
            connections.remove(connection)
        assign = getattr(pool, "_assign_requests_to_connections", None)
        closing = list(assign()) if callable(assign) else []
    return orphans, closing


async def reap_orphaned_connections(pool) -> int:
    """Close connections stranded in ``pool``; return how many were reaped."""
    orphans, closing = _take_orphaned_connections(pool)
    if not orphans and not closing:
        return 0
    # Runs while a cancellation is unwinding: finish closing the sockets.
    with anyio.CancelScope(shield=True):
        for connection in [*orphans, *closing]:
            try:
                await connection.aclose()
            except Exception:
                logger.debug("failed to close stranded pooled connection", exc_info=True)
    if orphans:
        logger.debug("reaped %d stranded pooled connection(s)", len(orphans))
    return len(orphans)


class CancellationSafeTransport(httpx.AsyncHTTPTransport):
    """``AsyncHTTPTransport`` that reaps connections stranded by cancellation."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return await super().handle_async_request(request)
        except BaseException:
            try:
                await reap_orphaned_connections(self._pool)
            except Exception:
                logger.debug("pooled connection reaping failed", exc_info=True)
            raise


def cancellation_safe_client(*, trust_env: bool = True, **kwargs) -> httpx.AsyncClient:
    """Build an ``AsyncClient`` whose default transport survives cancellation."""
    transport = CancellationSafeTransport(trust_env=trust_env)
    return httpx.AsyncClient(transport=transport, trust_env=trust_env, **kwargs)
