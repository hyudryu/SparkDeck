"""Close owned async streams in their iteration context during disconnect."""
import anyio


async def close_async_stream(stream) -> None:
    close = getattr(stream, "aclose", None)
    if close is not None:
        # Keep ContextVar resets in the iteration task, and shield Starlette's
        # AnyIO disconnect cancellation while awaiting transport cleanup.
        with anyio.CancelScope(shield=True):
            await close()
