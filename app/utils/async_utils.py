"""Bridges for blocking operations used by async request paths."""

import asyncio
from concurrent.futures import Executor
from typing import Any, Callable, TypeVar

T = TypeVar("T")


async def run_in_executor(
    executor: Executor | None,
    func: Callable[..., T],
    *args: Any,
    **kwargs: Any,
) -> T:
    """Run a sync callable in the supplied pool without blocking the event loop.

    StopIteration cannot be set on an asyncio.Future: convert it into an error
    that the awaiting request can handle instead of leaving it pending forever.
    Cancelling the await does not stop a callable already running in a worker.
    """

    def wrapper() -> T:
        try:
            return func(*args, **kwargs)
        except StopIteration as exc:
            raise RuntimeError from exc

    return await asyncio.get_running_loop().run_in_executor(executor, wrapper)
