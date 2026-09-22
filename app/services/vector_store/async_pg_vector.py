from typing import Callable, Optional, List, Sequence, Tuple, Dict, Any, TypeVar
import asyncio
from concurrent.futures import Executor
from langchain_core.documents import Document
from app.utils.async_utils import run_in_executor
from .extended_pg_vector import ExtendedPgVector

T = TypeVar("T")


class AsyncPgVector(ExtendedPgVector):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_pool = None

    def _get_thread_pool(self):
        if self._thread_pool is None:
            try:
                loop = asyncio.get_running_loop()
                self._thread_pool = getattr(loop, "_default_executor", None)
            except Exception:
                pass
        return self._thread_pool

    @staticmethod
    async def _run_in_executor(
        executor: Executor | None,
        func: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """Run a sync callable with the shared exception-safe executor bridge."""
        return await run_in_executor(executor, func, *args, **kwargs)

    async def get_all_ids(self, owners: Sequence[str], executor=None) -> list[str]:
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(executor, super().get_all_ids, owners)

    async def get_filtered_ids(
        self, ids: Sequence[str], owners: Sequence[str], executor=None
    ) -> list[str]:
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(
            executor, super().get_filtered_ids, ids, owners
        )

    async def get_documents_by_ids(
        self, ids: Sequence[str], owners: Sequence[str], executor=None
    ) -> list[Document]:
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(
            executor, super().get_documents_by_ids, ids, owners
        )

    async def delete_scoped(
        self, ids: Sequence[str], owners: Sequence[str], executor=None
    ) -> None:
        executor = executor or self._get_thread_pool()
        await self._run_in_executor(executor, super().delete_scoped, ids, owners)

    async def delete(
        self,
        ids: Optional[list[str]] = None,
        collection_only: bool = False,
        executor=None,
    ) -> None:
        executor = executor or self._get_thread_pool()
        await self._run_in_executor(
            executor, self._delete_multiple, ids, collection_only
        )

    async def delete_by_metadata(
        self, metadata_filter: Dict[str, Any], executor=None
    ) -> None:
        """Delete only rows in this collection that match metadata values."""
        executor = executor or self._get_thread_pool()
        await self._run_in_executor(executor, self._delete_by_metadata, metadata_filter)

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        executor=None,
    ) -> List[Tuple[Document, float]]:
        """Async version of similarity_search_with_score_by_vector"""
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(
            executor,
            super().similarity_search_with_score_by_vector,
            embedding,
            k,
            filter,
        )

    async def aadd_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        executor=None,
        **kwargs,
    ) -> List[str]:
        """Async version of add_documents"""
        executor = executor or self._get_thread_pool()
        return await self._run_in_executor(
            executor, super().add_documents, documents, ids=ids, **kwargs
        )
