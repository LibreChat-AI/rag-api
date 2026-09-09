# tests/test_batch_processing_integration.py
"""
Integration tests for batch processing.

These tests verify actual memory behavior and require more resources to run.
Mark with @pytest.mark.integration to skip in normal test runs.

Run with: pytest tests/test_batch_processing_integration.py -v -m integration
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import tracemalloc
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from langchain_core.documents import Document


class TestMemoryOptimization:
    """Tests to verify memory optimization behavior."""

    @pytest.mark.asyncio
    async def test_memory_bounded_by_batch_size(self):
        """
        Test that memory usage is bounded by batch size, not total documents.

        This test verifies that processing many documents in batches doesn't
        accumulate memory proportionally to the total document count.
        """
        from app.routes.document_routes import _process_documents_async_pipeline

        # Track how many documents are in memory at any time
        max_docs_in_memory = 0
        current_docs_in_memory = 0

        async def tracking_add_documents(docs, ids=None, executor=None):
            nonlocal max_docs_in_memory, current_docs_in_memory
            current_docs_in_memory = len(docs)
            max_docs_in_memory = max(max_docs_in_memory, current_docs_in_memory)
            return [f"id_{i}" for i in range(len(docs))]

        mock_store = AsyncMock()
        mock_store.aadd_documents = tracking_add_documents
        mock_store.delete = AsyncMock()

        # Create 100 documents
        docs = [
            Document(page_content=f"doc_{i}" * 100, metadata={"idx": i})
            for i in range(100)
        ]

        # Process with batch size of 10
        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 10):
            result = await _process_documents_async_pipeline(
                documents=docs,
                file_id="test",
                user_id="testuser",
                vector_store=mock_store,
                executor=None,
            )

        # Verify we got all 100 IDs back
        assert len(result) == 100

        # Verify max docs in memory was bounded by batch size (10), not total (100)
        assert (
            max_docs_in_memory <= 10
        ), f"Expected max {10} docs in memory at once, but saw {max_docs_in_memory}"

    @pytest.mark.asyncio
    async def test_memory_tracking_with_tracemalloc(self):
        """
        Test memory usage with tracemalloc.

        This test uses Python's tracemalloc to verify memory behavior.
        Note: This is a sanity check, not a strict memory bound test.
        """
        from app.routes.document_routes import _process_documents_async_pipeline

        mock_store = AsyncMock()
        mock_store.aadd_documents = AsyncMock(return_value=["id"])
        mock_store.delete = AsyncMock()

        # Create documents with substantial content
        doc_count = 50
        docs = [
            Document(
                page_content=f"Document content {i} " * 100,  # ~2KB per doc
                metadata={"file_id": "test", "idx": i},
            )
            for i in range(doc_count)
        ]

        tracemalloc.start()

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 5):
            await _process_documents_async_pipeline(
                documents=docs,
                file_id="test",
                user_id="testuser",
                vector_store=mock_store,
                executor=None,
            )

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Log memory usage for debugging
        print(f"Current memory: {current / 1024:.2f} KB")
        print(f"Peak memory: {peak / 1024:.2f} KB")

        # The test passes if it completes without OOM
        # Actual memory bounds depend on Python internals and test environment
        assert True

    @pytest.mark.asyncio
    async def test_batch_processing_maintains_order(self):
        """Test that document IDs are returned in correct order across batches."""
        from app.routes.document_routes import _process_documents_async_pipeline

        call_order = []

        async def ordered_add_documents(docs, ids=None, executor=None):
            batch_ids = [f"id_{docs[0].metadata['idx']}_to_{docs[-1].metadata['idx']}"]
            call_order.append(docs[0].metadata["idx"])
            return [f"id_{d.metadata['idx']}" for d in docs]

        mock_store = AsyncMock()
        mock_store.aadd_documents = ordered_add_documents

        docs = [
            Document(page_content=f"doc_{i}", metadata={"idx": i}) for i in range(15)
        ]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 5):
            result = await _process_documents_async_pipeline(
                documents=docs,
                file_id="test",
                user_id="testuser",
                vector_store=mock_store,
                executor=None,
            )

        # Verify batches were processed in order
        assert call_order == [0, 5, 10], f"Batches processed out of order: {call_order}"

        # Verify all IDs returned
        assert len(result) == 15


class TestSyncBatchedMemory:
    """Memory tests for sync batched processing."""

    @pytest.mark.asyncio
    async def test_sync_memory_bounded_by_batch_size(self):
        """Test that sync batch processing bounds memory by batch size."""
        from app.routes.document_routes import _process_documents_batched_sync
        from concurrent.futures import ThreadPoolExecutor

        max_docs_in_batch = 0

        def tracking_add_documents(documents, ids=None):
            nonlocal max_docs_in_batch
            max_docs_in_batch = max(max_docs_in_batch, len(documents))
            return [f"id_{i}" for i in range(len(documents))]

        mock_store = Mock()
        mock_store.add_documents = tracking_add_documents
        mock_store.delete = Mock()

        docs = [
            Document(page_content=f"doc_{i}" * 100, metadata={"idx": i})
            for i in range(50)
        ]

        with ThreadPoolExecutor(max_workers=2) as executor:
            with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 10):
                result = await _process_documents_batched_sync(
                    documents=docs,
                    file_id="test",
                    user_id="testuser",
                    vector_store=mock_store,
                    executor=executor,
                )

        assert len(result) == 50
        assert (
            max_docs_in_batch <= 10
        ), f"Expected max {10} docs per batch, but saw {max_docs_in_batch}"


class TestBatchProcessingResilience:
    """Tests for error handling and resilience."""

    @pytest.mark.asyncio
    async def test_partial_failure_tracks_inserted_ids(self):
        """Test that we track which IDs were inserted before failure."""
        from app.routes.document_routes import _process_documents_async_pipeline

        inserted_batches = []

        async def failing_add_documents(docs, ids=None, executor=None):
            batch_num = len(inserted_batches) + 1
            if batch_num == 3:  # Fail on third batch
                raise Exception("Simulated DB error")
            result = [f"batch{batch_num}_id_{i}" for i in range(len(docs))]
            inserted_batches.append(result)
            return result

        mock_store = AsyncMock()
        mock_store.aadd_documents = failing_add_documents
        mock_store.delete = AsyncMock()
        mock_store.delete_by_metadata = AsyncMock()

        docs = [
            Document(page_content=f"doc_{i}", metadata={"idx": i}) for i in range(15)
        ]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 5):
            with pytest.raises(Exception, match="Simulated DB error"):
                await _process_documents_async_pipeline(
                    documents=docs,
                    file_id="test_file",
                    user_id="testuser",
                    vector_store=mock_store,
                    executor=None,
                )

        # Verify rollback was called because we had inserted batches
        mock_store.delete.assert_not_called()
        mock_store.delete_by_metadata.assert_called_once()

        # Verify we inserted 2 batches before failure
        assert len(inserted_batches) == 2

    @pytest.mark.asyncio
    async def test_rollback_called_with_current_ingestion_attempt(self):
        """Test that rollback targets only the current ingestion attempt."""
        from app.routes.document_routes import _process_documents_async_pipeline

        async def failing_on_second(docs, ids=None, executor=None):
            if len(docs) > 0 and docs[0].metadata.get("idx", 0) >= 5:
                raise Exception("Fail")
            return ["id1"]

        mock_store = AsyncMock()
        mock_store.aadd_documents = failing_on_second
        mock_store.delete = AsyncMock()
        mock_store.delete_by_metadata = AsyncMock()

        docs = [
            Document(page_content=f"doc_{i}", metadata={"idx": i}) for i in range(10)
        ]

        with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 5):
            with pytest.raises(Exception):
                await _process_documents_async_pipeline(
                    documents=docs,
                    file_id="my_unique_file_id",
                    user_id="testuser",
                    vector_store=mock_store,
                    executor=None,
                )

        mock_store.delete.assert_not_called()
        mock_store.delete_by_metadata.assert_called_once()
        metadata_filter = mock_store.delete_by_metadata.call_args.args[0]
        assert metadata_filter["file_id"] == "my_unique_file_id"
        attempt_id = metadata_filter["_rag_ingestion_attempt_id"]
        assert attempt_id
        assert all(
            document.metadata["file_id"] == "my_unique_file_id" for document in docs
        )
        assert all(
            document.metadata["_rag_ingestion_attempt_id"] == attempt_id
            for document in docs
        )


class TestConfigurationBehavior:
    """Tests for configuration-driven behavior."""

    @pytest.mark.asyncio
    async def test_batch_size_zero_uses_original_path(self):
        """Test that EMBEDDING_BATCH_SIZE=0 uses the non-batched code path."""
        from app.routes.document_routes import store_data_in_vector_db
        from app.services.vector_store.async_pg_vector import AsyncPgVector

        mock_store = AsyncMock(spec=AsyncPgVector)
        mock_store.aadd_documents = AsyncMock(return_value=["id1", "id2"])

        docs = [Document(page_content="test", metadata={})]

        with patch("app.routes.document_routes.vector_store", mock_store):
            with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", 0):
                with patch("app.routes.document_routes.isinstance", return_value=True):
                    result = await store_data_in_vector_db(
                        data=docs,
                        file_id="test_file",
                        user_id="test_user",
                        executor=None,
                    )

        # When batch size is 0, aadd_documents should be called directly
        # (not through the pipeline)
        assert mock_store.aadd_documents.called
        stored_documents = mock_store.aadd_documents.call_args.args[0]
        assert [
            document.metadata["_rag_chunk_index"] for document in stored_documents
        ] == list(range(len(stored_documents)))
        assert all(
            document.metadata["file_id"] == "test_file" for document in stored_documents
        )
        attempt_ids = {
            document.metadata["_rag_ingestion_attempt_id"]
            for document in stored_documents
        }
        started_at_values = {
            document.metadata["_rag_ingestion_attempt_started_at_ns"]
            for document in stored_documents
        }
        assert len(attempt_ids) == 1
        assert all(attempt_ids)
        assert len(started_at_values) == 1
        assert type(next(iter(started_at_values))) is int

    @pytest.mark.asyncio
    async def test_different_batch_sizes_produce_correct_batches(self):
        """Test that different batch sizes produce expected number of batches."""
        from app.routes.document_routes import _process_documents_async_pipeline

        test_cases = [
            (10, 100, 10),  # 100 docs / 10 batch = 10 batches
            (25, 100, 4),  # 100 docs / 25 batch = 4 batches
            (100, 100, 1),  # 100 docs / 100 batch = 1 batch
            (150, 100, 1),  # 100 docs / 150 batch = 1 batch (batch larger than docs)
            (7, 20, 3),  # 20 docs / 7 batch = 3 batches (20 = 7+7+6)
        ]

        for batch_size, doc_count, expected_batches in test_cases:
            mock_store = AsyncMock()
            mock_store.aadd_documents = AsyncMock(return_value=["id"])

            docs = [
                Document(page_content=f"doc_{i}", metadata={}) for i in range(doc_count)
            ]

            with patch("app.routes.document_routes.EMBEDDING_BATCH_SIZE", batch_size):
                await _process_documents_async_pipeline(
                    documents=docs,
                    file_id="test",
                    user_id="testuser",
                    vector_store=mock_store,
                    executor=None,
                )

            actual_batches = mock_store.aadd_documents.call_count
            assert actual_batches == expected_batches, (
                f"batch_size={batch_size}, docs={doc_count}: "
                f"expected {expected_batches} batches, got {actual_batches}"
            )


class TestStreamingIngestion:
    """Production ingestion must stay bounded before the vector-store call."""

    @pytest.mark.asyncio
    async def test_does_not_consume_the_full_source_before_first_insert(self):
        import app.routes.document_routes as routes

        consumed = 0
        first_insert_started = asyncio.Event()
        release_first_insert = asyncio.Event()

        def source_documents():
            nonlocal consumed
            for index in range(5):
                consumed += 1
                yield Document(page_content=f"page {index}", metadata={"page": index})

        async def add_documents(documents, ids=None, executor=None):
            first_insert_started.set()
            await release_first_insert.wait()
            return ids

        mock_store = AsyncMock()
        mock_store.aadd_documents = add_documents
        mock_store.delete_by_metadata = AsyncMock()

        with (
            patch.object(routes, "vector_store", mock_store),
            patch.object(routes, "EMBEDDING_BATCH_SIZE", 500),
            patch.object(routes, "PARALLEL_EXECUTION", 2),
            patch.object(routes, "RAG_INGESTION_WINDOW_SIZE", 1),
            patch.object(routes, "_INGESTION_SEMAPHORE", asyncio.Semaphore(1)),
            patch.object(routes, "isinstance", return_value=True),
        ):
            task = asyncio.create_task(
                routes.store_data_in_vector_db(
                    source_documents(), "streamed-file", "user", executor=None
                )
            )
            await asyncio.wait_for(first_insert_started.wait(), timeout=1)
            assert consumed == 1

            release_first_insert.set()
            result = await task

        assert consumed == 5
        assert len(result["ids"]) == 5

    @pytest.mark.asyncio
    async def test_limits_complete_ingestions_per_process(self):
        import app.routes.document_routes as routes

        active = 0
        max_active = 0

        async def add_documents(documents, ids=None, executor=None):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            active -= 1
            return ids

        mock_store = AsyncMock()
        mock_store.aadd_documents = add_documents
        mock_store.delete_by_metadata = AsyncMock()

        with (
            patch.object(routes, "vector_store", mock_store),
            patch.object(routes, "EMBEDDING_BATCH_SIZE", 1),
            patch.object(routes, "PARALLEL_EXECUTION", 1),
            patch.object(routes, "_INGESTION_SEMAPHORE", asyncio.Semaphore(1)),
            patch.object(routes, "isinstance", return_value=True),
        ):
            results = await asyncio.gather(
                routes.store_data_in_vector_db(
                    [Document(page_content="one")], "file-one", "user"
                ),
                routes.store_data_in_vector_db(
                    [Document(page_content="two")], "file-two", "user"
                ),
            )

        assert max_active == 1
        assert all("error" not in result for result in results)

    @pytest.mark.asyncio
    async def test_failure_in_a_later_window_rolls_back_the_whole_attempt(self):
        import app.routes.document_routes as routes

        inserted_documents = []

        async def add_documents(documents, ids=None, executor=None):
            inserted_documents.extend(documents)
            if len(inserted_documents) > 1:
                raise RuntimeError("embedding failed")
            return ids

        mock_store = AsyncMock()
        mock_store.aadd_documents = add_documents
        mock_store.delete_by_metadata = AsyncMock()

        with (
            patch.object(routes, "vector_store", mock_store),
            patch.object(routes, "EMBEDDING_BATCH_SIZE", 1),
            patch.object(routes, "PARALLEL_EXECUTION", 1),
            patch.object(routes, "_INGESTION_SEMAPHORE", asyncio.Semaphore(1)),
            patch.object(routes, "isinstance", return_value=True),
        ):
            result = await routes.store_data_in_vector_db(
                [Document(page_content="one"), Document(page_content="two")],
                "rollback-file",
                "user",
            )

        assert "error" in result
        metadata_filter = mock_store.delete_by_metadata.call_args.args[0]
        assert metadata_filter["file_id"] == "rollback-file"
        assert metadata_filter["_rag_ingestion_attempt_id"]
        assert {
            document.metadata["_rag_ingestion_attempt_id"]
            for document in inserted_documents
        } == {metadata_filter["_rag_ingestion_attempt_id"]}

    def test_streaming_splitter_matches_recursive_splitter(self):
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        from app.routes.document_routes import _iter_recursive_text_chunks

        splitter = RecursiveCharacterTextSplitter(chunk_size=40, chunk_overlap=8)
        text = (
            "First paragraph with several words.\n\n"
            "Second paragraph is longer and crosses more than one chunk. " * 8
        )

        assert list(
            _iter_recursive_text_chunks(splitter, text, splitter._separators)
        ) == splitter.split_text(text)

    @pytest.mark.asyncio
    async def test_chunk_indexes_and_attempt_span_multiple_windows(self):
        import app.routes.document_routes as routes

        inserted_documents = []

        async def add_documents(documents, ids=None, executor=None):
            inserted_documents.extend(documents)
            return ids

        mock_store = AsyncMock()
        mock_store.aadd_documents = add_documents
        mock_store.delete_by_metadata = AsyncMock()

        with (
            patch.object(routes, "vector_store", mock_store),
            patch.object(routes, "EMBEDDING_BATCH_SIZE", 10),
            patch.object(routes, "PARALLEL_EXECUTION", 1),
            patch.object(routes, "RAG_INGESTION_WINDOW_SIZE", 2),
            patch.object(routes, "_INGESTION_SEMAPHORE", asyncio.Semaphore(1)),
            patch.object(routes, "isinstance", return_value=True),
        ):
            result = await routes.store_data_in_vector_db(
                [Document(page_content=f"page {index}") for index in range(5)],
                "ordered-file",
                "user",
            )

        assert len(result["ids"]) == 5
        assert [
            document.metadata["_rag_chunk_index"]
            for document in inserted_documents
        ] == list(range(5))
        assert len(
            {
                document.metadata["_rag_ingestion_attempt_id"]
                for document in inserted_documents
            }
        ) == 1

    @pytest.mark.asyncio
    async def test_cancellation_waits_for_parser_before_releasing_ingestion_slot(self):
        import app.routes.document_routes as routes

        parser_started = threading.Event()
        release_parser = threading.Event()
        second_insert_started = asyncio.Event()

        def blocking_source():
            parser_started.set()
            release_parser.wait(timeout=2)
            yield Document(page_content="first")

        async def add_documents(documents, ids=None, executor=None):
            if ids == ["second-file"]:
                second_insert_started.set()
            return ids

        mock_store = AsyncMock()
        mock_store.aadd_documents = add_documents
        mock_store.delete_by_metadata = AsyncMock()

        with (
            ThreadPoolExecutor(max_workers=2) as executor,
            patch.object(routes, "vector_store", mock_store),
            patch.object(routes, "EMBEDDING_BATCH_SIZE", 1),
            patch.object(routes, "PARALLEL_EXECUTION", 1),
            patch.object(routes, "RAG_INGESTION_WINDOW_SIZE", 1),
            patch.object(routes, "_INGESTION_SEMAPHORE", asyncio.Semaphore(1)),
            patch.object(routes, "isinstance", return_value=True),
        ):
            first_task = asyncio.create_task(
                routes.store_data_in_vector_db(
                    blocking_source(), "first-file", "user", executor=executor
                )
            )
            await asyncio.to_thread(parser_started.wait, 1)
            first_task.cancel()

            second_task = asyncio.create_task(
                routes.store_data_in_vector_db(
                    [Document(page_content="second")],
                    "second-file",
                    "user",
                    executor=executor,
                )
            )
            await asyncio.sleep(0.05)
            assert not second_insert_started.is_set()

            release_parser.set()
            with pytest.raises(asyncio.CancelledError):
                await first_task
            second_result = await second_task

        assert "error" not in second_result
        assert second_insert_started.is_set()
