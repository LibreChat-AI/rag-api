# main.py
import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from starlette.responses import JSONResponse

from app.config import (
    VectorDBType,
    debug_mode,
    RAG_HOST,
    RAG_PORT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    PDF_EXTRACT_IMAGES,
    VECTOR_DB_TYPE,
    LogMiddleware,
    logger,
    vector_store,
)
from app.middleware import security_middleware
from app.routes import document_routes, extraction_routes, pgvector_routes
from app.services.database import PSQLDatabase, ensure_vector_indexes
from app.services.vector_store.factory import close_vector_store_connections


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic goes here
    # Create bounded thread pool executor based on CPU cores
    max_workers = min(
        int(os.getenv("RAG_THREAD_POOL_SIZE", str(os.cpu_count()))), 8
    )  # Cap at 8
    app.state.thread_pool = ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="rag-worker"
    )
    logger.info(
        f"Initialized thread pool with {max_workers} workers (CPU cores: {os.cpu_count()})"
    )

    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        await PSQLDatabase.get_pool()  # Initialize the pool
        await ensure_vector_indexes()

    yield

    # Cleanup logic
    if VECTOR_DB_TYPE == VectorDBType.PGVECTOR:
        try:
            logger.info("Closing asyncpg connection pool")
            await PSQLDatabase.close_pool()
            logger.info("asyncpg connection pool closed")
        except Exception as e:
            logger.warning("Failed to close asyncpg pool: %s", e)

    # Drain in-flight work before closing backing resources
    logger.info("Shutting down thread pool")
    app.state.thread_pool.shutdown(wait=True)
    logger.info("Thread pool shutdown complete")

    # Close vector store connections (MongoDB client / SQLAlchemy engine)
    try:
        close_vector_store_connections(vector_store)
    except Exception as e:
        logger.warning("Failed to close vector store connections: %s", e)


app = FastAPI(lifespan=lifespan, debug=debug_mode)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LogMiddleware)

app.middleware("http")(security_middleware)

# Set state variables for use in routes
app.state.CHUNK_SIZE = CHUNK_SIZE
app.state.CHUNK_OVERLAP = CHUNK_OVERLAP
app.state.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

# Include routers
app.include_router(document_routes.router)
if os.getenv("RAG_EXTRACTION_API_ENABLED", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}:
    app.include_router(extraction_routes.router)
if debug_mode:
    app.include_router(router=pgvector_routes.router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.debug("Validation error: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "message": "Request validation failed"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=RAG_HOST, port=RAG_PORT, log_config=None)
