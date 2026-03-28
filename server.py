"""
RAG Chat Server (Starlette + Qdrant + FastEmbed)
High-throughput server supporting Ollama and Gemini LLM providers.
"""

import os
import sys
import json
import uuid
import asyncio
import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from abc import ABC, abstractmethod
from dotenv import load_dotenv
import uvicorn
import httpx
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, PayloadSchemaType
)
from fastembed import TextEmbedding


# ============================================================================================================
# 1. Configuration
# ============================================================================================================
load_dotenv()

@dataclass
class ServerConfig:
    """Starlette config"""
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "5500"))
    server_cors_origins: List[str] = field(default_factory=lambda: os.getenv("SERVER_CORS_ORIGINS", "*").split(","))
    server_static_dir: str = os.getenv("SERVER_STATIC_DIR", "./frontend/dist")
@dataclass
class EmbeddingConfig:
    """FastEmbed config"""
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_cache_dir: str = os.getenv("EMBEDDING_CACHE_DIR", "./.cache/embeddings")
    embedding_threadpool_size: int = int(os.getenv("EMBEDDING_THREADPOOL_SIZE", "64"))
    embedding_batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
@dataclass
class QdrantConfig:
    """Qdrant config"""
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
    qdrant_collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "rag")
    qdrant_vector_size: int = int(os.getenv("QDRANT_VECTOR_SIZE", "384"))
    qdrant_upsert_batch_size: int = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "16"))
    qdrant_asyncio_writer_count: int = int(os.getenv("QDRANT_ASYNCIO_WRITER_COUNT", "64"))
    qdrant_asyncio_task_queue_timeout: float = float(os.getenv("QDRANT_ASYNCIO_TASK_QUEUE_TIMEOUT", 5.0)) # seconds
    qdrant_top_k: int = int(os.getenv("QDRANT_TOP_K", "5"))
    qdrant_min_score: float = float(os.getenv("QDRANT_MIN_SCORE", "0.6"))
@dataclass
class LLMConfig:
    """LLM config base class"""
    host: str = None
    model: str = None
    api_key: str = None
    timeout: float = float(os.getenv("LLM_TIMEOUT", "300.0"))

@dataclass
class LLMConfigGemini(LLMConfig):
    """LLM config for Gemini"""
    host: str = os.getenv("GEMINI_HOST", "generativelanguage.googleapis.com")
    model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
    api_key: str = os.getenv("GEMINI_API_KEY", "")

@dataclass
class LLMConfigOllama(LLMConfig):
    """LLM config for Ollama"""
    host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "gemma3:4b")

def create_logger(name: str = "server", log_file: str = "logs/server.log") -> logging.Logger:
    """Logger config."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-6s | %(threadName)-6s | %(thread)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

# ============================================================================================================
# 2. Data Models
# ============================================================================================================

@dataclass
class Document:
    """Document model."""
    src_id: str
    doc_id: str
    title: str = ""
    author: str = ""
    page: int = -1
    line: int = -1
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class Reference:
    """Reference model."""
    ref_id: str
    title: str = ""
    author: str = ""
    page: int = -1
    line: int = -1
    content: str = ""
    score: float = 0.0

@dataclass
class ChatMessage:
    """Chat message model."""
    role: str
    content: str
    references: List[Reference] = field(default_factory=list)

# ============================================================================================================
# 3. Embedding Service
# ============================================================================================================

class EmbeddingService:
    """Embedding service using FastEmbed."""
    def __init__(self, config: EmbeddingConfig, logger: logging.Logger):
        """Initialize the embedding service.
        
        Args:
            config: Embedding config.
            logger: Logger instance.
        """
        self._config = config
        self._logger = logger
        self._model = None
        self._lock = None
        self._threadpool = None
        self._initialized = False
    
    def _get_lock(self) -> asyncio.Lock:
        """Lazily create lock (must be created within event loop)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
    
    async def initialize(self):
        """Initialize the embedding model (thread-safe)."""
        if self._initialized:
            return
        try:
            async with self._get_lock():
                if self._initialized:
                    return
                if self._threadpool is None:
                    self._threadpool = ThreadPoolExecutor(
                        max_workers=self._config.embedding_threadpool_size,
                        thread_name_prefix="embedding-"
                    )
                    self._logger.info(f"[Embedding] Threadpool initialized (max_workers={self._config.embedding_threadpool_size})")
                if self._model is None:
                    cache_path = Path(self._config.embedding_cache_dir)
                    cache_path.mkdir(parents=True, exist_ok=True)
                    loop = asyncio.get_event_loop()
                    self._model = await loop.run_in_executor(
                        self._threadpool,
                        lambda: TextEmbedding(
                            model_name=self._config.embedding_model,
                            cache_dir=str(cache_path)
                        )
                    )
                    self._logger.info(f"[Embedding] Model {self._config.embedding_model} initialized")
                self._initialized = True
        except Exception as e:
            self._logger.error(traceback.format_exc())
            self._logger.error(f"[Embedding] Error initializing: {e}")
            raise
     
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using parallel threads."""
        if not self._initialized:
            await self.initialize()
        if not texts:
            return []
        loop = asyncio.get_event_loop()
        # Split texts into chunks
        chunks = [texts[i:i + self._config.embedding_batch_size] for i in range(0, len(texts), self._config.embedding_batch_size)]
        # Submit each chunk as task to threadpool
        tasks = [
            loop.run_in_executor(
                self._threadpool,
                lambda t=chunk: [list(emb) for emb in self._model.embed(t)]
            )
            for chunk in chunks
        ]
        # Wait for all threads to complete
        results = await asyncio.gather(*tasks)
        # Flatten results
        embeddings = [emb for chunk_result in results for emb in chunk_result]
        self._logger.debug(f"[Embedding] Generated embeddings for {len(texts)} texts in {len(chunks)} chunks")
        return embeddings

    async def embed(self, text: str) -> List[float]:
        """Get embedding for a single text.
        
        Args:
            text: Input text to embed.
        
        Returns:
            List of embedding vectors.
        """
        embeddings = await self.embed_batch([text])
        return list(embeddings[0])
    
    async def shutdown(self):
        """Cleanup resources.
        
        Shutdown the threadpool and release model resources.
        """
        self._logger.info("[Embedding] Shutting down...")
        if self._threadpool is not None:
            self._threadpool.shutdown(wait=True)
            self._threadpool = None
        self._model = None
        self._lock = None
        self._initialized = False
        self._logger.info("[Embedding] Shutdown complete")

# ============================================================================================================
# 4. Qdrant Store
# ============================================================================================================

class QdrantStore:
    """Async Qdrant store with multiple batch writers."""
    
    def __init__(self, config: QdrantConfig, logger: logging.Logger):
        """Initialize Qdrant store.
        
        Args:
            config: Qdrant config.
            logger: Logger instance.
        """
        self._config = config
        self._logger = logger
        self._async_client: Optional[AsyncQdrantClient] = None
        self._batch_queue: Optional[asyncio.Queue] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        self._writer_tasks: List[asyncio.Task] = []
        self._initialized: bool = False
    
    async def initialize(self) -> None:
        """Initialize Qdrant client and start writers."""
        if self._initialized:
            return
        self._async_client = AsyncQdrantClient(
            host=self._config.qdrant_host,
            port=self._config.qdrant_port
        )
        self._batch_queue = asyncio.Queue()
        self._shutdown_event = asyncio.Event()
        await self._ensure_collection()
        for i in range(self._config.qdrant_asyncio_writer_count):
            task = asyncio.create_task(self._batch_writer(writer_id=i))
            self._writer_tasks.append(task)
        self._initialized = True
        self._logger.info(f"[Qdrant] Writers initialized (writer_count={self._config.qdrant_asyncio_writer_count})")
    
    async def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        exists = await self._async_client.collection_exists(self._config.qdrant_collection_name)
        if not exists:
            await self._async_client.create_collection(
                collection_name=self._config.qdrant_collection_name,
                vectors_config=VectorParams(
                    size=self._config.qdrant_vector_size,
                    distance=Distance.COSINE,
                    on_disk=True
                )
            )
            await self._async_client.create_payload_index(
                collection_name=self._config.qdrant_collection_name,
                field_name="title",
                field_schema=PayloadSchemaType.TEXT
            )
            await self._async_client.create_payload_index(
                collection_name=self._config.qdrant_collection_name,
                field_name="author",
                field_schema=PayloadSchemaType.KEYWORD
            )
            self._logger.info(f"[Qdrant] Collection {self._config.qdrant_collection_name} created")
        else:
            self._logger.info(f"[Qdrant] Collection {self._config.qdrant_collection_name} exists")

    async def _batch_writer(self, writer_id: int) -> None:
        """Background task that batches writes."""
        batch: List[tuple] = []
        
        while not self._shutdown_event.is_set():
            try:
                try:
                    item = await asyncio.wait_for(
                        self._batch_queue.get(),
                        timeout=self._config.qdrant_asyncio_task_queue_timeout
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass
                while len(batch) < self._config.qdrant_upsert_batch_size:
                    try:
                        item = self._batch_queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break
                if batch:
                    await self._flush_batch(batch, writer_id)
                    batch = []
            except asyncio.CancelledError:
                self._logger.info(f"[Qdrant] Writer {writer_id} cancelled")
                raise
            except Exception as e:
                self._logger.error(f"[Qdrant] Writer {writer_id} error: {e}")
                self._logger.error(traceback.format_exc())
                batch = []
    
    async def _flush_batch(self, batch: List[tuple], writer_id: int):
        """Flush a batch to Qdrant."""
        try:
            points = [
                PointStruct(id=doc_id, vector=vector, payload=payload)
                for doc_id, vector, payload in batch
            ]
            await self._async_client.upsert(
                collection_name=self._config.qdrant_collection_name,
                points=points,
                wait=False
            )
            self._logger.debug(f"[Qdrant] Writer {writer_id} flushed {len(points)} docs")
        except Exception as e:
            self._logger.error(f"[Qdrant] Writer {writer_id} flush error: {e}")
    
    async def add_document(self, doc_id: str, vector: List[float], payload: Dict[str, Any]):
        """Add a document to the batch queue."""
        if not self._initialized:
            await self.initialize()
        await self._batch_queue.put((doc_id, vector, payload))

    async def reset(self):
        """Delete all documents in the collection."""
        if not self._initialized:
            await self.initialize()
        await self._async_client.delete_collection(self._config.qdrant_collection_name)
        await self._ensure_collection()
        self._logger.debug(f"[Qdrant] Reset collection {self._config.qdrant_collection_name}")

    async def get_count(self) -> int:
        """Get the number of documents in the collection."""
        if not self._initialized:
            await self.initialize()
        count = await self._async_client.count(
            collection_name=self._config.qdrant_collection_name,
            exact=True
        )
        return count.count

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        min_score: float = 0.0,
        filter_conditions: Optional[Dict] = None
    ) -> List[Reference]:
        """Search for similar documents."""
        if not self._initialized:
            await self.initialize()
        query_filter = None
        if filter_conditions:
            must_conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_conditions.items()
            ]
            query_filter = Filter(must=must_conditions)
        
        results = await self._async_client.query_points(
            collection_name=self._config.qdrant_collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=min_score,
            query_filter=query_filter,
            with_payload=True
        )
        references = []
        for idx, point in enumerate(results.points):
            payload = point.payload
            references.append(Reference(
                ref_id=str(idx + 1),
                title=payload.get("title", ""),
                author=payload.get("author", ""),
                page=payload.get("page", -1),
                line=payload.get("line", -1),
                content=payload.get("content", ""),
                score=point.score
            ))
        references.sort(key=lambda x: x.score, reverse=True)
        for i, ref in enumerate(references):
            ref.ref_id = str(i + 1)
        return references
    
    async def shutdown(self):
        """Graceful shutdown."""
        if not self._initialized:
            return
        self._logger.info("[Qdrant] Shutting down...")
        self._shutdown_event.set()
        for task in self._writer_tasks:
            task.cancel()
        await asyncio.gather(*self._writer_tasks, return_exceptions=True)
        remaining = []
        while not self._batch_queue.empty():
            try:
                remaining.append(self._batch_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        
        if remaining:
            await self._flush_batch(remaining, writer_id=-1)
        await self._async_client.close()
        self._initialized = False
        self._logger.info("[Qdrant] Shutdown complete")


# ============================================================================================================
# 5. LLM Providers
# ============================================================================================================

class LLMProvider(ABC):
    """Base class for LLM providers."""
    
    def __init__(self, config: LLMConfig, logger: logging.Logger):
        self._config = config
        self._logger = logger
    
    @abstractmethod
    async def stream(
        self,
        client_request: Request,
        chat_history: List[ChatMessage],
        system_prompt: str = "",
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Stream LLM response.
        Yields tuples of (event_type, content).
        Event types: "text", "error"
        """
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Google Gemini API provider."""
    
    def __init__(self, config: LLMConfigGemini, logger: logging.Logger):
        super().__init__(config, logger)
        self._config: LLMConfigGemini = config
    
    async def stream(
        self,
        client_request: Request,
        chat_history: List[ChatMessage],
        system_prompt: str = "",
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Handle streaming from request arrival to completion.
        Yields tuples of (event_type, content).
        """
        # Validate API key
        if not self._config.api_key:
            self._logger.error("[Gemini] API key not configured")
            yield ("error", "Internal server error")
            return
        # Early check for user abort
        if await client_request.is_disconnected():
            self._logger.info("[Gemini] Client disconnected")
            return
        # Build contents from ChatMessage objects
        contents = self._build_contents(chat_history)
        # Build payload
        payload = {"contents": contents}
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        # Stream from Gemini
        async for event in self._stream_from_api(client_request, payload):
            yield event
    
    def _build_contents(self, chat_history: List[ChatMessage]) -> List[Dict[str, Any]]:
        """Build Gemini API contents from chat history."""
        contents = []
        for i, msg in enumerate(chat_history):
            text = msg.content
            # Add references to last user message
            if i == len(chat_history) - 1 and msg.role == "user":
                text += "\n\nReferences:\n"
                if msg.references:
                    for ref in msg.references:
                        text += f"[{ref.ref_id}] {ref.title} by {ref.author}, page {ref.page}, line {ref.line}: {ref.content}\n"
                else:
                    text += "None\n"
            contents.append({
                "role": "user" if msg.role == "user" else "model",
                "parts": [{"text": text}]
            })
        return contents
    
    async def _stream_from_api(
        self,
        client_request: Request,
        payload: Dict[str, Any]
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """Stream from Gemini API with error handling."""
        try:
            async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                async with client.stream(
                    method="POST",
                    url=f"https://{self._config.host}/v1beta/models/{self._config.model}:streamGenerateContent?alt=sse",
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": self._config.api_key
                    }
                ) as response:
                    # Check response status from Gemini API
                    if response.status_code != 200:
                        error_body = await response.aread()
                        self._logger.error(f"[Gemini] API error {response.status_code}: {error_body}")
                        yield ("error", f"Internal server error")
                        return
                    # Stream response lines
                    async for line in response.aiter_lines():
                        # Check for client disconnect
                        if await client_request.is_disconnected():
                            self._logger.info("[Gemini] Client disconnected")
                            return
                        # Skip non-data lines
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        # Skip empty data
                        if not data:
                            continue
                        # Check for end of stream
                        if data == "[DONE]":
                            break
                        # Parse JSON
                        try:
                            parsed = json.loads(data)
                            text = (
                                parsed
                                .get("candidates", [{}])[0]
                                .get("content", {})
                                .get("parts", [{}])[0]
                                .get("text", "")
                            )
                            if text:
                                yield ("text", text)
                        except json.JSONDecodeError:
                            self._logger.debug(f"[Gemini] JSON decode error: {data}")
        # Client abort
        except asyncio.CancelledError:
            self._logger.info("[Gemini] Stream aborted by client")
            raise   # Must raise to cancel stream
        # Connection errors (Server → Gemini)
        except httpx.ConnectError as e:
            self._logger.error(f"[Gemini] Connection error: {e}")
            yield ("error", "Internal server error")
        except httpx.ConnectTimeout as e:
            self._logger.error(f"[Gemini] Connection timeout: {e}")
            yield ("error", "Internal server error")
        # Read errors (Gemini → Server)
        except httpx.ReadError as e:
            self._logger.error(f"[Gemini] Read error: {e}")
            yield ("error", "Internal server error")
        except httpx.ReadTimeout:
            self._logger.error("[Gemini] Read timeout")
            yield ("error", "Internal server error")
        # Other HTTP errors
        except httpx.HTTPStatusError as e:
            self._logger.error(f"[Gemini] HTTP error: {e}")
            yield ("error", "Internal server error")
        # Unexpected errors
        except Exception as e:
            self._logger.error(f"[Gemini] Unexpected error: {e}")
            yield ("error", "Internal server error")

class OllamaProvider(LLMProvider):
    """Ollama API provider."""
    
    def __init__(self, config: LLMConfigOllama, logger: logging.Logger):
        super().__init__(config, logger)
        self._config: LLMConfigOllama = config
    
    async def stream(
        self,
        client_request: Request,
        chat_history: List[ChatMessage],
        system_prompt: str = "",
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """
        Handle streaming from request arrival to completion.
        Yields tuples of (event_type, content).
        """
        # Early check for user abort
        if await client_request.is_disconnected():
            self._logger.info("[Ollama] Client disconnected")
            return
        # Build messages from ChatMessage objects
        messages = self._build_messages(chat_history, system_prompt)
        # Build payload
        payload = {
            "model": self._config.model,
            "messages": messages,
            "stream": True
        }
        # Stream from Ollama
        async for event in self._stream_from_api(client_request, payload):
            yield event
    
    def _build_messages(self, chat_history: List[ChatMessage], system_prompt: str) -> List[Dict[str, Any]]:
        """Build Ollama API messages from chat history."""
        messages = []
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        # Add chat history
        for i, msg in enumerate(chat_history):
            content = msg.content
            # Add references to last user message
            if i == len(chat_history) - 1 and msg.role == "user":
                content += "\n\nReferences:\n"
                if msg.references:
                    for ref in msg.references:
                        content += f"[{ref.ref_id}] {ref.title} by {ref.author}, page {ref.page}, line {ref.line}: {ref.content}\n"
                else:
                    content += "None\n"
            messages.append({
                "role": msg.role,
                "content": content
            })
        return messages
    
    async def _stream_from_api(
        self,
        client_request: Request,
        payload: Dict[str, Any]
    ) -> AsyncGenerator[Tuple[str, str], None]:
        """Stream from Ollama API."""
        try:
            async with httpx.AsyncClient(timeout=self._config.timeout) as client:
                async with client.stream(
                    method="POST",
                    url=f"{self._config.host}/api/chat",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    # Check response status from Ollama API
                    if response.status_code != 200:
                        error_body = await response.aread()
                        self._logger.error(f"[Ollama] API error {response.status_code}: {error_body}")
                        yield ("error", "Internal server error")
                        return
                    # Stream response lines (Ollama returns NDJSON)
                    async for line in response.aiter_lines():
                        # Check for client disconnect
                        if await client_request.is_disconnected():
                            self._logger.info("[Ollama] Client disconnected")
                            return
                        # Skip empty lines
                        if not line.strip():
                            continue
                        # Parse JSON
                        try:
                            parsed = json.loads(line)
                            # Check if stream is done
                            if parsed.get("done", False):
                                break
                            # Extract text from message content
                            text = parsed.get("message", {}).get("content", "")
                            if text:
                                yield ("text", text)
                        except json.JSONDecodeError:
                            self._logger.debug(f"[Ollama] JSON decode error: {line}")
        # Client abort
        except asyncio.CancelledError:
            self._logger.info("[Ollama] Stream aborted by client")
            raise  # Must raise to cancel stream
        # Connection errors (Server → Ollama)
        except httpx.ConnectError as e:
            self._logger.error(f"[Ollama] Connection error: {e}")
            yield ("error", "Failed to connect to Ollama. Is it running?")
        except httpx.ConnectTimeout as e:
            self._logger.error(f"[Ollama] Connection timeout: {e}")
            yield ("error", "Connection to Ollama timed out")
        # Read errors (Ollama → Server)
        except httpx.ReadError as e:
            self._logger.error(f"[Ollama] Read error: {e}")
            yield ("error", "Internal server error")
        except httpx.ReadTimeout:
            self._logger.error("[Ollama] Read timeout")
            yield ("error", "Ollama response timed out")
        # Other HTTP errors
        except httpx.HTTPStatusError as e:
            self._logger.error(f"[Ollama] HTTP error: {e}")
            yield ("error", "Internal server error")
        # Unexpected errors
        except Exception as e:
            self._logger.error(f"[Ollama] Unexpected error: {e}")
            yield ("error", "Internal server error")
# ============================================================================================================
# 6. RAG Helpers
# ============================================================================================================

def create_system_prompt(rag_mode: bool, has_references: bool) -> str:
    """Create system prompt based on mode."""
    if not rag_mode:
        return "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."
    if has_references:
        return """Answer the user's question using ONLY the provided references.
Rules:
- Use ONLY information from the references. Never use outside knowledge.
- Cite sources using [1], [2], etc. For multiple sources, use [1][2][3] format, NOT [1, 2, 3].
- If references don't contain relevant information, respond exactly: "I don't know based on the provided references."
- Be concise and direct."""
    else:
        return """No relevant references were found for this question.
Respond exactly: "I don't know - no relevant documents found in the knowledge base."
Do not attempt to answer the question."""

# ============================================================================================================
# 7. Knowledge Base Builder
# ============================================================================================================

async def create_knowledge_base(
    file_path: str,
    embedder: EmbeddingService,
    qdrant_store: QdrantStore,
    logger: logging.Logger,
    batch_size: int = 1024,
) -> None:
    """Create knowledge base from a JSONL file with batched processing."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    logger.info(f"[Server] Loading local knowledge from {file_path}")
    stats = {"total": 0, "success": 0, "failed": 0}

    async def _process_batch(batch_docs: List[Dict[str, Any]], batch_content: List[str]) -> None:
        vectors = await embedder.embed_batch(batch_content)
        for doc, vector in zip(batch_docs, vectors):
            doc_id = str(uuid.uuid4())
            qdrant_store._batch_queue.put_nowait((doc_id, vector, doc))
            stats["success"] += 1
        logger.debug(f"[Server] Batch of {len(batch_docs)} documents processed")
    
    batch_docs: List[Dict[str, Any]] = []
    batch_content: List[str] = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                doc = json.loads(line)
                content = doc.get("content", "")
                if not content:
                    stats["failed"] += 1
                    continue
                payload = {
                    "title": doc.get("title", ""),
                    "author": doc.get("author", ""),
                    "page": doc.get("page", -1),
                    "line": doc.get("line", -1),
                    "content": content,
                }
                batch_docs.append(payload)
                batch_content.append(content)
                if len(batch_docs) >= batch_size:
                    await _process_batch(batch_docs, batch_content)
                    logger.info(f"[Server] Processed {stats['success']}/{stats['total']} documents...")
                    batch_docs.clear()
                    batch_content.clear()
            except json.JSONDecodeError as e:
                logger.error(f"[Server] JSON parse error: {e}")
                stats["failed"] += 1
    # Process remaining
    if batch_docs:
        await _process_batch(batch_docs, batch_content)
    logger.info(f"[Server] {stats['success']}/{stats['total']} documents indexed")

# ============================================================================================================
# 8. API Handlers
# ============================================================================================================
async def document_count_handler(request: Request) -> JSONResponse:
    """
    Get the number of documents in the knowledge base.
    """
    # Get services from app state
    logger: logging.Logger = request.app.state.logger
    qdrant_store: QdrantStore = request.app.state.qdrant_store
    try:
        result = await qdrant_store.get_count()
        return JSONResponse({"count": result})
    except Exception as e:
        logger.error(f"[Server][api/chat/count] Failed to get count: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

async def chat_stream_handler(request: Request) -> StreamingResponse:
    """
    Handle chat streaming with optional RAG.

    Request body:
    {
        "messages": json.dumps(List[ChatMessage]),
        "provider": "gemini" | "ollama",
        "rag_mode": true | false,
        "topk": int,
        "threshold": float
    }

    SSE Events:
    - event: reference, data: {"content": json.dumps(List[Reference])}
    - event: text, data: {"content": str}
    - event: error, data: {"content": str}
    - event: done, data: {"content": str}
    """
async def chat_stream_handler(request: Request) -> StreamingResponse:
    """
    Handle chat streaming with optional RAG.
    """
    # Get services from app state
    logger: logging.Logger = request.app.state.logger
    embedder: EmbeddingService = request.app.state.embedder
    qdrant_store: QdrantStore = request.app.state.qdrant_store
    qdrant_config: QdrantConfig = request.app.state.qdrant_config
    gemini_provider: GeminiProvider = request.app.state.gemini
    ollama_provider: OllamaProvider = request.app.state.ollama

    # Parse request body BEFORE generator
    logger.info(f"[DEBUG] Content-Type: {request.headers.get('content-type')}")
    body_bytes = await request.body()
    logger.info(f"[DEBUG] Raw body length: {len(body_bytes)}")
    logger.info(f"[DEBUG] Raw body: {body_bytes[:500]}")

    try:
        request_data = json.loads(body_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        logger.error(f"[Server][api/chat/stream] Failed to parse request: {e}")
        # Return error as StreamingResponse
        async def error_gen():
            error_payload = json.dumps({"content": "Invalid request body"})
            yield f"event: error\ndata: {error_payload}\n\n"
        return StreamingResponse(
            error_gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # Extract parameters here (before generator)
    messages = request_data.get("messages", [])
    provider_name = request_data.get("provider", "ollama").lower()
    rag_mode = request_data.get("rag_mode", True)
    top_k = request_data.get("topk", qdrant_config.qdrant_top_k)
    min_score = request_data.get("threshold", qdrant_config.qdrant_min_score)

    async def generate():
        try:
            # Validate messages
            if not messages:
                logger.error("[Server][api/chat/stream] No messages provided")
                error_payload = json.dumps({"content": "No message provided"})
                yield f"event: error\ndata: {error_payload}\n\n"
                return

            # Check provider and select the appropriate one
            if provider_name == "gemini":
                llm_provider = gemini_provider
            elif provider_name == "ollama":
                llm_provider = ollama_provider
            else:
                logger.error(f"[Server][api/chat/stream] Unknown LLM provider: {provider_name}")
                error_payload = json.dumps({"content": f"Invalid LLM provider: {provider_name}"})
                yield f"event: error\ndata: {error_payload}\n\n"
                return

            # Build chat history from request
            chat_history: List[ChatMessage] = []
            for msg in messages:
                chat_history.append(ChatMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    references=[]
                ))

            # Get last user message for RAG
            last_message = chat_history[-1] if chat_history else None
            references: List[Reference] = []

            # Perform RAG search if enabled
            if rag_mode and last_message and last_message.role == "user" and last_message.content:
                try:
                    query_vector = await embedder.embed(last_message.content)
                    references = await qdrant_store.search(
                        query_vector=query_vector,
                        top_k=top_k,
                        min_score=min_score
                    )
                    last_message.references = references
                    last_message.content = "YOU MUST FOLLOW SYSTEM INSTRUCTIONS TO RESPOND!\n\n" + last_message.content
                    reference_payload = json.dumps({"content": [asdict(ref) for ref in references]})
                    yield f"event: reference\ndata: {reference_payload}\n\n"
                    logger.info(f"[Server][api/chat/stream] RAG search found {len(references)} references")
                except Exception as e:
                    last_message.references = []
                    reference_payload = json.dumps({"content": []})
                    yield f"event: reference\ndata: {reference_payload}\n\n"
                    logger.error(f"[Server][api/chat/stream] RAG search error: {e}")

            # Create system prompt
            system_prompt = create_system_prompt(
                rag_mode=rag_mode,
                has_references=len(references) > 0
            )

            # Stream from the selected LLM provider
            async for event_type, content in llm_provider.stream(
                client_request=request,
                chat_history=chat_history,
                system_prompt=system_prompt
            ):
                payload = json.dumps({"content": content})
                yield f"event: {event_type}\ndata: {payload}\n\n"

            # Send done event
            done_payload = json.dumps({})
            yield f"event: done\ndata: {done_payload}\n\n"

        except asyncio.CancelledError:
            logger.info("[Server][api/chat/stream] Client disconnected")
        except Exception as e:
            logger.error(f"[Server][api/chat/stream] Unexpected error: {e}")
            logger.error(traceback.format_exc())
            error_payload = json.dumps({"content": "Internal server error"})
            yield f"event: error\ndata: {error_payload}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# ============================================================================================================
# 9. Application Setup
# ============================================================================================================

def create_app(knowledge_file_path: str, total_documents_count: int) -> Starlette:
    """Create and configure the Starlette application."""

    async def lifespan(app: Starlette):
        """Application lifespan handler for startup and shutdown.""" 
        # Startup
        logger = app.state.logger
        logger.info("[Server] Starting up...")
        # Initialize services
        embedder: EmbeddingService = app.state.embedder
        qdrant_store: QdrantStore = app.state.qdrant_store
        await embedder.initialize()
        await qdrant_store.initialize()
        logger.info("[Server] All services initialized")
        logger.info(f"[Server] Checking knowledge integrity...")
        count = await qdrant_store.get_count()
        logger.info(f"[Server] Qdrant store has {count} documents")
        if count != total_documents_count:
            logger.info(f"[Server] Discrepancy detected: {count} documents instead of {total_documents_count}")
            logger.info(f"[Server] Resetting knowledge base...")
            await qdrant_store.reset()
            # Create knowledge base
            await create_knowledge_base(
                file_path=knowledge_file_path,
                qdrant_store=app.state.qdrant_store,
                embedder=app.state.embedder,
                logger=app.state.logger,
            )
        logger.info("[Server] Knowledge base is up to date")
        yield  # Server runs

        # Shutdown
        logger.info("[Server] Shutting down...")
        await qdrant_store.shutdown()
        await embedder.shutdown()
        logger.info("[Server] Shutdown complete")

    # Create configs
    server_config = ServerConfig()
    embedding_config = EmbeddingConfig()
    qdrant_config = QdrantConfig()
    gemini_config = LLMConfigGemini()
    ollama_config = LLMConfigOllama()
    # Create logger
    logger = create_logger()
    # Create services (will be initialized in lifespan)
    embedder = EmbeddingService(config=embedding_config, logger=logger)
    qdrant_store = QdrantStore(config=qdrant_config, logger=logger)
    gemini = GeminiProvider(config=gemini_config, logger=logger)
    ollama = OllamaProvider(config=ollama_config, logger=logger)
    # Define routes
    routes = [
        Route("/api/chat/stream", chat_stream_handler, methods=["POST"]),
        Route("/api/document/count", document_count_handler, methods=["GET"]),
    ]
    # Add static files if directory exists
    static_dir = Path(server_config.server_static_dir)
    if static_dir.exists():
        routes.append(Mount("/", app=StaticFiles(directory=str(static_dir), html=True), name="static"))
    # Define middleware
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=server_config.server_cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ]
    # Create app
    app = Starlette(
        routes=routes,
        middleware=middleware,
        lifespan=lifespan,
    )
    
    # Attach services to app state
    app.state.logger = logger
    app.state.embedder = embedder
    app.state.qdrant_store = qdrant_store
    app.state.gemini = gemini
    app.state.ollama = ollama
    app.state.server_config = server_config
    app.state.embedding_config = embedding_config
    app.state.qdrant_config = qdrant_config
    app.state.gemini_config = gemini_config
    app.state.ollama_config = ollama_config
    
    return app


# ============================================================================================================
# 10. Entry Point
# ============================================================================================================

def main():
    """Main entry point."""
    total_documents_count = 13168
    app = create_app("./rag/output/total_snippets.jsonl", total_documents_count) 
    uvicorn.run(
        app,
        host=app.state.server_config.server_host,
        port=app.state.server_config.server_port,
        log_level="info",
    )

if __name__ == "__main__":
    main()