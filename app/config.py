import os
import dotenv
import json
from redis import Redis
from llama_index.llms.ollama import Ollama
from llama_index.llms.lmstudio import LMStudio
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.storage.index_store.redis import RedisIndexStore
from redisvl.schema import IndexSchema
from llama_index.core.storage import StorageContext
from llama_index.core import Settings

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

dotenv.load_dotenv()

# Add Phoenix
span_phoenix_processor = SimpleSpanProcessor(
    HTTPSpanExporter(endpoint="http://localhost:6006//v1/traces")
)

# Add them to the tracer
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

# Instrument the application
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


class Config:
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS = Redis.from_url(f"redis://{REDIS_HOST}:{REDIS_PORT}")
    LLM = Ollama(model="llama3.2", request_timeout=60.0)
    EMBED_MODEL = OllamaEmbedding(
        model_name="nomic-embed-text",
    )
    DOC_STORE = RedisDocumentStore.from_redis_client(
        REDIS, namespace="lola_document_store"
    )
    VECTOR_STORE = RedisVectorStore(
        schema=IndexSchema.from_dict(dict(json.load(open("custom_redis_vector_schema.json", "r")))),
        redis_client=REDIS,
    )
    INDEX_STORE = RedisIndexStore.from_redis_client(
        REDIS, namespace="lola_index"
    )
    STORAGE_CONTEXT = StorageContext.from_defaults(
        index_store=INDEX_STORE,
        docstore=DOC_STORE,
    )
    SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
    SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
    G_CREDENTIALS = dict(json.load(open("token.json", "r")))
    G_CLIENT_ID = G_CREDENTIALS["client_id"]
    G_CLIENT_SECRET = G_CREDENTIALS["client_secret"]
    GLOSSARY_DICT = {
        "HR": "https://docs.google.com/spreadsheets/d/1_sSt--3wTpUpJLfzQt3oiKQvw1DVNQ4M7i9xqpGVJmg/edit?gid=0#gid=0"
    }

    Settings.llm = LLM
    Settings.embed_model = EMBED_MODEL


config = Config()
