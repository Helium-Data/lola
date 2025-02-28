import os
import dotenv
import json
import boto3
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.storage.index_store.redis import RedisIndexStore
from redisvl.schema import IndexSchema
from llama_index.core.storage import StorageContext
from llama_index.core import Settings
from llama_index.core.ingestion import IngestionCache

from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPSpanExporter,
)
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.bedrock import BedrockInstrumentor

dotenv.load_dotenv()

# Add Phoenix
span_phoenix_processor = SimpleSpanProcessor(
    HTTPSpanExporter(endpoint=os.environ.get("PHOENIX_ENDPOINT"))
)

# Add them to the tracer
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(span_processor=span_phoenix_processor)

# Instrument the application
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


# TODO: fix: "unable to track token usage on phoenix"
# BedrockInstrumentor().instrument(tracer_provider=tracer_provider)


# class contain configuration parameters for the application
class Config:
    REDIS_HOST = os.environ.get("REDIS_HOST")
    REDIS_PORT = int(os.environ.get("REDIS_PORT"))
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}"
    BOTO_CLIENT = boto3.client(
        "bedrock-runtime",
        region_name=os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")

    )
    LLM = BedrockConverse(
        model=os.environ.get("BEDROCK_LLM_MODEL_ID"),
        client=BOTO_CLIENT,
        region_name=os.environ.get("AWS_REGION"),
    )
    EMBED_MODEL = BedrockEmbedding(
        model_name=os.environ.get("BEDROCK_EMBED_MODEL_ID"),
        client=BOTO_CLIENT,
        region_name=os.environ.get("AWS_REGION"),
    )
    CACHE = IngestionCache(
        cache=RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT),
        collection="lola_redis_cache",
    )
    DOC_STORE = RedisDocumentStore.from_host_and_port(
        host=REDIS_HOST, port=REDIS_PORT, namespace="lola_document_store"
    )
    VECTOR_STORE = RedisVectorStore(
        schema=IndexSchema.from_dict(dict(json.load(open("custom_redis_vector_schema.json", "r")))),
        redis_url=REDIS_URL,
    )
    INDEX_STORE = RedisIndexStore.from_host_and_port(
        host=REDIS_HOST, port=REDIS_PORT, namespace="lola_index"
    )
    STORAGE_CONTEXT = StorageContext.from_defaults(
        index_store=INDEX_STORE,
        docstore=DOC_STORE,
    )
    SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
    SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
    G_CLIENT_ID = os.environ.get("G_CLIENT_ID")
    G_CLIENT_SECRET = os.environ.get("G_CLIENT_SECRET")
    G_PROJECT_ID = os.environ.get("G_PROJECT_ID")
    G_PRIVATE_KEY_ID = os.environ.get("G_PRIVATE_KEY_ID")
    G_CLIENT_EMAIL = os.environ.get("G_CLIENT_EMAIL")
    G_CLIENT_SERVICE_ID = os.environ.get("G_CLIENT_SERVICE_ID")
    G_CLIENT_CERT_URI = os.environ.get("G_CLIENT_CERT_URI")
    G_PRIVATE_KEY = os.environ.get("G_PRIVATE_KEY")
    GLOSSARY_DICT = {
        "HR": "https://docs.google.com/spreadsheets/d/1_sSt--3wTpUpJLfzQt3oiKQvw1DVNQ4M7i9xqpGVJmg/edit?gid=0#gid=0"
    }
    G_CREDENTIALS = {
        "type": "service_account",
        "project_id": G_PROJECT_ID,
        "private_key_id": G_PRIVATE_KEY_ID,
        "private_key": G_PRIVATE_KEY.replace(r"\n", "\n"),
        "client_email": G_CLIENT_EMAIL,
        "client_id": G_CLIENT_SERVICE_ID,
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": G_CLIENT_CERT_URI,
        "universe_domain": "googleapis.com"
    }


config = Config()

Settings.llm = config.LLM
Settings.embed_model = config.EMBED_MODEL
