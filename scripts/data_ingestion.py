import os
import asyncio
import json
from typing import Dict, List
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from llama_index.readers.google import GoogleDriveReader
from llama_index.core import SummaryIndex, VectorStoreIndex, Document, SimpleKeywordTableIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage import StorageContext

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.storage.index_store.redis import RedisIndexStore
from redisvl.schema import IndexSchema

from dotenv import load_dotenv

load_dotenv()

DOC_DIRS = {
    "HR": "1a59en4nkmKnfI7vxO7_CyMEBSmEAeNFM",
    # "Legal": "",
    # "IT": ""
}
REDIS_HOST = "localhost"
REDIS_PORT = 6379
LLM = Ollama(model="phi4", request_timeout=60.0)
EMBED_MODEL = OllamaEmbedding(
    model_name="nomic-embed-text",
)
RESET_INDEX = True

# Set up the ingestion cache layer
cache = IngestionCache(
    cache=RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT),
    collection="lola_redis_cache",
)
doc_store = RedisDocumentStore.from_host_and_port(
    REDIS_HOST, REDIS_PORT, namespace="lola_document_store"
)
vector_store = RedisVectorStore(
    schema=IndexSchema.from_dict(dict(json.load(open("custom_redis_vector_schema.json", "r")))),
    redis_url=f"redis://{REDIS_HOST}:{REDIS_PORT}",
)

if RESET_INDEX:
    async def reset_indexes():
        vector_store.delete_index()
        doc_infos = await doc_store.aget_all_ref_doc_info()
        await asyncio.gather(*[doc_store.adelete_ref_doc(info) for info in doc_infos])


    asyncio.run(reset_indexes())

index_store = RedisIndexStore.from_host_and_port(
    host=REDIS_HOST, port=REDIS_PORT, namespace="lola_index"
)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(
            chunk_size=256,
            chunk_overlap=10,
        ),
        EMBED_MODEL,
    ],
    docstore=doc_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.DUPLICATES_ONLY,
)


async def load_resource(loader, resource, dir_team):
    docs = loader.load_resource(resource)
    for doc in docs:
        doc.metadata["file_name"] = " ".join(doc.metadata["file path"].split(".")[:-1])
        doc.id_ = doc.metadata["file_name"]
        doc.metadata["doc_team"] = dir_team
    return resource, docs


async def load_from_dir(dir_id: str, dir_team: str):
    print(f"Loading from {dir_team}...")
    reader = GoogleDriveReader(
        folder_id=dir_id,
        credentials_path="credentials.json",
    )
    dir_resources = reader.list_resources()
    dir_docs_resources = await asyncio.gather(
        *[load_resource(reader, resource, dir_team) for resource in dir_resources[:3]])
    dir_docs = {k: v for k, v in dir_docs_resources}
    return dir_docs


async def load_from_drive():
    print("Loading from drive...")
    drive_docs = {}
    for doc_dir in DOC_DIRS:
        dir_docs = await load_from_dir(DOC_DIRS[doc_dir], doc_dir)
        drive_docs.update(dir_docs)
    return drive_docs


async def create_nodes_single(drive_docs: Dict[str, List[Document]]):
    print("Create nodes (single)...")
    drive_nodes = pipeline.run(documents=[no for nods in drive_docs.values() for no in nods], show_progress=True)

    if not drive_nodes:
        return None

    storage_context = StorageContext.from_defaults(
        index_store=index_store,
        docstore=pipeline.docstore,
    )

    summary_index = SummaryIndex(nodes=drive_nodes, storage_context=storage_context)
    vector_index = VectorStoreIndex(nodes=drive_nodes, storage_context=storage_context, embed_model=EMBED_MODEL)
    keyword_table_index = SimpleKeywordTableIndex(
        drive_nodes, storage_context=storage_context, llm=LLM
    )

    storage_context.persist()
    return {
        "summary_index": summary_index.index_id,
        "vector_index": vector_index.index_id,
        "keyword_table_index": keyword_table_index.index_id
    }


async def create_nodes_multi(drive_docs: Dict[str, List[Document]]):
    print("Create nodes (Multi)...")
    drive_nodes = {}
    for doc_id in drive_docs:
        doc_nodes = await pipeline.arun(documents=drive_docs[doc_id], show_progress=True, num_workers=2)

        if not doc_nodes:
            continue

        storage_context = StorageContext.from_defaults(
            index_store=index_store,
            docstore=pipeline.docstore,
        )
        summary_index = SummaryIndex(nodes=doc_nodes, storage_context=storage_context)
        vector_index = VectorStoreIndex(nodes=doc_nodes, storage_context=storage_context, embed_model=EMBED_MODEL)
        keyword_table_index = SimpleKeywordTableIndex(
            doc_nodes, storage_context=storage_context, llm=LLM
        )

        storage_context.persist()
        drive_nodes[doc_id] = {
            "summary_index": summary_index.index_id,
            "vector_index": vector_index.index_id,
            "keyword_table_index": keyword_table_index
        }
    return drive_nodes


async def run_ingestion(strategy="single"):
    print("Running ingestor...")
    drive_docs = await load_from_drive()
    print(len(drive_docs))
    if strategy == "single":
        storage_details = await create_nodes_single(drive_docs)
    else:
        storage_details = await create_nodes_multi(drive_docs)
    # persist storage details
    # json.dump()
    return storage_details


if __name__ == '__main__':
    details = asyncio.run(run_ingestion())
    print("Details: ", details)
