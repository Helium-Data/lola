import os
import re
import asyncio
import json
import datetime
import unicodedata
from typing import Dict, List, Union
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
    IngestionCache,
)
from pydrive2.fs import GDriveFileSystem
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.google import GoogleDriveReader
from llama_index.core import (
    SummaryIndex,
    VectorStoreIndex,
    Document,
    SimpleKeywordTableIndex,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.storage import StorageContext

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

from llama_index.storage.kvstore.redis import RedisKVStore as RedisCache
from llama_index.storage.docstore.redis import RedisDocumentStore
from llama_index.vector_stores.redis import RedisVectorStore
from llama_index.storage.index_store.redis import RedisIndexStore
from redisvl.schema import IndexSchema
from llama_index.readers.file import PDFReader, DocxReader

from dotenv import load_dotenv

load_dotenv()

DOC_DIRS = {
    "HR": "1a59en4nkmKnfI7vxO7_CyMEBSmEAeNFM",
    # "Legal": "",
    # "IT": ""
}
REDIS_HOST = os.environ["REDIS_HOST"]
REDIS_PORT = int(os.environ["REDIS_PORT"])

LLM = Ollama(model="phi4", request_timeout=60.0)
EMBED_MODEL = OllamaEmbedding(
    model_name="nomic-embed-text:latest",
)
RESET_INDEX = False

# Set up the ingestion cache layer
cache = IngestionCache(
    cache=RedisCache.from_host_and_port(REDIS_HOST, REDIS_PORT),
    collection="lola_redis_cache",
)
doc_store = RedisDocumentStore.from_host_and_port(
    REDIS_HOST, REDIS_PORT, namespace="lola_document_store"
)
index_schema = IndexSchema.from_dict(dict(json.load(open("custom_redis_vector_schema.json", "r"))))
vector_store = RedisVectorStore(
    schema=index_schema,
    redis_url=f"redis://{REDIS_HOST}:{REDIS_PORT}",
)
# parser = PyMuPDFReader()
file_extractor = {".pdf": PDFReader(), ".docx": DocxReader()}

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
        EMBED_MODEL
    ],
    docstore=doc_store,
    cache=cache,
    docstore_strategy=DocstoreStrategy.UPSERTS,
    vector_store=vector_store
)

storage_context = StorageContext.from_defaults(
    index_store=index_store,
    docstore=pipeline.docstore,
)


def clean_text_func(text):
    clean_text = unicodedata.normalize("NFKD", text).strip()
    clean_text = clean_text.replace('\n', '')
    clean_text = " ".join(re.split(r'[ ]{3,}', clean_text))
    clean_text = clean_text.replace("●", "")
    return clean_text


async def load_resource(gfs, resource, dir_team):
    loader = SimpleDirectoryReader(
        input_files=[resource],
        fs=gfs,
        file_extractor=file_extractor,
        required_exts=[".pdf"]
    )
    pages = loader.load_data(fs=gfs)
    docs = []
    for page in pages:
        page.metadata["doc_team"] = dir_team
        page.metadata["updated_at"] = str(datetime.datetime.now())

        cleaned = clean_text_func(page.text)

        doc = Document(
            text=cleaned,
            metadata=page.metadata,
            id_=page.id_
        )
        docs.append(doc)
    return resource, docs


async def load_from_dir(dir_id: str, dir_team: str):
    print(f"Loading from {dir_team}...")
    gfs = GDriveFileSystem(
        dir_id,
        client_id="",
        client_secret=""
    )
    dir_resources = None
    for root, dnames, fnames in gfs.walk(dir_id):
        dir_resources = [f"{dir_id}/{res}" for res in fnames if res.split('.')[-1] == "pdf"]
        break

    dir_docs_resources = await asyncio.gather(
        *[load_resource(gfs, resource, dir_team) for resource in dir_resources[:5]])
    dir_docs = {k: v for k, v in dir_docs_resources}
    return dir_docs


async def load_from_drive():
    print("Loading from drive...")
    drive_docs = {}
    for doc_dir in DOC_DIRS:
        dir_docs = await load_from_dir(DOC_DIRS[doc_dir], doc_dir)
        drive_docs.update(dir_docs)
    return drive_docs


async def create_nodes(
        drive_docs: Dict[str, List[Document]],
        index_ids: Dict[str, str] = None
) -> Union[Dict[str, str], None]:
    print("Create nodes...")
    all_docs = [do for doc in drive_docs.values() for do in doc]
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=EMBED_MODEL)

    doc_nodes = pipeline.run(documents=all_docs)

    if not doc_nodes:
        return None

    if index_ids:
        # load indexes
        print("Loading indices")
        summary_index = load_index_from_storage(
            storage_context=storage_context, index_id=index_ids["summary_index"]
        )
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store, embed_model=EMBED_MODEL
        )
        keyword_table_index = load_index_from_storage(
            storage_context=storage_context, index_id=index_ids["keyword_table_index"]
        )

        # delete docs from all indexes
        print("Deleting old docs")
        for doc in doc_nodes:
            summary_index.delete(doc.id_)
            vector_index.delete(doc.id_)
            keyword_table_index.delete(doc.id_)

        # insert new doc nodes
        print("Add new docs")
        summary_index.insert_nodes(doc_nodes)
        vector_index.insert_nodes(doc_nodes)
        keyword_table_index.insert_nodes(doc_nodes)
    else:
        print("Creating new indices")
        summary_index = SummaryIndex(nodes=doc_nodes, storage_context=storage_context)

        keyword_table_index = SimpleKeywordTableIndex(
            doc_nodes, storage_context=storage_context, llm=LLM
        )

    # doc_summary = summary_index.as_query_engine(llm=LLM).query("Summarize document")
    # print("DOc summary: ", doc_summary)

    storage_context.persist()
    drive_indices = {
        "summary_index": summary_index.index_id,
        "vector_index": vector_index.index_id,
        "keyword_table_index": keyword_table_index.index_id,
    }

    # storage_context.persist()
    save_dict_to_json(drive_indices, "drive_indices.json")
    return drive_indices


def save_dict_to_json(indices: Dict[str, str], file_path: str):
    with open(file_path, 'w') as fp:
        json.dump(indices, fp)


def load_json_to_dict(file_path: str):
    with open(file_path, 'r') as fp:
        indices = json.load(fp)
    return indices


async def run_ingestion(indices_path: str):
    print("Running ingestor...")
    indices = None
    if os.path.exists(indices_path):
        indices = load_json_to_dict(indices_path)

    drive_docs = await load_from_drive()

    indices = await create_nodes(drive_docs, indices)
    return indices


if __name__ == '__main__':
    details = asyncio.run(run_ingestion("drive_indices.json"))
    print("Details: ", details)
