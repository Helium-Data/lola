import os
import re
import asyncio
import json
import datetime
import unicodedata
from typing import Dict, List, Union, Tuple
from llama_index.core.ingestion import (
    DocstoreStrategy,
    IngestionPipeline,
)
from pydrive2.fs import GDriveFileSystem
from llama_index.core import SimpleDirectoryReader
from llama_index.core import (
    SummaryIndex,
    VectorStoreIndex,
    Document,
    SimpleKeywordTableIndex,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.readers.file import PDFReader, DocxReader

from dotenv import load_dotenv
from config import config

load_dotenv()


class LolaIngestionPipeline:
    DOC_DIRS = {
        "HR": "1a59en4nkmKnfI7vxO7_CyMEBSmEAeNFM",
        # "Legal": "",
        # "IT": ""
    }
    RESET_INDEX = False
    PIPELINE = IngestionPipeline(
        transformations=[
            SentenceSplitter(
                chunk_size=256,
                chunk_overlap=10,
            ),
            config.EMBED_MODEL
        ],
        docstore=config.DOC_STORE,
        cache=config.CACHE,
        docstore_strategy=DocstoreStrategy.UPSERTS,
        vector_store=config.VECTOR_STORE
    )

    def __init__(self):
        self.vector_store = config.VECTOR_STORE
        self.doc_store = config.DOC_STORE
        self.file_extractor = {".pdf": PDFReader(), ".docx": DocxReader()}
        self.storage_context = config.STORAGE_CONTEXT

        if self.RESET_INDEX:  # Manually reset all the data from the vector index
            asyncio.run(self.reset_indexes())

    async def reset_indexes(self):
        """
        Function to manually delete and remove all documents in the vector store
        :return:
        """
        self.vector_store.delete_index()
        doc_infos = await self.doc_store.aget_all_ref_doc_info()
        await asyncio.gather(*[self.doc_store.adelete_ref_doc(info) for info in doc_infos])

    def clean_text_func(self, text: str) -> str:
        """
        Preprocessing function to clean the extracted text from pdf documents.
        :param text: clean preprocessed text
        :return:
        """
        clean_text = unicodedata.normalize("NFKD", text).strip()  # convert Unicode string to normal 'form'
        clean_text = " ".join(re.split(r'[ ]{3,}', clean_text))  # remove occurrences with multiple new lines
        clean_text = clean_text.replace("●", "")  # remove '●' symbol from text
        return clean_text

    async def load_resource(self, gfs: GDriveFileSystem, resource: str, dir_team: str) -> Tuple[str, List[Document]]:
        """
        Load and read the text from all pages of a single document.
        :param gfs: initialized google file system object
        :param resource: filename of the resource to load
        :param dir_team: the team string whose file we are reading
        :return: a tuple containing the resource filename and list of processed document pages.
        """
        loader = SimpleDirectoryReader(
            input_files=[resource],
            fs=gfs,
            file_extractor=self.file_extractor,
            required_exts=[".pdf"]
        )  # read resource from the drive using specified extractors and remote file system (GDrive).
        pages = loader.load_data(fs=gfs)
        docs = []
        for page in pages:
            page.metadata["doc_team"] = dir_team  # update the pages' metadata with the team name
            page.metadata["updated_at"] = str(datetime.datetime.now())  # add new metadata

            cleaned = self.clean_text_func(page.text)

            doc = Document(
                text=cleaned,
                metadata=page.metadata,
                id_=page.id_
            )  # Replace existing document object with cleaned text, keep existing metadata
            docs.append(doc)
        return resource, docs

    async def load_from_dir(self, dir_id: str, dir_team: str) -> Dict[str, List[Document]]:
        """
        Load texts in all documents from a particular team.
        :param dir_id: the unique directory name to retrieve documents from
        :param dir_team: the team name.
        :return: a dictionary with the filename as keys and list of document as values.
        """
        print(f"Loading from {dir_team}...")
        gfs = GDriveFileSystem(
            dir_id,
            client_id=config.G_CLIENT_ID,
            client_secret=config.G_CLIENT_SECRET
        )  # initialize Google Drive file system with credentials
        dir_resources = None
        for root, dnames, fnames in gfs.walk(dir_id):  # walk through the directory and filter by extension type
            dir_resources = [f"{dir_id}/{res}" for res in fnames if res.split('.')[-1] == "pdf"]
            break

        dir_docs_resources = await asyncio.gather(
            *[self.load_resource(gfs, resource, dir_team) for resource in
              dir_resources])  # Run the "load_resource" asynchronously for each resource
        dir_docs = {k: v for k, v in dir_docs_resources}
        return dir_docs

    async def load_from_drive(self) -> Dict[str, List[Document]]:
        """
        Load documents from every team's Google Drive.
        :return: a dictionary with the filename as keys and list of document as values.
        """
        print("Loading from drive...")
        drive_docs = {}
        for doc_dir in self.DOC_DIRS:
            dir_docs = await self.load_from_dir(self.DOC_DIRS[doc_dir], doc_dir)
            drive_docs.update(dir_docs)
        return drive_docs

    async def create_nodes(
            self, drive_docs: Dict[str, List[Document]],
            index_ids: Dict[str, str] = None
    ) -> Union[Dict[str, str], None]:
        """
        Method to generate nodes (chunks) and embeddings from all the extracted documents. i.e. Single document agent.
        :param drive_docs: a dictionary of filenames and list of processed documents.
        :param index_ids: (Optional) a dictionary containing existing vector and summary index ids.
        :return:
        """
        print("Create nodes...")
        all_docs = [do for doc in drive_docs.values() for do in doc]  # flatten dictionary to get all documents
        vector_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store,
                                                          embed_model=config.EMBED_MODEL)  # initialize vector database to store embedding and metadata

        doc_nodes = self.PIPELINE.run(
            documents=all_docs)  # apply initialized transformation pipeline to list of documents

        if not doc_nodes:  # if no document has changed
            return None

        if index_ids:
            # load existing index
            print("Loading indexes")
            summary_index = load_index_from_storage(
                storage_context=self.storage_context, index_id=index_ids["summary_index"]
            )
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, embed_model=config.EMBED_MODEL
            )
            keyword_table_index = load_index_from_storage(
                storage_context=self.storage_context, index_id=index_ids["keyword_table_index"]
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
            # Creating new indexes
            summary_index = SummaryIndex(nodes=doc_nodes, storage_context=self.storage_context)

            keyword_table_index = SimpleKeywordTableIndex(
                doc_nodes, storage_context=self.storage_context, llm=config.LLM
            )

        self.storage_context.persist()  # Persist indexes (save to file)  TODO: Check if necessary
        drive_indexes = {
            "summary_index": summary_index.index_id,
            "vector_index": vector_index.index_id,
            "keyword_table_index": keyword_table_index.index_id,
        }

        self.save_dict_to_json(drive_indexes, "drive_indexes.json")  # save index ids to json file
        return drive_indexes

    def save_dict_to_json(self, indexes: Dict[str, str], file_path: str):
        """
        Helper method to save created index ids to json file.
        :param indexes: dictionary containing the created index ids
        :param file_path: file path to save the json file
        :return:
        """
        with open(file_path, 'w') as fp:
            json.dump(indexes, fp)

    def load_json_to_dict(self, file_path: str) -> Dict[str, str]:
        """
        load the index ids from a json file
        :param file_path: path to the json file
        :return: dictionary containing the created index ids
        """
        with open(file_path, 'r') as fp:
            indexes = json.load(fp)
        return indexes

    async def run_ingestion(self, indexes_path: str) -> Dict[str, str]:
        """
        Main function to run the ingestion pipeline.
        :param indexes_path: path to the json file containing the created index ids
        :return: dictionary containing the created index ids
        """
        print("Running ingestor...")
        indexes = None
        if os.path.exists(indexes_path):  # check for existing index id file
            indexes = self.load_json_to_dict(indexes_path)

        drive_docs = await self.load_from_drive()  # load data from drive

        indexes = await self.create_nodes(drive_docs, indexes)  # create nodes from extracted documents
        return indexes


if __name__ == '__main__':
    ingestor = LolaIngestionPipeline()
    details = asyncio.run(ingestor.run_ingestion("drive_indexes.json"))
    print("Details: ", details)
