import json
import pandas as pd
from typing import Dict, List, Any, Union
from llama_index.core import (
    Response,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.vector_stores import (
    MetadataFilters,
    ExactMatchFilter,
    FilterCondition
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool, RetrieverTool

import gspread
from config import config


def prepare_tools(doc_indices: Dict[str, str]) -> Any:
    print("Preparing tools...")
    tools = []
    glossary = load_glossary(config.GLOSSARY_DICT)

    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=config.VECTOR_STORE, embed_model=config.EMBED_MODEL
    )
    summary_index = load_index_from_storage(
        config.STORAGE_CONTEXT, index_id=doc_indices["summary_index"]
    )
    keyword_table_index = load_index_from_storage(
        config.STORAGE_CONTEXT, index_id=doc_indices["keyword_table_index"]
    )

    tools.append(QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            llm=config.LLM,
            response_mode="tree_summarize",
            use_async=False,
        ),
        name=f"summary_tool",
        description=(
            "Useful for any requests or questions that require a holistic summary "
            "of EVERYTHING related to a document."
            f" To answer questions about more specific sections"
            f" of the document, please use vector_tool."
        ),
    ))

    # tools.append(RetrieverTool.from_defaults(
    #     retriever=keyword_table_index.as_retriever(),
    #     description=(
    #         "Useful for retrieving specific context using keywords from"
    #         " the documents."
    #     ),
    # ))

    def vector_search(query: str, filename: str = None) -> str:
        """
        Function useful for answering questions or queries related to a particular document.
        First get available documents from the 'get_team_glossary' tool before calling this tool.
        :param query: (Required) the detailed query string to answer. str
        :param filename: (Optional) the document filename to refine the query by.
        :return: Response object with answer to the provided query.
        """
        filters = None
        if filename:
            filters_ = [ExactMatchFilter(key="file_name", value=filename)]
            filters = MetadataFilters(
                filters=filters_,
                condition=FilterCondition.AND,
            )

        query_engine = vector_index.as_query_engine(
            llm=config.LLM,
            similarity_top_k=3,
            filters=filters,
        )


        response = query_engine.query(query)
        print("source: ", response.source_nodes)
        print("response: ", response.response)
        return response.response

    def get_team_glossary(team: str = "HR") -> dict[str, str] | list[dict[str, Any]]:
        """
        Function for getting a list of document filenames from a particular team name.
        Call this function first!
        :param team: the requested team name. e.g: 'HR', 'IT'
        :return: Response dictionary with filename (can be used as input into 'vector_search' tool) and a description of the document.
        """
        team_glossary = glossary.get(team)
        if not team_glossary:
            return {
                "error": "Team name does not exist."
            }

        return team_glossary

    tools.append(
        FunctionTool.from_defaults(
            fn=vector_search
        )
    )

    tools.append(
        FunctionTool.from_defaults(
            fn=get_team_glossary
        )
    )

    return tools


def get_glossary_sheet(sheet_key, worksheet: Union[int, str] = 0):
    gcloud = gspread.oauth(credentials_filename="credentials.json")
    sheet = gcloud.open_by_url(sheet_key)
    worksheet = sheet.get_worksheet(worksheet)
    list_rows_worksheet = worksheet.get_all_values()
    return pd.DataFrame(
        list_rows_worksheet[1:], columns=list_rows_worksheet[0]
    )


def load_glossary(glossary_dict: Dict[str, str]):
    glossary = {}
    for team in glossary_dict.keys():
        glossary_df = get_glossary_sheet(
            sheet_key=glossary_dict[team],
        )
        glossary[team] = [{
            "filename": row["filename"],
            "description": row["description"]
        } for idx, row in glossary_df.iterrows()]
    return glossary


def load_json_to_dict(file_path: str):
    with open(file_path, 'r') as fp:
        indices = json.load(fp)
    return indices
