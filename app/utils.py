import json
import pandas as pd
from typing import Dict, List, Any, Union
from llama_index.core import (
    VectorStoreIndex,
    load_indices_from_storage,
)
from llama_index.core.schema import IndexNode
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, BaseTool

import gspread
from .config import config


def prepare_tools() -> List[BaseTool] | None:
    """
    Function to convert indexes to tools (vector, summary), also create new functions that the AI agent can reference to extract information.
    :return: a list of tools for the LLM agent to use
    """
    print("Preparing tools...")
    tools = []

    # load indices
    indexes = load_indices_from_storage(
        config.STORAGE_CONTEXT
    )

    if indexes:
        # Build tools
        agents = build_document_agents(indexes)
        obj_qe = build_agent_objects(agents)
        sub_qe = build_sub_question_qe(obj_qe)  # Optional: build sub question query engine

        tools.append(
            QueryEngineTool(
                query_engine=sub_qe,
                metadata=ToolMetadata(
                    name="sub_query_engine",
                    description="Useful for getting context on different company policy documents.",
                ),
            )
        )

    return tools


def build_document_agents(indices: List[BaseIndex]) -> Dict[str, FunctionCallingAgent]:
    print("Building document agents...")
    summary_prompt = "Write a one sentence summary about the document"
    agents = {}  # Build agents dictionary
    for index in indices:
        fname = "_".join(index.index_id.split("_")[:-2])
        query_engine_tools = []

        if "summary_index" in index.index_id:
            sqe = index.as_query_engine()

            summary = sqe.query(summary_prompt)

            query_engine_tools.append(
                QueryEngineTool(
                    query_engine=sqe,
                    metadata=ToolMetadata(
                        name=f"{fname}_summary_tool",
                        description=(
                            f"Useful for summarization questions related to {fname}. \n"
                            f"Document summary: {summary}"
                        ),
                    ),
                )
            )

        if "vector_index" in index.index_id:
            vqe = index.as_query_engine()
            summary = vqe.query(summary_prompt)

            query_engine_tools.append(
                QueryEngineTool(
                    query_engine=vqe,
                    metadata=ToolMetadata(
                        name=f"{fname}_vector_tool",
                        description=(
                            f"Useful for retrieving specific context from {fname}. \n"
                            f"Document summary: {summary}"
                        ),
                    ),
                )
            )

        # build agent
        agent = FunctionCallingAgent.from_tools(
            query_engine_tools,
            llm=config.LLM,
            verbose=True,
        )

        agents[fname] = agent

    return agents


def build_agent_objects(agents_dict: Dict[str, FunctionCallingAgent]):
    objects = []
    for agent_label in agents_dict:
        # define index node that links to these agents
        policy_summary = (
            f"This content contains company policy documents about {agent_label}. Use"
            " this index if you need to lookup specific facts about"
            f" {agent_label}. \nDo not use this index if you want to analyze"
            " multiple documents."
        )
        node = IndexNode(
            text=policy_summary, index_id=f"{agent_label}_agent_object", obj=agents_dict[agent_label]
        )
        objects.append(node)

    # define top-level retriever
    vector_index = VectorStoreIndex(
        objects=objects,
    )
    objects_query_engine = vector_index.as_query_engine(similarity_top_k=1, verbose=True)
    return objects_query_engine


def build_sub_question_qe(objects_query_engine):
    query_engine_tools = [
        QueryEngineTool(
            query_engine=objects_query_engine,
            metadata=ToolMetadata(
                name="policy_engine",
                description="Useful for getting context on different company policy documents.",
            ),
        ),
    ]

    sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True,
        llm=config.LLM
    )
    return sub_query_engine


def get_glossary_sheet(sheet_key, worksheet: Union[int, str] = 0) -> pd.DataFrame:
    """
    Function to convert google sheet to a pandas dataframe
    :param sheet_key: the url to the Google spreadsheet
    :param worksheet: name or index of the workbook
    :return: a pandas dataframe
    """
    gcloud = gspread.service_account_from_dict(info=config.G_CREDENTIALS)
    sheet = gcloud.open_by_url(sheet_key)
    worksheet = sheet.get_worksheet(worksheet)
    list_rows_worksheet = worksheet.get_all_values()
    return pd.DataFrame(
        list_rows_worksheet[1:], columns=list_rows_worksheet[0]
    )


def load_glossary(glossary_dict: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    """
    Convert extracted glossary spreadsheet into a python dictionary for easy lookup
    :param glossary_dict: dictionary containing the team and location to team's glossary spreadsheet
    :return:
    """
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
    """
    Load index JSON file into python dictionary
    :param file_path: string path to json file
    :return: dictionary containing index ids
    """
    with open(file_path, 'r') as fp:
        indexes = json.load(fp)
    return indexes
