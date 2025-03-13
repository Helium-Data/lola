import json
import re
import asyncio
import pandas as pd
import nest_asyncio
from llama_index.core.indices.list.base import ListRetrieverMode
from tqdm import tqdm
from typing import Dict, List, Union, Tuple
from llama_index.core import (
    VectorStoreIndex,
    load_indices_from_storage,
)
from pydantic import ValidationError
from llama_index.core.schema import IndexNode
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector

from llama_index.core.query_engine import SubQuestionQueryEngine, RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, BaseTool
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.vector_stores import ExactMatchFilter, FilterCondition, MetadataFilters

import gspread
from .config import config

nest_asyncio.apply()


def prepare_tools() -> List[BaseTool] | None:
    """
    Function to convert indexes to tools (vector, summary), also create new functions that the AI agent can reference to extract information.
    :return: a list of tools for the LLM agent to use
    """
    print("Preparing tools...")
    tools: List[QueryEngineTool] = []

    # load indices
    indices = load_indices_from_storage(
        storage_context=config.STORAGE_CONTEXT
    )
    print(f"{len(indices)}: {[ind.index_id for ind in indices]}")

    if indices:
        # Build tools
        agents, summary = build_document_agents(indices)
        obj_qe = build_agent_objects(agents)
        # rqe_tool = build_router_engine(query_engine_tools)
        sub_qe = build_sub_question_qe(obj_qe)  # Optional: build sub question query engine
        description = (f"Use this tool to fetch answers, context and summaries on the company's "
                       f"policy and official documents.\n"
                       f"ALWAYS use this tool FIRST to check and retrieve information based on user's query!")
        # f"Available documents: {summary}")

        tools.append(
            QueryEngineTool(
                query_engine=sub_qe,
                metadata=ToolMetadata(
                    name="main_query_engine",
                    description=description,
                ),
            )
        )

    return tools


def build_document_agents(indices: List[BaseIndex]) -> Tuple[Dict[str, Dict[str, FunctionCallingAgent]], str]:
    print("Building document agents...")
    summary_prompt = "Describe in depth, the topics covered in the document."
    agents = {}  # Build agents dictionary
    all_doc_names: str = ""
    for index in tqdm(indices):
        fname = "_".join(index.index_id.split("_")[:-2])
        fname = fname.strip().replace("(", "").replace(")", "").replace(".", "")
        query_engine_tools: List[QueryEngineTool] = []

        if "summary_index" in index.index_id:
            print(f"index_id: {index.index_id}, {index.index_struct}")
            sqe = index.as_query_engine(llm=config.LLM, retriever_mode=ListRetrieverMode.EMBEDDING,
                                        embed_model=config.EMBED_MODEL)
            summary = self_retry(sqe.query, summary_prompt)
            all_doc_names += f"- Document: {fname}, Summary: {summary}\n"

            query_engine_tools.append(
                QueryEngineTool(
                    query_engine=sqe,
                    metadata=ToolMetadata(
                        name=f"{fname}_summary_tool",
                        description=(
                            f"Useful for summarization questions related to {fname}. \n"
                        ),
                    ),
                )
            )

            # Get query engine for single document
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=config.VECTOR_STORE, embed_model=config.EMBED_MODEL
            )
            filters_ = [ExactMatchFilter(key="tag_name", value=fname)]  # specify the filter type
            filters = MetadataFilters(
                filters=filters_,
                condition=FilterCondition.AND,
            )
            retriever = VectorIndexRetriever(
                index=vector_index,
                similarity_top_k=3,
                filters=filters
            )
            rqe = RetrieverQueryEngine(
                retriever=retriever
            )

            sub_qe = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=[
                    QueryEngineTool(
                        query_engine=rqe,
                        metadata=ToolMetadata(
                            name=f"{fname[:-5]}_base_vector_tool",
                            description=(
                                f"Useful for retrieving specific context from {fname}. \n"
                            ),
                        ),
                    )
                ],
                use_async=True,
                llm=config.LLM
            )

            query_engine_tools.append(
                QueryEngineTool(
                    query_engine=sub_qe,
                    metadata=ToolMetadata(
                        name=f"{fname[:-5]}_sub_vector_tool",
                        description=(
                            f"Useful for retrieving specific context from {fname}. \n"
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

            agents[fname] = {
                "agent": agent,
                "summary": f"{summary}"
            }

    return agents, all_doc_names


def build_agent_objects(agents_dict: Dict[str, Dict[str, FunctionCallingAgent]]):
    objects = []
    for agent_label in agents_dict:
        # define index node that links to these agents
        policy_summary = (
            f"This content contains company policy documents about {agent_label}. Use"
            " this index if you need to lookup specific facts about: "
            f"{agents_dict[agent_label]['summary']}."
            f"\nDo not use this index if you want to analyze multiple documents."
        )
        node = IndexNode(
            text=policy_summary, index_id=f"{agent_label}_agent_object", obj=agents_dict[agent_label]["agent"]
        )
        objects.append(node)

    # define top-level retriever
    vector_index = VectorStoreIndex(
        objects=objects,
    )
    objects_query_engine = vector_index.as_query_engine(similarity_top_k=1, verbose=True)
    return objects_query_engine


def build_router_engine(query_engine_tools):
    query_engine = RouterQueryEngine(
        selector=LLMMultiSelector.from_defaults(llm=config.LLM),
        query_engine_tools=query_engine_tools,
        llm=config.LLM
    )
    tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="policy_engine",
            description="Useful for getting context on different company policy documents.",
        ),
    )
    return tool


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


def self_retry(func, *args, n_retries=5):
    """
    Invoke the model 'n_loops' times to ensure we get a valid response from the model.
    :param func: the function generating the response
    :param n_retries: maximum number of retries
    :param args: extra parameters for the 'func' method
    :return:
    """
    for _ in range(n_retries):
        try:
            response = func(*args)
            if response:
                return response
        except ValidationError as e:
            print(f"Validation error: {e}")
        except Exception as e:
            print(f"Other error: {e}")
    return None


async def get_bot_id(app_):
    # get the bot's own user ID so it can tell when somebody is mentioning it
    auth_response = await app_.client.auth_test()
    bot_user_id_ = auth_response["user_id"]
    return bot_user_id_


async def clean_response(response_text, async_func):
    if "assistant" in response_text:
        if async_func:
            await async_func("Typing...")
        # pattern = r'assistant\s*.*?\n'
        # response_text = re.sub(pattern, '', response_text)
        response_text = response_text.replace("assistant: ", "")
        response_text = response_text.replace("**", "*")  # format to slack's bold syntax. *Bold*
        return response_text


async def get_user_name(user_id, app_):
    user_info = await app_.client.users_info(user=user_id)
    user_name = user_info['user']['name']
    user_display_name = user_info['user']['profile']['display_name']
    return user_name, user_display_name
