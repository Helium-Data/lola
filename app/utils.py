import json
import re
import asyncio
import pandas as pd
import nest_asyncio
import requests
from llama_index.core.indices.list.base import ListRetrieverMode
from tqdm import tqdm
from typing import Dict, List, Union, Tuple
from llama_index.core import (
    VectorStoreIndex,
    load_indices_from_storage, Document, Response,
)
from pydantic import ValidationError
from bs4 import BeautifulSoup
from llama_index.core.schema import IndexNode
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector

from llama_index.core.query_engine import SubQuestionQueryEngine, RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, BaseTool, FunctionTool
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.vector_stores import ExactMatchFilter, FilterCondition, MetadataFilters

import gspread
from .config import config
from .prompts import DOC_AGENT_SYSTEM_PROMPT

nest_asyncio.apply()


def prepare_tools() -> List[BaseTool] | None:
    """
    Function to convert indexes to tools (vector, summary), also create new functions that the AI agent can reference to extract information.
    :return: a list of tools for the LLM agent to use
    """
    print("Preparing tools...")
    tools: List[BaseTool] = []

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

        tools.append(
            FunctionTool.from_defaults(async_fn=query_sage_kb)
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

        if "FAQ" in fname:
            continue

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
                system_prompt=DOC_AGENT_SYSTEM_PROMPT.format(
                    filename=fname,
                    summary_tool=f"{fname}_summary_tool",
                    vector_tool=f"{fname[:-5]}_sub_vector_tool"
                )
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


async def get_request(url: str):
    """
    Send a GET request to the specified URL and return the response text.

    :param url: The URL of the resource to retrieve.
    :return: The text content of the response, or None if the request fails.
    """
    try:
        print(f"getting req for {url}")
        r = requests.get(url)

        # Raise an exception for HTTP errors (4xx/5xx)
        r.raise_for_status()

        return r.text
    except Exception as e:
        print(f"error getting {url}: {e}")
        return None


async def parse_search_results(raw_html, k=5):
    """
    Parse the search results HTML to extract URLs of relevant pages.

    :param k: number of results to return
    :param raw_html: The HTML content of the search results page.
    :return: A list of URLs of relevant pages, or None if no results are found.
    """
    print(f"parsing search results")
    result_urls = []
    # Parse the HTML using BeautifulSoup
    html = BeautifulSoup(raw_html, "html.parser")
    # Find the div with class 'flex flex-col gap-3' (search results container)
    results_div = html.find("div", class_="flex flex-col gap-3")
    if not results_div:
        return None
    # Extract URLs from relevant pages
    results = results_div.find_all("div", class_="w-full")
    result_urls.extend([urls.find("a")["href"] for urls in results[:k]])
    return [f"https://support.sage.hr{url}" for url in result_urls]


async def parse_single_page(page_url):
    """
    Parse a single page and extract its content.

    :param page_url: The URL of the page to parse.
    :return: The text content of the page, or None if parsing fails.
    """
    print(f"parsing single {page_url}")
    page_raw = await get_request(page_url)
    if not page_raw:
        return None

    html = BeautifulSoup(page_raw, "html.parser")
    # Find the div with class 'article intercom-force-break' (content container)
    content = html.find("div", class_="article intercom-force-break")
    return content.get_text(strip=True) if content else None


def build_vector_index(documents):
    """
    Build a vector index from the given documents.

    :param documents: A list of Document objects to index.
    :return: The constructed VectorStoreIndex object.
    """
    print("Building index")
    llama_docs = [Document(text=doc) for doc in documents]
    vector_index = VectorStoreIndex.from_documents(documents=llama_docs)
    return vector_index.as_query_engine(llm=config.LLM)


async def query_sage_kb(query: str) -> Union[Response, None]:
    """
    This function utilizes the SageHR search functionality to extract relevant content related to application use, troubleshooting, and support.
    The information returned may include guides for common issues, step-by-step instructions, and links to additional
    resources such as FAQs, user manuals, or contact information for support teams.
    Query must be at least 2 words.

    :param query: The relevant search query, minimum of 2 words.
    :return: A Response object containing the query results, or None if no results found.
    """
    print("Quering SageKB")
    sage_kb_search_url = f"https://support.sage.hr/en/?q={query.replace(' ', '+')}"
    search_results_raw = await get_request(sage_kb_search_url)
    if not search_results_raw:
        return None

    search_results_urls = await parse_search_results(search_results_raw)
    if not search_results_urls:
        return None

    # Extract documents from each relevant page and build a vector index
    documents = await asyncio.gather(*[parse_single_page(url) for url in search_results_urls])

    query_engine = build_vector_index(documents=documents)
    return await query_engine.aquery(query)


if __name__ == '__main__':
    resp = asyncio.run(query_sage_kb("sagehr"))
    print(resp)
