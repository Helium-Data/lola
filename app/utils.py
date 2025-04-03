import json
import asyncio

import pandas as pd
import nest_asyncio
import requests
from llama_index.core.indices.list.base import ListRetrieverMode
from requests import HTTPError
from tqdm import tqdm
from typing import Dict, List, Union, Tuple, Any
from llama_index.core import (
    VectorStoreIndex,
    load_indices_from_storage, Document, Response,
    load_index_from_storage
)
from pydantic import ValidationError
from bs4 import BeautifulSoup
from llama_index.core.schema import IndexNode
from llama_index.core.agent import FunctionCallingAgent
from llama_index.core.indices.base import BaseIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.question_gen.llm_generators import LLMQuestionGenerator
from llama_index.core.query_engine import SubQuestionQueryEngine, RetrieverQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata, BaseTool, FunctionTool
from llama_index.core.indices.vector_store.retrievers.retriever import VectorIndexRetriever
from llama_index.core.vector_stores import ExactMatchFilter, FilterCondition, MetadataFilters

import gspread
from .config import config
from .prompts import DOC_AGENT_SYSTEM_PROMPT, DOC_SUMMARY_PROMPT, MAIN_QUERY_ENGINE_PROMPT, \
    CUSTOM_SUB_QUESTION_PROMPT_TMPL

nest_asyncio.apply()
SAGE_BASE_URL = "https://heliumhealthnigeria.sage.hr/api/"


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
    indices_index_ids = [ind.index_id for ind in indices]
    print(f"{len(indices)}: {indices_index_ids}")

    try:
        doc_vec_index = load_index_from_storage(
            storage_context=config.STORAGE_CONTEXT, index_id="doc_agent_vector_store"
        )
    except ValueError:
        doc_vec_index = None

    new_indices = get_doc_vector_indices(
        doc_vec_index=doc_vec_index,
        indices_index_ids=indices_index_ids
    )
    print(f"{len(new_indices)}: {new_indices}")

    if indices:
        if doc_vec_index is None or len(new_indices) > 0:
            # Build tools
            agents, summary = build_document_agents(indices)
            obj_qe = build_agent_objects(agents)
            # sub_qe = build_sub_question_qe(obj_qe)  # Optional: build sub question query engine for doc agent router
        else:
            # Load doc agent vector index from storage
            obj_qe = doc_vec_index.as_query_engine(similarity_top_k=2, verbose=True)

        tools.append(
            QueryEngineTool(
                query_engine=obj_qe,
                metadata=ToolMetadata(
                    name="main_query_engine",
                    description=MAIN_QUERY_ENGINE_PROMPT,
                ),
            )
        )

    tools.append(
        FunctionTool.from_defaults(async_fn=query_sage_kb)
    )

    sage_tools = build_sage_api_tools()
    tools.extend(sage_tools)

    return tools


def get_doc_vector_indices(doc_vec_index, indices_index_ids):
    if doc_vec_index is None:
        return []

    structs_node_ids = list(doc_vec_index.index_struct.to_dict()["nodes_dict"].keys())

    struct_docs = []
    for ids in structs_node_ids:
        try:
            struct_id = config.DOC_STORE.get_node(ids, raise_error=False)
            struct_docs.append(struct_id)
        except ValueError:
            print(f"Struct ID error: {ids}")

    doc_vec_index_ids = [
        doc.to_dict()["index_id"].split("_")[0]
        for doc in struct_docs
    ]

    new_indices = []
    for idx in indices_index_ids:
        if idx in ["doc_agent_vector_store", "For_LolaHR_-_FAQs_Document_summary_index"]:
            continue

        if idx.split("_")[0] not in doc_vec_index_ids:
            new_indices.append(idx)

    return new_indices

def build_sage_api_tools():
    sage_tools = []

    async def _request_api(url, method="GET", data=None) -> Union[None, Dict[str, Any]]:
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Auth-Token': config.SAGE_HR_API_KEY
        }
        try:
            resp = requests.request(method, url, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except HTTPError as e:
            print(f"HTTPError requesting {url}: {e}")
            return {
                "error": f"HTTPError requesting {url}: {e}"
            }
        except Exception as e:
            print(f"Other Exception requesting {url}: {e}")
            return {
                "error": f"Other Exception requesting {url}: {e}"
            }

    async def get_employee_name(employee_id: int) -> Union[str, Dict[str, Any]]:
        """
        Use this function to get the details of a single active employee in the company given the employee id.
        :param employee_id: an integer value representing the employee
        :return: a dictionary with the employee details.
        """
        url = f"{SAGE_BASE_URL}employees/{employee_id}"
        json_resp = await _request_api(url)
        if "error" in json_resp:
            return json_resp["error"]

        details = json_resp.get("data", {})
        return {
            "email": details.get("email"),
            "first_name": details.get("first_name"),
            "last_name": details.get("last_name"),
            "position": details.get("position"),
            "country": details.get("country"),
        }

    async def get_list_of_recently_terminated_employees(employee_name=None):
        """
        Use this function to fetch the list of recently terminated employees or to check for a particular terminated employee.
        Can also be useful to check if an employee is still within in the company.
        :param employee_name: (Optional) the employee name to filter list by.
        :return: detailed list of recently terminated employees.
        """
        url = f"{SAGE_BASE_URL}terminated-employees"
        json_resp = await _request_api(url)
        if "error" in json_resp:
            return json_resp["error"]

        employees = []
        data = json_resp.get("data", {})
        employees.extend(data)
        details = [
            {
                "email": details.get("email"),
                "first_name": details.get("first_name"),
                "last_name": details.get("last_name"),
                "position": details.get("position"),
                "country": details.get("country"),
                "termination_date": details.get("termination_date")
            } for details in employees
        ]

        if employee_name:
            details = [
                detail for detail in details if
                employee_name.lower() in f'{detail["first_name"].lower()} {detail["last_name"].lower()}'
            ]

        return details

    async def get_company_teams_list(team_name: str) -> Union[str, List[Dict[str, Any]]]:
        """
        Use this function to get the list of functional teams in the company.
        :param team_name: string containing the requested team name. E.g. "Data", "Public Health", "Product Management", "Operations & Strategy"
        :return: list of dictionary containing team names and a list of manager names.
        """
        if "team" in team_name:
            team_name = team_name.replace("team", "").strip()

        url = f"{SAGE_BASE_URL}teams"
        all_data = []

        json_resp = await _request_api(url)
        if "error" in json_resp:
            return json_resp["error"]

        all_data.extend(json_resp.get("data", []))
        meta: dict[str, Any] = json_resp.get("meta", {})
        if meta:
            total_pages = meta.get("total_pages", 0)
            current_page = meta.get("current_page", 0)
            while current_page < total_pages:
                url = f"{url}?page={int(current_page) + 1}"
                json_resp = await _request_api(url)
                all_data.extend(json_resp.get("data", []))

                meta = json_resp.get("meta", {})
                current_page = meta.get("current_page", 0)

        all_data = [{
            "team_id": data["id"],
            "team_name": data["name"],
            "managers": await asyncio.gather(*[get_employee_name(manager_id) for manager_id in data["manager_ids"]]),
            "employees": await asyncio.gather(
                *[get_employee_name(employee_id) for employee_id in data["employee_ids"]]),
        } for data in all_data if team_name in data["name"]]

        if not all_data:
            return f"Cannot find team with team name: {team_name}"

        return all_data

    sage_tools.append(
        FunctionTool.from_defaults(async_fn=get_company_teams_list)
    )

    sage_tools.append(
        FunctionTool.from_defaults(async_fn=get_list_of_recently_terminated_employees)
    )

    return sage_tools


def build_document_agents(indices: List[BaseIndex]) -> Tuple[Dict[str, Dict[str, FunctionCallingAgent]], str]:
    print("Building document agents...")
    agents: Dict[str, Dict[str, FunctionCallingAgent | str]] = {}  # Build agents dictionary
    all_doc_names: str = ""
    for index in tqdm(indices):
        fname = "_".join(index.index_id.split("_")[:-2])
        fname = fname.strip().replace("(", "").replace(")", "").replace(".", "")

        if "FAQ" in fname:
            fname = "FAQ_document"

        if "summary_index" in index.index_id:
            agent, doc_names = build_single_agent(
                index=index,
                fname=fname
            )
            all_doc_names += doc_names
            agents[fname] = agent

    return agents, all_doc_names


def build_single_agent(index: BaseIndex, fname=None, return_agent=True) -> Union[
    Tuple[Dict[str, FunctionCallingAgent | str], str], List[QueryEngineTool]]:
    if not fname:
        fname = "_".join(index.index_id.split("_")[:-2])
        fname = fname.strip().replace("(", "").replace(")", "").replace(".", "")

    query_engine_tools: List[QueryEngineTool] = []
    sqe = index.as_query_engine(llm=config.LLM, retriever_mode=ListRetrieverMode.EMBEDDING,
                                embed_model=config.EMBED_MODEL, choice_batch_size=3, similarity_top_k=3)
    summary = self_retry(sqe.query, DOC_SUMMARY_PROMPT)
    all_doc_names = f"- Document: {fname}, Summary: {summary}\n"
    print(f"index_id: {index.index_id}; fname: {fname}; Summary: {summary}")

    query_engine_tools.append(
        QueryEngineTool(
            query_engine=sqe,
            metadata=ToolMetadata(
                name=f"{fname}_summary_tool",
                description=(
                    f"Utilize this response template to effectively answer summarization questions that relate to "
                    f"the content of {fname}. This might involve distilling key points, highlighting main concepts, "
                    f"or rephrasing complex information into concise and easily digestible summaries."
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

    question_gen = LLMQuestionGenerator.from_defaults(llm=config.LLM,
                                                      prompt_template_str=CUSTOM_SUB_QUESTION_PROMPT_TMPL)
    sub_qe = SubQuestionQueryEngine.from_defaults(
        question_gen=question_gen,
        query_engine_tools=[
            QueryEngineTool(
                query_engine=rqe,
                metadata=ToolMetadata(
                    name=f"{fname[:-5]}_base_vector_tool",
                    description=(
                        f"Leverage this tool to access specific information within the {fname}, enabling users to "
                        f"quickly and accurately obtain context on similar topics or issues."
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
                    f"Leverage this tool to access specific information within the {fname}, enabling users to quickly "
                    f"and accurately obtain context on similar topics or issues."
                ),
            ),
        )
    )

    if return_agent:
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

        return {
            "agent": agent,
            "summary": f"{summary}"
        }, all_doc_names

    else:
        faq_description = ("This FAQ document provides answers to common questions related to workplace policies and "
                           "technical support. Here are the key topics: Onboarding & Documentation, Payroll & Benefits "
                           "Work Hours & Leave Policies, Probation & Performance Evaluation, and many more.")
        return [
            QueryEngineTool(
                query_engine=sqe,
                metadata=ToolMetadata(
                    name=f"{fname}_summary_tool",
                    description=(
                        f"Use this tool to get summaries about the FAQ document {fname}. {summary}"
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=sub_qe,
                metadata=ToolMetadata(
                    name=f"{fname[:-5]}_sub_vector_tool",
                    description=(
                        f"Use this tool to get specific FAQ context from {fname}. {summary}"
                    ),
                ),
            )
        ]


def build_agent_objects(agents_dict: Dict[str, Dict[str, FunctionCallingAgent]]):
    objects = []
    for agent_label in agents_dict:
        # define index node that links to these agents
        policy_summary = f"""
        This content contains company documents related to {agent_label}. 
        Summary: {agents_dict[agent_label]['summary']}
        These tool provide comprehensive information about {agent_label} and is intended for:
        - Referencing specific facts and guidelines outlined in the document.
        - Retrieving supporting context and evidence.
        
        Important: Please note that this index should be used when you need to look up detailed information on a specific topic or section within the documents. 
        """
        node = IndexNode(
            text=agents_dict[agent_label]['summary'], index_id=f"{agent_label}_agent_object",
            obj=agents_dict[agent_label]["agent"]
        )
        objects.append(node)

    # define top-level retriever
    vector_index = VectorStoreIndex(
        objects=objects,
        transformations=[
            SemanticSplitterNodeParser.from_defaults(
                embed_model=config.EMBED_MODEL
            )
        ],
        storage_context=config.STORAGE_CONTEXT
    )
    vector_index.set_index_id("doc_agent_vector_store")
    print(vector_index.index_struct)
    objects_query_engine = vector_index.as_query_engine(similarity_top_k=2, verbose=True)
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
                name="company_and_document_engine_vector_tool",
                description=("This tool is particularly useful for: "
                             "- Quickly obtaining concise summaries of various company policies and procedures. \n"
                             "- Gaining specific context or insights related to particular company documents, such as "
                             "employee handbooks, benefit guides, or regulatory compliance materials. \n"
                             "- Finding answers to frequently asked questions about company policies and practices. \n"
                             "- Streamlining the process of researching and understanding complex company information.\n"
                             "- Enhancing overall knowledge of a company's policies and procedures for employees, "
                             "partners, or stakeholders.")
            ),
        ),
    ]

    sub_query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=query_engine_tools,
        use_async=True,
        llm=config.LLM,
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
