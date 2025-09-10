import json
import asyncio
import re
import io
import pandas as pd
import nest_asyncio
import requests
from rapidfuzz import fuzz
from llama_index.core.indices.list.base import ListRetrieverMode
from requests import HTTPError
from tqdm import tqdm
from typing import Dict, List, Union, Tuple, Any, Optional
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
from config import config
from prompts import (
    DOC_AGENT_SYSTEM_PROMPT, DOC_SUMMARY_PROMPT, MAIN_QUERY_ENGINE_DESCRIPTION,
    CUSTOM_SUB_QUESTION_PROMPT_TMPL, FAQ_QUERY_ENGINE_DESCRIPTION
)
import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

nest_asyncio.apply()
SAGE_BASE_URL = "https://heliumhealthnigeria.sage.hr/api/"
EMPLOYEE_DIRECTORY: Union[pd.DataFrame | None] = None
EMPLOYEE_DIRECTORY_PATH = "1XnoZnA2jKP_pnEuq8OIbDYM06GLO58DE"


def prepare_tools() -> List[BaseTool] | None:
    """
    Function to convert indexes to tools (vector, summary), also create new functions that the AI agent can reference to extract information.
    :return: a list of tools for the LLM agent to use
    """
    global EMPLOYEE_DIRECTORY

    EMPLOYEE_DIRECTORY = download_excel_from_drive(EMPLOYEE_DIRECTORY_PATH)
    print("Preparing tools...")
    tools: List[BaseTool] = []

    # load indices
    indices = load_indices_from_storage(
        storage_context=config.STORAGE_CONTEXT
    )
    indices_index_ids = [ind.index_id for ind in indices if
                         "summary" in ind.index_id and "faq" not in ind.index_id.lower()]
    indices_faq_index_ids = [ind.index_id for ind in indices if
                             "summary" in ind.index_id and "faq" in ind.index_id.lower()]
    print(f"{len(indices)}: {indices_index_ids}, {indices_faq_index_ids}")

    try:
        doc_vec_index = load_index_from_storage(
            storage_context=config.STORAGE_CONTEXT, index_id="doc_agent_vector_store"
        )
        faq_doc_vec_index = load_index_from_storage(
            storage_context=config.STORAGE_CONTEXT, index_id="faq_doc_agent_vector_store"
        )
    except ValueError:
        doc_vec_index = None
        faq_doc_vec_index = None

    new_indices = get_doc_vector_indices(
        doc_vec_index=doc_vec_index,
        faq_vec_index=faq_doc_vec_index,
        indices_index_ids=indices_index_ids,
        faq_indices_index_ids=indices_faq_index_ids
    )
    print(f"{len(new_indices)}: {new_indices}")

    if indices:
        if doc_vec_index is None or len(new_indices) > 0:
            # Build tools
            agents, summary = build_document_agents(indices)
            obj_qe, faq_obj_qe = build_agent_objects(agents)
            # sub_qe = build_sub_question_qe(obj_qe)  # Optional: build sub question query engine for doc agent router
        else:
            # Load doc agent vector index from storage
            obj_qe = doc_vec_index.as_query_engine(similarity_top_k=2, verbose=True)
            faq_obj_qe = faq_doc_vec_index.as_query_engine(similalrity=1, verbose=True)

        tools.append(
            QueryEngineTool(
                query_engine=obj_qe,
                metadata=ToolMetadata(
                    name="main_query_engine",
                    description=MAIN_QUERY_ENGINE_DESCRIPTION,
                ),
            )
        )
        tools.append(
            QueryEngineTool(
                query_engine=faq_obj_qe,
                metadata=ToolMetadata(
                    name="faq_query_engine",
                    description=FAQ_QUERY_ENGINE_DESCRIPTION,
                ),
            )
        )

    tools.extend([
        FunctionTool.from_defaults(async_fn=query_sage_kb),
        FunctionTool.from_defaults(async_fn=get_core_values),
        FunctionTool.from_defaults(fn=search_employee_directory)
    ])

    # sage_tools = build_sage_api_tools()
    # tools.extend(sage_tools)

    return tools


def get_doc_vector_indices(
        doc_vec_index: Union[VectorStoreIndex | None],
        faq_vec_index: Union[VectorStoreIndex | None],
        indices_index_ids: List[str],
        faq_indices_index_ids: List[str],
):
    if doc_vec_index is None:
        return []

    structs_node_ids = list(doc_vec_index.index_struct.to_dict()["nodes_dict"].values())
    faq_structs_node_ids = list(faq_vec_index.index_struct.to_dict()["nodes_dict"].values())

    struct_docs = []
    for ids in structs_node_ids:
        try:
            struct_id = doc_vec_index.docstore.get_node(ids, raise_error=False)
            struct_docs.append(struct_id)
        except ValueError:
            print(f"Struct ID error: {ids}")

    for ids in faq_structs_node_ids:
        try:
            faq_id = faq_vec_index.docstore.get_node(ids, raise_error=False)
            struct_docs.append(faq_id)
        except ValueError:
            print(f"Faq ID error: {ids}")

    doc_vec_index_ids = [
        doc.to_dict()["index_id"].split("_")[0]
        for doc in struct_docs
    ]

    new_indices = []
    for idx in indices_index_ids:
        if idx.split("_")[0] not in doc_vec_index_ids + faq_indices_index_ids:
            new_indices.append(idx)

    return new_indices


def build_sage_api_tools():
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
        Use this tool to retrieve basic profile information about a single **active** employee,
        given their unique employee ID. The data returned includes non-sensitive identifying details
        such as the employee’s name, email, position, and country of work.

        Important:
        - This tool should only be used to retrieve **basic public-facing or role-related information**
          about employees within the company.
        - It will not return details for deactivated or former employees.
        - Do not use this tool to fetch sensitive HR data (e.g., salary, personal identifiers, or performance records).

        Parameters:
        - employee_id (int): The unique numerical ID of the employee.
        Returns:
        - A dictionary containing:
            - `first_name`
            - `last_name`
            - `email`
            - `position`
            - `country`
          Or a string error message if the employee is not found or inactive.
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
        Use this tool to retrieve a list of employees who have recently been terminated from the company.
        This can be used to verify if a specific employee has been offboarded or to fetch the full list
        of recent terminations for auditing or reporting purposes.

        Important:
        - This tool should only be used for retrieving official termination records.
        - It can also be used to check if an employee is no longer with the company.
        - It does **not** return information about current or active employees unless cross-checked manually.

        Parameters:
        - employee_name (str, optional): An optional full or partial name to filter the list by a specific individual.
        Returns:
        - A list of terminated employee records, each including:
            - first name
            - last name
            - email
            - position
            - country
            - termination date
          If an employee name is provided, the results will be filtered accordingly.
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

    async def get_company_teams_list(
            team_name: Optional[str] = None,
            page: int = 1,
            page_size: int = 10
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Use this tool to retrieve information about a specific functional team within the company.
        It returns details about the requested team including the team name, list of managers, and employees.

        Important:
        - The tool supports pagination. Use the `page` and `page_size` parameters to control results.
        - Partial matches on `team_name` are supported (e.g., "Product" will match "Product Management").
        - team_name query should be words not initials (e.g., "Human Resources" instead of "HR").
        - This function queries internal organizational data.

        Parameters:
        - team_name (str, optional): Name or partial name of the team (e.g., "Operations", "Product", None).
        - page (int, optional): The page number of results to return (default: 1).
        - page_size (int, optional): The number of teams per page (default: 10)
        Returns:
        - A list of dictionaries. Each dictionary contains:
            - team_id (str): Unique identifier of the team.
            - team_name (str): Name of the team.
            - managers (List[str]): Names of managers in the team.
            - employees (List[str]): Names of employees in the team.
        - If no teams match the name, returns a message indicating no results.
        """

        if team_name and "team" in team_name.lower():
            team_name = team_name.lower().replace("team", "").strip()

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
                current_page += 1
                next_url = f"{url}?page={current_page}"
                json_resp = await _request_api(next_url)
                all_data.extend(json_resp.get("data", []))
                meta = json_resp.get("meta", {})
                current_page = meta.get("current_page", current_page)

        # Filter by team name if provided
        if team_name:
            filtered_teams = [
                data for data in all_data if team_name.lower() in data["name"].lower()
            ]
        else:
            filtered_teams = all_data

        if not filtered_teams:
            return f"No teams found matching: {team_name}" if team_name else "No teams available."

        # Apply pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_teams = filtered_teams[start_idx:end_idx]

        # Enrich team data
        enriched_teams = []
        for data in paginated_teams:
            managers = await asyncio.gather(
                *[get_employee_name(mid) for mid in data.get("manager_ids", [])]
            )
            employees = await asyncio.gather(
                *[get_employee_name(eid) for eid in data.get("employee_ids", [])]
            )
            enriched_teams.append({
                "team_id": data["id"],
                "team_name": data["name"],
                "managers": managers,
                "employees": employees,
            })

        return enriched_teams

    async def find_employee_by_name(name: str, threshold: int = 80) -> List[Dict[str, Any]]:
        """
        Use this tool to search for employees by their full name (first name + last name). It returns detailed
        employee information including contact info, team, position, and employment status.

        Parameters:
        - name (str): Partial or full name of the employee to search for.
          Example: "John Doe", "Jane", "Adeola Smith"
        Returns:
        - A list of matching employee records, each as a dictionary. If no match is found, returns an empty list.
        """

        matches = []
        page = 1
        has_more = True

        while has_more:
            url = f"{SAGE_BASE_URL}employees?page={page}"
            response = await _request_api(url)

            if "error" in response:
                break  # Optionally raise an exception or return error

            employees = response.get("data", [])
            for employee in employees:
                full_name = f"{employee.get('first_name', '')} {employee.get('last_name', '')}".strip().lower()
                if name.lower() in full_name:
                    matches.append(employee)
                else:
                    match_score = fuzz.token_set_ratio(name, full_name)
                    if match_score >= threshold:
                        matches.append(employee)

            meta = response.get("meta", {})
            current_page = meta.get("current_page", page)
            total_pages = meta.get("total_pages", current_page)
            has_more = current_page < total_pages
            page += 1

        return matches

    async def find_employee_by_role(role: str, threshold: int = 80) -> List[Dict[str, Any]]:
        """
        Use this tool to retrieve employees whose job titles match a given role using fuzzy matching.
        It returns detailed employee records including position, team, and contact information.

        Parameters:
        - role (str): The job title or partial role to search for. Example: "Product Manager", "Engineer".
        - threshold (int, optional): Fuzzy match score threshold (0–100). Defaults to 80.
          Only employees whose position title matches the role above this threshold are returned.
        Returns:
        - A list of dictionaries for matching employees. Each dictionary includes employee details and their match score.
          If no match is found, returns an empty list.
        """

        matches = []
        page = 1
        has_more = True
        role = role.strip().lower()

        while has_more:
            url = f"{SAGE_BASE_URL}employees?page={page}"
            response = await _request_api(url)

            if "error" in response:
                break

            employees = response.get("data", [])
            for employee in employees:
                position = employee.get("position", "")
                if not position:
                    continue

                match_score = fuzz.token_set_ratio(role, position.lower())
                if match_score >= threshold:
                    matches.append({
                        **employee,
                        "match_score": match_score
                    })

            meta = response.get("meta", {})
            current_page = meta.get("current_page", page)
            total_pages = meta.get("total_pages", current_page)
            has_more = current_page < total_pages
            page += 1

        return matches

    sage_tools = [
        FunctionTool.from_defaults(async_fn=get_employee_name),
        # FunctionTool.from_defaults(async_fn=get_company_teams_list),
        FunctionTool.from_defaults(async_fn=get_list_of_recently_terminated_employees),
        FunctionTool.from_defaults(async_fn=find_employee_by_name),
        FunctionTool.from_defaults(async_fn=find_employee_by_role)
    ]

    return sage_tools


def build_document_agents(indices: List[BaseIndex]) -> Tuple[Dict[str, Dict[str, FunctionCallingAgent]], str]:
    print("Building document agents...")
    agents: Dict[str, Dict[str, FunctionCallingAgent | str]] = {}  # Build agents dictionary
    all_doc_names: str = ""
    for index in tqdm(indices):
        fname = "_".join(index.index_id.split("_")[:-2])
        fname = fname.strip().replace("(", "").replace(")", "").replace(".", "")

        # if "FAQ" in fname:
        #     fname = "FAQ_document"

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
    objects, faq_objects = [], []
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
        if "faq" in agent_label.lower():
            faq_objects.append(node)
        else:
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

    faq_vector_index = VectorStoreIndex(
        objects=faq_objects,
        transformations=[
            SemanticSplitterNodeParser.from_defaults(
                embed_model=config.EMBED_MODEL
            )
        ],
        storage_context=config.STORAGE_CONTEXT
    )

    vector_index.set_index_id("doc_agent_vector_store")
    faq_vector_index.set_index_id("faq_doc_agent_vector_store")

    objects_query_engine = vector_index.as_query_engine(similarity_top_k=2, verbose=True)
    faq_objects_query_engine = faq_vector_index.as_query_engine(similarity_top_k=1, verbose=True)
    return objects_query_engine, faq_objects_query_engine


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


def download_excel_from_drive(file_id: str) -> pd.DataFrame:
    """Download an Excel file from Google Drive and return as DataFrame."""
    creds, _ = google.auth.load_credentials_from_dict(info=config.G_CREDENTIALS)

    try:
        # Create Drive API client
        service = build("drive", "v3", credentials=creds)

        # Download file as bytes
        request = service.files().get_media(fileId=file_id)
        file_bytes = io.BytesIO()
        downloader = MediaIoBaseDownload(file_bytes, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Download {int(status.progress() * 100)}%.")

        # Reset pointer before reading
        file_bytes.seek(0)

        # Load into pandas
        df = pd.read_excel(file_bytes)
        df.columns = ['S/N', 'First name', 'Last name', 'Work email', 'Entity',
       'Department', 'Team', 'Job Title', 'Line Manager Position',
       'Line Manager Name', 'Gender', 'Nationality', 'Location', 'Country']
        return df

    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


def search_employee_directory(
        first_name: str = None,
        last_name: str = None,
        work_email: str = None,
        entity: str = None,
        department: str = None,
        team: str = None,
        job_title: str = None,
        line_manager_position: str = None,
        line_manager_name: str = None,
        gender: str = None,
        nationality: str = None,
        location: str = None,
        country: str = None,
        threshold: int = 70
) -> List[Dict[str, Any]]:
    """
    Search an employee dataframe with flexible partial and fuzzy matching.

    Parameters:
        Employee directory with the following columns (parameters):
            ['First name', 'Last name', 'Work email', 'Entity', 'Department', 'Team',
             'Job Title', 'Line Manager Position', 'Line Manager Name', 'Gender',
             'Nationality', 'Location', 'Country']
        Each additional parameter is an optional search filter (string).
        threshold (int): Minimum similarity score (0-100) for fuzzy matching.

    Returns:
        pd.DataFrame: Filtered employee records that match the criteria.
    """
    df_filtered = EMPLOYEE_DIRECTORY.copy().fillna("").astype(str)

    # Filters mapping
    filters = {
        "First name": first_name,
        "Last name": last_name,
        "Work email": work_email,
        "Entity": entity,
        "Department": department,
        "Team": team,
        "Job Title": job_title,
        "Line Manager Position": line_manager_position,
        "Line Manager Name": line_manager_name,
        "Gender": gender,
        "Nationality": nationality,
        "Location": location,
        "Country": country,
    }

    for col, val in filters.items():
        if val:  # apply only if user passed something
            df_filtered = df_filtered[
                df_filtered[col].apply(
                    lambda x: fuzz.partial_ratio(str(val).lower(), x.lower()) >= threshold
                )
            ]

    return df_filtered.reset_index(drop=True).to_dict("records")


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


async def get_core_values():
    """Retrieves the core values of Helium.

    This function returns a dictionary containing the core values
    of Helium Health, along with descriptions explaining what each value means within
    the company's context. The descriptions provide insight into how these
    values are implemented and reflected in the organization's culture and practices.

    :return: A dictionary where keys are the names of the core values (strings)
             and values are their corresponding descriptions (strings).
    """
    return {
        "Simplicity": ("Simplicity is a key factor in driving the adoption of innovation, and thus, our products "
                       "are simple and easy to use.\nOur organizational structure is tailored in the same fashion. "
                       "No one is barricaded by bureaucracy and every staff member is easily accessible "
                       "and approachable"),
        "Boldness": "Take smart risks and make tough decisions without excessive agonizing.",
        "Innovation": ("We take on the biggest challenges with passion and extreme attention to detail. \n"
                       "You re- conceptualize issues to discover practical solutions to hard problems and "
                       "challenge prevailing assumptions when warranted and suggest better approaches."),
        "Camaraderie": ("We want everyone that works at Helium to feel that they are in a place where they "
                        "are comfortable enough to excel at their jobs and be who they’re meant to be.")
    }


async def query_sage_kb(query: str) -> Union[Response, None]:
    """
    Use this tool to search the SageHR public support knowledge base for user-facing help articles.
    It returns documentation relevant to application usage, troubleshooting, configuration guidance, and
    frequently asked questions. This includes step-by-step instructions, feature explanations, and links
    to relevant help resources.

    Important:
    - This tool is intended **only** for retrieving public support content about the SageHR product.
    - It should **not** be used to look up internal company information, employee data, or proprietary staff records.
    - The query must be at least two words to ensure meaningful search results.

    Parameters:
    - query (str): A user query describing the support issue or information need (e.g., "reset password",
      "leave approval workflow").
    Returns:
    - A Response object containing relevant support documentation, or None if no results are found.
    """
    print("Querying SageKB")
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


def remove_thinking_tags(text: str) -> str:
    """
    Removes all <thinking>...</thinking> blocks (including tags) from the given text.
    Works even if there are multiple occurrences or multiline content inside the tags.
    """
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


def clean_content(content: str) -> str:
    if "warm regards" in content.lower():
        string_list = content.split("\n")
        content = "\n".join(string_list[:-2])

    if "cakehr" in content.lower():
        content = content.replace("CakeHR", "SageHR")

    if "**response:**" in content:
        content = content.replace("**response:**", "")

    if "assistant:" in content:
        content = content.replace("assistant:", "")

    content = remove_thinking_tags(content.strip())

    content = content.replace("**", "*")
    content = content.replace("Best,\nLola", "")
    return content.strip()


if __name__ == '__main__':
    # resp = asyncio.run(query_sage_kb("sagehr"))
    # print(resp)

    resp = download_excel_from_drive(EMPLOYEE_DIRECTORY_PATH)
    print(resp.columns)
    EMPLOYEE_DIRECTORY = resp

    print(search_employee_directory(first_name="osasu"))
