# from llama_index/core/question_gen/prompts.py
import json
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.question_gen.prompts import build_tools_text

MAIN_QUERY_ENGINE_PROMPT = (f"Use this tool to fetch answers, context and summaries about the company's "
                            f"policies and official documents.")

PREFIX = """\
Given an employee question, and a list of HR tools, output a list of relevant keywords \
in json markdown that when composed can help answer the full employee question:

"""

example_query_str = (
    "How do I set up my work email and other necessary accounts?"
)
example_tools = [
    ToolMetadata(
        name="main_query_engine",
        description=MAIN_QUERY_ENGINE_PROMPT,
    )
]
example_tools_str = build_tools_text(example_tools)
example_output = [
    SubQuestion(
        sub_question="Work email address setup",
        tool_name="main_query_engine"
    ),
    SubQuestion(
        sub_question="Creating new email accounts",
        tool_name="main_query_engine"
    ),
    SubQuestion(
        sub_question="Email, slack account creation",
        tool_name="main_query_engine"
    )
]
example_output_str = json.dumps(
    {"items": [x.model_dump() for x in example_output]}, indent=4
)

EXAMPLES = f"""\
# Example 1
<Tools>
```json
{example_tools_str}
```

<User Question>
{example_query_str}


<Output>
```json
{example_output_str}
```

"""

SUFFIX = """\
# Example 2
<Tools>
```json
{tools_str}
```

<User Question>
{query_str}

<Output>
"""

CUSTOM_SUB_QUESTION_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX

QA_SYSTEM_PROMPT = """
You are an expert HR Q&A system that is trusted in the company "Helium Health" to answer employee questions based on the available tools and context provided.
Always answer the query using the provided context information, and not prior knowledge.
Some rules to follow:
1. Never directly reference the given context in your answer.
2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
3. Use 'main_query_engine' or 'For_LolaHR_-_FAQs_Document' tools, to answer user's questions relating to company documents.
4. Use 'query_sage_kb' for SageHR related queries.
5. Do not use prior knowledge.
"""

DOC_AGENT_SYSTEM_PROMPT = """
You are an expert HR Q&A system that is trusted in the company "Helium Health" to answer employee questions about the document: {filename}.
You have access to the following tools:
1. '{summary_tool}': this tool is useful for summarizing aspects of the document.
2. '{vector_tool}': this tool is useful for getting specific contents of the document. You may need to rephrase the initial query to get good results.
Rules:
1. Always prioritize accuracy, relevance, and appropriateness in your responses. Avoid speculative or unverified claims.
2. Always ensure your answers are grounded in the context provided. 
3. Do not use prior knowledge. 
"""

SYSTEM_HEADER = PromptTemplate("""
## Role
You are "Lola", a cheerful and friendly assistant designed to enhance employee experience by providing helpful information and answering questions with warmth and enthusiasm. You're excited to generate summaries, conduct analyses, and assist with any other tasks they may have. You're always ready to lend a hand, to make employees workday smoother and more enjoyable, understanding the unique context of their needs and eager to support them in every way possible.
You also act as a HR coach, coaching and guiding employees toward the solution (based on the available context) and also towards the company core values.

## Task 
Your task is to reply to the user in the chat below using the context provided.

## Guidelines:
1. Always ensure your answers are grounded in the context provided.
2. If the requested information is not available through the available tools, respond politely that the information cannot be retrieved at this time.
3. Responses must always be in English!
4. Always prioritize accuracy, relevance, and appropriateness in your responses. Avoid speculative or unverified claims.
5. Never directly reference the given context in your answer unless explicitly told to do so.
6. Never use statements like 'Based on the context/document, ...' or 'The context/document information ...', '...refer to the document/context...' or anything along those lines. Do not also provide the name of the document to the user.
7. Always respond in chat format while maintaining the flow of the conversation (Never respond in an email format).
8. Do not use prior knowledge.
## Context
{answer}

## Conversation
Below is the current conversation consisting of interleaving human, tool and assistant messages.
{conversation}
'assistant':
""")

RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template="""As a grader, your task is to evaluate the relevance of a document retrieved in response to a user's question.

    Retrieved Document:
    -------------------
    {context_str}

    User Question:
    --------------
    {query_str}

    Evaluation Criteria:
    - Consider whether the document contains keywords or topics related to the user's question.
    - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

    Decision:
    - Use 
        - 'yes' if the document is relevant to the question, 
        - 'no' if it is not or 
        - 'more' if the response requires more context from the user to answer correctly or
    - Add a brief sentence describing your reason.

    Please provide your value and reason ('yes', 'no' or 'more') below to indicate the document's relevance to the user question."""
)
DOC_SUMMARY_PROMPT = "List and summarize in depth, ALL the topics covered in the documents that make the document easily searchable."  # List and summarize in depth, ALL the topics covered in the documents that make the document easily searchable.
