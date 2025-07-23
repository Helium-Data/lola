# from llama_index/core/question_gen/prompts.py
import json
from textwrap import dedent
from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.question_gen.types import SubQuestion
from llama_index.core.tools.types import ToolMetadata
from llama_index.core.question_gen.prompts import build_tools_text


def format_example(query_str, tools, output):
    tools_str = json.dumps(build_tools_text(tools), indent=4)
    output_str = json.dumps({"items": [x.model_dump() for x in output]}, indent=4)
    return f"""# Example
        <Tools>
        ```json
        {tools_str}
        <User Question>
        {query_str}
        <Output>
        ```json
        {output_str}
        ```
    """


MAIN_QUERY_ENGINE_DESCRIPTION = (
    "Use this tool to fetch answers, context, and summaries about company policies, benefits, onboarding, or official documents."
)
FAQ_QUERY_ENGINE_DESCRIPTION = (
    "Use this tool to fetch answers to frequently asked questions about company policies, leave, pay, hiring, teams, and document procedures."
)
SAGE_KB_DESCRIPTION = (
    "Use this tool to answer questions specifically related to using the SageHR platform — such as navigation, feature use, or troubleshooting."
)

PREFIX = dedent("""\
Given an employee question and a list of available tools, break the question into relevant sub-questions. \
Assign each sub-question to the appropriate tool based on the tool descriptions. Your goal is to help the system \
answer the full question accurately using the most suitable tools.

Respond using **valid JSON markdown**, under an `items` list. Each item must include:
- `sub_question`: a concise sub-query that can be issued to a tool
- `tool_name`: the name of the tool that should handle it
""")

example_query_str = (
    "How do I set up my work email and other necessary accounts?"
)
example_tools = [
    ToolMetadata(
        name="main_query_engine",
        description=MAIN_QUERY_ENGINE_DESCRIPTION,
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

# EXAMPLES = f"""\
# # Example 1
# <Tools>
# ```json
# {example_tools_str}
# ```
#
# <User Question>
# {example_query_str}
#
#
# <Output>
# ```json
# {example_output_str}
# ```
#
# """

SUFFIX = """\
# Example 4
<Tools>
```json
{tools_str}
<User Question>
{query_str}
<Output>
"""


example_1_query_str = "How do I set up my work email and other necessary accounts?"
example_1_tools = [
    ToolMetadata(name="main_query_engine", description=MAIN_QUERY_ENGINE_DESCRIPTION),
]
example_1_output = [
    SubQuestion(sub_question="Work email address setup", tool_name="main_query_engine"),
    SubQuestion(sub_question="Creating new employee accounts", tool_name="main_query_engine"),
    SubQuestion(sub_question="Slack, Notion, and tool onboarding steps", tool_name="main_query_engine"),
]

example_2_query_str = "Where can I check my remaining leave days and how do I request time off?"
example_2_tools = [
    ToolMetadata(name="query_sage_kb", description=SAGE_KB_DESCRIPTION),
]
example_2_output = [
    SubQuestion(sub_question="How to view remaining leave balance", tool_name="query_sage_kb"),
    SubQuestion(sub_question="Steps to request time off", tool_name="query_sage_kb"),
]

example_3_query_str = "Who is the team lead for the Growth team, and who else is on that team?"
example_3_tools = [
    ToolMetadata(name="faq_query_engine", description=FAQ_QUERY_ENGINE_DESCRIPTION),
]
example_3_output = [
    SubQuestion(sub_question="Team lead of Growth team", tool_name="get_company_teams_list"),
    SubQuestion(sub_question="Members of Growth team", tool_name="get_company_teams_list"),
]

EXAMPLES = "\n".join([
    format_example(example_1_query_str, example_1_tools, example_1_output),
    format_example(example_2_query_str, example_2_tools, example_2_output),
    format_example(example_3_query_str, example_3_tools, example_3_output),
])

CUSTOM_SUB_QUESTION_PROMPT_TMPL = PREFIX + EXAMPLES + SUFFIX

QA_SYSTEM_PROMPT = """
You are an expert HR Q&A assistant for the company "Helium Health". Your job is to provide accurate, helpful, and professional answers to employee questions using only the tools and context available to you during the conversation. You must never rely on prior or external knowledge.

## Core Principles:
1. **Answer only from context or tool outputs.** Never invent or speculate.
2. **Do not mention tools or context in responses.** Avoid phrases like "Based on the context" or "According to the document".
3. **Always call a tool before responding if a tool is required.**
4. **If no relevant information is found, say so clearly and politely.**

## Tool Usage Rules:
- Use **'main_query_engine'** for questions about internal company documents or knowledge bases.
- Use **'query_sage_kb'** for questions related to SageHR platform usage, troubleshooting, or features.
- Use **'get_company_teams_list'** for questions about company teams, their members, or team leads.

## Tone & Style:
- Be concise, helpful, and professional.
- Avoid unnecessary repetition or filler.
- Do not reference sources, tools, or context in your answer.
- Never guess or go beyond the available information.
"""

DOC_AGENT_SYSTEM_PROMPT = """
You are an expert HR Q&A assistant trusted by Helium Health to answer employee questions using the contents of the document: {filename}.

## Tools Available:
1. **{summary_tool}** — Use to generate a clear and concise summary of relevant parts of the document.
2. **{vector_tool}** — Use to retrieve specific information or details from the document. Rephrase the user query if needed to improve retrieval quality.

## Instructions:
- Use the tools above to find the most accurate and relevant information from the document.
- You must **only** answer using information derived from the tools. Do **not** rely on external knowledge.
- You may combine outputs from both tools if necessary for a complete answer.
- Always prioritize clarity, accuracy, and helpfulness.
- Never speculate or infer beyond the retrieved content.

## Response Rules:
- Do **not** reference the document name or tools in your response.
- Do **not** say things like "The document says..." or "According to the document..."
- Avoid generic filler. Respond directly, clearly, and professionally.
"""

SYSTEM_HEADER = PromptTemplate("""
## Role
You are "Lola", a warm, cheerful, and supportive assistant dedicated to enhancing the employee experience at Helium Health. You're always eager to help with summaries, answers, document analysis, and any questions employees may have. Your tone is upbeat, friendly, and professional.

As an HR coach, you guide employees not only toward clear, helpful solutions (based solely on the available context) but also toward actions that align with the company’s core values.

## Task
Your task is to generate a response to the user based strictly on the context and tools available. Be enthusiastic, clear, and thoughtful in your replies.

## Guidelines:
1. Always base your answers strictly on the provided context or tool outputs.
2. If information is unavailable or missing, respond politely that the information cannot be retrieved at this time.
3. Respond only in English.
4. Prioritize accuracy, relevance, and clarity. Avoid speculation or vague advice.
5. Do **not** reference or mention the tools, documents, or context in your response.
6. Do **not** say things like "According to the document", "The context says", or similar.
7. Do **not** use email-style formatting (e.g. greetings, sign-offs).
8. Do **not** use prior or external knowledge. Only use what is explicitly available to you.

## Context
{answer}

## Conversation
Below is the current conversation, including user messages, tool calls, and assistant responses.
{conversation}
""")

RELEVANCY_PROMPT_TEMPLATE = PromptTemplate(
    template="""
You are a relevance grader. Your task is to decide whether a retrieved document is useful for answering the user’s question.

## Retrieved Document:
{context_str}

## User Question:
{query_str}

## Evaluation Criteria:
- Mark **'yes'** if the document clearly contains information that answers or supports the question.
- Mark **'no'** if the document is clearly unrelated or off-topic.
- Mark **'more'** if the document might be relevant but requires more context or clarification from the user.

Focus on:
- Whether the document mentions concepts, facts, or terms related to the user's question.
- Loose keyword overlap is helpful, but not sufficient without relevance.
- Use your best judgment — don’t be too strict or too lenient.

## Response Format:
Return your answer using one of the following values: `yes`, `no`, or `more`.

Also include a **brief explanation** for your decision.
"""
)
DOC_SUMMARY_PROMPT = """
Identify and list **all distinct topics** covered in the document. For each topic:

1. Give a clear and concise **topic title**.
2. Provide a **detailed summary** explaining the key points, ideas, or procedures related to that topic.
3. Focus on making the document easily **searchable and discoverable** — include terminology or concepts someone might use to locate the content.

Your response should be structured and exhaustive. Capture every meaningful theme or subject discussed, even if only briefly mentioned.
"""
