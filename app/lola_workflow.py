import asyncio
from typing import Any, List
import time

from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.utils.workflow import draw_all_possible_flows
from llama_index.core.query_pipeline import QueryPipeline
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool, AsyncBaseTool
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
)

from .utils import prepare_tools
from .config import config
from .prompts import SYSTEM_HEADER, RELEVANCY_PROMPT_TEMPLATE, QA_SYSTEM_PROMPT, SYSTEM_HEADER_PROMPT


class PrepEvent(Event):
    pass


class InputEvent(Event):
    input: list[ChatMessage]


class RelevancyEvent(Event):
    tool_msgs: list[ChatMessage]


class ResponseEvent(Event):
    answer: ChatMessage
    history: list[ChatMessage]


class ToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class FunctionOutputEvent(Event):
    output: ToolOutput


class LolaAgent(Workflow):
    def __init__(
            self,
            *args: Any,
            llm: FunctionCallingLLM | None = None,
            tools: List[BaseTool] | None = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []
        self.memory_token_limit = 30000

        self.llm = llm
        self.chat_llm = llm
        assert self.llm.metadata.is_function_calling_model

        self.relevancy_pipeline = QueryPipeline(chain=[RELEVANCY_PROMPT_TEMPLATE, llm])
        self.response_pipeline = QueryPipeline(chain=[SYSTEM_HEADER, llm])

    @step
    async def prepare_chat_history(
            self, ctx: Context, ev: StartEvent
    ) -> InputEvent:
        # clear sources
        await ctx.set("sources", [])

        session_id = ev.get("session_id", None)
        if not session_id:
            session_id = f"default-{time.time_ns()}"

        # init memory
        await ctx.set("memory", None)
        memory = ChatMemoryBuffer.from_defaults(
            llm=self.llm,
            token_limit=self.memory_token_limit,
            chat_store=config.CHAT_STORE,
            chat_store_key=session_id,
        )

        # get user input
        user_input = ev.input
        user_input += "\n (USE at least ONE tool to answer query.)"
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)
        await ctx.set("query_str", user_input)

        # get chat history
        chat_history = memory.get()

        has_system_message = False
        for msg in chat_history:
            if msg.role == "system":
                has_system_message = True

        if not has_system_message:
            system_msg = ChatMessage(role="system", content=SYSTEM_HEADER_PROMPT)
            chat_history.insert(0, system_msg)
            memory.set(chat_history)

        # update context
        await ctx.set("memory", memory)

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
            self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | ResponseEvent:
        chat_history = ev.input

        # stream the response
        response = await self.llm.achat_with_tools(
            self.tools, chat_history=chat_history, allow_parallel_tool_calls=True
        )

        # get tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        # save the final response, which should have all content
        memory = await ctx.get("memory")
        memory.put(response.message)
        await ctx.set("memory", memory)

        if not tool_calls:
            return ResponseEvent(
                history=memory.get(),
                answer=response.message
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
            self, ctx: Context, ev: ToolCallEvent
    ) -> RelevancyEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        tool_msgs = []
        sources = await ctx.get("sources", default=[])

        # call tools -- safely!
        for tool_call in tool_calls:
            tool: AsyncBaseTool = tools_by_name.get(tool_call.tool_name)
            additional_kwargs = {
                "tool_call_id": tool_call.tool_id,
                "name": tool.metadata.get_name(),
            }
            if not tool:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Tool {tool_call.tool_name} does not exist",
                        additional_kwargs=additional_kwargs,
                    )
                )
                continue

            try:
                tool_output = await tool.acall(**tool_call.tool_kwargs)
                sources.append(tool_output)
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=tool_output.content,
                        additional_kwargs=additional_kwargs,
                    )
                )
            except Exception as e:
                tool_msgs.append(
                    ChatMessage(
                        role="tool",
                        content=f"Encountered error in tool call: {e}",
                        additional_kwargs=additional_kwargs,
                    )
                )

        await ctx.set("sources", sources)

        return RelevancyEvent(tool_msgs=tool_msgs)

    @step
    async def eval_relevance(
            self, ctx: Context, ev: RelevancyEvent
    ) -> InputEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        tool_msgs = ev.tool_msgs
        query_str = await ctx.get("query_str")

        relevancy_results = []
        for msg in tool_msgs:
            relevancy = await self.relevancy_pipeline.arun(
                context_str=msg.content, query_str=query_str
            )
            relevance_response = f"Tool output: {msg.content} \nRelevancy: {relevancy.message.content.lower().strip()}"
            relevancy_results.append(
                ChatMessage(
                    role="tool",
                    content=relevance_response,
                    additional_kwargs=msg.additional_kwargs,
                )
            )

        # update memory
        memory = await ctx.get("memory")
        for msg in relevancy_results:
            memory.put(msg)

        await ctx.set("memory", memory)

        chat_history = memory.get()
        return InputEvent(input=chat_history)

    @step
    async def handle_response(
            self, ctx: Context, ev: ResponseEvent
    ) -> StopEvent:
        """Evaluate relevancy of retrieved documents with the query."""
        chat_history = ev.history
        answer = ev.answer

        # new_chat_history = ""
        # for chat in chat_history[:-1]:
        #     if chat.role == MessageRole.USER:
        #         role = "user"
        #     elif chat.role == MessageRole.TOOL:
        #         role = "tool"
        #     else:
        #         role = "assistant"
        #     new_chat_history += f"\n'{role}': {chat.content.strip()}"
        #
        # response = await self.response_pipeline.arun(conversation=new_chat_history, answer=answer)
        response_message = str(answer)

        if "warm regards" in response_message.lower():
            string_list = response_message.split("\n")
            response_message = "\n".join(string_list[:-2])

        if "cakehr" in response_message.lower():
            response_message = response_message.replace("CakeHR", "SageHR")

        # save the final response
        memory = await ctx.get("memory")
        memory.put(response_message)
        await ctx.set("memory", memory)

        sources = await ctx.get("sources", default=[])
        return StopEvent(result={"response": response_message, "sources": [*sources]})


def initialize_workflow(visualize_workflow=False) -> LolaAgent:
    print("Initializing workflow...")
    print("Loading indexes...")
    tools = prepare_tools()

    print("Calling agent...")
    agent = LolaAgent(
        llm=config.LLM, tools=tools, verbose=True, timeout=1000
    )
    if visualize_workflow:
        draw_all_possible_flows(LolaAgent, filename="lola_workflow.html")
    return agent


async def run_agent(text):
    agent = initialize_workflow()

    start_time = time.time()
    print(f"Running agent at: {start_time}")
    response = await agent.run(input=text)
    end_time = time.time()
    print(f"Completed run at: {end_time} for {end_time - start_time}")

    print("Sources: ", response["sources"])
    print("Response: ", response["response"])


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Insert query",
                        type=str)
    args = parser.parse_args()
    asyncio.run(run_agent(str(args.query)))
