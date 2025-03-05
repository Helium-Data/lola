import asyncio
from typing import Any, List, Union, Dict
import time

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
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
from .prompts import SYSTEM_HEADER


class PrepEvent(Event):
    pass


class InputEvent(Event):
    input: list[ChatMessage]


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

        self.llm = llm
        assert self.llm.metadata.is_function_calling_model

    @step
    async def prepare_chat_history(
            self, ctx: Context, ev: StartEvent
    ) -> InputEvent:
        # clear sources
        await ctx.set("sources", [])

        session_id = ev.get("session_id", f"default-{time.time_ns()}")

        # init memory
        await ctx.set("memory", None)
        memory = ChatMemoryBuffer.from_defaults(
            llm=self.llm,
            token_limit=3000,
            chat_store=config.CHAT_STORE,
            chat_store_key=session_id,
        )

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        memory.put(user_msg)

        # get chat history
        chat_history = memory.get()

        system_msg = ChatMessage(role="system", content=SYSTEM_HEADER)
        chat_history.insert(0, system_msg)

        # update context
        await ctx.set("memory", memory)

        return InputEvent(input=chat_history)

    @step
    async def handle_llm_input(
            self, ctx: Context, ev: InputEvent
    ) -> ToolCallEvent | StopEvent:
        chat_history = ev.input

        # stream the response
        response = await self.llm.achat_with_tools(
            self.tools, chat_history=chat_history
        )

        # save the final response, which should have all content
        memory = await ctx.get("memory")
        memory.put(response.message)
        await ctx.set("memory", memory)

        # get tool calls
        tool_calls = self.llm.get_tool_calls_from_response(
            response, error_on_no_tool_call=False
        )

        if not tool_calls:
            sources = await ctx.get("sources", default=[])
            return StopEvent(
                result={"response": response, "sources": [*sources]}
            )
        else:
            return ToolCallEvent(tool_calls=tool_calls)

    @step
    async def handle_tool_calls(
            self, ctx: Context, ev: ToolCallEvent
    ) -> InputEvent:
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

        # update memory
        memory = await ctx.get("memory")
        for msg in tool_msgs:
            memory.put(msg)

        await ctx.set("sources", sources)
        await ctx.set("memory", memory)

        chat_history = memory.get()
        return InputEvent(input=chat_history)


def initialize_workflow() -> LolaAgent:
    print("Initializing workflow...")
    print("Loading indexes...")
    tools = prepare_tools()

    print("Calling agent...")
    agent = LolaAgent(
        llm=config.LLM, tools=tools, verbose=True, timeout=1000
    )
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
#     python app/lola_workflow.py "What are the eligibility criteria for transport allowance?"
