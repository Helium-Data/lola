import asyncio
from typing import Any, List, Union, Dict
import time

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.workflow import Event
from llama_index.core.agent.react import ReActChatFormatter, ReActOutputParser
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
            llm: LLM | None = None,
            tools: list[BaseTool] | None = None,
            extra_context: str | None = None,
            **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.formatter = ReActChatFormatter(context=extra_context or "")
        self.formatter.system_header = SYSTEM_HEADER.replace(
            "{context_prompt}",
            """
        Here is some context to help you answer the question and plan:
        {context}
        """,
            1,
        )
        self.output_parser = ReActOutputParser()
        self.sources = []

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> PrepEvent:
        # clear sources, reset existing memory
        self.sources = []
        self.memory.reset()

        conversation_history: List[Dict[str, str]] = ev.get("history", [])
        for hist in conversation_history:
            self.memory.put(ChatMessage(role=hist["role"], content=hist["content"]))

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role="user", content=user_input)
        self.memory.put(user_msg)

        # clear current reasoning
        await ctx.set("current_reasoning", [])

        return PrepEvent()

    @step
    async def prepare_chat_history(
            self, ctx: Context, ev: PrepEvent
    ) -> InputEvent:
        # get chat history
        chat_history = self.memory.get()
        current_reasoning = await ctx.get("current_reasoning", default=[])
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        return InputEvent(input=llm_input)

    @step
    async def handle_llm_input(
            self, ctx: Context, ev: InputEvent
    ) -> Union[ToolCallEvent, PrepEvent, StopEvent]:
        chat_history = ev.input

        response = await self.llm.achat(chat_history)

        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            (await ctx.get("current_reasoning", default=[])).append(
                reasoning_step
            )
            if reasoning_step.is_done:
                self.memory.put(
                    ChatMessage(
                        role="assistant", content=reasoning_step.response
                    )
                )
                return StopEvent(
                    result={
                        "response": reasoning_step.response,
                        "sources": [*self.sources],
                        "reasoning": await ctx.get(
                            "current_reasoning", default=[]
                        ),
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id="fake",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            (await ctx.get("current_reasoning", default=[])).append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )

        # if no tool calls or final response, iterate again
        return PrepEvent()

    @step
    async def handle_tool_calls(
            self, ctx: Context, ev: ToolCallEvent
    ) -> PrepEvent:
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        # call tools -- safely!
        for tool_call in tool_calls:
            tool: AsyncBaseTool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                continue

            try:
                tool_output = await tool.acall(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
            except Exception as e:
                (await ctx.get("current_reasoning", default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )

        # prep the next iteraiton
        return PrepEvent()


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
