import asyncio

from slack_bolt.async_app import (
    AsyncApp,
    AsyncAssistant,
    AsyncSetTitle,
    AsyncSetStatus,
    AsyncSetSuggestedPrompts,
    AsyncSay,
    AsyncGetThreadContext,
)
from slack_sdk.web.async_client import AsyncWebClient


class LolaSlackListener:
    def __init__(self, app: AsyncApp, assistant: AsyncAssistant, agent):
        self.app = app
        self.assistant = assistant

        # self.bot_user_id = asyncio.run(self._get_bot_id())
        self.agent = agent

    async def _get_bot_id(self):
        # get the bot's own user ID so it can tell when somebody is mentioning it
        auth_response = await self.app.client.auth_test()
        bot_user_id_ = auth_response["user_id"]
        return bot_user_id_

    async def reply_message(self, message, say: AsyncSay):
        print("Saw a fact: ", message.get('text'))
        pass

    async def handle_assistant_thread_started(
            self,
            say: AsyncSay,
            get_thread_context: AsyncGetThreadContext,
            set_suggested_prompts: AsyncSetSuggestedPrompts
    ):
        await say(":wave: Hi, I'm Lola. How can I help you today?")
        await set_suggested_prompts(
            prompts=[
                "How can we protect against Intellectual Property theft?",
                "How does an employee default on the transport allowance?",
            ]
        )
        return

    async def _prepare_history(self, history):
        prepared_history = []
        for idx, message in enumerate(history["messages"][:-1]):
            role = "user" if message.get("bot_id") is None else "assistant"
            message_text = message["text"]
            if idx == 0:
                message_text = "Starting new conversation"
            prepared_history.append({"role": role, "content": message_text})
        return prepared_history

    async def handle_assistant_message(
            self, payload: dict,
            logger: None,
            set_title: AsyncSetTitle,
            set_status: AsyncSetStatus,
            say: AsyncSay,
            get_thread_context: AsyncGetThreadContext,
            client: AsyncWebClient,
    ):
        thread_ts = payload.get("thread_ts", "")
        query = payload.get("text", "No message")
        channel = payload.get("channel", "")

        await set_title(query)
        await set_status("Typing...")
        await set_status("Typing...")

        history = await client.conversations_replies(
            ts=thread_ts, channel=channel, limit=50
        )
        prepared_history = await self._prepare_history(history)

        response = await self.agent.run(input=query, history=prepared_history)
        await set_status("Still typing...")
        await say(
            response["response"],
            thread_ts=thread_ts
        )
        print(response["sources"])
        return


def load_listeners(app: AsyncApp, assistant: AsyncAssistant, agent):
    lola_listener = LolaSlackListener(
        app=app, assistant=assistant, agent=agent
    )

    app.event("message")(ack=lola_listener.reply_message, lazy=[lola_listener.reply_message])

    assistant.thread_started(
        lola_listener.handle_assistant_thread_started
    )
    assistant.user_message(
        lola_listener.handle_assistant_message
    )
