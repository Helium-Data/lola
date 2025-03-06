import asyncio
import nest_asyncio
import json
import uvicorn
from fastapi import FastAPI, Request
from slack_bolt.async_app import AsyncApp, AsyncAssistant

from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from .config import config
from .lola_listeners import load_listeners
from .lola_workflow import initialize_workflow
from .utils import get_bot_id, clean_response, get_user_name

nest_asyncio.apply()

app = AsyncApp(
    token=config.SLACK_BOT_TOKEN,
    signing_secret=config.SLACK_SIGNING_SECRET
)
BOT_APP_ID = asyncio.run(get_bot_id(app))
AGENT = initialize_workflow()
assistant = AsyncAssistant()
load_listeners(app, assistant=assistant, agent=AGENT)

app.use(assistant)
handler = AsyncSlackRequestHandler(app)

fast_app = FastAPI()


async def answer_question(query, thread_ts):
    response = await AGENT.run(input=query, session_id=thread_ts)

    response_text = str(response["response"])
    response_text = await clean_response(response_text, None)
    return response_text


# this is the challenge route required by Slack
# if it's not the challenge it's something for Bolt to handle
@fast_app.post("/")
async def slack_challenge(request: Request):
    json_obj = await request.json()
    if json_obj and "challenge" in json_obj:
        return json.dumps({"challenge": json_obj["challenge"]})
    return await handler.handle(request)


@app.message()
async def handle_messages(message, say):
    if message.get('blocks'):
        for block in message.get('blocks'):
            if block.get('type') == 'rich_text':
                for rich_text_section in block.get('elements'):
                    for element in rich_text_section.get('elements'):
                        if element.get('type') == 'user' and element.get('user_id') == BOT_APP_ID:
                            for element in rich_text_section.get('elements'):
                                if element.get('type') == 'text':
                                    query = element.get('text')
                                    print(f"Somebody asked the bot: {query}")
                                    channel = message.get("channel", None)
                                    user = await get_user_name(message.get('user'), app)
                                    thread_ts = f"{channel}-{user[0]}"
                                    response_text = answer_question(query, thread_ts)
                                    say(
                                        response_text,
                                        channel=channel,
                                        mrkdwn=True
                                    )
                                    return

    # if it's not a question, it might be a threaded reply
    # if it's a reply to the bot, we treat it as if it were a question
    if message.get('thread_ts'):
        thread_ts = message.get("thread_ts")
        if message.get('parent_user_id') == BOT_APP_ID:
            query = message.get('text')
            response = answer_question(query, thread_ts)
            await app.client.chat_postMessage(
                channel=message.get('channel'),
                text=str(response),
                thread_ts=message.get('thread_ts')
            )
            return


if __name__ == "__main__":
    uvicorn.run(port=3000, app=fast_app)
