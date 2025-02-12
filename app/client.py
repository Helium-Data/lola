import asyncio
import json
import uvicorn
from slack_bolt import App
from flask import jsonify
from fastapi import FastAPI, Request
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.flask import SlackRequestHandler
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from config import config
from lola_workflow import initialize_workflow

app = AsyncApp(
    token=config.SLACK_BOT_TOKEN,
    signing_secret=config.SLACK_SIGNING_SECRET
)
handler = AsyncSlackRequestHandler(app)

flask_app = FastAPI()


async def load_channel():
    channel_list = await app.client.conversations_list()
    channel_list = channel_list.data
    print(channel_list)
    channel = next(
        (channel for channel in channel_list.get("channels") if channel.get("name") == "helium-data-research"), None)
    channel_id = channel.get('id')
    await app.client.conversations_join(channel=channel_id)
    print(f"Found the channel {channel_id} and joined it")

    # get the bot's own user ID so it can tell when somebody is mentioning it
    auth_response = await app.client.auth_test()
    bot_user_id_ = auth_response["user_id"]
    return bot_user_id_


#
AGENT = initialize_workflow()
bot_user_id = asyncio.run(load_channel())


# this is the challenge route required by Slack
# if it's not the challenge it's something for Bolt to handle
@flask_app.post("/")
async def slack_challenge(request: Request):
    json_obj = await request.json()
    if json_obj and "challenge" in json_obj:
        print("Received challenge")
        return json.dumps({"challenge": json_obj["challenge"]})
    else:
        print("Incoming event:")
        print(json_obj)
    return await handler.handle(request)


@app.message()
async def reply(message, say):
    if message.get('blocks'):
        for block in message.get('blocks'):
            if block.get('type') == 'rich_text':
                for rich_text_section in block.get('elements'):
                    for element in rich_text_section.get('elements'):
                        if element.get('type') == 'user' and element.get('user_id') == bot_user_id:
                            for element in rich_text_section.get('elements'):
                                if element.get('type') == 'text':
                                    query = element.get('text')
                                    print(f"Somebody asked the bot: {query}")
                                    response = await AGENT.run(input=query)
                                    await say(response["response"])
                                    return
    print("Saw a fact: ", message.get('text'))


if __name__ == "__main__":
    uvicorn.run(port=3000, app=flask_app)
