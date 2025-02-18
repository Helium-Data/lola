import json
import uvicorn
from fastapi import FastAPI, Request
from slack_bolt.async_app import AsyncApp, AsyncAssistant

from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from config import config
from .lola_listeners import load_listeners
from .lola_workflow import initialize_workflow

app = AsyncApp(
    token=config.SLACK_BOT_TOKEN,
    signing_secret=config.SLACK_SIGNING_SECRET
)
assistant = AsyncAssistant()
agent = initialize_workflow()
load_listeners(app, assistant=assistant, agent=agent)

app.use(assistant)
handler = AsyncSlackRequestHandler(app)

fast_app = FastAPI()


# this is the challenge route required by Slack
# if it's not the challenge it's something for Bolt to handle
@fast_app.post("/")
async def slack_challenge(request: Request):
    json_obj = await request.json()
    if json_obj and "challenge" in json_obj:
        return json.dumps({"challenge": json_obj["challenge"]})
    return await handler.handle(request)


if __name__ == "__main__":
    uvicorn.run(port=3000, app=fast_app)
