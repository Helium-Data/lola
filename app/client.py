from slack_bolt import App
from flask import Flask, request, jsonify
from slack_bolt.adapter.flask import SlackRequestHandler

from config import config
from lola_workflow import initialize_workflow

app = App(
    token=config.SLACK_BOT_TOKEN,
    signing_secret=config.SLACK_SIGNING_SECRET
)
handler = SlackRequestHandler(app)

flask_app = Flask(__name__)

# join the #bot-testing channel so we can listen to messages
channel_list = app.client.conversations_list().data
channel = next((channel for channel in channel_list.get('channels') if channel.get("name") == "bot-testing"), None)
channel_id = channel.get('id')
app.client.conversations_join(channel=channel_id)
print(f"Found the channel {channel_id} and joined it")

# get the bot's own user ID so it can tell when somebody is mentioning it
auth_response = app.client.auth_test()
bot_user_id = auth_response["user_id"]

AGENT = initialize_workflow()


# this is the challenge route required by Slack
# if it's not the challenge it's something for Bolt to handle
@flask_app.route("/", methods=["POST"])
def slack_challenge():
    if request.json and "challenge" in request.json:
        print("Received challenge")
        return jsonify({"challenge": request.json["challenge"]})
    else:
        print("Incoming event:")
        print(request.json)
    return handler.handle(request)


# this handles any incoming message the bot can hear
# we want it to only respond when somebody messages it directly
# otherwise it listens and stores every message as future context
@app.message()
def reply(message, say):
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
                                    response = AGENT.run(input=query)
                                    print("Context was:")
                                    # print(response.source_nodes)
                                    print(f"Response was: {response}")
                                    say(str(response))
                                    return
    # otherwise treat it as a document to store
    # index.insert(Document(text=message.get('text')))
    # print("Stored message", message.get('text'))


if __name__ == "__main__":
    flask_app.run(port=3000)
