import json
import os
from typing import Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import ValidationError

from courtbee_synth.models import Player, PlayerInvite

load_dotenv()

MODEL = os.getenv("LLM_MODEL", "mlx-community/Qwen3.8-27B-8bit")
SERVER = os.getenv("LLM_SERVER", "http://localhost:8080/v1")
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "5"))

client = OpenAI(base_url=SERVER, api_key=os.getenv("OPENAI_API_KEY"))


SYSTEM_PROMPT = f"""
You generate one synthetic Courtbee player and, when that player allows invites, one play invitation. Follow the schemas below. Return only that JSON. Do not wrap it in Markdown.

Player schema:
{json.dumps(Player.model_json_schema(), indent=2)}

Invite schema:
{json.dumps(PlayerInvite.model_json_schema(), indent=2)}


Players are people in Slovakia. Vary the person on every request.

Player:
- Use a realistic Slovak given name and surname, about half men and half women.
- Build the email from the name: lowercase ASCII, diacritics removed, only letters and digits, first.last.<first 6 characters of the id>@example.com.
- Always set an age. Pick a band with these relative weights, then an age inside that band: 5–12 weight 8, 13–17 weight 10, 18–29 weight 22, 30–49 weight 40, 50–69 weight 17, 70–85 weight 3.
- Leave the phone number empty when age is under 12. From age 12, about 90% have a number. Use +421, then one of 901, 903, 905, 908, 910, 911, 915, 917, 940, 944, 948, 949, 950, 951, then two groups of three digits.
- Leave skill level empty about 20% of the time. Otherwise keep it within the age limit: under 10 the maximum is 3, under 13 it is 5, under 16 it is 8, and from 16 it is 10. Prefer lower and middle levels. Relative weights for levels 1 through 10 are 10, 14, 16, 16, 14, 11, 8, 6, 3, 2; ignore weights past the age maximum.
- Set allow_play_invites true or false about half the time each.

Invite:
- If allow_play_invites is false, do not generate an invite.
- author_id is the player's id.
- The date is 1 to 14 days after the reference date. The reference date is 2026-10-01 unless the user gives another one.
- time_from is from 07:00 to 20:00 inclusive, on a 30-minute boundary. The reservation lasts 60, 90, or 120 minutes, and time_to is time_from plus that duration.
- If the player's skill level is empty, use the full level range allowed by the schema. Otherwise level_min is the skill level minus 0, 1, or 2, and level_max is the skill level plus 0, 1, or 2, each kept inside the schema bounds.
- Choose each allowed player count about equally often.
- text_sk is a short invitation in natural Slovak, consistent with the sport, the time, and the level range.

Return one JSON object with a player key and an invite key. Each value is an object filled with the fields from the matching schema above, not an empty object. When allow_play_invites is false, set invite to null.

A later user message may be a validation error for the JSON you just returned. Correct that output and return the full JSON object again. Do not explain the error.

{{
  "player": {{ <fields from the Player schema> }},
  "invite": {{ <fields from the Invite schema> }}
}}
"""


def generate_player_invate() -> Tuple[Player, PlayerInvite]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Generate a player and an invite."},
    ]
    error = None
    for i in range(MAX_RETRIES):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
        print(content)

        # Extract the JSON from the content. This is a workround if model still returns something else then plain JSON.
        start = content.find("{")
        end = content.rfind("}")
        data = json.loads(content[start : end + 1])
        try:
            return Player(**data["player"]), PlayerInvite(**data["invite"])
        except ValidationError as caught:
            error = caught
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": str(caught)})
    raise error
