import asyncio
import json

from courtbee_synth.llm import client, generate_player_invate
from courtbee_synth.models import Player, PlayerInvite


def _with_age(content: str, age: int) -> str:
    start = content.find("{")
    end = content.rfind("}")
    data = json.loads(content[start : end + 1])
    data["player"]["age"] = age
    return json.dumps(data)


def test_model_corrects_age_after_validation_error(monkeypatch, capsys):
    original = client.chat.completions.create
    calls = []

    async def create(**kwargs):
        calls.append(kwargs["messages"])
        response = await original(**kwargs)
        if len(calls) == 1:
            response.choices[0].message.content = _with_age(
                response.choices[0].message.content, 2
            )
        return response

    monkeypatch.setattr(client.chat.completions, "create", create)

    player, invite = asyncio.run(generate_player_invate())

    assert len(calls) == 2
    assert calls[1][-2]["role"] == "assistant"
    assert calls[1][-1]["role"] == "user"
    assert "2" in calls[1][-1]["content"]
    printed = capsys.readouterr().out
    assert printed.count("{") >= 2
    assert isinstance(player, Player)
    assert isinstance(invite, PlayerInvite)
    assert player.age is None or player.age >= 5
