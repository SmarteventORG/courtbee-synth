import asyncio
import json
import os
from datetime import date
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from faker import Faker
from huggingface_hub import login

from courtbee_synth.faker import make_invite, make_player
from courtbee_synth.huggingface import average_similarity
from courtbee_synth.llm import generate_player_invate

load_dotenv()

OUTPUT_PATHS = {
    "Faker": (
        Path("data") / "synthetic_faker_player.json",
        Path("data") / "synthetic_faker_invite.json",
    ),
    "Local model": (
        Path("data") / "synthetic_llm_player.json",
        Path("data") / "synthetic_llm_invite.json",
    ),
}
TODAY = date(2026, 10, 1)


async def generate(source: str, count: int) -> tuple[str, list[str], float | None]:
    count = int(count)
    players = []
    invites = []
    if source == "Faker":
        fake = Faker("en_US")
        pairs = []
        for _ in range(count):
            player = make_player(fake)
            invite = (
                make_invite(fake, player, TODAY) if player.allow_play_invites else None
            )
            pairs.append((player, invite))
    else:
        pairs = await asyncio.gather(*(generate_player_invate() for _ in range(count)))

    for player, invite in pairs:
        players.append(player.model_dump(mode="json"))
        if invite is not None:
            invites.append(invite.model_dump(mode="json"))

    player_path, invite_path = OUTPUT_PATHS[source]
    player_path.parent.mkdir(exist_ok=True)
    player_path.write_text(json.dumps(players, indent=2, ensure_ascii=False) + "\n")
    invite_path.write_text(json.dumps(invites, indent=2, ensure_ascii=False) + "\n")
    texts = [invite["text_en"] for invite in invites if invite.get("text_en")]
    similarity = average_similarity(texts)
    if similarity is not None:
        similarity = round(similarity, 3)
    status = (
        f"Wrote {len(players)} players to {player_path} "
        f"and {len(invites)} invites to {invite_path}"
    )
    return status, [str(player_path), str(invite_path)], similarity


def main() -> None:
    with gr.Blocks(title="Courtbee synthetic data") as demo:
        source = gr.Radio(["Faker", "Local model"], value="Faker", label="Generator")
        count = gr.Number(value=5, precision=0, minimum=1, label="Players")
        button = gr.Button("Generate")
        status = gr.Textbox(label="Status")
        diversity = gr.Number(
            label="Rozmanitosť (priemerná podobnosť, nižšia = rozmanitejšie)",
            precision=3,
        )
        output_file = gr.File(label="JSON files", file_count="multiple")
        button.click(generate, [source, count], [status, output_file, diversity])
    demo.launch()


if __name__ == "__main__":
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(hf_token)

    main()
