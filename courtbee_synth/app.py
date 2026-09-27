import json
from datetime import date
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from faker import Faker

from courtbee_synth.faker import make_invite, make_player
from courtbee_synth.llm import generate_player_invate

load_dotenv()

OUTPUT_PATHS = {
    "Faker": Path("data") / "synthetic_faker.jsonl",
    "Local model": Path("data") / "synthetic_llm.jsonl",
}
TODAY = date(2026, 10, 1)


def generate(source: str, count: int) -> tuple[str, str]:
    count = int(count)
    fake = Faker("sk_SK")
    records = []
    for _ in range(count):
        if source == "Faker":
            player = make_player(fake)
            invite = (
                make_invite(fake, player, TODAY) if player.allow_play_invites else None
            )
        else:
            player, invite = generate_player_invate()

        records.append(
            {
                "player": player.model_dump(mode="json"),
                "invite": None if invite is None else invite.model_dump(mode="json"),
            }
        )

    output_path = OUTPUT_PATHS[source]
    output_path.parent.mkdir(exist_ok=True)

    text = "\n".join(json.dumps(record, indent=2, ensure_ascii=False) for record in records)
    output_path.write_text(text + "\n")
    return f"Wrote {count} records to {output_path}", str(output_path)


def main() -> None:
    with gr.Blocks(title="Courtbee synthetic data") as demo:
        source = gr.Radio(["Faker", "Local model"], value="Faker", label="Generator")
        count = gr.Number(value=5, precision=0, minimum=1, label="Players")
        button = gr.Button("Generate")
        status = gr.Textbox(label="Status")
        output_file = gr.File(label="JSONL file")
        button.click(generate, [source, count], [status, output_file])
    demo.launch()


if __name__ == "__main__":
    main()
