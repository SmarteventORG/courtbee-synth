# courtbee-synth

Generates synthetic Courtbee players and play invites with several LLMs (local and cloud), behind a Gradio UI. The output is JSONL, which a seed script in the Courtbee repo imports into a local or demo database.

## Setup

```bash
uv sync
cp .env.example .env   # fill in the keys
```

Generated files go to `data/` (git-ignored).
