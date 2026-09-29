# courtbee-synth

Generates synthetic Courtbee players and play invites, behind a Gradio UI. Two generators are available:

- **Faker**: plain Python with weighted random values. Fast, no model needed.
- **Local model**: Qwen3.8-27B on a local OpenAI-compatible server. The model writes the player and the invite; Python adds the IDs, validates the output with Pydantic and sends validation errors back to the model for a retry.

Output goes to `data/` (git-ignored) as two JSON files per generator: players and invites.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- For the local model: one of the two MLX servers below (Apple Silicon)

## Setup

```bash
uv sync
cp .env.example .env
```

`.env` settings:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_SERVER` | `http://localhost:8080/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `mlx-community/Qwen3.8-27B-8bit` | model id the server expects |
| `OPENAI_API_KEY` | `local` | any value works for the local servers |
| `LLM_MAX_RETRIES` | `5` | attempts per player before giving up |
| `LLM_MAX_CONCURRENCY` | `16` | concurrent requests to the server |

## Local LLM server

Both servers run the same model and expose the same API. Run only one at a time: each loads its own copy of the model.

### mlx-lm: bulk generation with many concurrent requests

Batches concurrent requests and computes them together on the GPU. Use this one when generating many players.

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen3.8-27B-8bit --port 8080 \
  --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 32000 \
  --decode-concurrency 16 --prompt-cache-bytes 4GB \
  --chat-template-args '{"enable_thinking":false}'
```

`.env`:

```
LLM_SERVER=http://localhost:8080/v1
LLM_MODEL=mlx-community/Qwen3.8-27B-8bit
```

Stop it with Ctrl-C.

### MTPLX: single requests and prompt experiments

Faster for a single request, so better for trying prompts one at a time; use mlx-lm for bulk runs. Install from [MTPLX](https://github.com/youssofal/MTPLX).

```bash
mtplx serve --model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality --port 8000 \
  --no-auth --reasoning off --scheduler-mode ar_batch \
  --max-active-requests 16 --decode-batch-max 16 \
  --no-ngram-prewarm --ssd-session-cache off --yes
```

`.env`:

```
LLM_SERVER=http://localhost:8000/v1
LLM_MODEL=mtplx-qwen38-27b-optimized-quality
LLM_MAX_CONCURRENCY=1
```

Stop it with `mtplx stop --port 8000`.

### Limits

- Thinking must stay off. With thinking on, Qwen writes thousands of reasoning tokens per player.
- If the server runs out of GPU memory, lower `LLM_MAX_CONCURRENCY`. mlx-lm can keep running while every request hangs; the client uses a 300 s timeout for that reason.

## Run

```bash
uv run python -m courtbee_synth.app
```

Open the URL Gradio prints (by default http://127.0.0.1:7860), pick a generator and the number of players, and press **Generate**.

## Tests

`tests/test_llm_retry.py` is an integration test: it calls the real model, so a server must be running.

```bash
uv run pytest tests/test_llm_retry.py
```
