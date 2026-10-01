# courtbee-synth

Generates synthetic test data for [Courtbee](https://courtbee.com), a booking platform for padel, pickleball and tennis clubs: **players** and the **play invites** they post when looking for a partner. A Gradio UI runs the generators and writes the results to JSON files.

## How it works

There are two generators:

- **Faker**: plain Python. Values are drawn from weighted random distributions (age, skill level, time window, …). Fast and needs no model.
- **LLM** (*Local model* in the UI): any chat model behind an OpenAI-compatible API, local or in the cloud. The model writes the player and the invite as JSON. Python then:
  1. adds the fields a model should not invent (IDs, unique e-mails),
  2. validates the result against the Pydantic models in `courtbee_synth/models.py`,
  3. sends any validation error back to the model and asks it to correct its answer.

After each run, the app reports a **diversity score**: the average cosine similarity between all invite texts, computed with a Hugging Face embedding model ([`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) by default). Lower means more varied texts. Use it to compare generators, models or prompts on the same number of players.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- For the LLM generator: access to an OpenAI-compatible endpoint (see [Choosing an LLM](#choosing-an-llm))

## Setup

```bash
uv sync
cp .env.example .env
```

`.env` settings:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_SERVER` | `http://127.0.0.1:8080/v1` | base URL of the OpenAI-compatible API |
| `LLM_MODEL` | `mlx-community/Qwen3.8-27B-8bit` | model name the server expects |
| `OPENAI_API_KEY` | – | API key for the server; local servers accept any value |
| `LLM_MAX_RETRIES` | `5` | attempts per player before giving up |
| `LLM_MAX_CONCURRENCY` | `16` | requests sent to the server at the same time |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Hugging Face model for the diversity score |
| `HF_TOKEN` | – | optional Hugging Face access token; needed only for gated or private embedding models |

The embedding model is downloaded to the Hugging Face cache on the first run. Any sentence-embedding model from the Hub can replace the default; the score averages the model's token vectors (mean pooling), which suits most `sentence-transformers` models.

## Choosing an LLM

Any server that implements the OpenAI chat completions API works: a local server (mlx-lm, MTPLX, Ollama, LM Studio, vLLM, …) or a cloud API. Set `LLM_SERVER`, `LLM_MODEL` and `OPENAI_API_KEY` accordingly.

Things to know when picking one:

- **The model must follow JSON instructions.** Small models fail validation more often, which costs retries.
- **Reasoning stays off.** A reasoning ("thinking") model writes many extra tokens per player before the JSON. The client always sends `chat_template_kwargs: {"enable_thinking": false}`, which the Qwen chat template understands. Other servers may ignore it, so turn reasoning off on the server side too. Only mlx-lm and MTPLX have been tested.
- **Prefer `127.0.0.1` over `localhost`** for local servers. `localhost` can resolve to IPv6 `::1` first and reach a different program listening on the same port.
- **Lower `LLM_MAX_CONCURRENCY`** if a local server runs out of memory. Some servers keep running but stop answering; the client gives up after 300 s.

### Example: Qwen3.8-27B with MLX (Apple Silicon)

The project was developed with this model on two MLX servers. Run only one at a time: each loads its own copy of the model.

**mlx-lm** batches concurrent requests on the GPU, so it is the better choice for generating many players:

```bash
uv tool install mlx-lm
mlx_lm.server --model mlx-community/Qwen3.8-27B-8bit --port 8080 \
  --temp 0.7 --top-p 0.8 --top-k 20 --max-tokens 32000 \
  --decode-concurrency 16 --prompt-cache-bytes 4GB \
  --chat-template-args '{"enable_thinking":false}'
```

```
LLM_SERVER=http://127.0.0.1:8080/v1
LLM_MODEL=mlx-community/Qwen3.8-27B-8bit
OPENAI_API_KEY=local
```

Stop it with Ctrl-C.

**[MTPLX](https://github.com/youssofal/MTPLX)** is faster for a single request, so it suits trying prompts one at a time:

```bash
mtplx serve --model Youssofal/Qwen3.8-27B-MTPLX-Optimized-Quality --port 8000 \
  --no-auth --reasoning off --scheduler-mode ar_batch \
  --max-active-requests 16 --decode-batch-max 16 \
  --no-ngram-prewarm --ssd-session-cache off --yes
```

```
LLM_SERVER=http://127.0.0.1:8000/v1
LLM_MODEL=mtplx-qwen38-27b-optimized-quality
OPENAI_API_KEY=local
LLM_MAX_CONCURRENCY=1
```

Stop it with `mtplx stop --port 8000`.

## Run

```bash
uv run python -m courtbee_synth.app
```

Open the URL Gradio prints (by default http://127.0.0.1:7860). Pick a generator, set the number of players and press **Generate**.

Each generator writes two files to `data/` (git-ignored), overwriting the previous run:

| Generator | Players | Invites |
|---|---|---|
| Faker | `synthetic_faker_player.json` | `synthetic_faker_invite.json` |
| Local model | `synthetic_llm_player.json` | `synthetic_llm_invite.json` |

## Tests

`tests/test_llm_retry.py` is an integration test: it calls the configured LLM, so the server must be running.

```bash
uv run pytest tests/test_llm_retry.py
```
