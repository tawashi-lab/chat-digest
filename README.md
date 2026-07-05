# chat-digest

[日本語版 README はこちら / Japanese README](README.ja.md)

**chat-digest** summarizes busy Discord / Telegram chat channels by topic using LLMs, and posts the digest to Discord via webhooks — designed to run periodically from cron.

<!-- TODO: add a screenshot of an example digest post -->

## How it works

```
fetch history ──► preprocess ──► detect topics ──► classify every message
 (Discord Bot /     (group          (multiple LLM      (parallel LLM calls,
  selfbot /          consecutive     candidates +       prompt caching)
  Telegram)          messages)       vote)
                                                          │
post to Discord ◄── build embeds ◄── enrich ◄── summarize each topic
 (webhooks,          (split to fit    (condense, titles,
  headline digest,    field limits)    50-char highlight,
  activity graph)                      fact-check via Grok search,
                                       reference links via embeddings)
```

- **Pluggable sources**: official Discord Bot API (recommended), Discord selfbot (⚠️ against Discord ToS, use at your own risk), or Telegram (Telethon).
- **Job presets in YAML**: each channel × time window is a named job (`chat-digest run --job general-24h`).
- **Prompt profiles**: bundled profiles for crypto / stocks / casual / AI communities, overridable per job or via local YAML files.
- **Multi-provider LLM**: OpenAI / Anthropic / Gemini via [LiteLLM](https://github.com/BerriAI/litellm), with per-step model selection and prompt caching for the high-volume classification step.

## Installation

Python 3.10+.

```bash
pip install 'chat-digest[discord] @ git+https://github.com/tawashi-lab/chat-digest'
# extras: discord (official bot) / selfbot / telegram / grok
```

> `discord` and `selfbot` extras cannot be installed together (both provide the `discord` module).

## Setup

1. **Create a Discord webhook** in the channel where digests should be posted (channel settings → Integrations → Webhooks).
2. **Prepare a source**:
   - *Discord Bot*: create a bot in the [Developer Portal](https://discord.com/developers/applications), enable **Message Content Intent**, and invite it to your server with read-message-history permission.
   - *Telegram*: get `api_id` / `api_hash` from [my.telegram.org](https://my.telegram.org). The first run asks for phone auth interactively.
3. **Configure**:

```bash
cp config/config.example.yaml config/config.yaml   # edit server/channel IDs and jobs
cp .env.example .env                               # fill in API keys, tokens, webhook URLs
```

4. **Validate and run**:

```bash
chat-digest validate
chat-digest run --job general-24h
```

### Cron example

```cron
0 4 * * * cd /path/to/chat-digest && .venv/bin/chat-digest run --job general-24h >> cron.log 2>&1
```

## Configuration reference

See [config/config.example.yaml](config/config.example.yaml) for a commented example. Key concepts:

| Section | Purpose |
|---|---|
| `sources` | Where to read messages from (type, token env var, channel IDs) |
| `outputs` | Discord webhooks to post to (referenced by env var name) |
| `jobs` | One summarization preset: source channel, time window, topic settings, features, prompt profile, destinations |
| `llm.models` | Which model to use for each pipeline step (supports `gpt-5.x-hh/hm/hl/mh` reasoning aliases) |
| `branding` | Webhook username / avatar / footer |

Feature flags per job: `condense` (shorten summaries), `highlight` (50-char headline), `graph` (topic activity chart), `references` (link summary phrases to source messages via embeddings), `topic_context` / `fact_check` (Grok live search, requires `GROK_API_KEY`).

### Custom prompts

Bundled profiles: `crypto`, `stocks`, `casual`, `ai`, `crypto_airdrop`. To customize, put `<profile>.yaml` files in a directory listed under `prompt_dirs:` in your config — they take precedence over bundled ones. Individual keys can also be overridden per job with `prompt_overrides:`.

## Development

```bash
pip install -e '.[discord,telegram,grok,dev]'
pytest
ruff check src tests
```

## Disclaimer

- The selfbot source automates a user account, which **violates Discord's Terms of Service** and may get the account banned. It exists for channels where you cannot invite a bot; prefer the official Bot API.
- Summaries are LLM-generated and may contain mistakes. The optional fact-check step reduces but does not eliminate errors.

## License

[MIT](LICENSE)
