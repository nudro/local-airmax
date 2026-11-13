# Local AirMax

Local AirMax is a small control panel that runs everyday business writing on this Mac. Nothing in a task is sent to a cloud model. A savings counter shows what that work would have cost at cloud prices.

It is built for an Apple MacBook Air with an M3 chip and 16 GB of memory. Only one writing model is loaded at a time.

## Install

```bash
cd /Users/catherineordun/Documents/local-airmax
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

EmbeddingGemma is a gated Google model. Sign in once so document search can download it:

```bash
hf auth login
```

## Run

```bash
source .venv/bin/activate
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Web search stays off until you turn it on. Research starts with it on. Only the search query leaves the Mac, and the privacy line turns amber while it is on. The Logs button keeps a timeline of each run and can download or delete those rows. Two Booz Allen articles are staged in `demo/` the first time the app starts.

The first time you run a task, the Mac downloads that task's model.

- Most tasks use Gemma 4 12B (about 6.3 GB), from `DreamFoundries/gemma-4-12B-it-4bit`.
- Multi-Step Deep Research also downloads Llama 3.2 3B (about 1.8 GB) to pull facts, then unloads it before Gemma writes the answer.

Later runs use the copies already on the Mac.

## Optional adapters

Adapters are optional. If a folder is missing, that task runs the base Gemma model.

Every adapter must be a LoRA trained on `gemma-4-12B-it`. Point `--source` at a PEFT folder that contains `adapter_config.json` and `adapter_model.safetensors`.

```bash
python convert_lora.py --source /path/to/exec_summary_peft --dest adapters/brief
python convert_lora.py --source /path/to/tabular_peft --dest adapters/csv
python convert_lora.py --source /path/to/code_peft --dest adapters/code
python convert_lora.py --source /path/to/brand_voice_peft --dest adapters/memo
```

## What the five cards do

- **Executive Document Brief** reads a long PDF in pieces and writes a short brief.
- **Spreadsheet & Data Analyst** uses column names, a row count, and summary numbers. It does not send the raw spreadsheet.
- **Data Science & SQL Helper** drafts Python or SQL, then checks the grammar without running the code.
- **Client & Internal Comms** drafts an email, memo, or proposal.
- **Multi-Step Deep Research** searches your files, lists the facts, then writes the answer.

Smart Mode picks a card from your sentence. While Smart Mode is on, a spreadsheet file opens the spreadsheet card, and two or more files open deep research. A card you click runs that task even if a spreadsheet is still attached.

The price toggle compares the work with everyday cloud prices ($2 / $10 per million tokens, Claude Sonnet 5.5) or premium cloud prices ($10 / $50 per million tokens, GPT-6 Astra short context). Each finished task subtracts $0.0005 for local electricity. The ledger is `savings.db` in this folder.
