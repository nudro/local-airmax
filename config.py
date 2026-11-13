# Settings for Local AirMax.
# Every task, model name, and price lives in the dictionaries below.

# How many characters we treat as one token when a tokenizer is not loaded yet.
CHARS_PER_TOKEN = 4

# A document this short is read once. Longer text is searched and summarized in pieces.
SHORT_PASS_CHARS = 24000

# Local electricity for one click on this MacBook Air. About 15 watts for a short run.
LOCAL_COST = 0.0005

# How many times the local model may answer during one web-search task.
MAX_AGENT_TURNS = 3

# How many public web snippets one search may bring back.
MAX_SEARCH_RESULTS = 4

# Sample articles used by the demo chips. Each row is a page and the local file.
DEMO_URLS = []

cyber_demo = {}
cyber_demo["url"] = "https://www.boozallen.com/insights/velocity/achieving-real-time-cyber-defense.html"
cyber_demo["path"] = "demo/achieving-real-time-cyber-defense.md"
cyber_demo["title"] = "Achieving Real-Time Cyber Defense"
DEMO_URLS.append(cyber_demo)

space_demo = {}
space_demo["url"] = "https://www.boozallen.com/insights/space/large-language-models-in-space-and-beyond.html"
space_demo["path"] = "demo/large-language-models-in-space-and-beyond.md"
space_demo["title"] = "Large Language Models in Space and Beyond"
DEMO_URLS.append(space_demo)

# Language models. Gemma 4 12B is the only one that accepts a LoRA adapter.
MODELS = {}

gemma12 = {}
gemma12["repo"] = "DreamFoundries/gemma-4-12B-it-4bit"
gemma12["alternate_repo"] = "mlx-community/gemma-4-12B-it-4bit"
gemma12["family"] = "gemma-4-12B-it"
gemma12["accepts_adapter"] = True
MODELS["gemma12"] = gemma12

# Analysis model. IBM's 4-bit MLX build of Granite 4.2 8B. No adapter.
granite8 = {}
granite8["repo"] = "ibm-granite/granite-4.2-8b-q4-mlx"
granite8["family"] = "granite-4.2-8b"
granite8["accepts_adapter"] = False
MODELS["granite8"] = granite8

# Fast Gemma. No adapter. The 12B model stays reserved for LoRA mode.
gemma4b = {}
gemma4b["repo"] = "mlx-community/gemma-4-e4b-it-4bit"
gemma4b["family"] = "gemma-4-e4b-it"
gemma4b["accepts_adapter"] = False
MODELS["gemma4b"] = gemma4b

llama3 = {}
llama3["repo"] = "mlx-community/Llama-3.2-3B-Instruct-4bit"
llama3["family"] = "llama-3.2-3b"
llama3["accepts_adapter"] = False
MODELS["llama3"] = llama3

# The switch the business user sees. Names describe the job, not the model file.
# Precision loads the text Gemma 4 12B already used here. The mlx-community 12B file is a vision build.
# Analysis loads IBM's 4-bit MLX Granite 4.2 8B. The mlx-community 8B id is not published.
ENGINE_MODES = {}

summaries_mode = {}
summaries_mode["label"] = "Summaries"
summaries_mode["hint"] = "Fastest • Best for quick briefs & emails"
summaries_mode["model_id"] = "mlx-community/Llama-3.2-3B-Instruct-4bit"
summaries_mode["base_model"] = "llama3"
summaries_mode["use_lora"] = False
ENGINE_MODES["summaries"] = summaries_mode

writing_mode = {}
writing_mode["label"] = "Writing"
writing_mode["hint"] = "Balanced • Best for memos & general drafting"
writing_mode["model_id"] = "mlx-community/gemma-4-e4b-it-4bit"
writing_mode["base_model"] = "gemma4b"
writing_mode["use_lora"] = False
ENGINE_MODES["writing"] = writing_mode

analysis_mode = {}
analysis_mode["label"] = "Analysis"
analysis_mode["hint"] = "Structured • Best for spreadsheets & data facts"
analysis_mode["model_id"] = "ibm-granite/granite-4.2-8b-q4-mlx"
analysis_mode["base_model"] = "granite8"
analysis_mode["use_lora"] = False
ENGINE_MODES["analysis"] = analysis_mode

precision_mode = {}
precision_mode["label"] = "Precision"
precision_mode["hint"] = "Specialist • Uses your custom-trained templates"
precision_mode["model_id"] = "DreamFoundries/gemma-4-12B-it-4bit"
precision_mode["base_model"] = "gemma12"
precision_mode["use_lora"] = True
ENGINE_MODES["precision"] = precision_mode

# Small models used for routing and for document search. They are not chat models.
EMBED_MODEL = "google/embeddinggemma-300m"
ROUTER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cloud list prices per million tokens, used only to estimate what was not spent.
# Everyday is Claude Sonnet 5.5. Premium is GPT-6 Astra at its short-context rate.
PRICES = {}

everyday = {}
everyday["input_per_million"] = 2.0
everyday["output_per_million"] = 10.0
everyday["label"] = "Everyday cloud prices"
PRICES["everyday"] = everyday

premium = {}
premium["input_per_million"] = 10.0
premium["output_per_million"] = 50.0
premium["label"] = "Premium cloud prices"
PRICES["premium"] = premium

# The five business tasks, in the order they appear on screen.
TASKS = {}

brief = {}
brief["name"] = "Executive Document Brief"
brief["blurb"] = "Turn a long document into a short brief."
brief["keywords"] = ["brief", "summary", "summarize", "report", "paper", "executive"]
brief["utterances"] = [
    "Summarize this report for an executive.",
    "Give me a short brief of this PDF.",
    "What are the main points of this research paper?",
    "Turn this long document into a one-page brief.",
    "Summarize the findings in plain English.",
]
brief_embed = {}
brief_embed["kind"] = "embed"
brief_embed["context_budget"] = 32000
brief_embed["chunk_chars"] = 1200
brief_map = {}
brief_map["kind"] = "map_reduce"
brief_map["base_model"] = "gemma12"
# Precision reads specialists/exec-brief for this task.
brief_map["adapter_dir"] = "brief"
brief_map["context_budget"] = 32000
brief_map["forward_cap"] = 2000
brief_map["max_kv_size"] = 4096
brief_map["map_chars"] = 8000
brief_map["map_output_guess"] = 180
brief_map["reduce_output_guess"] = 450
brief_map["max_tokens"] = 400
brief_map["map_max_tokens"] = 220
brief_map["reduce_max_tokens"] = 400
brief_map["map_instructions"] = (
    "Summarize this section in a few plain sentences. "
    "Keep names, dates, and numbers. Do not add facts that are not written here."
)
brief_map["reduce_instructions"] = (
    "Write a short executive brief in plain English. "
    "Start with the main point, then the few details a busy manager needs."
)
brief["steps"] = [brief_embed, brief_map]
brief["default_web_search"] = False
TASKS["brief"] = brief

csv_task = {}
csv_task["name"] = "Spreadsheet & Data Analyst"
csv_task["blurb"] = "Explain a spreadsheet in plain English."
csv_task["keywords"] = ["spreadsheet", "csv", "table", "anomaly", "trend", "column", "rows"]
csv_task["utterances"] = [
    "Look at this spreadsheet and tell me the trends.",
    "Find anything odd in these numbers.",
    "Summarize this CSV for a manager.",
    "Which columns matter in this table?",
    "Explain the highs and lows in this data.",
]
csv_step = {}
csv_step["kind"] = "generate"
csv_step["base_model"] = "gemma12"
# Precision reads specialists/data-analyst for this task.
csv_step["adapter_dir"] = "csv"
csv_step["forward_cap"] = 8000
csv_step["max_kv_size"] = 4096
csv_step["output_guess"] = 450
csv_step["max_tokens"] = 700
csv_step["check_syntax"] = False
csv_step["input_from"] = "material"
csv_step["instructions"] = (
    "Help a business person understand the attached spreadsheet. "
    "The prompt already contains the row count, column names, a markdown histogram table, "
    "the first 5 rows, and summary statistics. The data is present. "
    "Do not say the spreadsheet is missing. "
    "Do not repeat the histogram and do not redraw the bins. The screen already shows that table. "
    "Answer the user's question in three short paragraphs, then stop on a complete sentence. "
    "Use only numbers from the profile. Do not invent numbers. "
    "Do not write an internal monologue, a chain of thought, or any thinking tags."
)
csv_task["steps"] = [csv_step]
csv_task["default_web_search"] = False
TASKS["csv"] = csv_task

code_task = {}
code_task["name"] = "Data Science & SQL Helper"
code_task["blurb"] = "Draft Python or SQL."
code_task["keywords"] = ["python", "pandas", "sql", "query", "script", "select", "dataframe", "numpy", "gini"]
code_task["utterances"] = [
    "Write a Python script to clean this data.",
    "Write a SQL query for monthly revenue.",
    "Help me with a pandas filter.",
    "Draft a SELECT statement for these columns.",
    "Write code that groups rows and counts them.",
    "Write a decision tree totally in NumPy and calculate Gini impurity.",
]
code_step = {}
code_step["kind"] = "generate"
code_step["base_model"] = "gemma12"
# Precision reads specialists/ds-coder for this task.
code_step["adapter_dir"] = "code"
code_step["forward_cap"] = 8000
code_step["max_kv_size"] = 4096
code_step["output_guess"] = 900
code_step["max_tokens"] = 2000
code_step["check_syntax"] = True
code_step["input_from"] = "material"
code_step["instructions"] = (
    "Write the Python or SQL the user asked for. "
    "Answer with that program. Do not summarize an attached spreadsheet or repeat its histogram. "
    "Use column names only if the code needs them. "
    "Write the whole program and finish every function. Do not say the request is too complex. "
    "If the user asks for a decision tree, include the split, the leaves, and a Gini impurity calculation for each leaf. "
    "The answer is always two parts and nothing else. "
    "Part one is a single Markdown code fence. The opening line is three backticks and the language, either python or sql. "
    "Use clean indentation, descriptive variable names, and brief inline comments. "
    "Never nest backticks. Never put explanatory prose inside the fence. "
    "Close the fence with a line of three backticks. "
    "Part two is outside the fence: two or three sentences of plain English that explain how to run the program."
)
code_task["steps"] = [code_step]
code_task["default_web_search"] = False
TASKS["code"] = code_task

memo_task = {}
memo_task["name"] = "Client & Internal Comms"
memo_task["blurb"] = "Write an email, memo, or proposal."
memo_task["keywords"] = ["email", "memo", "proposal", "client", "tone", "draft"]
memo_task["utterances"] = [
    "Draft an email to the client.",
    "Write an internal memo about the delay.",
    "Write a short proposal in our company voice.",
    "Reply to this customer in a warm professional tone.",
    "Draft a note I can send to the team.",
]
memo_step = {}
memo_step["kind"] = "generate"
memo_step["base_model"] = "gemma12"
memo_step["adapter_dir"] = "memo"
memo_step["forward_cap"] = 4000
memo_step["max_kv_size"] = 4096
memo_step["output_guess"] = 400
memo_step["max_tokens"] = 500
memo_step["check_syntax"] = False
memo_step["input_from"] = "material"
memo_step["instructions"] = (
    "Draft the email, memo, or proposal the user asked for. "
    "Use a warm, clear company voice. Make it ready to send."
)
memo_task["steps"] = [memo_step]
memo_task["default_web_search"] = False
TASKS["memo"] = memo_task

research = {}
research["name"] = "Multi-Step Deep Research"
research["blurb"] = "Answer a question from several files."
research["keywords"] = ["research", "compare", "cross", "several files", "sources"]
research["utterances"] = [
    "Compare these files and answer my question.",
    "Look across these documents and pull the facts.",
    "What do all of these files say about the budget?",
    "Cross-check these notes and write an answer.",
    "Research this question using the uploaded files.",
]
research_embed = {}
research_embed["kind"] = "embed"
research_embed["context_budget"] = 8000
research_embed["chunk_chars"] = 1200
research_extract = {}
research_extract["kind"] = "generate"
research_extract["base_model"] = "llama3"
research_extract["adapter_dir"] = ""
research_extract["forward_cap"] = 8000
research_extract["max_kv_size"] = 8192
research_extract["output_guess"] = 400
research_extract["max_tokens"] = 500
research_extract["check_syntax"] = False
research_extract["input_from"] = "material"
research_extract["instructions"] = (
    "Read the material and list the facts that help answer the user's question. "
    "Use short lines. Do not add facts that are not in the material."
)
research_write = {}
research_write["kind"] = "generate"
research_write["base_model"] = "gemma12"
research_write["adapter_dir"] = ""
research_write["forward_cap"] = 4096
research_write["max_kv_size"] = 4096
research_write["output_guess"] = 500
research_write["max_tokens"] = 600
research_write["map_max_tokens"] = 220
research_write["check_syntax"] = False
research_write["input_from"] = "previous"
research_write["map_trigger_tokens"] = 4096
research_write["map_chars"] = 8000
research_write["map_output_guess"] = 180
research_write["reduce_output_guess"] = 500
research_write["reduce_max_tokens"] = 600
research_write["map_instructions"] = (
    "Summarize these facts in a few plain sentences. Keep names and numbers."
)
research_write["reduce_instructions"] = (
    "Using only these facts, write a clear answer to the user's question. "
    "Say when the files do not contain the answer."
)
research_write["web_instructions"] = (
    "The notes include web search snippets gathered for this question. "
    "Write the answer from those snippets. Also use file facts when they are present. "
    "Ignore a note that says the files do not contain the answer when a snippet covers it. "
    "Give the points in plain sentences and name the source for each one. "
    "If the snippets do not contain the answer, say that the search did not return it."
)
research_write["instructions"] = research_write["reduce_instructions"]
research["steps"] = [research_embed, research_extract, research_write]
research["default_web_search"] = True
TASKS["research"] = research
