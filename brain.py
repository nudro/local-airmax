# All of the thinking for Local AirMax lives in this file.
# The web page calls these functions. They talk to the local models and the savings ledger.

import ast
import calendar
import gc
import hashlib
import json
import multiprocessing
import os
import sqlite3
import threading
import time
import uuid
from datetime import date, datetime

import config
from pydantic_ai.models import Model

# The one language model currently in memory. A second model is never loaded beside it.
LOADED_STATE = {}
LOADED_STATE["model"] = None
LOADED_STATE["tokenizer"] = None
LOADED_STATE["model_id"] = ""
LOADED_STATE["adapter_path"] = ""
LOADED_STATE["adapter_config"] = ""

# One saved document prefix. It is dropped when the resident model changes.
PROMPT_CACHE = {}
PROMPT_CACHE["model_id"] = ""
PROMPT_CACHE["adapter_path"] = ""
PROMPT_CACHE["prefix"] = ""
PROMPT_CACHE["token_count"] = 0
PROMPT_CACHE["cache"] = None

# Metal's spare buffer is capped once, before the first weights are loaded.
METAL_LIMIT_SET = False
# True when this load reused weights that were already in memory.
LAST_LOAD_WAS_REUSE = False

# Tokenizers are small. We keep them so the savings badge does not load the big model.
TOKENIZER_CACHE = {}
# Remember a failed tokenizer load so the badge does not retry on every keystroke.
TOKENIZER_FAILED = {}

# One language model is allowed in memory. Stop does not take this lock.
MODEL_LOCK = threading.Lock()
# The run id currently writing, and the id Stop asked to cancel.
ACTIVE_RUN = ""
STOP_RUN = ""
# The search helper, so Stop can end that process while the page is waiting.
EMBED_PROCESS = None
STOP_LOCK = threading.Lock()

# The task picker is built the first time Smart Mode needs it.
ROUTER = None

# A short sentence about why the last pick_task choice was made.
LAST_ROUTE_DETAIL = ""

# The run that is in progress. Tools read these instead of closing over the request.
ACTIVE_TRACE = []
ACTIVE_CHUNKS = []
AGENT_STEP = None
LAST_MODEL_TEXT = ""

# Remember what this run has already written into the timeline.
TRACE_MODEL_KEY = ""
TRACE_PREFILL_DONE = False
LOADED_DURING_RUN = []


# Folder that holds this app, the ledger, and the adapter folders.
def app_folder():
    return os.path.dirname(os.path.abspath(__file__))


# Where the savings ledger is stored.
def database_path():
    return os.path.join(app_folder(), "savings.db")


# Create the savings ledger the first time we need it.
def ensure_database():
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS runs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "created_at TEXT, "
            "task_id TEXT, "
            "task_name TEXT, "
            "input_tokens INTEGER, "
            "output_tokens INTEGER, "
            "rate_tier TEXT, "
            "cloud_cost REAL, "
            "local_cost REAL, "
            "savings REAL)"
        )
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS run_logs ("
            "run_id TEXT PRIMARY KEY, "
            "created_at TEXT, "
            "task_id TEXT, "
            "task_title TEXT, "
            "base_model TEXT, "
            "adapter_used TEXT, "
            "web_search_enabled INTEGER, "
            "prefill_tokens INTEGER, "
            "output_tokens INTEGER, "
            "latency_seconds REAL, "
            "estimated_cloud_cost REAL, "
            "local_cost REAL, "
            "net_savings REAL, "
            "events_json TEXT, "
            "rate_tier TEXT)"
        )
        cursor.execute("PRAGMA table_info(run_logs)")
        names = []
        for info in cursor.fetchall():
            names.append(info[1])
        has_tier = False
        for name in names:
            if name == "rate_tier":
                has_tier = True
        if has_tier == False:
            cursor.execute("ALTER TABLE run_logs ADD COLUMN rate_tier TEXT")
        cursor.execute(
            "UPDATE run_logs SET rate_tier = 'everyday' WHERE rate_tier IS NULL OR rate_tier = ''"
        )
        cursor.execute("UPDATE runs SET savings = 0 WHERE savings < 0")
        cursor.execute("UPDATE run_logs SET net_savings = 0 WHERE net_savings < 0")
        connection.commit()
    finally:
        connection.close()


# Pick the price row the user chose. Unknown names fall back to everyday prices.
def price_for(rate_tier):
    if rate_tier in config.PRICES:
        return config.PRICES[rate_tier]
    return config.PRICES["everyday"]


# Count how well a sentence matches one task's plain keywords.
def keyword_score(user_prompt, task_id):
    lowered = user_prompt.lower()
    score = 0
    task = config.TASKS[task_id]
    for word in task["keywords"]:
        if word in lowered:
            score = score + 1
    return score


# Choose a task from keywords when the smart picker is not available.
def keyword_route(user_prompt):
    best_id = ""
    best_score = 0
    for task_id in config.TASKS:
        score = keyword_score(user_prompt, task_id)
        if score > best_score:
            best_score = score
            best_id = task_id
    return best_id


# Build the meaning-based task picker. This downloads a tiny local model the first time.
def build_router():
    from semantic_router import IndexedRouter, Route
    from semantic_router.encoders import FastEmbedEncoder

    routes = []
    for task_id in config.TASKS:
        task = config.TASKS[task_id]
        route = Route(name=task_id, utterances=task["utterances"])
        routes.append(route)
    encoder = FastEmbedEncoder(name=config.ROUTER_MODEL)
    router = IndexedRouter(encoder=encoder, routes=routes)
    return router


# Ask the smart picker which task fits this sentence.
def route_with_model(user_prompt):
    global ROUTER, LAST_ROUTE_DETAIL
    LAST_ROUTE_DETAIL = ""
    if user_prompt.strip() == "":
        return ""
    try:
        if ROUTER is None:
            ROUTER = build_router()
        choice = ROUTER(user_prompt)
    except Exception:
        return keyword_route(user_prompt)
    if choice is None:
        return keyword_route(user_prompt)
    name = ""
    if hasattr(choice, "name"):
        if choice.name is not None:
            name = str(choice.name)
    if name == "":
        return keyword_route(user_prompt)
    if name not in config.TASKS:
        return ""
    score_text = ""
    if hasattr(choice, "similarity_score"):
        if choice.similarity_score is not None:
            score_text = str(choice.similarity_score)
    if score_text == "":
        if hasattr(choice, "score"):
            if choice.score is not None:
                score_text = str(choice.score)
    detail = "Smart Mode chose " + config.TASKS[name]["name"]
    if score_text != "":
        detail = detail + " with similarity " + score_text
    LAST_ROUTE_DETAIL = detail + "."
    return name


# Decide which task card to run. A card the person clicked wins. Smart Mode may still follow the files.
def pick_task(user_prompt, filenames, auto_route, chosen_task_id):
    global LAST_ROUTE_DETAIL
    file_count = 0
    csv_count = 0
    for name in filenames:
        file_count = file_count + 1
        lower = name.lower()
        if lower.endswith(".csv") or lower.endswith(".xlsx"):
            csv_count = csv_count + 1
    if auto_route:
        if csv_count > 0 and csv_count == file_count:
            LAST_ROUTE_DETAIL = "The person attached a spreadsheet, so this runs as " + config.TASKS["csv"]["name"] + "."
            return "csv"
        if file_count >= 2:
            LAST_ROUTE_DETAIL = "The person attached several files, so this runs as " + config.TASKS["research"]["name"] + "."
            return "research"
        routed = route_with_model(user_prompt)
        if routed != "":
            if LAST_ROUTE_DETAIL == "":
                LAST_ROUTE_DETAIL = "The wording matched " + config.TASKS[routed]["name"] + "."
            return routed
    if chosen_task_id in config.TASKS:
        LAST_ROUTE_DETAIL = "The person chose " + config.TASKS[chosen_task_id]["name"] + "."
        return chosen_task_id
    LAST_ROUTE_DETAIL = "This runs as " + config.TASKS["brief"]["name"] + "."
    return "brief"


# Read a plain text or markdown file.
def read_text_file(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    return text


# Raised when Stop is pressed, so the run can quit without saving a finished task.
class RunStopped(Exception):
    pass


# Remember which run is writing, so Stop can target that click.
def begin_run(run_id):
    global ACTIVE_RUN
    STOP_LOCK.acquire()
    try:
        ACTIVE_RUN = run_id
    finally:
        STOP_LOCK.release()


# Forget the run id once that click has left the model.
def end_run(run_id):
    global ACTIVE_RUN
    STOP_LOCK.acquire()
    try:
        if ACTIVE_RUN == run_id:
            ACTIVE_RUN = ""
    finally:
        STOP_LOCK.release()


# True when Stop was pressed for the run that is writing now.
def run_was_stopped():
    STOP_LOCK.acquire()
    try:
        if ACTIVE_RUN == "":
            return False
        if STOP_RUN == ACTIVE_RUN:
            return True
        return False
    finally:
        STOP_LOCK.release()


# Leave the run at the next pause when Stop has been pressed.
def raise_if_stopped():
    if run_was_stopped():
        raise RunStopped()


# Stop between prompt chunks. The model calls this while it is still reading.
def stop_during_prefill(done_tokens, total_tokens):
    if done_tokens < 0 and total_tokens < 0:
        return
    raise_if_stopped()


# Ask the running click to quit, and end the search helper if it is still up.
def request_stop(run_id):
    global STOP_RUN
    STOP_LOCK.acquire()
    try:
        if run_id == "":
            run_id = ACTIVE_RUN
        STOP_RUN = run_id
        process = EMBED_PROCESS
    finally:
        STOP_LOCK.release()
    if process is None:
        return
    if process.is_alive():
        process.terminate()


# Keep a handle to the search helper so Stop can end it.
def remember_embed_process(process):
    global EMBED_PROCESS
    STOP_LOCK.acquire()
    try:
        EMBED_PROCESS = process
    finally:
        STOP_LOCK.release()


# Drop the search helper handle after it has exited.
def forget_embed_process():
    global EMBED_PROCESS
    STOP_LOCK.acquire()
    try:
        EMBED_PROCESS = None
    finally:
        STOP_LOCK.release()


# Pull the words out of a PDF, one page after another.
def read_pdf(path):
    from pypdf import PdfReader

    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        raise_if_stopped()
        extracted = page.extract_text()
        if extracted is not None:
            text = text + extracted + "\n"
    return text


# Read a CSV or Excel file into a table. The raw sheet stays on this Mac.
def load_spreadsheet(path):
    import pandas as pd

    lower = path.lower()
    if lower.endswith(".xlsx"):
        frame = pd.read_excel(path, engine="openpyxl")
        return frame
    # Many business spreadsheets are saved as Windows text, not UTF-8.
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_error = None
    for encoding in encodings:
        try:
            frame = pd.read_csv(path, encoding=encoding)
            return frame
        except UnicodeDecodeError as err:
            last_error = err
    if last_error is not None:
        raise last_error
    frame = pd.read_csv(path)
    return frame


# Turn a bin edge into a short label. Dates are calendar days, and large numbers keep their digits.
def format_bin_edge(value, is_date):
    if is_date:
        import pandas as pd

        seconds = int(round(float(value)))
        stamp = pd.to_datetime(seconds, unit="s")
        return stamp.strftime("%Y-%m-%d")
    number = float(value)
    if abs(number) >= 100:
        return format(number, ",.0f")
    if abs(number) >= 1:
        text = format(number, ",.2f")
    else:
        text = format(number, ".3f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


# Turn date values into seconds so five bins stay on the real calendar.
def seconds_from_dates(values):
    import pandas as pd

    stamps = pd.to_datetime(values)
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    epoch = pd.Timestamp("1970-01-01")
    seconds = (stamps - epoch).dt.total_seconds()
    return seconds.to_numpy().astype("float64")


# Count rows in five bins for every numeric or date column.
def histogram_block(frame):
    import numpy as np
    import pandas as pd

    lines = []
    for column in frame.columns:
        series = frame[column]
        values = None
        label = ""
        edges_are_dates = False
        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna().astype(float)
            label = "number"
        elif pd.api.types.is_datetime64_any_dtype(series):
            values = series.dropna()
            label = "date"
            edges_are_dates = True
        else:
            non_null = series.dropna()
            as_number = pd.to_numeric(series, errors="coerce").dropna()
            if len(non_null) > 0 and len(as_number) * 2 >= len(non_null):
                values = as_number.astype(float)
                label = "number"
            else:
                sample = ""
                for item in non_null.head(8):
                    sample = str(item)
                    break
                looks_like_date = False
                if "-" in sample or "/" in sample or ":" in sample:
                    looks_like_date = True
                if looks_like_date:
                    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
                    valid = parsed.dropna()
                    if len(non_null) > 0 and len(valid) * 2 >= len(non_null):
                        values = valid
                        label = "date"
                        edges_are_dates = True
        if values is None:
            continue
        count_values = int(len(values))
        if count_values == 0:
            continue
        if edges_are_dates:
            numbers = seconds_from_dates(values)
        else:
            numbers = values.to_numpy()
        low = float(np.min(numbers))
        high = float(np.max(numbers))
        column_name = str(column).replace("|", "/")
        if low == high:
            shown = format_bin_edge(low, edges_are_dates)
            row = {}
            row["column"] = column_name
            row["kind"] = label
            row["left"] = shown
            row["right"] = shown
            row["count"] = str(count_values)
            row["share"] = "100%"
            lines.append(row)
            continue
        counts, edges = np.histogram(numbers, bins=5)
        index = 0
        while index < len(counts):
            left = format_bin_edge(edges[index], edges_are_dates)
            right = format_bin_edge(edges[index + 1], edges_are_dates)
            bin_count = int(counts[index])
            share = (float(bin_count) / float(count_values)) * 100.0
            row = {}
            row["column"] = column_name
            row["kind"] = label
            row["left"] = left
            row["right"] = right
            row["count"] = str(bin_count)
            row["share"] = format(share, ".1f") + "%"
            lines.append(row)
            index = index + 1
    return histogram_markdown(lines)


# Turn histogram rows into the pipe table the result page already knows how to draw.
def histogram_markdown(rows):
    if len(rows) == 0:
        return "No numeric or date columns were found."
    lines = []
    lines.append("| Column | Kind | From | To | Rows | Share |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    last_column = ""
    for row in rows:
        column = row["column"]
        kind = row["kind"]
        if column == last_column:
            column = ""
            kind = ""
        else:
            last_column = row["column"]
        line = "| " + column + " | " + kind + " | " + row["left"] + " | " + row["right"]
        line = line + " | " + row["count"] + " | " + row["share"] + " |"
        lines.append(line)
    text = ""
    for line in lines:
        if text == "":
            text = line
        else:
            text = text + "\n" + line
    return text


# Column names for a code draft. The histogram belongs to the spreadsheet task.
def spreadsheet_column_note(path):
    frame = load_spreadsheet(path)
    column_line = ""
    for column in frame.columns:
        name = str(column)
        if column_line == "":
            column_line = name
        else:
            column_line = column_line + ", " + name
    text = "Spreadsheet file: " + os.path.basename(path) + "\n"
    text = text + "Row count: " + str(int(frame.shape[0])) + "\n"
    text = text + "Columns: " + column_line
    return text


# Pull the histogram pipe table back out of a spreadsheet profile.
def histogram_table_from_profile(profile):
    if profile is None:
        return ""
    lines = str(profile).split("\n")
    started = False
    kept = []
    for line in lines:
        trimmed = line.strip()
        if started == False:
            if trimmed.startswith("| Column |"):
                started = True
                kept.append(trimmed)
            continue
        if trimmed.startswith("|"):
            kept.append(trimmed)
        else:
            break
    if len(kept) < 3:
        return ""
    text = ""
    for line in kept:
        if text == "":
            text = line
        else:
            text = text + "\n" + line
    return text


# Put the real histogram table above the model's answer so the bins stay in rows.
def answer_with_histogram_table(answer, profile):
    table = histogram_table_from_profile(profile)
    if table == "":
        return answer
    if answer.find("| Column |") != -1:
        return answer
    return "Quantified histogram\n\n" + table + "\n\n" + answer


# Turn a spreadsheet into the facts Gemma should use, without sending every raw row.
def build_csv_profile(path):
    frame = load_spreadsheet(path)
    column_line = ""
    for column in frame.columns:
        name = str(column)
        if column_line == "":
            column_line = name
        else:
            column_line = column_line + ", " + name
    row_count = int(frame.shape[0])
    preview = frame.head(5).to_string(index=False)
    try:
        stats = frame.describe(include="all").to_string()
    except Exception:
        stats = "Summary statistics were not available."
    if len(stats) > 8000:
        stats = stats[0:8000]
    histograms = histogram_block(frame)
    text = "Spreadsheet file: " + os.path.basename(path) + "\n"
    text = text + "Row count: " + str(row_count) + "\n"
    text = text + "Columns: " + column_line + "\n\n"
    text = text + "Quantified 5-bin histograms:\n" + histograms + "\n\n"
    text = text + "First 5 rows:\n" + preview + "\n\n"
    text = text + "Summary statistics:\n" + stats
    return text


# Read every uploaded file into one block of text the task can use.
def read_materials(file_paths, task_id):
    text_parts = []
    csv_parts = []
    for path in file_paths:
        lower = path.lower()
        try:
            if lower.endswith(".csv") or lower.endswith(".xlsx"):
                if task_id == "code":
                    csv_parts.append(spreadsheet_column_note(path))
                else:
                    csv_parts.append(build_csv_profile(path))
            elif lower.endswith(".pdf"):
                text_parts.append(read_pdf(path))
            elif lower.endswith(".txt") or lower.endswith(".md"):
                text_parts.append(read_text_file(path))
        except Exception:
            note = "This file could not be read: " + os.path.basename(path)
            if lower.endswith(".csv") or lower.endswith(".xlsx"):
                csv_parts.append(note)
            else:
                text_parts.append(note)
    parts = []
    if task_id == "csv":
        for part in csv_parts:
            parts.append(part)
        for part in text_parts:
            parts.append(part)
    else:
        for part in text_parts:
            parts.append(part)
        for part in csv_parts:
            parts.append(part)
    combined = ""
    for part in parts:
        if combined == "":
            combined = part
        else:
            combined = combined + "\n\n" + part
    return combined


# Split a long document into pieces that are each about the same size.
def chunk_text(text, chunk_chars):
    paragraphs = text.split("\n\n")
    packed = []
    current = ""
    for paragraph in paragraphs:
        cleaned = paragraph.strip()
        if cleaned == "":
            continue
        if current != "" and len(current) + len(cleaned) + 2 > chunk_chars:
            packed.append(current)
            current = cleaned
        else:
            if current == "":
                current = cleaned
            else:
                current = current + "\n\n" + cleaned
    if current != "":
        packed.append(current)
    chunks = []
    for piece in packed:
        if len(piece) <= chunk_chars:
            chunks.append(piece)
        else:
            start = 0
            while start < len(piece):
                chunks.append(piece[start:start + chunk_chars])
                start = start + chunk_chars
    return chunks


# Keep only the first part of a document when it is longer than the task allows.
def limit_text(text, token_budget):
    max_chars = token_budget * config.CHARS_PER_TOKEN
    warning = ""
    if len(text) > max_chars:
        text = text[0:max_chars]
        warning = "This uses the first part of a very long document."
    return text, warning


# Glue chunk texts together until the character budget is full.
def text_from_chunks(chunks, max_chars):
    chosen = ""
    for chunk in chunks:
        if len(chosen) >= max_chars:
            break
        if chosen == "":
            chosen = chunk
        else:
            chosen = chosen + "\n\n" + chunk
    if len(chosen) > max_chars:
        chosen = chosen[0:max_chars]
    return chosen


# Turn one embedding row into a plain list of numbers.
def vector_to_list(vector):
    if hasattr(vector, "tolist"):
        raw = vector.tolist()
    else:
        raw = vector
    if len(raw) > 0:
        first = raw[0]
        if isinstance(first, list):
            raw = first
    values = []
    for value in raw:
        values.append(float(value))
    return values


# Multiply two equal-length lists and add the results. This is a simple similarity score.
def dot_product(left, right):
    total = 0.0
    index = 0
    length = len(left)
    if len(right) < length:
        length = len(right)
    while index < length:
        total = total + (left[index] * right[index])
        index = index + 1
    return total


# Order scored chunks from the closest match to the weakest match.
def highest_first(scored):
    ordered = []
    used = []
    for item in scored:
        used.append(False)
    guard = 0
    while guard < len(scored):
        best_at = -1
        best_score = -999999.0
        index = 0
        while index < len(scored):
            if used[index] == False:
                if scored[index]["score"] > best_score:
                    best_score = scored[index]["score"]
                    best_at = index
            index = index + 1
        if best_at == -1:
            break
        used[best_at] = True
        ordered.append(scored[best_at])
        guard = guard + 1
    return ordered


# This runs in a short-lived helper process so the document model can leave memory when it finishes.
def embed_worker(query, chunks, max_chars, result_queue):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.EMBED_MODEL)
        query_vector = model.encode_query(query)
        query_list = vector_to_list(query_vector)
        scored = []
        for index in range(len(chunks)):
            doc_vector = model.encode_document(chunks[index])
            doc_list = vector_to_list(doc_vector)
            row = {}
            row["score"] = dot_product(query_list, doc_list)
            row["index"] = index
            scored.append(row)
        ordered = highest_first(scored)
        ranked_chunks = []
        for row in ordered:
            ranked_chunks.append(chunks[row["index"]])
        text = text_from_chunks(ranked_chunks, max_chars)
        message = {}
        message["text"] = text
        result_queue.put(message)
        del model
        gc.collect()
    except Exception as err:
        message = {}
        message["error"] = str(err)
        result_queue.put(message)


# Search the document pieces for the parts that match the question, then free that memory.
def embed_and_pick_chunks(query, chunks, max_chars):
    fallback = text_from_chunks(chunks, max_chars)
    if len(chunks) == 0:
        return ""
    search_query = query.strip()
    if search_query == "":
        search_query = "main points of this document"
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=embed_worker,
        args=(search_query, chunks, max_chars, result_queue),
    )
    remember_embed_process(process)
    process.start()
    started = time.perf_counter()
    while process.is_alive():
        if run_was_stopped():
            process.terminate()
            process.join(2)
            if process.is_alive():
                process.kill()
                process.join(2)
            forget_embed_process()
            raise RunStopped()
        if time.perf_counter() - started > 600:
            process.terminate()
            process.join(10)
            forget_embed_process()
            return fallback
        process.join(0.2)
    forget_embed_process()
    raise_if_stopped()
    try:
        message = result_queue.get(timeout=5)
    except Exception:
        return fallback
    if "text" in message:
        if message["text"] != "":
            return message["text"]
    return fallback


# Build the instruction the model sees. The file comes before the question so a repeat can reuse it.
def build_prompt(instructions, user_prompt, source_text):
    prompt = instructions
    if source_text.strip() != "":
        prompt = prompt + "\n\nThe attached file is included below. Use it. Do not say the data is missing.\n"
        prompt = prompt + source_text.strip()
    if user_prompt.strip() != "":
        prompt = prompt + "\n\nThe user asked:\n" + user_prompt.strip()
    return prompt


# Remember a model folder we already found on this Mac.
LOCAL_FOLDERS = {}


# The Hugging Face cache folder for one model repo.
def huggingface_snapshot_root(repo):
    safe = "models--" + repo.replace("/", "--")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", safe, "snapshots")


# True when this folder holds the big weight file, not just a tokenizer.
def snapshot_has_weights(folder):
    if os.path.isdir(folder) == False:
        return False
    for filename in os.listdir(folder):
        if filename.startswith("model") and filename.endswith(".safetensors"):
            return True
    return False


# True when this folder can count tokens.
def snapshot_has_tokenizer(folder):
    if os.path.isdir(folder) == False:
        return False
    json_path = os.path.join(folder, "tokenizer.json")
    if os.path.exists(json_path):
        return True
    model_path = os.path.join(folder, "tokenizer.model")
    return os.path.exists(model_path)


# Find a snapshot that is already on disk. This does not contact Hugging Face.
def find_snapshot(repo, need_weights):
    key = repo + "|tokenizer"
    if need_weights:
        key = repo + "|weights"
    if key in LOCAL_FOLDERS:
        return LOCAL_FOLDERS[key]
    root = huggingface_snapshot_root(repo)
    found = ""
    if os.path.isdir(root):
        for name in os.listdir(root):
            folder = os.path.join(root, name)
            matches = False
            if need_weights:
                matches = snapshot_has_weights(folder)
            else:
                matches = snapshot_has_tokenizer(folder)
            if matches:
                found = folder
    if found != "":
        LOCAL_FOLDERS[key] = found
    return found


# The weight folder for a model. Download it once, then always reuse this Mac's copy.
def local_weight_dir(repo):
    found = find_snapshot(repo, True)
    if found != "":
        return found
    from huggingface_hub import snapshot_download

    folder = snapshot_download(repo_id=repo)
    LOCAL_FOLDERS[repo + "|weights"] = folder
    return folder


# A folder that contains the tokenizer. Prefer the full model folder when it is already here.
def local_tokenizer_dir(repo):
    found = find_snapshot(repo, True)
    if found != "":
        return found
    found = find_snapshot(repo, False)
    if found != "":
        return found
    from huggingface_hub import snapshot_download

    patterns = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
    ]
    folder = snapshot_download(repo_id=repo, allow_patterns=patterns)
    LOCAL_FOLDERS[repo + "|tokenizer"] = folder
    return folder


# Load just the tokenizer for a model so we can count tokens without loading the model itself.
def get_tokenizer(base_model):
    if base_model in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[base_model]
    if base_model in TOKENIZER_FAILED:
        raise RuntimeError(TOKENIZER_FAILED[base_model])
    from mlx_lm.utils import load_tokenizer

    repo = config.MODELS[base_model]["repo"]
    try:
        folder = local_tokenizer_dir(repo)
        tokenizer = load_tokenizer(folder)
    except Exception as err:
        TOKENIZER_FAILED[base_model] = str(err)
        raise
    TOKENIZER_CACHE[base_model] = tokenizer
    return tokenizer


# Wrap the user's text in the model's chat format when that model has one.
def render_prompt(tokenizer, user_text):
    messages = []
    message = {}
    message["role"] = "user"
    message["content"] = user_text
    messages.append(message)
    has_template = False
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            has_template = True
    if has_template:
        try:
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    enable_thinking=False,
                )
            except Exception:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            if isinstance(rendered, str):
                rendered = rendered.replace("<|think|>", "")
                return rendered
        except Exception:
            return user_text
    return user_text


# Count tokens in a piece of text. If the tokenizer is missing, estimate from the length.
def count_tokens(text, base_model):
    if text == "":
        return 0
    tokenizer = None
    try:
        tokenizer = get_tokenizer(base_model)
    except Exception:
        tokenizer = None
    if tokenizer is None:
        return len(text) // config.CHARS_PER_TOKEN
    rendered = render_prompt(tokenizer, text)
    try:
        return len(tokenizer.encode(rendered))
    except Exception:
        return len(rendered) // config.CHARS_PER_TOKEN


# Shorten text so it stays under a token cap.
def trim_to_tokens(tokenizer, text, cap):
    try:
        token_ids = tokenizer.encode(text)
    except Exception:
        max_chars = cap * config.CHARS_PER_TOKEN
        return text[0:max_chars]
    if len(token_ids) <= cap:
        return text
    kept = []
    index = 0
    while index < cap and index < len(token_ids):
        kept.append(token_ids[index])
        index = index + 1
    try:
        return tokenizer.decode(kept)
    except Exception:
        max_chars = cap * config.CHARS_PER_TOKEN
        return text[0:max_chars]


# Shorten source text to the token budget for one model call.
def trim_for_budget(text, token_cap, base_model):
    if text == "":
        return text
    max_chars = token_cap * config.CHARS_PER_TOKEN
    if len(text) > max_chars:
        text = text[0:max_chars]
    try:
        tokenizer = get_tokenizer(base_model)
    except Exception:
        return text
    return trim_to_tokens(tokenizer, text, token_cap)


# Cloud dollar amount for this many tokens. Electricity is subtracted later, once per click.
def calculate_savings(input_tokens, output_tokens, rate_tier):
    rates = price_for(rate_tier)
    input_cost = (float(input_tokens) / 1000000.0) * rates["input_per_million"]
    output_cost = (float(output_tokens) / 1000000.0) * rates["output_per_million"]
    cloud = input_cost + output_cost
    return cloud


# Dollar guess shown before a run. Same prices as the receipt.
def estimate_savings(input_tokens, output_guess, rate_tier):
    cloud = calculate_savings(input_tokens, output_guess, rate_tier)
    return net_after_local(cloud)


# Savings after local electricity. A tiny cloud bill stays at zero.
def net_after_local(cloud):
    saved = cloud - config.LOCAL_COST
    if saved < 0.0:
        saved = 0.0
    return saved


# Guess the token totals for a map-reduce step without running the model.
def estimate_map_reduce(step, source_text, user_prompt):
    pieces = chunk_text(source_text, step["map_chars"])
    if len(pieces) == 0:
        pieces.append("")
    total_in = 0
    total_out = 0
    if len(pieces) == 1:
        trimmed = trim_for_budget(pieces[0], step["forward_cap"], step["base_model"])
        prompt = build_prompt(step["reduce_instructions"], user_prompt, trimmed)
        total_in = count_tokens(prompt, step["base_model"])
        total_out = step["reduce_output_guess"]
        return total_in, total_out
    for piece in pieces:
        trimmed = trim_for_budget(piece, step["forward_cap"], step["base_model"])
        prompt = build_prompt(step["map_instructions"], user_prompt, trimmed)
        total_in = total_in + count_tokens(prompt, step["base_model"])
        total_out = total_out + step["map_output_guess"]
    reduce_prompt = build_prompt(step["reduce_instructions"], user_prompt, "")
    total_in = total_in + count_tokens(reduce_prompt, step["base_model"])
    total_in = total_in + (step["map_output_guess"] * len(pieces))
    total_out = total_out + step["reduce_output_guess"]
    return total_in, total_out


# Guess the savings for a click before any big model is loaded.
def estimate_run(task_id, user_prompt, file_paths, rate_tier, use_web_search=False, model_choice="summaries"):
    source = read_materials(file_paths, task_id)
    task = config.TASKS[task_id]
    steps = steps_for_engine(task, model_choice, task_id)
    working = source
    previous_guess = 0
    total_in = 0
    total_out = 0
    warning = ""
    for step in steps:
        kind = step["kind"]
        if kind == "embed":
            limited, limit_warning = limit_text(working, step["context_budget"])
            if limit_warning != "":
                warning = limit_warning
            working = limited
            continue
        if kind == "map_reduce":
            if len(working) < config.SHORT_PASS_CHARS:
                instructions = ""
                if "reduce_instructions" in step:
                    instructions = step["reduce_instructions"]
                elif "instructions" in step:
                    instructions = step["instructions"]
                prompt = build_prompt(instructions, user_prompt, working)
                step_in = count_tokens(prompt, step["base_model"])
                step_out = step["reduce_output_guess"]
                if task_id == "brief":
                    step_out = 400
            else:
                step_in, step_out = estimate_map_reduce(step, working, user_prompt)
            total_in = total_in + step_in
            total_out = total_out + step_out
            previous_guess = step_out
            continue
        material = working
        use_map = False
        if step["input_from"] == "previous":
            material = ""
            if "map_trigger_tokens" in step:
                if previous_guess > step["map_trigger_tokens"]:
                    use_map = True
        if use_map:
            stand_in = "fact " * previous_guess
            step_in, step_out = estimate_map_reduce(step, stand_in, user_prompt)
            total_in = total_in + step_in
            total_out = total_out + step_out
        else:
            if step["input_from"] == "previous":
                prompt = build_prompt(step["instructions"], user_prompt, "")
                total_in = total_in + count_tokens(prompt, step["base_model"])
                total_in = total_in + previous_guess
            else:
                trimmed = trim_for_budget(material, step["forward_cap"], step["base_model"])
                prompt = build_prompt(step["instructions"], user_prompt, trimmed)
                total_in = total_in + count_tokens(prompt, step["base_model"])
            total_out = total_out + step["output_guess"]
        previous_guess = step["output_guess"]
    if use_web_search:
        total_in = total_in + 400
        total_out = total_out + 200
    saved = estimate_savings(total_in, total_out, rate_tier)
    preview = {}
    preview["estimated_savings"] = saved
    preview["input_tokens"] = total_in
    preview["output_tokens"] = total_out
    preview["task_id"] = task_id
    preview["task_name"] = task["name"]
    preview["warning"] = warning
    return preview


# Keep only the four job names. Anything else becomes Summaries.
def clean_model_choice(model_choice):
    if model_choice in config.ENGINE_MODES:
        return model_choice
    return "summaries"


# The settings for the job the person picked.
def engine_mode(model_choice):
    return config.ENGINE_MODES[clean_model_choice(model_choice)]


# The language model key for this job.
def base_for_engine(model_choice):
    return engine_mode(model_choice)["base_model"]


# Point one task step at the chosen job. Precision attaches that task's template.
def step_for_engine(step, model_choice, task_id):
    copied = copy_step(step)
    if "base_model" not in copied:
        return copied
    mode = engine_mode(model_choice)
    copied["base_model"] = mode["base_model"]
    if mode["use_lora"] == True:
        copied["adapter_dir"] = task_id
    else:
        copied["adapter_dir"] = ""
    return copied


# Rewrite every step in a task for the job switch.
def steps_for_engine(task, model_choice, task_id):
    steps = []
    for step in task["steps"]:
        steps.append(step_for_engine(step, model_choice, task_id))
    return steps


# Which job matches the model currently in memory.
def engine_from_loaded_repo():
    if LOADED_STATE["model_id"] == "":
        return ""
    for key in config.ENGINE_MODES:
        mode = config.ENGINE_MODES[key]
        repo = config.MODELS[mode["base_model"]]["repo"]
        if LOADED_STATE["model_id"] == repo:
            return key
    return ""


# Tell the page whether the local model is sitting in memory.
def model_power_status():
    status = {}
    choice = engine_from_loaded_repo()
    status["model_choice"] = choice
    if LOADED_STATE["model"] is None:
        status["power"] = "off"
        status["label"] = "AirMax Off"
        status["detail"] = "Turn on to download the model once."
    else:
        status["power"] = "on"
        status["label"] = "AirMax On"
        if choice != "":
            status["detail"] = config.ENGINE_MODES[choice]["label"] + " stays in memory until you turn AirMax off."
        else:
            status["detail"] = "In memory until you turn AirMax off."
    return status


# Load the chosen model and keep it, or drop every weight so the Mac gets that memory back.
def set_model_power(turn_on, model_choice, task_id=""):
    MODEL_LOCK.acquire()
    try:
        if turn_on:
            step = {}
            step["base_model"] = base_for_engine(model_choice)
            step["adapter_dir"] = ""
            mode = engine_mode(model_choice)
            if mode["use_lora"] == True and task_id != "":
                step["adapter_dir"] = task_id
            load_model_for_step(step)
            if step["adapter_dir"] != "":
                folder = adapter_folder(step["adapter_dir"])
                print(
                    "Precision template is in memory from "
                    + os.path.relpath(folder, app_folder())
                    + ".",
                    flush=True,
                )
        else:
            unload_model()
        return model_power_status()
    finally:
        MODEL_LOCK.release()


# Forget the saved document prefix. The next question reads the file again.
def clear_prompt_cache():
    PROMPT_CACHE["model_id"] = ""
    PROMPT_CACHE["adapter_path"] = ""
    PROMPT_CACHE["prefix"] = ""
    PROMPT_CACHE["token_count"] = 0
    PROMPT_CACHE["cache"] = None


# Keep Metal's spare buffer at 1.5 GB so a long answer cannot push the Mac into swap.
def limit_metal_cache():
    global METAL_LIMIT_SET
    if METAL_LIMIT_SET:
        return
    import mlx.core as mx

    mx.set_cache_limit(1536 * 1024 * 1024)
    METAL_LIMIT_SET = True


# Drop the language model so the Mac can have that memory back.
def unload_model():
    clear_prompt_cache()
    LOADED_STATE["model"] = None
    LOADED_STATE["tokenizer"] = None
    LOADED_STATE["model_id"] = ""
    LOADED_STATE["adapter_path"] = ""
    LOADED_STATE["adapter_config"] = ""
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        return


# Read the small template description next to a custom adapter.
def read_adapter_config(folder):
    path = os.path.join(folder, "adapter_config.json")
    if os.path.exists(path) == False:
        return ""
    handle = open(path, "r", encoding="utf-8")
    try:
        text = handle.read()
    finally:
        handle.close()
    return text


# Replace only the small template weights on a model that already has those layers.
def swap_adapter_weights(model, folder):
    import mlx.core as mx

    weights = mx.load(os.path.join(folder, "adapters.safetensors"))
    pairs = []
    for name in weights:
        pairs.append((name, weights[name]))
    model.load_weights(pairs, strict=False)


# Attach a Precision template, or swap its weights when the shape already matches.
def attach_adapter(adapter):
    config_text = read_adapter_config(adapter)
    if config_text == "":
        return False
    weights_file = os.path.join(adapter, "adapters.safetensors")
    if os.path.exists(weights_file) == False:
        return False
    model = LOADED_STATE["model"]
    if model is None:
        return False
    if LOADED_STATE["adapter_config"] == "":
        from mlx_lm.tuner.utils import load_adapters

        load_adapters(model, adapter)
        LOADED_STATE["adapter_path"] = adapter
        LOADED_STATE["adapter_config"] = config_text
        clear_prompt_cache()
        return True
    if LOADED_STATE["adapter_config"] != config_text:
        return False
    swap_adapter_weights(model, adapter)
    LOADED_STATE["adapter_path"] = adapter
    clear_prompt_cache()
    return True


# Make a plain copy of a step dictionary so we can change one setting without touching the original.
def copy_step(step):
    copied = {}
    for key in step:
        copied[key] = step[key]
    return copied


# The folder for one custom template. Each specialist lives under specialists/.
def adapter_folder(folder_name):
    if folder_name == "brief":
        return os.path.join(app_folder(), "specialists", "exec-brief")
    if folder_name == "csv":
        return os.path.join(app_folder(), "specialists", "data-analyst")
    if folder_name == "code":
        return os.path.join(app_folder(), "specialists", "ds-coder")
    return os.path.join(app_folder(), "adapters", folder_name)


# Load the one model this step needs. The same weights stay until the base model changes.
def load_model_for_step(step):
    global LAST_LOAD_WAS_REUSE
    limit_metal_cache()
    base_model = step["base_model"]
    info = config.MODELS[base_model]
    repo = info["repo"]
    adapter = ""
    # Adapters belong to Gemma 4 12B only. Other models always run with no adapter.
    if info["accepts_adapter"] == True:
        folder_name = step["adapter_dir"]
        if folder_name != "":
            folder = adapter_folder(folder_name)
            config_file = os.path.join(folder, "adapter_config.json")
            weights_file = os.path.join(folder, "adapters.safetensors")
            if os.path.exists(config_file) and os.path.exists(weights_file):
                adapter = folder
    model = LOADED_STATE["model"]
    if model is not None:
        if LOADED_STATE["model_id"] == repo and LOADED_STATE["adapter_path"] == adapter:
            LAST_LOAD_WAS_REUSE = True
            return
        if LOADED_STATE["model_id"] == repo and adapter != "":
            try:
                swapped = attach_adapter(adapter)
            except Exception as err:
                print("Adapter could not be swapped: " + str(err))
                swapped = False
            if swapped:
                LAST_LOAD_WAS_REUSE = True
                return
    unload_model()
    from mlx_lm import load

    folder = local_weight_dir(repo)
    model, tokenizer = load(folder)
    LOADED_STATE["model"] = model
    LOADED_STATE["tokenizer"] = tokenizer
    LOADED_STATE["model_id"] = repo
    TOKENIZER_CACHE[base_model] = tokenizer
    LOADED_STATE["adapter_path"] = ""
    LOADED_STATE["adapter_config"] = ""
    if adapter != "":
        try:
            attached = attach_adapter(adapter)
        except Exception as err:
            print("Adapter could not be attached: " + str(err))
            attached = False
        if attached == False:
            print("Custom template folder was skipped. Using the base model.")
    LAST_LOAD_WAS_REUSE = False


# Put the Executive Brief specialist in memory when the panel starts. A failed load must not stop the panel.
def warm_default_model():
    limit_metal_cache()
    step = {}
    step["base_model"] = "gemma12"
    step["adapter_dir"] = "brief"
    try:
        load_model_for_step(step)
        print("Executive brief model is in memory from specialists/exec-brief.")
    except Exception as err:
        print("Executive brief model could not be warmed: " + str(err))
        unload_model()


# Add one line to the timeline for this run.
def log_event(trace_list, stage, detail):
    event = {}
    event["step"] = len(trace_list) + 1
    event["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    event["stage"] = stage
    event["detail"] = detail
    trace_list.append(event)


# Clear the timeline markers at the start of a click.
def begin_trace():
    global TRACE_MODEL_KEY, TRACE_PREFILL_DONE, LOADED_DURING_RUN
    global ACTIVE_TRACE, ACTIVE_CHUNKS, AGENT_STEP, LAST_MODEL_TEXT
    TRACE_MODEL_KEY = ""
    TRACE_PREFILL_DONE = False
    LOADED_DURING_RUN = []
    ACTIVE_TRACE = []
    ACTIVE_CHUNKS = []
    AGENT_STEP = None
    LAST_MODEL_TEXT = ""


# Write the model load into the timeline the first time each model is used.
def remember_loaded_model(trace_list):
    global TRACE_MODEL_KEY
    key = LOADED_STATE["model_id"] + "|" + LOADED_STATE["adapter_path"]
    if key == TRACE_MODEL_KEY:
        return
    TRACE_MODEL_KEY = key
    if LAST_LOAD_WAS_REUSE:
        detail = "Reused the model already in memory."
    else:
        if LOADED_STATE["adapter_path"] == "":
            adapter_words = "base model fallback"
        else:
            adapter_words = os.path.basename(LOADED_STATE["adapter_path"])
        detail = "Loaded " + LOADED_STATE["model_id"] + ". Adapter: " + adapter_words + "."
    log_event(trace_list, "MODEL_LOAD", detail)
    LOADED_DURING_RUN.append(LOADED_STATE["model_id"])


# Write the first input-size guess, before the model answers.
def remember_prefill(trace_list, user_text, base_model):
    global TRACE_PREFILL_DONE
    if TRACE_PREFILL_DONE:
        return
    TRACE_PREFILL_DONE = True
    count = count_tokens(user_text, base_model)
    detail = "About " + str(count) + " tokens will be read before the answer."
    log_event(trace_list, "PREFILL_ESTIMATE", detail)


# True when this click attached a PDF, a spreadsheet, or a staged demo article.
def files_need_embed_log(file_paths):
    for path in file_paths:
        lower = path.lower()
        if lower.endswith(".pdf") or lower.endswith(".csv"):
            return True
        if demo_file_path(os.path.basename(path)) != "":
            return True
    return False


# The custom template for this task, or a plain fallback when that folder is missing.
def template_label(task_id):
    folder = adapter_folder(task_id)
    config_file = os.path.join(folder, "adapter_config.json")
    weights_file = os.path.join(folder, "adapters.safetensors")
    if os.path.exists(config_file) and os.path.exists(weights_file):
        return os.path.basename(folder)
    return "base model fallback"


# The adapter folder for this task, or a plain fallback label when that folder is missing.
def adapter_label(task):
    label = "base model fallback"
    for step in task["steps"]:
        if "adapter_dir" not in step:
            continue
        folder_name = step["adapter_dir"]
        if folder_name == "":
            continue
        folder = adapter_folder(folder_name)
        config_file = os.path.join(folder, "adapter_config.json")
        weights_file = os.path.join(folder, "adapters.safetensors")
        if os.path.exists(config_file) and os.path.exists(weights_file):
            label = os.path.basename(folder)
    return label


# Join the models this click actually loaded.
def base_model_label(task):
    names = []
    for repo in LOADED_DURING_RUN:
        already = False
        for seen in names:
            if seen == repo:
                already = True
        if already == False:
            names.append(repo)
    if len(names) == 0:
        for step in task["steps"]:
            if "base_model" in step:
                info = config.MODELS[step["base_model"]]
                return info["repo"]
        return config.MODELS["gemma12"]["repo"]
    label = ""
    for name in names:
        if label == "":
            label = name
        else:
            label = label + " then " + name
    return label


# Find one of the two known demo articles. Any other name is refused.
def demo_file_path(name):
    if name is None or name == "":
        return ""
    safe = os.path.basename(name)
    for item in config.DEMO_URLS:
        if os.path.basename(item["path"]) == safe:
            return os.path.join(app_folder(), item["path"])
    return ""


# List the staged demo articles for the chips under the drop zone.
def list_demo_files():
    rows = []
    for item in config.DEMO_URLS:
        row = {}
        row["name"] = os.path.basename(item["path"])
        row["title"] = item["title"]
        rows.append(row)
    return rows


# Turn a page into markdown from its title, headings, and paragraphs.
def article_to_markdown(html, page_url):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    lines = []
    title = ""
    if soup.title is not None:
        if soup.title.string is not None:
            title = soup.title.string.strip()
    if title != "":
        lines.append("# " + title)
        lines.append("")
        lines.append("Source: " + page_url)
        lines.append("")
    skip_lines = [
        "Advanced tech. Deep domain know-how.",
        "Real-world missions. Real-time results",
        "Bold ideas. Expert takes.",
        "Whether you love solving problems",
        "Driven by purpose. Powered by people.",
        "You can update your communications choices",
        "You can learn more about Booz Allen",
        "Thank you for contacting Booz Allen.",
    ]
    stop_heads = ["tags", "related content", "information for", "industries", "featured solutions", "about us"]
    for tag in soup.find_all(["h1", "h2", "h3", "p"]):
        text = tag.get_text(" ", strip=True)
        if text == "":
            continue
        lowered = text.lower()
        name = tag.name
        if name == "h2" or name == "h3":
            stopped = False
            for head in stop_heads:
                if lowered == head:
                    stopped = True
            if stopped:
                break
        skipped = False
        for prefix in skip_lines:
            if text.startswith(prefix):
                skipped = True
        if skipped:
            continue
        if name == "h1":
            lines.append("# " + text)
            lines.append("")
        elif name == "h2":
            lines.append("## " + text)
            lines.append("")
        elif name == "h3":
            lines.append("### " + text)
            lines.append("")
        elif len(text) >= 40:
            lines.append(text)
            lines.append("")
    body = ""
    for line in lines:
        body = body + line + "\n"
    return body.strip()


# Download a missing demo article. A failed download becomes a short note and does not stop the app.
def stage_demo_files():
    import urllib.request

    folder = os.path.join(app_folder(), "demo")
    if os.path.isdir(folder) == False:
        os.makedirs(folder)
    for item in config.DEMO_URLS:
        dest = os.path.join(app_folder(), item["path"])
        if os.path.exists(dest):
            if os.path.getsize(dest) > 0:
                continue
        note = (
            "This demo article could not be downloaded.\n\n"
            "Open the page in a browser when this Mac is online:\n"
            + item["url"]
            + "\n"
        )
        try:
            request = urllib.request.Request(item["url"])
            request.add_header(
                "User-Agent",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                raw = response.read()
            html = raw.decode("utf-8", errors="replace")
            text = article_to_markdown(html, item["url"])
            if len(text) < 200:
                text = note + "\nThe page did not include enough readable text to stage.\n"
        except Exception:
            text = note
        with open(dest, "w", encoding="utf-8") as handle:
            handle.write(text)


# Look up a short question on the public web and return a few snippets.
def search_web(query: str) -> str:
    """Search the public web and return a few short snippets. The query is the only text that leaves this Mac."""
    from ddgs import DDGS

    backends = ["bing", "google", "yahoo", "duckduckgo"]
    found = []
    last_error = ""
    for backend in backends:
        raise_if_stopped()
        try:
            found = DDGS().text(query, max_results=config.MAX_SEARCH_RESULTS, backend=backend)
        except RunStopped:
            raise
        except Exception as err:
            last_error = str(err)
            found = []
        if found is None:
            found = []
        if len(found) > 0:
            break
    if len(found) == 0:
        detail = "Web search failed for \"" + query + "\"."
        if last_error != "":
            detail = detail + " " + last_error
        log_event(ACTIVE_TRACE, "TOOL_WEB_SEARCH", detail)
        return "Web search could not be completed."
    urls = []
    lines = []
    for item in found:
        title = ""
        href = ""
        body = ""
        if "title" in item and item["title"] is not None:
            title = str(item["title"])
        if "href" in item and item["href"] is not None:
            href = str(item["href"])
        elif "url" in item and item["url"] is not None:
            href = str(item["url"])
        if "body" in item and item["body"] is not None:
            body = str(item["body"])
        if href != "":
            urls.append(href)
        lines.append(title + "\n" + href + "\n" + body)
    detail = "Searched the web for \"" + query + "\"."
    if len(urls) == 0:
        detail = detail + " No pages were returned."
    else:
        joined = ""
        for url in urls:
            if joined == "":
                joined = url
            else:
                joined = joined + ", " + url
        detail = detail + " Pages: " + joined + "."
    log_event(ACTIVE_TRACE, "TOOL_WEB_SEARCH", detail)
    if len(lines) == 0:
        return "No web results were found."
    text = ""
    for line in lines:
        if text == "":
            text = line
        else:
            text = text + "\n\n" + line
    return text


# Search the files already collected for this run and return the closest passages.
def query_local_docs(query: str) -> str:
    """Search the uploaded files and demo articles already on this Mac."""
    scored = []
    words = query.lower().split()
    for chunk in ACTIVE_CHUNKS:
        lower = chunk.lower()
        score = 0
        for word in words:
            if len(word) < 3:
                continue
            if word in lower:
                score = score + 1
        row = {}
        row["score"] = score
        row["text"] = chunk
        scored.append(row)
    ordered = highest_first(scored)
    chosen = []
    for row in ordered:
        if row["score"] <= 0:
            continue
        chosen.append(row)
        if len(chosen) >= 3:
            break
    if len(chosen) == 0 and len(ordered) > 0:
        chosen.append(ordered[0])
    snippets = []
    for row in chosen:
        snippets.append(row["text"].strip().split("\n")[0][:80])
    if len(snippets) == 0:
        log_event(ACTIVE_TRACE, "TOOL_LOCAL_DOCS", "Looked through the attached files for \"" + query + "\" and found nothing.")
        return "No attached document text was available."
    joined = ""
    for snippet in snippets:
        if joined == "":
            joined = snippet
        else:
            joined = joined + " | " + snippet
    log_event(ACTIVE_TRACE, "TOOL_LOCAL_DOCS", "Looked through the attached files for \"" + query + "\". Retrieved: " + joined + ".")
    text = ""
    for row in chosen:
        if text == "":
            text = row["text"]
        else:
            text = text + "\n\n" + row["text"]
    return text


# Turn message parts into one plain string the local model can read.
def content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    text = ""
    try:
        for item in content:
            if isinstance(item, str):
                piece = item
            else:
                piece = str(getattr(item, "content", item))
            if text == "":
                text = piece
            else:
                text = text + "\n" + piece
    except TypeError:
        return str(content)
    return text


# Read a TOOL line at the start of a reply. A normal answer is left alone.
def parse_tool_reply(text):
    if text is None:
        return "", ""
    lines = str(text).split("\n")
    first = ""
    for line in lines:
        stripped = line.strip()
        if stripped != "":
            first = stripped
            break
    upper = first.upper()
    if upper.startswith("TOOL ") == False:
        return "", ""
    tool_name = first[5:].strip()
    query_lines = []
    if ":" in tool_name:
        halves = tool_name.split(":", 1)
        tool_name = halves[0].strip()
        if halves[1].strip() != "":
            query_lines.append(halves[1].strip())
    passed_first = False
    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue
        if passed_first == False:
            passed_first = True
            continue
        upper_line = stripped.upper()
        if upper_line.startswith("QUERY:"):
            rest = stripped[6:].strip()
            if rest != "":
                query_lines.append(rest)
        elif upper_line.startswith("QUERY "):
            rest = stripped[6:].strip()
            if rest != "":
                query_lines.append(rest)
        else:
            query_lines.append(stripped)
    query = ""
    for piece in query_lines:
        if query == "":
            query = piece
        else:
            query = query + " " + piece
    return tool_name, query


# Names of the tools this request is allowed to call.
def tool_names_from(parameters):
    names = []
    tools = []
    if parameters is not None:
        if hasattr(parameters, "function_tools"):
            tools = parameters.function_tools
    for tool in tools:
        if hasattr(tool, "name"):
            names.append(tool.name)
    return names


# The one PydanticAI model. It calls the MLX model that is already loaded.
class LocalMLXModel(Model):
    model_name = "local-mlx"
    system = "mlx"

    def __init__(self):
        super().__init__()
        self.input_tokens = 0
        self.output_tokens = 0

    async def request(self, messages, model_settings, model_request_parameters):
        global LAST_MODEL_TEXT
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.usage import RequestUsage

        prompt = messages_to_prompt(messages, model_request_parameters)
        result = None
        for event in run_one_call(AGENT_STEP, prompt, ACTIVE_TRACE):
            if event["kind"] == "result":
                result = event
        self.input_tokens = self.input_tokens + result["input_tokens"]
        self.output_tokens = self.output_tokens + result["output_tokens"]
        LAST_MODEL_TEXT = result["text"]
        usage = RequestUsage(input_tokens=result["input_tokens"], output_tokens=result["output_tokens"])
        tool_name, query = parse_tool_reply(result["text"])
        allowed = tool_names_from(model_request_parameters)
        if tool_name != "" and query != "":
            permitted = False
            for name in allowed:
                if name == tool_name:
                    permitted = True
            if permitted:
                args = {}
                args["query"] = query
                part = ToolCallPart(tool_name=tool_name, args=args)
                return ModelResponse(parts=[part], usage=usage)
        part = TextPart(content=result["text"])
        return ModelResponse(parts=[part], usage=usage)


# Flatten the agent messages, and tell the model the two-line tool reply.
def messages_to_prompt(messages, parameters):
    lines = []
    lines.append("You are a local assistant running on this Mac.")
    names = tool_names_from(parameters)
    if len(names) == 0:
        lines.append("Write the answer in plain text. Do not call a tool.")
    else:
        joined = ""
        for name in names:
            if joined == "":
                joined = name
            else:
                joined = joined + ", " + name
        lines.append("If you need a tool, reply with exactly two lines and nothing else:")
        lines.append("TOOL tool_name")
        lines.append("QUERY the short question")
        lines.append("The only tool names you may use are: " + joined + ".")
        lines.append("If you can answer from the notes already in this prompt, write the answer and do not start with TOOL.")
    for message in messages:
        instructions = getattr(message, "instructions", None)
        if instructions is not None and str(instructions) != "":
            lines.append("Instructions: " + str(instructions))
        parts = getattr(message, "parts", [])
        for part in parts:
            kind = getattr(part, "part_kind", "")
            if kind == "system-prompt":
                lines.append("System: " + content_to_text(getattr(part, "content", "")))
            elif kind == "user-prompt":
                lines.append("User: " + content_to_text(getattr(part, "content", "")))
            elif kind == "tool-return":
                tool_name = getattr(part, "tool_name", "tool")
                lines.append("Tool " + str(tool_name) + " returned:\n" + content_to_text(getattr(part, "content", "")))
            elif kind == "text":
                lines.append("Assistant: " + content_to_text(getattr(part, "content", "")))
            elif kind == "tool-call":
                lines.append("Assistant called " + str(getattr(part, "tool_name", "tool")))
            else:
                content = getattr(part, "content", "")
                if content != "" and content is not None:
                    lines.append(content_to_text(content))
    text = ""
    for line in lines:
        if text == "":
            text = line
        else:
            text = text + "\n" + line
    return text


# Run the final answer through PydanticAI so web search can be used, at most a few turns.
def run_with_agent(step, user_prompt, source_text, use_web_search):
    global AGENT_STEP, LAST_MODEL_TEXT
    from pydantic_ai import Agent, UsageLimits

    AGENT_STEP = step
    active_tools = []
    if use_web_search:
        active_tools.append(search_web)
    if len(ACTIVE_CHUNKS) > 0:
        active_tools.append(query_local_docs)
    instructions = ""
    if "instructions" in step:
        instructions = step["instructions"]
    elif "reduce_instructions" in step:
        instructions = step["reduce_instructions"]
    model = LocalMLXModel()
    agent = Agent(model, tools=active_tools, instructions=instructions)
    trimmed = trim_for_budget(source_text, step["forward_cap"], step["base_model"])
    prompt = build_prompt("", user_prompt, trimmed)
    limits = UsageLimits(request_limit=config.MAX_AGENT_TURNS)
    text = ""
    try:
        result = agent.run_sync(prompt, usage_limits=limits)
        if result.output is None:
            text = ""
        elif isinstance(result.output, str):
            text = result.output
        else:
            text = str(result.output)
    except Exception as err:
        print("Agent stopped: " + str(err))
        text = LAST_MODEL_TEXT
        if text.strip().upper().startswith("TOOL"):
            text = "The local model stopped before it finished the answer."
    text = clean_gemma_output(text)
    wrapped = {}
    wrapped["text"] = text
    wrapped["input_tokens"] = model.input_tokens
    wrapped["output_tokens"] = model.output_tokens
    return wrapped


# Drop Gemma's hidden thinking channel so the screen shows the answer only.
def clean_gemma_output(text):
    if text is None:
        return ""
    cleaned = str(text)
    marker = "<channel|>"
    guard = 0
    while guard < 8:
        end_at = cleaned.find(marker)
        if end_at == -1:
            break
        cleaned = cleaned[end_at + len(marker):]
        guard = guard + 1
    cleaned = cleaned.replace("<|channel>thought", "")
    cleaned = cleaned.replace("<|think|>", "")
    return cleaned.strip()


# Turn a rendered prompt into token ids the same way generation does.
def encode_prompt_ids(tokenizer, text):
    add_special_tokens = True
    bos = None
    if hasattr(tokenizer, "bos_token"):
        bos = tokenizer.bos_token
    if bos is not None:
        if text.startswith(bos):
            add_special_tokens = False
    raw = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    ids = []
    for item in raw:
        ids.append(int(item))
    return ids


# True when the document tokens are the start of the full prompt.
def ids_are_prefix(prefix_ids, full_ids):
    if len(prefix_ids) == 0:
        return False
    if len(prefix_ids) >= len(full_ids):
        return False
    index = 0
    while index < len(prefix_ids):
        if prefix_ids[index] != full_ids[index]:
            return False
        index = index + 1
    return True


# Count tokens that land exactly on the cut between the file and the question.
def prefix_token_count(tokenizer, full_ids, full_text, cut_at):
    low = 1
    high = len(full_ids) - 1
    found = 0
    while low <= high:
        mid = (low + high) // 2
        decoded = tokenizer.decode(full_ids[:mid])
        if full_text.startswith(decoded) == False:
            high = mid - 1
        elif len(decoded) < cut_at:
            found = mid
            low = mid + 1
        elif len(decoded) == cut_at:
            return mid
        else:
            high = mid - 1
    if found == 0:
        return 0
    decoded = tokenizer.decode(full_ids[:found])
    if len(decoded) == cut_at:
        if full_text.startswith(decoded):
            return found
    return 0


# Split a prompt into the saved document and the new question.
def split_prompt_tokens(tokenizer, user_text):
    marker = "\n\nThe user asked:\n"
    if user_text.find(marker) == -1:
        return None
    full = render_prompt(tokenizer, user_text)
    cut_at = full.find(marker)
    if cut_at == -1:
        return None
    cut_at = cut_at + len(marker)
    full_ids = encode_prompt_ids(tokenizer, full)
    if len(full_ids) < 2:
        return None
    prefix_text = full[:cut_at]
    prefix_ids = encode_prompt_ids(tokenizer, prefix_text)
    aligned = ids_are_prefix(prefix_ids, full_ids)
    if aligned == False:
        count = prefix_token_count(tokenizer, full_ids, full, cut_at)
        if count < 1:
            return None
        prefix_ids = full_ids[:count]
    suffix_ids = []
    index = len(prefix_ids)
    while index < len(full_ids):
        suffix_ids.append(full_ids[index])
        index = index + 1
    if len(suffix_ids) == 0:
        return None
    split = {}
    split["prefix"] = prefix_text
    split["prefix_ids"] = prefix_ids
    split["suffix_ids"] = suffix_ids
    return split


# How many tokens a saved prompt cache currently holds.
def cache_token_count(cache):
    if cache is None:
        return 0
    if len(cache) == 0:
        return 0
    first = cache[0]
    if hasattr(first, "offset") == False:
        return 0
    return int(first.offset)


# Remember one document prefix in memory.
def store_prompt_cache(prefix_text, token_count, cache):
    PROMPT_CACHE["model_id"] = LOADED_STATE["model_id"]
    PROMPT_CACHE["adapter_path"] = LOADED_STATE["adapter_path"]
    PROMPT_CACHE["prefix"] = prefix_text
    PROMPT_CACHE["token_count"] = token_count
    PROMPT_CACHE["cache"] = cache


# Shorten a cache back to the document so the next question can reuse it.
def trim_cache_to_prefix(cache, token_count):
    from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

    if can_trim_prompt_cache(cache) == False:
        return False
    held = cache_token_count(cache)
    if held < token_count:
        return False
    extra = held - token_count
    if extra > 0:
        trim_prompt_cache(cache, extra)
    if cache_token_count(cache) != token_count:
        return False
    return True


# File name for a short saved prefix. Long documents stay in memory only.
def prefix_disk_path(model_id, adapter_path, prefix_text):
    folder = os.path.join(app_folder(), "cache", "prefixes")
    if os.path.isdir(folder) == False:
        os.makedirs(folder)
    raw = model_id + "\n" + adapter_path + "\n" + prefix_text
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return os.path.join(folder, digest + ".safetensors")


# Write a short prefix to disk so task instructions survive a restart.
def save_prefix_to_disk(prefix_text, token_ids, cache):
    if len(token_ids) > 512:
        return
    from mlx_lm.models.cache import save_prompt_cache

    path = prefix_disk_path(
        LOADED_STATE["model_id"],
        LOADED_STATE["adapter_path"],
        prefix_text,
    )
    meta = {}
    meta["model_id"] = LOADED_STATE["model_id"]
    meta["adapter_path"] = LOADED_STATE["adapter_path"]
    meta["token_count"] = str(len(token_ids))
    meta["prefix"] = prefix_text
    try:
        save_prompt_cache(path, cache, meta)
    except Exception as err:
        print("Prompt cache could not be saved: " + str(err))


# Read a short prefix back from disk when it still matches this model.
def load_prefix_from_disk(prefix_text, token_count):
    if token_count > 512:
        return None
    path = prefix_disk_path(
        LOADED_STATE["model_id"],
        LOADED_STATE["adapter_path"],
        prefix_text,
    )
    if os.path.exists(path) == False:
        return None
    from mlx_lm.models.cache import load_prompt_cache

    try:
        loaded = load_prompt_cache(path, return_metadata=True)
    except Exception as err:
        print("Saved prompt could not be read: " + str(err))
        return None
    cache = loaded[0]
    meta = loaded[1]
    if "model_id" not in meta:
        return None
    if meta["model_id"] != LOADED_STATE["model_id"]:
        return None
    if "adapter_path" not in meta:
        return None
    if meta["adapter_path"] != LOADED_STATE["adapter_path"]:
        return None
    if "token_count" not in meta:
        return None
    if meta["token_count"] != str(token_count):
        return None
    if "prefix" not in meta:
        return None
    if meta["prefix"] != prefix_text:
        return None
    return cache


# Return the saved document prefix when this question is about the same file.
def recall_prefix_cache(prefix_text, token_ids):
    token_count = len(token_ids)
    same = False
    if PROMPT_CACHE["cache"] is not None:
        if PROMPT_CACHE["prefix"] == prefix_text:
            if PROMPT_CACHE["model_id"] == LOADED_STATE["model_id"]:
                if PROMPT_CACHE["adapter_path"] == LOADED_STATE["adapter_path"]:
                    if PROMPT_CACHE["token_count"] == token_count:
                        same = True
    if same:
        trimmed = trim_cache_to_prefix(PROMPT_CACHE["cache"], token_count)
        if trimmed:
            return PROMPT_CACHE["cache"]
    loaded = load_prefix_from_disk(prefix_text, token_count)
    if loaded is None:
        return None
    store_prompt_cache(prefix_text, token_count, loaded)
    return loaded


# Read the document once and keep that prefix for the next question.
def prefill_prefix_cache(model, token_ids):
    from mlx_lm.models.cache import make_prompt_cache
    import mlx.core as mx

    cache = make_prompt_cache(model)
    start = 0
    total = len(token_ids)
    while start < total:
        raise_if_stopped()
        end = start + 2048
        if end > total:
            end = total
        piece = token_ids[start:end]
        tokens = mx.array(piece)
        model(tokens[None], cache=cache)
        mx.eval([c.state for c in cache])
        start = end
    return cache


# Prepare a reusable document prefix, or report that this prompt should be read in full.
def ready_prefix_cache(step, user_text, model, tokenizer):
    split = split_prompt_tokens(tokenizer, user_text)
    if split is None:
        return None, None, 0, False
    prefix_text = split["prefix"]
    prefix_ids = split["prefix_ids"]
    suffix_ids = split["suffix_ids"]
    prefix_count = len(prefix_ids)
    if prefix_count > step["max_kv_size"]:
        return None, None, 0, False
    if len(suffix_ids) == 0:
        return None, None, 0, False
    found = recall_prefix_cache(prefix_text, prefix_ids)
    if found is not None:
        return found, suffix_ids, prefix_count, True
    found = prefill_prefix_cache(model, prefix_ids)
    if cache_token_count(found) != prefix_count:
        return None, None, 0, False
    store_prompt_cache(prefix_text, prefix_count, found)
    save_prefix_to_disk(prefix_text, prefix_ids, found)
    return found, suffix_ids, prefix_count, False


# Hold back a thinking tag until it is finished, and return the words that are safe to show.
def take_visible_piece(held):
    cleaned = held
    marker = "<channel|>"
    guard = 0
    while guard < 8:
        end_at = cleaned.find(marker)
        if end_at == -1:
            break
        cleaned = cleaned[end_at + len(marker):]
        guard = guard + 1
    cleaned = cleaned.replace("<|channel>thought", "")
    cleaned = cleaned.replace("<|think|>", "")
    markers = ["<channel|>", "<|channel>thought", "<|think|>"]
    hold_from = len(cleaned)
    window_start = 0
    if len(cleaned) > 48:
        window_start = len(cleaned) - 48
    index = window_start
    while index < len(cleaned):
        suffix = cleaned[index:]
        mark_index = 0
        while mark_index < len(markers):
            tag = markers[mark_index]
            if tag.startswith(suffix):
                if suffix != tag:
                    if index < hold_from:
                        hold_from = index
            mark_index = mark_index + 1
        index = index + 1
    safe = cleaned[0:hold_from]
    rest = cleaned[hold_from:]
    return safe, rest


# Ask the loaded model for one answer and record the real token counts.
def run_one_call(step, user_text, trace_list=None, reuse_prefix=False, keep_full_prompt=False):
    raise_if_stopped()
    load_model_for_step(step)
    raise_if_stopped()
    if trace_list is not None:
        remember_loaded_model(trace_list)
    from mlx_lm import stream_generate
    import mlx.core as mx

    model = LOADED_STATE["model"]
    tokenizer = LOADED_STATE["tokenizer"]
    prompt = render_prompt(tokenizer, user_text)
    max_tokens = step["max_tokens"]
    prompt_cache = None
    suffix_ids = None
    prefix_count = 0
    reused_prefix = False
    if reuse_prefix:
        try:
            ready = ready_prefix_cache(step, user_text, model, tokenizer)
            prompt_cache = ready[0]
            suffix_ids = ready[1]
            prefix_count = ready[2]
            reused_prefix = ready[3]
        except RunStopped:
            raise
        except Exception as err:
            print("Saved prompt could not be reused: " + str(err))
            prompt_cache = None
            prefix_count = 0
            reused_prefix = False
    if trace_list is not None:
        if reused_prefix:
            log_event(
                trace_list,
                "PREFILL_ESTIMATE",
                "Reused the saved prompt so the document was not read again.",
            )
        else:
            remember_prefill(trace_list, user_text, step["base_model"])
    # The model uses its normal answering style. The screen has no style slider.
    # 2048-token prefill steps keep a long prompt moving on this Mac.
    if prompt_cache is not None:
        token_stream = stream_generate(
            model,
            tokenizer,
            suffix_ids,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            prefill_step_size=2048,
            prompt_progress_callback=stop_during_prefill,
        )
    elif keep_full_prompt:
        prefix_count = 0
        token_stream = stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            prefill_step_size=2048,
            prompt_progress_callback=stop_during_prefill,
        )
    else:
        prefix_count = 0
        token_stream = stream_generate(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            max_kv_size=step["max_kv_size"],
            prefill_step_size=2048,
            prompt_progress_callback=stop_during_prefill,
        )
    raw_parts = []
    held = ""
    last = None
    try:
        for chunk in token_stream:
            raise_if_stopped()
            piece = chunk.text
            if piece is None:
                piece = ""
            raw_parts.append(piece)
            held = held + piece
            safe, held = take_visible_piece(held)
            if safe != "":
                shown = {}
                shown["kind"] = "piece"
                shown["text"] = safe
                yield shown
            last = chunk
    except RunStopped:
        try:
            clear_prompt_cache()
            mx.clear_cache()
        except Exception:
            pass
        raise
    if held != "":
        tail = clean_gemma_output(held)
        if tail != "":
            shown = {}
            shown["kind"] = "piece"
            shown["text"] = tail
            yield shown
    text = ""
    for piece in raw_parts:
        text = text + piece
    text = clean_gemma_output(text)
    input_tokens = 0
    output_tokens = 0
    if last is not None:
        input_tokens = int(last.prompt_tokens)
        output_tokens = int(last.generation_tokens)
    if input_tokens == 0:
        try:
            input_tokens = len(tokenizer.encode(prompt))
        except Exception:
            input_tokens = len(prompt) // config.CHARS_PER_TOKEN
        prefix_count = 0
    if prefix_count > 0:
        input_tokens = input_tokens + prefix_count
    mx.clear_cache()
    result = {}
    result["kind"] = "result"
    result["text"] = text
    result["input_tokens"] = input_tokens
    result["output_tokens"] = output_tokens
    if trace_list is not None:
        detail = "The model read " + str(input_tokens) + " tokens and wrote " + str(output_tokens) + " tokens."
        log_event(trace_list, "LLM_GENERATE", detail)
    yield result


# Pass each visible word onward, then hand back the token counts from that call.
def forward_model_call(step, prompt, trace_list, reuse_prefix, keep_full_prompt, stream_words):
    result = None
    for event in run_one_call(step, prompt, trace_list, reuse_prefix, keep_full_prompt):
        if event["kind"] == "piece":
            if stream_words:
                yield event
        else:
            result = event
    if result is None:
        result = {}
        result["kind"] = "result"
        result["text"] = ""
        result["input_tokens"] = 0
        result["output_tokens"] = 0
    return result


# Summarize a long document in small pieces, then stitch those notes into one answer.
def run_map_reduce(step, source_text, user_prompt, trace_list=None):
    pieces = chunk_text(source_text, step["map_chars"])
    if len(pieces) == 0:
        pieces.append("")
    map_step = copy_step(step)
    map_step["max_tokens"] = step["map_max_tokens"]
    reduce_step = copy_step(step)
    reduce_step["max_tokens"] = step["reduce_max_tokens"]
    total_in = 0
    total_out = 0
    if len(pieces) == 1:
        trimmed = trim_for_budget(pieces[0], step["forward_cap"], step["base_model"])
        prompt = build_prompt(step["reduce_instructions"], user_prompt, trimmed)
        result = yield from forward_model_call(reduce_step, prompt, trace_list, False, False, True)
        return result
    notes = []
    for piece in pieces:
        trimmed = trim_for_budget(piece, step["forward_cap"], step["base_model"])
        prompt = build_prompt(step["map_instructions"], user_prompt, trimmed)
        result = yield from forward_model_call(map_step, prompt, trace_list, False, False, False)
        notes.append(result["text"])
        total_in = total_in + result["input_tokens"]
        total_out = total_out + result["output_tokens"]
    combined = ""
    for note in notes:
        if combined == "":
            combined = note
        else:
            combined = combined + "\n\n" + note
    guard = 0
    while guard < 4:
        rough = len(combined) // config.CHARS_PER_TOKEN
        if rough <= step["max_kv_size"]:
            break
        groups = chunk_text(combined, step["map_chars"])
        new_notes = []
        for group in groups:
            prompt = build_prompt(step["reduce_instructions"], user_prompt, group)
            result = yield from forward_model_call(map_step, prompt, trace_list, False, False, False)
            new_notes.append(result["text"])
            total_in = total_in + result["input_tokens"]
            total_out = total_out + result["output_tokens"]
        combined = ""
        for note in new_notes:
            if combined == "":
                combined = note
            else:
                combined = combined + "\n\n" + note
        guard = guard + 1
    final_prompt = build_prompt(step["reduce_instructions"], user_prompt, combined)
    final = yield from forward_model_call(reduce_step, final_prompt, trace_list, False, False, True)
    total_in = total_in + final["input_tokens"]
    total_out = total_out + final["output_tokens"]
    final["input_tokens"] = total_in
    final["output_tokens"] = total_out
    return final


# Find fenced code blocks in an answer. Each block remembers its language label.
def pull_code_blocks(text):
    blocks = []
    marker = "```"
    start = 0
    while True:
        open_at = text.find(marker, start)
        if open_at == -1:
            break
        line_end = text.find("\n", open_at)
        if line_end == -1:
            break
        language = text[open_at + len(marker):line_end].strip().lower()
        close_at = text.find(marker, line_end + 1)
        block = {}
        block["language"] = language
        if close_at == -1:
            block["code"] = text[line_end + 1:]
            block["closed"] = False
            blocks.append(block)
            break
        block["code"] = text[line_end + 1:close_at]
        block["closed"] = True
        blocks.append(block)
        start = close_at + len(marker)
    return blocks


# Decide whether a block should be checked as Python or as SQL.
def block_kind(language, code):
    if language == "python" or language == "py":
        return "python"
    if language == "sql":
        return "sql"
    stripped = code.strip()
    upper = stripped.upper()
    if upper.startswith("SELECT") or upper.startswith("WITH"):
        return "sql"
    if "def " in code or "import " in code or "pd." in code:
        return "python"
    return ""


# Check one SQL statement without running it against any real data.
def sql_problem(code):
    stripped = code.strip()
    if stripped == "":
        return ""
    upper = stripped.upper()
    if upper.startswith("SELECT") == False and upper.startswith("WITH") == False:
        return ""
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("EXPLAIN " + stripped)
    except sqlite3.Error as err:
        message = str(err).lower()
        # A missing table name is not a grammar mistake. The check has no spreadsheet loaded.
        if "no such table" in message or "no such column" in message:
            return ""
        return "This draft has a syntax error: " + str(err)
    finally:
        connection.close()
    return ""


# Check one Python snippet without running it.
def python_problem(code):
    try:
        ast.parse(code)
    except SyntaxError as err:
        return "This draft has a syntax error: " + str(err)
    return ""


# Look through an answer for code mistakes. This never runs the code.
def check_syntax(text):
    blocks = pull_code_blocks(text)
    if len(blocks) == 0:
        kind = block_kind("", text)
        if kind != "":
            block = {}
            block["language"] = ""
            block["code"] = text
            blocks.append(block)
    notes = []
    for block in blocks:
        kind = block_kind(block["language"], block["code"])
        problem = ""
        if "closed" in block and block["closed"] == False:
            problem = "This draft stopped before the code block closed."
        elif kind == "python":
            problem = python_problem(block["code"])
        elif kind == "sql":
            problem = sql_problem(block["code"])
        if problem != "":
            notes.append(problem)
    message = ""
    for note in notes:
        if message == "":
            message = note
        else:
            message = message + "\n" + note
    return message


# Search the public web and place those snippets after any notes already gathered.
def notes_with_web_search(user_prompt, notes):
    snippets = search_web(user_prompt)
    text = ""
    if notes.strip() != "":
        text = "Notes already gathered:\n" + notes.strip() + "\n\n"
    text = text + "Web search results:\n" + snippets
    return text


# Build the web-search answer without telling the model the notes are a missing file.
def web_answer_prompt(step, user_prompt, notes):
    instructions = ""
    if "web_instructions" in step and step["web_instructions"] != "":
        instructions = step["web_instructions"]
    elif "instructions" in step:
        instructions = step["instructions"]
    instructions = (
        instructions
        + " The notes include web search results. Use those results in the answer."
        + " Do not say the answer is missing when a snippet covers it."
    )
    prompt = instructions
    if notes.strip() != "":
        prompt = prompt + "\n\nNotes for this answer:\n" + notes.strip()
    if user_prompt.strip() != "":
        prompt = prompt + "\n\nThe user asked:\n" + user_prompt.strip()
    return prompt


# Search the web, then write the answer from those snippets and the notes.
def answer_from_web(step, user_prompt, notes, trace_list):
    trimmed = notes
    if "forward_cap" in step:
        trimmed = trim_for_budget(notes, step["forward_cap"], step["base_model"])
    combined = notes_with_web_search(user_prompt, trimmed)
    prompt = web_answer_prompt(step, user_prompt, combined)
    result = yield from forward_model_call(step, prompt, trace_list, True, True, True)
    return result


# Run one task and yield each visible word, then the savings and the log for that click.
def run_task(task_id, user_prompt, file_paths, rate_tier, use_web_search=False, model_choice="summaries"):
    global ACTIVE_TRACE, ACTIVE_CHUNKS
    started = time.perf_counter()
    trace = []
    begin_trace()
    ACTIVE_TRACE = trace
    task = config.TASKS[task_id]
    choice = clean_model_choice(model_choice)
    mode = engine_mode(choice)
    detail = LAST_ROUTE_DETAIL
    if detail == "":
        detail = "This runs as " + task["name"] + "."
    detail = detail + " " + mode["label"] + " mode. " + mode["hint"] + "."
    if mode["use_lora"] == True:
        detail = detail + " Custom template folder: " + os.path.relpath(adapter_folder(task_id), app_folder()) + "."
    log_event(trace, "ROUTER", detail)
    steps = steps_for_engine(task, choice, task_id)
    source = read_materials(file_paths, task_id)
    if len(file_paths) == 0:
        file_note = "No files were attached."
    else:
        names = []
        for path in file_paths:
            names.append(os.path.basename(path))
        joined = ""
        for name in names:
            if joined == "":
                joined = name
            else:
                joined = joined + ", " + name
        file_note = "Prepared " + str(len(file_paths)) + " file(s): " + joined + "."
        if source.strip() == "":
            file_note = file_note + " Nothing from the file could be read."
        else:
            file_note = file_note + " Read " + str(len(source)) + " characters into the prompt."
    log_event(trace, "FILE_PREP", file_note)
    ACTIVE_CHUNKS = []
    if source.strip() != "":
        ACTIVE_CHUNKS = chunk_text(source, 1200)
    working = source
    previous_output = ""
    answer = ""
    total_in = 0
    total_out = 0
    warning = ""
    step_index = 0
    step_count = len(steps)
    for step in steps:
        raise_if_stopped()
        is_last = False
        if step_index == step_count - 1:
            is_last = True
        step_index = step_index + 1
        kind = step["kind"]
        if kind == "embed":
            short_document = False
            if len(working) < config.SHORT_PASS_CHARS:
                short_document = True
            if short_document:
                log_event(trace, "EMBEDDING_WORKER", "The document fits in one pass, so the search step was skipped.")
            else:
                unload_model()
                limited, limit_warning = limit_text(working, step["context_budget"])
                if limit_warning != "":
                    warning = limit_warning
                chunks = chunk_text(limited, step["chunk_chars"])
                max_chars = step["context_budget"] * config.CHARS_PER_TOKEN
                attached = files_need_embed_log(file_paths)
                if attached:
                    log_event(trace, "EMBEDDING_WORKER", "Started a helper process to search the attached file.")
                working = embed_and_pick_chunks(user_prompt, chunks, max_chars)
                if attached:
                    log_event(trace, "EMBEDDING_WORKER", "The helper process finished and its memory was released.")
        elif kind == "map_reduce":
            short_document = False
            if len(working) < config.SHORT_PASS_CHARS:
                short_document = True
            if short_document and use_web_search == False:
                instructions = ""
                if "reduce_instructions" in step:
                    instructions = step["reduce_instructions"]
                elif "instructions" in step:
                    instructions = step["instructions"]
                prompt = build_prompt(instructions, user_prompt, working)
                call_step = copy_step(step)
                if task_id == "brief":
                    call_step["max_tokens"] = 400
                log_event(trace, "FILE_PREP", "One pass. The document was read whole, not in pieces.")
                result = yield from forward_model_call(call_step, prompt, trace, True, True, True)
            elif short_document and use_web_search and is_last:
                call_step = copy_step(step)
                call_step["forward_cap"] = config.SHORT_PASS_CHARS // config.CHARS_PER_TOKEN
                result = yield from answer_from_web(call_step, user_prompt, working, trace)
            else:
                result = yield from run_map_reduce(step, working, user_prompt, trace)
            if use_web_search and is_last and short_document == False:
                total_in = total_in + result["input_tokens"]
                total_out = total_out + result["output_tokens"]
                result = yield from answer_from_web(step, user_prompt, result["text"], trace)
            answer = result["text"]
            previous_output = result["text"]
            total_in = total_in + result["input_tokens"]
            total_out = total_out + result["output_tokens"]
        elif kind == "generate":
            if use_web_search and is_last == False and working.strip() == "":
                log_event(trace, "FILE_PREP", "No files were attached, so this step waited for the web search.")
                continue
            material = working
            if step["input_from"] == "previous":
                material = previous_output
            short_text = False
            if len(material) < config.SHORT_PASS_CHARS:
                short_text = True
            use_map = False
            if short_text == False:
                if "map_trigger_tokens" in step:
                    rough = len(material) // config.CHARS_PER_TOKEN
                    if rough > step["map_trigger_tokens"]:
                        use_map = True
            if use_web_search and is_last:
                notes = material
                if use_map:
                    mapped = yield from run_map_reduce(step, material, user_prompt, trace)
                    notes = mapped["text"]
                    total_in = total_in + mapped["input_tokens"]
                    total_out = total_out + mapped["output_tokens"]
                result = yield from answer_from_web(step, user_prompt, notes, trace)
            elif use_map:
                result = yield from run_map_reduce(step, material, user_prompt, trace)
            elif short_text:
                prompt = build_prompt(step["instructions"], user_prompt, material)
                result = yield from forward_model_call(step, prompt, trace, True, True, True)
            else:
                trimmed = trim_for_budget(material, step["forward_cap"], step["base_model"])
                prompt = build_prompt(step["instructions"], user_prompt, trimmed)
                # Reuse the document when the same file is asked again.
                result = yield from forward_model_call(step, prompt, trace, True, False, True)
            answer = result["text"]
            if task_id == "csv":
                answer = answer_with_histogram_table(answer, material)
            previous_output = answer
            total_in = total_in + result["input_tokens"]
            total_out = total_out + result["output_tokens"]
            if "check_syntax" in step:
                if step["check_syntax"] == True:
                    note = check_syntax(answer)
                    if note != "":
                        answer = answer + "\n\n" + note
    cloud = calculate_savings(total_in, total_out, rate_tier)
    saved = net_after_local(cloud)
    elapsed = time.perf_counter() - started
    cloud_text = "$" + format(cloud, ".4f")
    saved_text = "$" + format(saved, ".4f")
    local_text = "$" + format(config.LOCAL_COST, ".4f")
    savings_detail = (
        "Cloud " + cloud_text
        + " minus local " + local_text
        + " leaves " + saved_text
        + " after " + format(elapsed, ".2f") + " seconds."
    )
    log_event(trace, "SAVINGS_CALC", savings_detail)
    row = {}
    row["task_id"] = task_id
    row["task_name"] = task["name"]
    row["input_tokens"] = total_in
    row["output_tokens"] = total_out
    row["rate_tier"] = rate_tier
    row["cloud_cost"] = cloud
    row["local_cost"] = config.LOCAL_COST
    row["savings"] = saved
    save_run(row)
    record = {}
    record["run_id"] = str(uuid.uuid4())
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["task_id"] = task_id
    record["task_title"] = task["name"]
    record["base_model"] = base_model_label(task)
    if engine_mode(choice)["use_lora"] == True:
        record["adapter_used"] = template_label(task_id)
    else:
        record["adapter_used"] = "none"
    record["web_search_enabled"] = use_web_search
    record["rate_tier"] = rate_tier
    record["prefill_tokens"] = total_in
    record["output_tokens"] = total_out
    record["latency_seconds"] = elapsed
    record["estimated_cloud_cost"] = cloud
    record["local_cost"] = config.LOCAL_COST
    record["net_savings"] = saved
    record["events"] = trace
    save_run_log(record)
    outcome = {}
    outcome["text"] = clean_gemma_output(answer)
    outcome["warning"] = warning
    outcome["input_tokens"] = total_in
    outcome["output_tokens"] = total_out
    outcome["cloud_cost"] = cloud
    outcome["local_cost"] = config.LOCAL_COST
    outcome["savings"] = saved
    outcome["task_id"] = task_id
    outcome["task_name"] = task["name"]
    outcome["log_entry"] = record
    outcome["kind"] = "done"
    yield outcome


# Write one finished click into the savings ledger.
def save_run(row):
    ensure_database()
    created = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO runs ("
            "created_at, task_id, task_name, input_tokens, output_tokens, "
            "rate_tier, cloud_cost, local_cost, savings"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                created,
                row["task_id"],
                row["task_name"],
                int(row["input_tokens"]),
                int(row["output_tokens"]),
                row["rate_tier"],
                float(row["cloud_cost"]),
                float(row["local_cost"]),
                float(row["savings"]),
            ),
        )
        connection.commit()
    finally:
        connection.close()


# Read one sum from the ledger. Missing rows count as zero.
def sum_column(cursor, sql, argument):
    cursor.execute(sql, (argument,))
    row = cursor.fetchone()
    if row is None:
        return 0.0
    if row[0] is None:
        return 0.0
    return float(row[0])


# Saved today, the month projection, and how many local tasks ran today.
def savings_summary():
    ensure_database()
    today = date.today()
    today_key = today.strftime("%Y-%m-%d")
    month_key = today.strftime("%Y-%m")
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        saved_today = sum_column(
            cursor,
            "SELECT SUM(savings) FROM runs WHERE created_at LIKE ?",
            today_key + "%",
        )
        month_total = sum_column(
            cursor,
            "SELECT SUM(savings) FROM runs WHERE created_at LIKE ?",
            month_key + "%",
        )
        cursor.execute(
            "SELECT COUNT(*) FROM runs WHERE created_at LIKE ?",
            (today_key + "%",),
        )
        count_row = cursor.fetchone()
        tasks_today = 0
        if count_row is not None:
            if count_row[0] is not None:
                tasks_today = int(count_row[0])
    finally:
        connection.close()
    days_in_month = calendar.monthrange(today.year, today.month)[1]
    projected = 0.0
    if today.day > 0:
        projected = (month_total / today.day) * days_in_month
    summary = {}
    summary["saved_today"] = saved_today
    summary["projected_month"] = projected
    summary["tasks_today"] = tasks_today
    return summary


# The newest ledger rows, for the savings receipt popup.
def recent_receipts():
    ensure_database()
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT created_at, task_name, input_tokens, output_tokens, "
            "cloud_cost, local_cost, savings FROM runs ORDER BY id DESC LIMIT 20"
        )
        fetched = cursor.fetchall()
    finally:
        connection.close()
    rows = []
    for item in fetched:
        row = {}
        row["created_at"] = item[0]
        row["task_name"] = item[1]
        row["input_tokens"] = item[2]
        row["output_tokens"] = item[3]
        row["cloud_cost"] = item[4]
        row["local_cost"] = item[5]
        row["savings"] = item[6]
        rows.append(row)
    return rows


# Turn one stored log row back into the dictionary the page shows.
def record_from_row(item):
    record = {}
    record["run_id"] = item[0]
    record["timestamp"] = item[1]
    record["task_id"] = item[2]
    record["task_title"] = item[3]
    record["base_model"] = item[4]
    record["adapter_used"] = item[5]
    enabled = False
    if item[6] == 1:
        enabled = True
    record["web_search_enabled"] = enabled
    record["prefill_tokens"] = item[7]
    record["output_tokens"] = item[8]
    record["latency_seconds"] = item[9]
    record["estimated_cloud_cost"] = item[10]
    record["local_cost"] = item[11]
    record["net_savings"] = item[12]
    record["rate_tier"] = "everyday"
    if len(item) > 14:
        if item[14] is not None and item[14] != "":
            record["rate_tier"] = item[14]
    events = []
    try:
        parsed = json.loads(item[13])
        if isinstance(parsed, list):
            events = parsed
    except Exception:
        events = []
    record["events"] = events
    return record


# Store the timeline for one finished click.
def save_run_log(record):
    ensure_database()
    enabled = 0
    if record["web_search_enabled"]:
        enabled = 1
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO run_logs ("
            "run_id, created_at, task_id, task_title, base_model, adapter_used, "
            "web_search_enabled, prefill_tokens, output_tokens, latency_seconds, "
            "estimated_cloud_cost, local_cost, net_savings, events_json, rate_tier"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record["run_id"],
                record["timestamp"],
                record["task_id"],
                record["task_title"],
                record["base_model"],
                record["adapter_used"],
                enabled,
                int(record["prefill_tokens"]),
                int(record["output_tokens"]),
                float(record["latency_seconds"]),
                float(record["estimated_cloud_cost"]),
                float(record["local_cost"]),
                float(record["net_savings"]),
                json.dumps(record["events"]),
                record["rate_tier"],
            ),
        )
        connection.commit()
    finally:
        connection.close()


# Newest run logs first.
def list_run_logs():
    ensure_database()
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT run_id, created_at, task_id, task_title, base_model, adapter_used, "
            "web_search_enabled, prefill_tokens, output_tokens, latency_seconds, "
            "estimated_cloud_cost, local_cost, net_savings, events_json, rate_tier "
            "FROM run_logs ORDER BY created_at DESC"
        )
        fetched = cursor.fetchall()
    finally:
        connection.close()
    rows = []
    for item in fetched:
        rows.append(record_from_row(item))
    return rows


# One log, or an empty dictionary when that id is not in the ledger.
def get_run_log(run_id):
    ensure_database()
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT run_id, created_at, task_id, task_title, base_model, adapter_used, "
            "web_search_enabled, prefill_tokens, output_tokens, latency_seconds, "
            "estimated_cloud_cost, local_cost, net_savings, events_json, rate_tier "
            "FROM run_logs WHERE run_id = ?",
            (run_id,),
        )
        item = cursor.fetchone()
    finally:
        connection.close()
    if item is None:
        return {}
    return record_from_row(item)


# Remove one run log.
def delete_run_log(run_id):
    ensure_database()
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM run_logs WHERE run_id = ?", (run_id,))
        connection.commit()
        removed = cursor.rowcount
    finally:
        connection.close()
    return removed


# Remove every run log.
def delete_all_run_logs():
    ensure_database()
    connection = sqlite3.connect(database_path())
    try:
        cursor = connection.cursor()
        cursor.execute("DELETE FROM run_logs")
        connection.commit()
    finally:
        connection.close()


# The logs the download route should send.
def logs_for_download(run_id):
    if run_id is None or run_id == "":
        return list_run_logs()
    one = get_run_log(run_id)
    rows = []
    if len(one) > 0:
        rows.append(one)
    return rows


# Plain JSON text for a log download.
def logs_as_json(rows):
    return json.dumps(rows, indent=2)


# A readable Markdown copy of the same logs.
def logs_as_markdown(rows):
    text = "# Local AirMax run log\n\n"
    if len(rows) == 0:
        text = text + "No runs are stored.\n"
        return text
    for record in rows:
        text = text + "## " + str(record["timestamp"]) + " — " + str(record["task_title"]) + "\n\n"
        text = text + "- Run: " + str(record["run_id"]) + "\n"
        text = text + "- Base model: " + str(record["base_model"]) + "\n"
        text = text + "- Adapter: " + str(record["adapter_used"]) + "\n"
        if record["web_search_enabled"]:
            text = text + "- Web search: on\n"
        else:
            text = text + "- Web search: off\n"
        text = text + "- Rate tier: " + str(record["rate_tier"]) + "\n"
        text = text + "- Prefill tokens: " + str(record["prefill_tokens"]) + "\n"
        text = text + "- Output tokens: " + str(record["output_tokens"]) + "\n"
        text = text + "- Cloud cost: $" + format(float(record["estimated_cloud_cost"]), ".4f") + "\n"
        text = text + "- Local cost: $" + format(float(record["local_cost"]), ".4f") + "\n"
        text = text + "- Net savings: $" + format(float(record["net_savings"]), ".4f") + "\n"
        text = text + "- Latency seconds: " + format(float(record["latency_seconds"]), ".2f") + "\n\n"
        events = record["events"]
        for event in events:
            text = text + str(event["step"]) + ". " + str(event["stage"]) + " — " + str(event["detail"]) + "\n"
        text = text + "\n"
    return text
