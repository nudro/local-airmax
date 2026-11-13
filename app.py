# The small web server for Local AirMax.
# It shows the page, saves uploads for one request, and calls the functions in brain.py.

import json
import os
import traceback
import uuid

os.environ["PYDANTIC_AI_NO_BANNER"] = "1"

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import brain
import config

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


# Folder where an upload sits until the request is finished.
def uploads_folder():
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    if os.path.isdir(folder) == False:
        os.makedirs(folder)
    return folder


# Keep spreadsheets, PDFs, and text files. Ignore anything else.
def allowed_upload(name):
    lower = name.lower()
    if lower.endswith(".pdf") or lower.endswith(".csv") or lower.endswith(".xlsx"):
        return True
    if lower.endswith(".txt") or lower.endswith(".md"):
        return True
    return False


# Save each file from the Run form so the task can read it.
def save_uploads():
    saved = []
    incoming = request.files.getlist("files")
    if len(incoming) == 0:
        incoming = []
        for key in request.files:
            for storage in request.files.getlist(key):
                incoming.append(storage)
    for storage in incoming:
        if storage is None:
            continue
        original = storage.filename
        if original is None or original == "":
            continue
        safe = secure_filename(original)
        if safe == "":
            continue
        if allowed_upload(safe) == False:
            continue
        unique = str(uuid.uuid4()) + "_" + safe
        path = os.path.join(uploads_folder(), unique)
        storage.save(path)
        saved.append(path)
    return saved


# Delete the files from this request so they do not pile up on the Mac.
def delete_uploads(paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


# Read the form fields that every request sends.
def form_choices():
    prompt = request.form.get("prompt", "")
    if prompt is None:
        prompt = ""
    auto_route = False
    if request.form.get("auto_route", "true") == "true":
        auto_route = True
    chosen = request.form.get("task_id", "brief")
    if chosen is None or chosen == "":
        chosen = "brief"
    rate_tier = request.form.get("rate_tier", "everyday")
    if rate_tier not in config.PRICES:
        rate_tier = "everyday"
    use_web_search = False
    if request.form.get("use_web_search", "false") == "true":
        use_web_search = True
    model_choice = brain.clean_model_choice(request.form.get("model_choice", "summaries"))
    choices = {}
    choices["prompt"] = prompt
    choices["auto_route"] = auto_route
    choices["task_id"] = chosen
    choices["rate_tier"] = rate_tier
    choices["use_web_search"] = use_web_search
    choices["model_choice"] = model_choice
    return choices


# Attach a staged demo article when the chip sends one of the two known names.
def demo_paths():
    paths = []
    names = request.form.getlist("demo_files")
    for name in names:
        path = brain.demo_file_path(name)
        if path != "":
            if os.path.exists(path):
                paths.append(path)
    return paths


# File names for routing, including a demo chip the person turned on.
def uploaded_names(paths):
    names = []
    for path in paths:
        names.append(os.path.basename(path))
    return names


# The control panel itself. Do not let the browser keep an old copy of the page.
@app.route("/")
def home():
    folder = os.path.dirname(os.path.abspath(__file__))
    response = send_from_directory(folder, "index.html")
    response.headers["Cache-Control"] = "no-store"
    return response


# Whether the local model is in memory, and the switch that loads or releases it.
@app.route("/api/model-power", methods=["GET", "POST"])
def model_power_route():
    try:
        if request.method == "POST":
            body = request.get_json(silent=True)
            if body is None:
                body = {}
            turn_on = False
            if "power" in body:
                if body["power"] == "on":
                    turn_on = True
            model_choice = "summaries"
            if "model_choice" in body:
                model_choice = body["model_choice"]
            task_id = ""
            if "task_id" in body:
                task_id = body["task_id"]
            status = brain.set_model_power(turn_on, model_choice, task_id)
        else:
            status = brain.model_power_status()
        return jsonify(status)
    except Exception as err:
        print("Model power failed: " + str(err))
        body = {}
        body["error"] = "The local model could not be changed."
        body["power"] = "off"
        return jsonify(body), 400


# Today's savings, the month projection, and how many tasks ran today.
@app.route("/api/summary", methods=["GET"])
def summary_route():
    try:
        return jsonify(brain.savings_summary())
    except Exception as err:
        print("Summary failed: " + str(err))
        body = {}
        body["error"] = "The savings totals could not be read."
        return jsonify(body), 400


# Recent rows for the savings receipt.
@app.route("/api/receipt", methods=["GET"])
def receipt_route():
    try:
        body = {}
        body["rows"] = brain.recent_receipts()
        return jsonify(body)
    except Exception as err:
        print("Receipt failed: " + str(err))
        body = {}
        body["error"] = "The receipt could not be read."
        return jsonify(body), 400


# Guess the savings before a run. This does not load a language model.
@app.route("/api/estimate", methods=["POST"])
def estimate_route():
    paths = save_uploads()
    materials = paths + demo_paths()
    try:
        choices = form_choices()
        names = uploaded_names(materials)
        task_id = brain.pick_task(
            choices["prompt"],
            names,
            choices["auto_route"],
            choices["task_id"],
        )
        preview = brain.estimate_run(
            task_id,
            choices["prompt"],
            materials,
            choices["rate_tier"],
            choices["use_web_search"],
            choices["model_choice"],
        )
        return jsonify(preview)
    except Exception as err:
        print("Estimate failed: " + str(err))
        body = {}
        body["error"] = "The savings estimate could not be calculated."
        return jsonify(body), 400
    finally:
        delete_uploads(paths)


# Turn one run event into a line the browser can read as it arrives.
def event_line(payload):
    return "data: " + json.dumps(payload) + "\n\n"


# Tell the running model to quit. This returns at once so the page does not wait.
@app.route("/api/stop", methods=["POST"])
def stop_route():
    body = request.get_json(silent=True)
    if body is None:
        body = {}
    run_id = ""
    if "run_id" in body:
        if body["run_id"] is not None:
            run_id = str(body["run_id"])
    brain.request_stop(run_id)
    reply = {}
    reply["stopped"] = True
    return jsonify(reply)


# Run the chosen task on this Mac and send each word as it is written.
@app.route("/api/run", methods=["POST"])
def run_route():
    paths = save_uploads()
    materials = paths + demo_paths()
    try:
        choices = form_choices()
        names = uploaded_names(materials)
        task_id = brain.pick_task(
            choices["prompt"],
            names,
            choices["auto_route"],
            choices["task_id"],
        )
    except Exception as err:
        print("Run failed: " + str(err), flush=True)
        traceback.print_exc()
        delete_uploads(paths)
        body = {}
        body["error"] = "This task could not be finished on this Mac."
        return jsonify(body), 400
    run_id = request.form.get("client_run_id", "")
    if run_id is None or run_id == "":
        run_id = str(uuid.uuid4())

    # Yields each server-sent line so the page can show words before the run finishes.
    def event_stream():
        brain.MODEL_LOCK.acquire()
        try:
            brain.begin_run(run_id)
            if brain.run_was_stopped():
                payload = {}
                payload["stopped"] = True
                yield event_line(payload)
            else:
                for event in brain.run_task(
                    task_id,
                    choices["prompt"],
                    materials,
                    choices["rate_tier"],
                    choices["use_web_search"],
                    choices["model_choice"],
                ):
                    if event["kind"] == "piece":
                        payload = {}
                        payload["text_piece"] = event["text"]
                        yield event_line(payload)
                    elif event["kind"] == "done":
                        payload = {}
                        payload["done"] = True
                        payload["text"] = event["text"]
                        payload["warning"] = event["warning"]
                        payload["input_tokens"] = event["input_tokens"]
                        payload["output_tokens"] = event["output_tokens"]
                        payload["cloud_cost"] = event["cloud_cost"]
                        payload["local_cost"] = event["local_cost"]
                        payload["savings"] = event["savings"]
                        payload["task_id"] = event["task_id"]
                        payload["task_name"] = event["task_name"]
                        payload["log_entry"] = event["log_entry"]
                        yield event_line(payload)
        except brain.RunStopped:
            payload = {}
            payload["stopped"] = True
            yield event_line(payload)
        except Exception as err:
            print("Run failed: " + str(err), flush=True)
            traceback.print_exc()
            payload = {}
            payload["error"] = "This task could not be finished on this Mac."
            yield event_line(payload)
        finally:
            brain.end_run(run_id)
            brain.MODEL_LOCK.release()
            delete_uploads(paths)

    headers = {}
    headers["Cache-Control"] = "no-cache"
    headers["X-Accel-Buffering"] = "no"
    return Response(event_stream(), mimetype="text/event-stream", headers=headers)


# The two sample articles, listed for the chips under the drop zone.
@app.route("/demo-files", methods=["GET"])
def demo_files_route():
    return jsonify(brain.list_demo_files())


# Newest run logs first.
@app.route("/logs", methods=["GET"])
def logs_route():
    try:
        return jsonify(brain.list_run_logs())
    except Exception as err:
        print("Logs failed: " + str(err))
        body = {}
        body["error"] = "The run logs could not be read."
        return jsonify(body), 400


# Download every log, or one log when run_id is set.
@app.route("/logs/download", methods=["GET"])
def logs_download_route():
    run_id = request.args.get("run_id", "")
    export_format = request.args.get("format", "json")
    rows = brain.logs_for_download(run_id)
    if export_format == "md":
        body = brain.logs_as_markdown(rows)
        mime = "text/markdown"
        filename = "airmax-log.md"
    else:
        body = brain.logs_as_json(rows)
        mime = "application/json"
        filename = "airmax-log.json"
    if run_id != "":
        if export_format == "md":
            filename = "airmax-" + run_id + ".md"
        else:
            filename = "airmax-" + run_id + ".json"
    headers = {}
    headers["Content-Disposition"] = "attachment; filename=\"" + filename + "\""
    return Response(body, mimetype=mime, headers=headers)


# Remove every stored run log.
@app.route("/logs", methods=["DELETE"])
def logs_delete_all_route():
    try:
        brain.delete_all_run_logs()
        body = {}
        body["deleted"] = "all"
        return jsonify(body)
    except Exception as err:
        print("Delete logs failed: " + str(err))
        body = {}
        body["error"] = "The logs could not be deleted."
        return jsonify(body), 400


# Remove one stored run log.
@app.route("/logs/<run_id>", methods=["DELETE"])
def logs_delete_one_route(run_id):
    try:
        brain.delete_run_log(run_id)
        body = {}
        body["deleted"] = run_id
        return jsonify(body)
    except Exception as err:
        print("Delete log failed: " + str(err))
        body = {}
        body["error"] = "That log could not be deleted."
        return jsonify(body), 400


if __name__ == "__main__":
    # One process. Threads let Stop arrive while a run is writing. The model lock keeps a single model.
    try:
        brain.stage_demo_files()
    except Exception as err:
        print("Demo files could not be staged: " + str(err))
    try:
        brain.warm_default_model()
    except Exception as err:
        print("Executive brief model could not be warmed: " + str(err))
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True, use_reloader=False)
