#!/usr/bin/env python3
"""
DeclineCraft annotation tool.

Run it from this folder:

    python annotate.py          (or: python3 annotate.py)

It opens the annotation page in your web browser. Every answer you give is written
straight into your CSV file in this folder. Stop it with Ctrl+C; run it again to
carry on where you left off. When you have finished, send back the CSV file.

Needs Python 3.8 or later and nothing else.
"""

import csv
import glob
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

csv.field_size_limit(2**31 - 1)   # long passages; sys.maxsize overflows on Windows
HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "annotate.html")
PORTS = range(8765, 8800)
CHOICES = {
    "b_response": {"refusal", "premise", "attempt"},
    "b_correct": {"1", "0.5", "0"},
}
LABELS = list(CHOICES) + ["note"]
TASK_OF = {"b_response": "R", "b_correct": "C"}   # part 1: responses (R); part 2: accuracy (C)
KEEP_BACKUPS = 10
BACKUP_EVERY = 25                              # saves between backups
LOCK = threading.Lock()                        # one read-modify-write of the CSV at a time
STATE = {"saves": 0}


def find_csv():
    if len(sys.argv) > 1:
        return os.path.abspath(sys.argv[1])
    found = sorted(glob.glob(os.path.join(HERE, "labels_*.csv")))
    if len(found) != 1:
        sys.exit("Expected exactly one labels_*.csv file next to annotate.py "
                 f"(found {len(found)}). You can also run: python annotate.py <your csv file>")
    return found[0]


def load():
    with open(CSV_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write(fields, rows):
    """Write to a temporary file, then swap it in, so the CSV is never left half written."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CSV_PATH), prefix=".saving-", suffix=".csv")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(5):   # Windows: a virus scanner or indexer can hold the file briefly
            try:
                os.replace(tmp, CSV_PATH)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.2)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def backup():
    """Keep the last few copies of the CSV in backups/, a safety net whatever goes wrong."""
    folder = os.path.join(os.path.dirname(CSV_PATH), "backups")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(CSV_PATH))[0]
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    shutil.copy2(CSV_PATH, os.path.join(folder, f"{stem}_{stamp}.csv"))
    for old in sorted(glob.glob(os.path.join(folder, f"{stem}_*.csv")))[:-KEEP_BACKUPS]:
        os.remove(old)


def public(row):
    item = {"id": row["item_id"], "task": row["task"], "question": row["question"],
            "labels": {k: row[k] for k in LABELS if row.get(k)}}
    if row["task"] == "A":
        item["passages"] = [{"title": row[f"passage{i}_title"], "text": row[f"passage{i}_text"]}
                            for i in range(1, 6) if row.get(f"passage{i}_text")]
    else:
        item["answer"] = row["answer"]
        if row.get("reference"):
            item["reference"] = row["reference"]
    return item


def apply_patch(item_id, changes):
    """Set only the fields the page changed. The CSV is re-read first, so nothing written
    by another browser tab is overwritten with an older copy."""
    fields, rows = load()
    row = next((r for r in rows if r["item_id"] == item_id), None)
    if row is None:
        raise ValueError(f"Unknown item {item_id}")
    for k, v in changes.items():
        if TASK_OF.get(k, row["task"]) != row["task"]:
            raise ValueError(f"{k} does not apply to a task {row['task']} item")
        row[k] = v
    row["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write(fields, rows)
    STATE["saves"] += 1
    if STATE["saves"] % BACKUP_EVERY == 0:
        backup()
    return {k: row[k] for k in LABELS if row[k]}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/", "/index.html", "/annotate.html"):
            with open(PAGE, "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if self.path == "/api/whoami":
            return self._send(200, {"csv": CSV_PATH})
        if self.path == "/api/items":
            with LOCK:
                _, rows = load()
            return self._send(200, {"annotator": rows[0].get("annotator", "") if rows else "",
                                    "file": os.path.basename(CSV_PATH),
                                    "items": [public(r) for r in rows]})
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(404, {"ok": False, "error": "not found"})
        try:
            msg = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            item_id, changes = msg["id"], msg["changes"]
            if not isinstance(item_id, str) or not isinstance(changes, dict) or not changes:
                raise ValueError("expected an item id and at least one change")
            for k, v in changes.items():
                if k not in LABELS or not isinstance(v, str) or (k in CHOICES and v not in CHOICES[k] | {""}):
                    raise ValueError(f"unexpected label {k}={v!r}")
            changes = {k: v.replace("\x00", "") for k, v in changes.items()}
        except (ValueError, KeyError, TypeError) as e:
            return self._send(400, {"ok": False, "error": f"Bad request: {e}"})
        try:
            with LOCK:
                saved = apply_patch(item_id, changes)
        except ValueError as e:
            return self._send(400, {"ok": False, "error": str(e)})
        except OSError as e:
            return self._send(500, {"ok": False, "error":
                                    "Couldn't write the CSV file. If it is open in another program "
                                    f"(such as Excel), close it. The page keeps retrying. ({e.strerror or e})"})
        self._send(200, {"ok": True, "labels": saved})

    def log_message(self, *args):   # keep the terminal quiet
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = False   # on Ctrl+C, let a save that has started finish before exiting
    request_queue_size = 128 # the default of 5 refuses connections when many saves arrive at once


def already_running():
    """If this CSV is already being served (the program was started twice), return its address."""
    for port in PORTS:
        url = f"http://127.0.0.1:{port}/"
        try:
            with urllib.request.urlopen(url + "api/whoami", timeout=0.3) as r:
                if json.load(r).get("csv") == CSV_PATH:
                    return url
        except Exception:
            continue
    return None


def main():
    global CSV_PATH
    CSV_PATH = find_csv()
    if not os.path.exists(PAGE):
        sys.exit("annotate.html must be in the same folder as annotate.py")
    fields, rows = load()
    missing = [k for k in ["item_id", "task", "question", "updated_at"] + LABELS if k not in fields]
    if missing:
        sys.exit(f"{os.path.basename(CSV_PATH)} is missing columns: {', '.join(missing)}")

    running = already_running()
    if running:
        print(f"The annotation program is already running for this file at {running}")
        print("Opening it in your browser. (Only one copy should run at a time.)", flush=True)
        webbrowser.open(running)
        return

    backup()
    server = None
    for port in PORTS:
        try:
            server = Server(("127.0.0.1", port), Handler)
            break
        except OSError:
            continue
    if server is None:
        sys.exit("Could not find a free port between 8765 and 8799.")
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"DeclineCraft annotation - {os.path.basename(CSV_PATH)}, {len(rows)} items")
    print(f"Open {url} in your browser if it doesn't open by itself.")
    print("Your answers are saved to the CSV as you go. Press Ctrl+C here to stop.", flush=True)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
        server.server_close()   # waits for any save in progress
        print("Stopped. Your answers are saved in", os.path.basename(CSV_PATH), flush=True)


if __name__ == "__main__":
    main()
