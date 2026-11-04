#!/usr/bin/env bash
# REVIEW CASES (rev-mlx-lm-2034) built on the design lane's harness of mlx-lm#2032 (same oracle,
# same verdict rules; only the cell list, the server args and a concurrent cell are added).
# Acceptance script for ml-explore/mlx-lm#2032: a prompt longer than the model
# context must get a fast client error that names the limit.
#
# Starts mlx_lm.server (one fresh server per cell) with mlx-community/Qwen3-0.6B-4bit
# from the checkout in $SRC, sends each request class, prints one RESULT row per cell.
# Oracle: the limit is read from the model's config.json and prompt lengths are
# counted with the HF tokenizer, both outside mlx_lm. Every wait is bounded
# (CELL_WAIT per request, BOOT_WAIT per server start).
#
# Env: SRC (checkout under test, default ./src), PY, SERVER_BIN, OUT (log dir),
#      CELL_WAIT (default 45), BOOT_WAIT (default 300), ONLY (regex on cell ids),
#      MODEL_DIR (use a local model dir instead of a download; harness control only).
# Exit: 0 all verdict rows green; 1 a red row; 2 a harness fault.

set -u

SRC="${SRC:-src}"
PY="${PY:-$SRC/.venv/bin/python}"
SERVER_BIN="${SERVER_BIN:-$SRC/.venv/bin/mlx_lm.server}"
OUT="${OUT:-$(mktemp -d)}"
export SRC PY SERVER_BIN OUT
export CELL_WAIT="${CELL_WAIT:-45}"
export BOOT_WAIT="${BOOT_WAIT:-300}"
mkdir -p "$OUT"

HARNESS="$OUT/harness.py"
cat > "$HARNESS" <<'PYEOF'
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

MODEL = "mlx-community/Qwen3-0.6B-4bit"
WAIT = float(os.environ["CELL_WAIT"])
BOOT = float(os.environ["BOOT_WAIT"])
SERVER = os.environ["SERVER_BIN"]
OUT = Path(os.environ["OUT"])
ONLY = re.compile(os.environ.get("ONLY", ".*"))
FAR = 60009  # tokens of the issue's repro prompt ("w " * 60000 in one user message)
CHAT, ALIAS, TEXT = "/v1/chat/completions", "/chat/completions", "/v1/completions"

model_dir = os.environ.get("MODEL_DIR") or snapshot_download(MODEL)
tok = AutoTokenizer.from_pretrained(model_dir)


def declared_limit(directory):
    v = json.loads((Path(directory) / "config.json").read_text())[
        "max_position_embeddings"
    ]
    return v


def chat_len(n):
    text = tok.apply_chat_template(
        [{"role": "user", "content": "w " * n}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return len(tok.encode(text, add_special_tokens=False))


def text_len(n):
    return len(tok.encode("w " * n))


def words_for(count, target):
    n = target
    for _ in range(8):
        c = count(n)
        if c == target:
            return n
        n = max(n + target - c, 0)
    raise SystemExit(f"generator cannot build a prompt of {target} tokens")


VARIANTS = {
    "cfg2048": (2048, None),
    "cfg2048yarn": (2048, {"rope_type": "yarn", "factor": 4.0,
                           "original_max_position_embeddings": 2048}),
    "cfg2048linear": (2048, {"type": "linear", "factor": 2.0}),
    "cfg2048rpdef": (2048, ("rope_parameters", {"rope_type": "default", "rope_theta": 1000000.0})),
    "cfg2048rpyarn": (2048, ("rope_parameters", {"rope_type": "yarn", "factor": 4.0,
                           "original_max_position_embeddings": 2048})),
    "cfgnull": (None, None),
    "cfgstr": ("2048", None),
}


def variant_dir(name):
    limit, rope = VARIANTS[name]
    d = OUT / name
    d.mkdir(exist_ok=True)
    for f in Path(model_dir).iterdir():
        if f.name != "config.json" and not (d / f.name).exists():
            (d / f.name).symlink_to(f.resolve())
    cfg = json.loads((Path(model_dir) / "config.json").read_text())
    cfg["max_position_embeddings"] = limit
    if isinstance(rope, tuple):
        cfg[rope[0]] = rope[1]
    elif rope:
        cfg["rope_scaling"] = rope
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


class Cell:
    def __init__(self, cid, route, stream, target, expect, extra=None, variant=None,
                 curl=False, max_tokens=8, check_usage=False, args=None, kind=None):
        self.cid, self.route, self.stream, self.target = cid, route, stream, target
        self.expect, self.extra, self.variant = expect, extra or {}, variant
        self.curl, self.max_tokens, self.check_usage = curl, max_tokens, check_usage
        self.args, self.kind = args or [], kind


def build_cells():
    cells = []
    L = declared_limit(model_dir)
    seed = {"seed": 7}
    cells.append(Cell("x-edge-alias-ns", ALIAS, False, L, "STARTED"))
    cells.append(Cell("x-over-alias-ns", ALIAS, False, L + 1, "REFUSED"))
    cells.append(Cell("x-over-alias-st", ALIAS, True, L + 1, "REFUSED"))
    cells.append(Cell("x-over-alias-seed-ns", ALIAS, False, L + 1, "REFUSED", seed))
    cells.append(Cell("x-fits-draft-ns", CHAT, False, 12, "OK", args=["--draft-model", MODEL]))
    cells.append(Cell("x-over-draft-ns", CHAT, False, L + 1, "REFUSED", args=["--draft-model", MODEL]))
    cells.append(Cell("x-over-draft-st", CHAT, True, L + 1, "REFUSED", args=["--draft-model", MODEL]))
    cells.append(Cell("x-edge-draft-st", CHAT, True, L, "STARTED", args=["--draft-model", MODEL]))
    cells.append(Cell("x-cfg2048-edge-chat-ns", CHAT, False, 2048, "OK", variant="cfg2048"))
    cells.append(Cell("x-cfg2048-over-seed-st", CHAT, True, 2049, "REFUSED", seed, variant="cfg2048"))
    cells.append(Cell("x-cfg2048-over-alias-ns", ALIAS, False, 2049, "REFUSED", variant="cfg2048"))
    cells.append(Cell("x-cfg2048yarn-over-seed-ns", CHAT, False, 2049, "OK", seed, variant="cfg2048yarn"))
    cells.append(Cell("x-cfg2048yarn-over-text-st", TEXT, True, 2049, "OK", variant="cfg2048yarn"))
    cells.append(Cell("x-cfg2048linear-over-chat-ns", CHAT, False, 2049, "OK", variant="cfg2048linear"))
    cells.append(Cell("x-cfg2048rpyarn-over-chat-ns", CHAT, False, 2049, "OK", variant="cfg2048rpyarn"))
    cells.append(Cell("x-cfg2048rpdef-over-chat-ns", CHAT, False, 2049, "REFUSED", variant="cfg2048rpdef"))
    cells.append(Cell("x-cfg2048rpdef-edge-chat-ns", CHAT, False, 2048, "OK", variant="cfg2048rpdef"))
    cells.append(Cell("x-cfgnull-over-chat-ns", CHAT, False, 3000, "OK", variant="cfgnull"))
    cells.append(Cell("x-cfgstr-over-chat-ns", CHAT, False, 3000, "OK", variant="cfgstr"))
    cells.append(Cell("x-concurrent", CHAT, False, L + 1, "REFUSED", kind="concurrent"))
    return [c for c in cells if ONLY.search(c.cid)]


def body_for(cell, model_name):
    if cell.route == TEXT:
        n = words_for(text_len, cell.target)
        body = {"model": model_name, "prompt": "w " * n}
        count = text_len(n)
    else:
        n = words_for(chat_len, cell.target)
        body = {"model": model_name, "messages": [{"role": "user", "content": "w " * n}]}
        count = chat_len(n)
    body.update(max_tokens=cell.max_tokens, stream=cell.stream, **cell.extra)
    return body, count


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post(port, route, body, wait, first_line_only=False):
    start = time.monotonic()
    deadline = start + wait
    res = dict(status=None, body="", lines=0, keepalive=0, data=0, done=False, err=None)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        conn.request("POST", route, body=json.dumps(body).encode(),
                     headers={"content-type": "application/json"})
        sock = conn.sock
        sock.settimeout(max(0.1, deadline - time.monotonic()))
        resp = conn.getresponse()
        res["status"] = resp.status
        if "text/event-stream" in resp.getheader("Content-Type", ""):
            while time.monotonic() < deadline:
                sock.settimeout(max(0.1, deadline - time.monotonic()))
                raw = resp.fp.readline()
                if raw == b"":
                    break
                line = raw.decode(errors="replace").strip()
                if not line:
                    continue
                res["lines"] += 1
                if line.startswith(": keepalive"):
                    res["keepalive"] += 1
                elif line.startswith("data: [DONE]"):
                    res["done"] = True
                    break
                elif line.startswith("data:"):
                    res["data"] += 1
                if first_line_only:
                    break
        else:
            sock.settimeout(max(0.1, deadline - time.monotonic()))
            res["body"] = resp.read().decode(errors="replace")
    except (TimeoutError, socket.timeout):
        res["err"] = "deadline"
    except Exception as e:
        res["err"] = f"{type(e).__name__}: {e}"
    finally:
        conn.close()
    res["elapsed"] = round(time.monotonic() - start, 1)
    return res


def curl_post(port, route, body, wait):
    body_file = OUT / "curl-body.json"
    body_file.write_text(json.dumps(body))
    start = time.monotonic()
    p = subprocess.run(
        ["curl", "-m", str(int(wait)), "-s", "-o", str(OUT / "curl-out.txt"),
         "-w", "%{http_code}", "-X", "POST", f"http://127.0.0.1:{port}{route}",
         "-H", "content-type: application/json", "-d", f"@{body_file}"],
        capture_output=True, text=True, timeout=wait + 15, stdin=subprocess.DEVNULL,
    )
    code = int(p.stdout.strip() or 0)
    out = (OUT / "curl-out.txt").read_text(errors="replace") if code else ""
    return dict(status=code or None, body=out, lines=0, keepalive=0, data=0,
                done=False, err=None if code else "deadline",
                elapsed=round(time.monotonic() - start, 1))


def start_server(model_arg, cid, extra=()):
    port = free_port()
    log = open(OUT / f"server-{cid}.log", "wb")
    proc = subprocess.Popen(
        [SERVER, "--model", model_arg, "--host", "127.0.0.1", "--port", str(port), *extra],
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    return proc, port, OUT / f"server-{cid}.log"


def stop_server(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def last_progress(log_path):
    last = ""
    for line in log_path.read_text(errors="replace").splitlines():
        if "Prompt processing progress" in line:
            last = line.split("progress:")[-1].strip()
    return last or "-"


def usage_ok(cell, res, count):
    if not cell.check_usage:
        return True
    try:
        return json.loads(res["body"])["usage"]["prompt_tokens"] == count
    except Exception:
        return False


def verdict(cell, res, limit, count):
    s = res["status"]
    if cell.expect == "OPEN":
        return "OPEN"
    if cell.expect == "OK":
        if cell.stream:
            ok = s == 200 and res["done"]
        else:
            try:
                ok = s == 200 and "choices" in json.loads(res["body"])
            except Exception:
                ok = False
        return "GREEN" if ok and usage_ok(cell, res, count) else "RED"
    if cell.expect == "STARTED":
        pending = s is None and res["err"] == "deadline"
        ok = s == 200 if cell.stream else (pending or s == 200)
        return "GREEN" if ok else "RED"
    try:
        msg = str(json.loads(res["body"]).get("error", ""))
    except Exception:
        msg = ""
    ok = s == 400 and str(limit) in msg and "context" in msg.lower()
    return "GREEN" if ok else "RED"


def describe(res):
    if res["status"] is None:
        return f"NO-RESPONSE({res['err']})"
    if res["err"] and res["err"] != "deadline":
        return f"{res['status']} harness-error({res['err']})"
    if res["lines"] or res["data"] or res["keepalive"] or res["done"]:
        return (f"{res['status']} stream lines={res['lines']} keepalive={res['keepalive']}"
                f" data={res['data']} done={res['done']}")
    return str(res["status"])


def run_concurrent(cell, port, model_arg, over_body, limit, count, log_path):
    """A long stream is in flight (batch active); an over-limit request arrives; the stream must finish."""
    import threading
    first = {"model": model_arg, "messages": [{"role": "user", "content": "Write the numbers from 1 to 300 separated by commas."}],
             "max_tokens": 200, "stream": True}
    state = dict(data=0, done=False, status=None, err=None, t_data=None)
    started = threading.Event()

    def stream():
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
        try:
            conn.request("POST", CHAT, body=json.dumps(first).encode(), headers={"content-type": "application/json"})
            resp = conn.getresponse()
            state["status"] = resp.status
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                raw = resp.fp.readline()
                if raw == b"":
                    break
                line = raw.decode(errors="replace").strip()
                if line.startswith("data: [DONE]"):
                    state["done"] = True
                    break
                if line.startswith("data:"):
                    state["data"] += 1
                    if state["data"] == 3:
                        started.set()
        except Exception as e:
            state["err"] = f"{type(e).__name__}: {e}"
        finally:
            started.set()
            conn.close()

    t = threading.Thread(target=stream)
    t.start()
    started.wait(60)
    mid = state["data"]
    over = post(port, cell.route, over_body, WAIT)
    t.join(130)
    v_over = verdict(Cell("o", cell.route, False, 0, "REFUSED"), over, limit, count)
    ok = v_over == "GREEN" and mid >= 3 and state["done"] and state["status"] == 200 and over["elapsed"] < 5
    print(f"RESULT id={cell.cid} route={cell.route} stream=0 path=batch limit={limit} prompt_tokens={count}"
          f" expect=REFUSED-while-stream-in-flight actual={describe(over)} elapsed_s={over['elapsed']}"
          f" inflight_data_before_over={mid} inflight_data_total={state['data']} inflight_done={state['done']}"
          f" inflight_status={state['status']} inflight_err={state['err']}"
          f" verdict={'GREEN' if ok else 'RED'} body={over['body'][:200]}", flush=True)
    return "GREEN" if ok else "RED"


def run_cell(cell):
    model_arg = variant_dir(cell.variant) if cell.variant else MODEL
    limit = declared_limit(model_arg if cell.variant else model_dir)
    body, count = body_for(cell, model_arg)
    proc, port, log_path = start_server(model_arg, cell.cid, cell.args)
    try:
        warm, _ = body_for(Cell("warm", cell.route, False, 12, "OK", cell.extra), model_arg)
        warm["max_tokens"] = 1
        ready = None
        t_end = time.monotonic() + BOOT
        while time.monotonic() < t_end and proc.poll() is None:
            ready = post(port, cell.route, warm, min(30, t_end - time.monotonic()))
            if ready["status"] == 200:
                break
            time.sleep(2)
        if ready is None or ready["status"] != 200:
            print(f"RESULT id={cell.cid} verdict=FAULT actual=server-not-ready "
                  f"warmup={ready and describe(ready)} exit={proc.poll()}", flush=True)
            return "FAULT"
        if cell.kind == "concurrent":
            return run_concurrent(cell, port, model_arg, body, limit, count, log_path)
        if cell.curl:
            res = curl_post(port, cell.route, body, WAIT)
        else:
            res = post(port, cell.route, body, WAIT,
                       first_line_only=cell.expect in ("STARTED", "OPEN"))
        v = verdict(cell, res, limit, count)
        snippet = res["body"].replace("\n", " ")[:240] if res["body"] else "-"
        path = "single" if "seed" in cell.extra else "batch"
        print(
            f"RESULT id={cell.cid} route={cell.route} stream={int(cell.stream)} path={path}"
            f" limit={limit} prompt_tokens={count} max_tokens={cell.max_tokens}"
            f" expect={cell.expect} actual={describe(res)} elapsed_s={res['elapsed']}"
            f" verdict={v} server_progress={last_progress(log_path)} body={snippet}",
            flush=True,
        )
        return v
    finally:
        stop_server(proc)


def main():
    cells = build_cells()
    print(f"CELLS n={len(cells)} model={MODEL} declared_limit={declared_limit(model_dir)}"
          f" cell_wait_s={WAIT:g} boot_wait_s={BOOT:g}", flush=True)
    counts = {"GREEN": 0, "RED": 0, "OPEN": 0, "FAULT": 0}
    red = []
    for cell in cells:
        v = run_cell(cell)
        counts[v] += 1
        if v in ("RED", "FAULT"):
            red.append(cell.cid)
    total = sum(counts.values())
    print(f"SUMMARY cells={total} green={counts['GREEN']} red={counts['RED']}"
          f" open={counts['OPEN']} fault={counts['FAULT']}", flush=True)
    print("RED-IDS " + " ".join(red), flush=True)
    return 2 if counts["FAULT"] else (1 if counts["RED"] else 0)


sys.exit(main())
PYEOF

"$PY" "$HARNESS" </dev/null
rc=$?
echo "LEFTOVER-SERVERS $(pgrep -f "$SERVER_BIN" | wc -l | tr -d ' ')"
echo "ACCEPTANCE-DONE rc=$rc"
exit "$rc"
