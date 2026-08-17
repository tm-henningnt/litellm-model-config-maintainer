"""
End-to-end tester for a *running* litellm proxy.

Unlike test_models.py (which calls litellm directly, bypassing the proxy
process), this script sends real HTTP requests to the proxy's
/v1/chat/completions endpoint using the proxy's master key. This is the
only way to catch bugs that live in the proxy process itself rather than
in config.yaml -- e.g. the proxy's environment not actually matching the
.env file on disk. See docs/gotchas.md, "The proxy environment can
differ from your .env file".

Usage:
    litellm --config config.yaml --host 127.0.0.1 --port 4000   # in another terminal
    python3 test_proxy.py                 # test every model
    python3 test_proxy.py claude-cline     # filter by substring in model_name

Env vars:
    LITELLM_CONFIG_DIR   default: ~/.config/litellm
    LITELLM_CONFIG_PATH  default: $LITELLM_CONFIG_DIR/config.yaml
    LITELLM_ENV_PATH     default: $LITELLM_CONFIG_DIR/.env
    LITELLM_PROXY_BASE   default: http://127.0.0.1:4000
"""

import json
import os
import re
import sys
import concurrent.futures as cf
import urllib.request
import urllib.error
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("LITELLM_CONFIG_DIR", str(Path.home() / ".config" / "litellm")))
CONFIG_PATH = Path(os.environ.get("LITELLM_CONFIG_PATH", str(CONFIG_DIR / "config.yaml")))
ENV_PATH = Path(os.environ.get("LITELLM_ENV_PATH", str(CONFIG_DIR / ".env")))
PROXY_BASE = os.environ.get("LITELLM_PROXY_BASE", "http://127.0.0.1:4000")


def load_env(path):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = load_env(ENV_PATH)
MASTER_KEY = ENV.get("LITELLM_MASTER_KEY")
REDACT = {v: f"***{k}***" for k, v in ENV.items() if v and len(v) > 6}


def redact(text):
    if not isinstance(text, str):
        text = str(text)
    for secret, label in REDACT.items():
        if secret in text:
            text = text.replace(secret, label)
    text = re.sub(r"(sk-[A-Za-z0-9_\-]{10,})", "***REDACTED***", text)
    text = re.sub(r"(Bearer\s+)[A-Za-z0-9_\-\.]{10,}", r"\1***REDACTED***", text)
    return text


import yaml  # noqa: E402

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

models = cfg["model_list"]
if len(sys.argv) > 1:
    filt = sys.argv[1]
    models = [m for m in models if filt in m["model_name"]]


def test_one(entry):
    name = entry["model_name"]
    body = json.dumps({
        "model": name,
        "messages": [{"role": "user", "content": "Reply with just: OK"}],
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        f"{PROXY_BASE}/v1/chat/completions",
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
            return name, "PASS", "ok"
    except urllib.error.HTTPError as e:
        msg = redact(e.read().decode(errors="replace")).replace("\n", " ")[:220]
        return name, "FAIL", f"HTTP {e.code}: {msg}"
    except Exception as e:
        return name, "FAIL", redact(f"{type(e).__name__}: {e}")[:220]


results = []
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(test_one, entry) for entry in models]
    for fut in cf.as_completed(futs):
        results.append(fut.result())

order = {e["model_name"]: i for i, e in enumerate(models)}
results.sort(key=lambda r: order.get(r[0], 999))

for name, status, msg in results:
    print(f"{status:5s} | {name:45s} | {msg}")

passc = sum(1 for r in results if r[1] == "PASS")
failc = sum(1 for r in results if r[1] == "FAIL")
print(f"\nTOTAL: {len(results)}  PASS={passc}  FAIL={failc}")
