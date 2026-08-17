"""
Direct-SDK model tester for a litellm config.yaml.

Calls litellm.completion()/litellm.responses() for every entry in
model_list (or a filtered subset), resolving os.environ/VAR references
from the config's .env file itself rather than the calling shell's
environment. This is the "does the underlying credential + model id
actually work" check -- it bypasses the litellm proxy process entirely,
so it can't tell you whether the *proxy* is configured correctly (use
test_proxy.py for that), only whether the model/credential/endpoint
combination in config.yaml is valid.

Usage:
    python3 test_models.py                      # test every model
    python3 test_models.py claude-cline          # filter by substring in model_name

Env vars (all optional, sensible defaults assume the standard layout):
    LITELLM_CONFIG_DIR   default: ~/.config/litellm
    LITELLM_CONFIG_PATH  default: $LITELLM_CONFIG_DIR/config.yaml
    LITELLM_ENV_PATH     default: $LITELLM_CONFIG_DIR/.env
    LITELLM_PYTHON       default: whatever python has litellm importable
                         (litellm may be installed as an isolated uv
                         tool install, not importable from system python3;
                         run this script with that interpreter directly,
                         e.g. ~/.local/share/uv/tools/litellm/bin/python)

Findings from the 2026-07-25 config audit that this harness itself had to
route around (see docs/gotchas.md for the full story):
  - Claude 5-family models reject temperature=0 (only temperature=1 is
    supported) -- don't pass temperature unless you mean it.
  - The chatgpt/ (Responses API) provider needs `input` as a list of
    message dicts, not a bare string.
  - A custom "cline" provider must be registered (see cline_provider.py)
    before Cline-routed models will resolve at all.
"""

import concurrent.futures as cf
import os
import re
import sys
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("LITELLM_CONFIG_DIR", str(Path.home() / ".config" / "litellm")))
CONFIG_PATH = Path(os.environ.get("LITELLM_CONFIG_PATH", str(CONFIG_DIR / "config.yaml")))
ENV_PATH = Path(os.environ.get("LITELLM_ENV_PATH", str(CONFIG_DIR / ".env")))


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


def resolve(val):
    if isinstance(val, str) and val.startswith("os.environ/"):
        return ENV.get(val[len("os.environ/"):])
    return val


os.environ.update(ENV)

import yaml  # noqa: E402
import litellm  # noqa: E402

litellm.suppress_debug_info = True

# Register the custom "cline" provider the same way the litellm proxy would
# when it sees litellm_settings.custom_provider_map in config.yaml.
try:
    sys.path.insert(0, str(CONFIG_DIR))
    import cline_provider  # noqa: E402
    from litellm.utils import custom_llm_setup  # noqa: E402

    litellm.custom_provider_map = [{"provider": "cline", "custom_handler": cline_provider.cline_llm}]
    custom_llm_setup()
except ImportError:
    pass  # config doesn't use the custom cline provider (yet) -- fine

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

models = cfg["model_list"]
if len(sys.argv) > 1:
    filt = sys.argv[1]
    models = [m for m in models if filt in m["model_name"]]


def test_one(entry):
    name = entry["model_name"]
    lp = dict(entry.get("litellm_params", {}))
    model = lp.get("model")
    api_key = resolve(lp.get("api_key"))
    api_base = resolve(lp.get("api_base"))
    mode = (entry.get("model_info") or {}).get("mode")

    for raw in (entry.get("litellm_params", {}).get("api_key"), entry.get("litellm_params", {}).get("api_base")):
        if isinstance(raw, str) and raw.startswith("os.environ/"):
            varname = raw[len("os.environ/"):]
            if not ENV.get(varname):
                return name, model, "SKIP", f"missing env var {varname}"

    kwargs = dict(model=model, timeout=25)
    if api_key:
        kwargs["api_key"] = api_key
    if api_base:
        kwargs["api_base"] = api_base

    try:
        if mode == "responses":
            litellm.responses(
                input=[{"role": "user", "content": "Reply with just: OK"}],
                max_output_tokens=16,
                **kwargs,
            )
        else:
            litellm.completion(
                messages=[{"role": "user", "content": "Reply with just: OK"}],
                max_tokens=8,
                **kwargs,
            )
        return name, model, "PASS", "ok"
    except Exception as e:
        msg = redact(f"{type(e).__name__}: {e}").replace("\n", " ")[:220]
        return name, model, "FAIL", msg


results = []
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    futs = [ex.submit(test_one, entry) for entry in models]
    for fut in cf.as_completed(futs):
        results.append(fut.result())

order = {e["model_name"]: i for i, e in enumerate(models)}
results.sort(key=lambda r: order.get(r[0], 999))

for name, model, status, msg in results:
    print(f"{status:5s} | {name:45s} | {redact(model):40s} | {msg}")

passc = sum(1 for r in results if r[2] == "PASS")
failc = sum(1 for r in results if r[2] == "FAIL")
skipc = sum(1 for r in results if r[2] == "SKIP")
print(f"\nTOTAL: {len(results)}  PASS={passc}  FAIL={failc}  SKIP={skipc}")
