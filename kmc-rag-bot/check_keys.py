"""Tests each AI provider exactly the way the chatbot calls it, and explains any problem in plain words.
Run in the VS Code terminal (with (.venv) showing):   python check_keys.py"""
import os, sys, tomllib
from pathlib import Path

import openai

secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
if not secrets_path.exists():
    sys.exit("No .streamlit/secrets.toml found. Create it first.")
try:
    secrets = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
except Exception as e:
    sys.exit(f"secrets.toml has a typing mistake: {e}\nEach line must look like:  GROQ_API_KEY = \"gsk_...\"  (with quotes)")
for k, v in secrets.items():  # make the engine read the same values
    os.environ[k] = str(v)

from engine import LLM, PROVIDERS  # noqa: E402

print(f"Python {sys.version.split()[0]} · openai library {openai.__version__}\n")
HINTS = {
    "AuthenticationError": "the key is wrong or expired. Copy it again from the provider's website.",
    "PermissionDeniedError": "the key is valid but not allowed to use this model.",
    "NotFoundError": "this model name isn't available to your key. Pick one from the list below.",
    "RateLimitError": "free limit reached, or (OpenAI) no credit on the account. The bot will use the next provider.",
    "APIConnectionError": "your computer can't reach the provider (network or firewall). Try a mobile hotspot.",
    "APITimeoutError": "the provider didn't reply in time. Try again, or set a faster model below.",
    "BadRequestError": "the provider rejected the request (send this to Claude).",
}
CHAT_HINT = {"groq": ("gpt", "qwen", "llama", "kimi", "compound"), "gemini": ("flash", "pro"), "openai": ("gpt", "o4", "o3")}
SKIP = ("whisper", "guard", "orpheus", "tts", "embedding", "image", "audio", "realtime", "transcribe", "search", "moderation")

working = []
for name, cfg in PROVIDERS.items():
    llm = LLM(only=name)
    if not llm.providers:
        print(f"=== {name.upper()}: SKIPPED (no {cfg['key']} in secrets.toml)\n")
        continue
    p = llm.providers[0]
    print(f"=== {name.upper()}")
    ok_all = True
    for label, fast in (("answer model", False), ("intent model", True)):
        model = p["fast"] if fast else p["model"]
        try:
            out = llm.json('Return JSON exactly like {"ok": true}.', "Test message. Reply in JSON.", fast=fast)
            print(f"   {label:13s} {model:28s} WORKING  -> {out}")
        except Exception:
            ok_all = False
            f = llm.failures[-1] if llm.failures else "unknown error"
            kind = f.split("-> ")[-1].split(":")[0]
            print(f"   {label:13s} {model:28s} FAILED   -> {kind}")
            print(f"      meaning: {HINTS.get(kind, 'see details')}")
            print(f"      details: {f.split('-> ', 1)[-1][:220]}")
    if ok_all:
        working.append(name)
    else:
        try:
            ids = sorted(m.id.replace("models/", "") for m in p["client"].models.list())
            ids = [i for i in ids if any(h in i for h in CHAT_HINT[name]) and not any(s in i for s in SKIP)]
            print(f"   Models your key can use ({len(ids)}): " + ", ".join(ids[:25]))
            print(f"   To use one, add to secrets.toml:  {name.upper()}_MODEL = \"<model>\"   and   {name.upper()}_FAST_MODEL = \"<model>\"")
        except Exception as e:
            print(f"   (could not list models: {type(e).__name__})")
    print()

print("SUMMARY:", ("working providers: " + ", ".join(working)) if working else "no provider is working yet.")
