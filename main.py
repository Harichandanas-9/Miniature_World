import os
import uuid
import asyncio
import random
import urllib.request
import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Miniature ASMR Bot")

HF_TOKEN   = os.getenv("HF_TOKEN", "")
SSL_VERIFY = os.getenv("SSL_VERIFY", "false").lower() != "false"
DEMO_MODE  = os.getenv("DEMO_MODE", "false").lower() == "true"

def _detect_proxy() -> dict | None:
    sys_proxies = urllib.request.getproxies()
    proxy_url = (
        os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or sys_proxies.get("https")
        or sys_proxies.get("http")
    )
    if proxy_url:
        print(f"[proxy] Using: {proxy_url}")
        return {"https://": proxy_url, "http://": proxy_url}
    return None

_PROXIES = _detect_proxy()

def _make_client(**kwargs):
    return httpx.AsyncClient(verify=SSL_VERIFY, proxies=_PROXIES, **kwargs)

os.makedirs("static/videos", exist_ok=True)

mode_label = "DEMO MODE (no video generation)" if DEMO_MODE else "LIVE MODE (HuggingFace video)"
print(f"[startup] {mode_label}")

ALLOWED_KEYWORDS = {
    "tiny","miniature","mini","small","little","micro","doll","dollhouse",
    "bakery","kitchen","shop","garden","farm","forest","cafe","library",
    "workshop","studio","cottage","bedroom","classroom",
    "knead","bake","cook","brew","plant","water","craft","sew","paint",
    "carve","fold","stamp","pour","slice","chop","stir","mix","roll",
    "dough","bread","cake","cookie","tea","soil","seed","wood","clay",
    "fabric","paper","glass","flower","leaf","stone","sand",
    "asmr","soothing","relaxing","satisfying","cozy","calm","peaceful",
}

BLOCKED_KEYWORDS = {
    "kill","weapon","bomb","gun","blood","violence","hack","exploit",
    "nude","sex","adult","drug","steal","politics","war","attack",
}

def is_allowed(text: str) -> bool:
    words = set(text.lower().replace("-", " ").replace(",", " ").split())
    if words & BLOCKED_KEYWORDS:
        return False
    return bool(words & ALLOWED_KEYWORDS)

ASMR_TEMPLATES = [
    "Extreme close-up macro shot of a 1:12 scale {scene}. Tiny hands {actions}. "
    "Soft ambient light, shallow depth of field, satisfying textures, gentle ASMR sounds.",

    "Miniature world close-up: a cozy tiny {scene}. Small delicate hands carefully {actions}. "
    "Pastel warm lighting, hyper-detailed textures, calming and satisfying.",

    "A beautiful miniature {scene} filmed in macro. Tiny objects, hands {actions}. "
    "Soft bokeh background, warm golden hour light, deeply satisfying ASMR aesthetics.",
]

def build_video_prompt(user_input: str) -> str:
