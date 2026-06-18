import os
import io
import uuid
import asyncio
import random
import urllib.parse
import urllib.request
import httpx
import numpy as np
from PIL import Image
import imageio
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Miniature ASMR Bot")

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

mode_label = "DEMO MODE (no video generation)" if DEMO_MODE else "LIVE MODE (Pollinations.ai → video)"
print(f"[startup] {mode_label}")


# ─────────────────────────────────────────────
# Local guardrails — no external API needed
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Local prompt builder — template-based
# ─────────────────────────────────────────────

ASMR_TEMPLATES = [
    "Extreme close-up macro shot of a 1:12 scale {scene}. Tiny hands {actions}. "
    "Soft ambient light, shallow depth of field, satisfying textures, gentle ASMR sounds.",

    "Miniature world close-up: a cozy tiny {scene}. Small delicate hands carefully {actions}. "
    "Pastel warm lighting, hyper-detailed textures, calming and satisfying.",

    "A beautiful miniature {scene} filmed in macro. Tiny objects, hands {actions}. "
    "Soft bokeh background, warm golden hour light, deeply satisfying ASMR aesthetics.",
]

def build_video_prompt(user_input: str) -> str:
    parts = user_input.replace("–", "-").replace("—", "-").split("-", 1)
    scene   = parts[0].strip()
    actions = parts[1].strip() if len(parts) > 1 else "work with tiny tools and objects"
    return random.choice(ASMR_TEMPLATES).format(scene=scene, actions=actions)


# ─────────────────────────────────────────────
# Pollinations.ai — free image generation
# Frames are stitched into an MP4 via imageio
# ─────────────────────────────────────────────

FRAME_COUNT  = 6
FRAME_FPS    = 2
FRAME_WIDTH  = 512
FRAME_HEIGHT = 320

async def _fetch_frame(client: httpx.AsyncClient, prompt: str, seed: int) -> bytes:
    encoded = urllib.parse.quote(prompt)
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={FRAME_WIDTH}&height={FRAME_HEIGHT}&seed={seed}&nologo=true"
    )
    resp = await client.get(url)
    resp.raise_for_status()
    return resp.content

async def generate_video(prompt: str) -> str:
    seeds = [random.randint(1, 99999) + i * 1000 for i in range(FRAME_COUNT)]

    async with _make_client(timeout=120) as client:
        # Fetch all frames concurrently
        frame_bytes_list = await asyncio.gather(
            *[_fetch_frame(client, prompt, seed) for seed in seeds]
        )

    frames = []
    for fb in frame_bytes_list:
        img = Image.open(io.BytesIO(fb)).convert("RGB").resize((FRAME_WIDTH, FRAME_HEIGHT))
        frames.append(np.array(img))

    filename = f"{uuid.uuid4().hex}.mp4"
    path = os.path.join("static", "videos", filename)
    imageio.mimsave(path, frames, fps=FRAME_FPS, codec="libx264", quality=8)
    return f"/static/videos/{filename}"


# ─────────────────────────────────────────────
# API routes
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    user_input = req.message.strip()
    if not user_input:
        return JSONResponse({"type": "error", "error": "Empty input."}, status_code=400)

    try:
        if not is_allowed(user_input):
            return JSONResponse({
                "type": "refusal",
                "message": (
                    "✋ I only create miniature ASMR videos! "
                    "Try something like: 'tiny bakery - knead dough, shape bread, bake and serve' "
                    "or 'miniature garden - plant seeds, water soil, watch sprout grow'. "
                    "Keep it small, cozy, and satisfying 🎋"
                )
            })

        video_prompt = build_video_prompt(user_input)

        if DEMO_MODE:
            return JSONResponse({
                "type": "demo",
                "prompt": video_prompt,
                "message": (
                    "✅ Guardrails passed! Your ASMR video prompt is ready.\n\n"
                    "🎬 In production this generates a real video via Pollinations.ai.\n\n"
                    f"📝 Prompt:\n{video_prompt}"
                )
            })

        try:
            video_url = await generate_video(video_prompt)
            return JSONResponse({"type": "video", "prompt": video_prompt, "video_url": video_url})
        except Exception as err:
            print(f"[video] failed: {err} — falling back to demo response")
            return JSONResponse({
                "type": "demo",
                "prompt": video_prompt,
                "message": (
                    "✅ Guardrails passed! Your ASMR video prompt is ready.\n\n"
                    "🎬 Video generation is temporarily unavailable.\n\n"
                    f"📝 Prompt that would be used:\n{video_prompt}"
                )
            })

    except Exception as e:
        return JSONResponse({"type": "error", "error": str(e)}, status_code=500)


# ─────────────────────────────────────────────
# Serve the chat UI
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
@app.head("/")
async def root():
    with open("static/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


app.mount("/static", StaticFiles(directory="static"), name="static")
