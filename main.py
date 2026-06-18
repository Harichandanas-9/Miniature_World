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
 
REPLICATE_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
SSL_VERIFY = os.getenv("SSL_VERIFY", "false").lower() != "false"
# Set DEMO_MODE=true in .env to skip video generation locally
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
 
mode_label = "DEMO MODE (no video generation)" if DEMO_MODE else "LIVE MODE (Replicate video)"
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
# Replicate — zeroscope-v2-xl text-to-video
# ─────────────────────────────────────────────
 
REPLICATE_MODEL_VERSION = "9f747673945c62801b13b84701c783929c0ee784e4748ec062204894dda1a351"
 
async def generate_video(prompt: str) -> str:
    headers = {
        "Authorization": f"Token {REPLICATE_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "version": REPLICATE_MODEL_VERSION,
        "input": {
            "prompt": prompt,
            "num_frames": 24,
            "width": 576,
            "height": 320,
            "num_inference_steps": 50,
            "guidance_scale": 17.5,
        }
    }
    async with _make_client(timeout=360) as client:
        # Start prediction
        resp = await client.post(
            "https://api.replicate.com/v1/predictions",
            json=payload,
            headers=headers,
        )
        resp.raise_for_status()
        prediction = resp.json()
        poll_url = prediction["urls"]["get"]
 
        # Poll until done (up to 5 min)
        for _ in range(60):
            await asyncio.sleep(5)
            poll = await client.get(poll_url, headers=headers)
            poll.raise_for_status()
            data = poll.json()
            status = data["status"]
            if status == "succeeded":
                video_url = data["output"][0]
                video_resp = await client.get(video_url)
                video_resp.raise_for_status()
                filename = f"{uuid.uuid4().hex}.mp4"
                path = os.path.join("static", "videos", filename)
                with open(path, "wb") as f:
                    f.write(video_resp.content)
                return f"/static/videos/{filename}"
            elif status == "failed":
                raise RuntimeError(f"Replicate failed: {data.get('error', 'unknown error')}")
 
    raise TimeoutError("Replicate prediction timed out after 5 minutes.")
 
 
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
        # Guardrail (local, no network)
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
 
        # Prompt builder (local, no network)
        video_prompt = build_video_prompt(user_input)
 
        # Demo mode — skip HuggingFace, return prompt preview
        if DEMO_MODE:
            return JSONResponse({
                "type": "demo",
                "prompt": video_prompt,
                "message": (
                    "✅ Guardrails passed! Your ASMR video prompt is ready.\n\n"
                    "🎬 In production this generates a real video via HuggingFace.\n\n"
                    f"📝 Prompt:\n{video_prompt}"
                )
            })
 
        # Live mode — call HuggingFace, fall back to demo if unreachable
        try:
            video_url = await generate_video(video_prompt)
            return JSONResponse({"type": "video", "prompt": video_prompt, "video_url": video_url})
        except Exception as rep_err:
            print(f"[Replicate] failed: {rep_err} — falling back to demo response")
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
