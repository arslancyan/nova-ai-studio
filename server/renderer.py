"""NOVA v0.5 render-preview worker.

This is an infrastructure renderer, not the trained NOVA generative model.
It creates a real MP4 so the full product pipeline can be tested end-to-end.
The future native NOVA model can replace this renderer without changing the API.
"""
from __future__ import annotations
import hashlib, math, os, shutil, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(os.getenv("NOVA_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FPS = 12
MAX_RENDER_SECONDS = 15
WIDTHS = {"16:9": (640, 360), "9:16": (360, 640), "1:1": (512, 512)}

def _font(size: int):
    candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"]
    for path in candidates:
        if Path(path).exists(): return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def _wrap(draw, text, font, max_width):
    words = text.split(); lines = []; line = ""
    for word in words:
        candidate = (line + " " + word).strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width: line = candidate
        else:
            if line: lines.append(line)
            line = word
    if line: lines.append(line)
    return lines[:5]

def render_preview(job_id: str, prompt: str, duration_seconds: int, aspect_ratio: str = "16:9", camera: str = "auto") -> str:
    duration = max(1, min(int(duration_seconds), MAX_RENDER_SECONDS))
    width, height = WIDTHS.get(aspect_ratio, WIDTHS["16:9"])
    frame_count = duration * FPS
    seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16)
    accent = (150 + seed % 90, 220, 70)
    workdir = Path(tempfile.mkdtemp(prefix=f"nova-{job_id}-"))
    output = OUTPUT_DIR / f"{job_id}.mp4"
    title_font = _font(max(18, min(width, height) // 15))
    small_font = _font(max(12, min(width, height) // 32))
    try:
        for i in range(frame_count):
            t = i / FPS; phase = t / max(duration, 1)
            img = Image.new("RGB", (width, height), (5, 8, 11)); px = img.load()
            for y in range(height):
                for x in range(width):
                    nx = x / max(width - 1, 1); ny = y / max(height - 1, 1)
                    wave = math.sin(nx * 4.0 + phase * 2.2 + seed * 0.0001)
                    glow = max(0.0, 1.0 - math.hypot(nx - 0.52, ny - 0.43) * 1.7)
                    px[x, y] = (int(5 + 18 * glow + 7 * (wave + 1)), int(8 + 28 * glow), int(11 + 8 * (1 - ny) + 10 * glow))
            draw = ImageDraw.Draw(img, "RGBA")
            motion = {"slow dolly": 0.55, "orbit": 1.25, "handheld": 2.5, "crane up": 0.8}.get(camera.lower(), 0.9)
            cx = width * (0.5 + 0.24 * math.sin(phase * math.tau * motion))
            cy = height * (0.43 + 0.15 * math.cos(phase * math.tau * motion * 0.7))
            radius = max(width, height) * 0.12
            for r in range(int(radius), 5, -5):
                alpha = int(70 * (1 - r / radius)); draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=accent + (max(alpha, 0),))
            margin = max(18, min(width, height) // 12)
            lines = _wrap(draw, prompt.strip(), small_font, width - 2 * margin)
            y = height - margin - len(lines) * (small_font.size + 5)
            draw.rectangle((0, y - 18, width, height), fill=(0, 0, 0, 105))
            for line in lines:
                draw.text((margin, y), line, font=small_font, fill=(245, 247, 251, 235)); y += small_font.size + 5
            draw.text((margin, margin), "NOVA · RENDER PREVIEW", font=title_font, fill=(220, 255, 160, 225))
            img.save(workdir / f"frame_{i:05d}.png", optimize=True)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg: raise RuntimeError("ffmpeg is required for NOVA MP4 rendering")
        command = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(workdir / "frame_%05d.png"), "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        subprocess.run(command, check=True)
        return str(output)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
