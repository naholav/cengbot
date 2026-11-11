#!/usr/bin/env python3
"""Render docs/assets/demo.gif: a Telegram-style chat built from real benchmark answers.

The answers are taken verbatim (only surrounding whitespace is trimmed) from a
benchmark run's ``answers.jsonl``, so the animation shows what the model really
said. It is a rendering, not a screenshot: the footer names the run it came from.

Usage (from the project root, Pillow is in config/requirements.txt):

    .venv/bin/python docs/assets/make_demo_gif.py                    # newest run, default items
    .venv/bin/python docs/assets/make_demo_gif.py --ids auto         # best-scoring TR/EN items with a link
    .venv/bin/python docs/assets/make_demo_gif.py --run-dir benchmark/results/<run> --ids tr-007,en-024

Regenerate it after a model change: run the benchmark for the new model
(benchmark/README.md), then run this script (``--ids auto`` picks new items if
the default ones are no longer good answers).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "benchmark" / "results"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "assets" / "demo.gif"
# Correct answers with a link, two Turkish and two English (run of 2026-11-11).
DEFAULT_IDS = ("tr-016", "en-024", "tr-007", "en-039")
FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu", "/usr/local/share/fonts")

# Geometry at 1x; everything is drawn at SCALE x and downsampled (smooth corners).
SCALE = 2
WIDTH, HEIGHT = 560, 700
HEADER_H, FOOTER_H = 58, 30
PAD, GAP = 12, 10
BUBBLE_MAX = 410          # max bubble width
BODY_SIZE, SMALL_SIZE = 14, 12
LINE_H = 20

COLORS = {
    "header": (62, 106, 142), "header_text": (255, 255, 255), "header_sub": (205, 222, 236),
    "avatar": (233, 160, 64), "chat_bg": (223, 230, 235), "footer": (245, 247, 249),
    "footer_text": (105, 115, 125), "user": (222, 246, 202), "bot": (255, 255, 255),
    "text": (24, 28, 32), "link": (35, 110, 190), "name": (45, 125, 190),
    "quote_bar": (45, 125, 190), "quote_bg": (234, 243, 251), "quote_text": (80, 90, 100),
    "shadow": (196, 205, 212), "button": (104, 126, 143), "button_text": (255, 255, 255),
    "dot": (150, 160, 170),
}
URL_RE = re.compile(r"https?://\S+")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
@dataclass
class Exchange:
    item_id: str
    lang: str
    question: str
    answer: str


def find_run_dir(explicit: Optional[str]) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else (Path.cwd() / path)
    runs = [d for d in RESULTS_DIR.iterdir() if (d / "answers.jsonl").is_file()] if RESULTS_DIR.is_dir() else []
    if not runs:
        sys.exit(f"No benchmark run with answers.jsonl in {RESULTS_DIR} - run benchmark/run_benchmark.py first")

    def created(d: Path) -> str:
        try:
            return json.loads((d / "summary.json").read_text(encoding="utf-8")).get("created_at", "")
        except (OSError, ValueError):
            return ""

    return max(runs, key=lambda d: (created(d), d.name))


def read_jsonl(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_auto(answers: Dict[str, dict], scores: Dict[str, dict], per_lang: int = 2, max_chars: int = 650) -> List[str]:
    """Best-scoring answers that contain a URL, ``per_lang`` per language, alternating TR/EN."""
    ranked: Dict[str, List[str]] = {"tr": [], "en": []}
    for item_id, row in sorted(scores.items(), key=lambda kv: -float(kv[1].get("score") or 0)):
        ans = answers.get(item_id)
        if not ans or ans.get("status") != "ok" or row.get("lang") not in ranked:
            continue
        text = (ans.get("answer") or "").strip()
        if URL_RE.search(text) and len(text) <= max_chars:
            ranked[row["lang"]].append(item_id)
    picked: List[str] = []
    for i in range(per_lang):
        for lang in ("tr", "en"):
            if i < len(ranked[lang]):
                picked.append(ranked[lang][i])
    return picked


def load_exchanges(run_dir: Path, ids_arg: str) -> Tuple[List[Exchange], dict]:
    answers = {row["id"]: row for row in read_jsonl(run_dir / "answers.jsonl")}
    summary: dict = {}
    if (run_dir / "summary.json").is_file():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if ids_arg == "auto":
        scores_path = run_dir / "scores.jsonl"
        if not scores_path.is_file():
            sys.exit(f"--ids auto needs {scores_path}")
        ids = pick_auto(answers, {row["id"]: row for row in read_jsonl(scores_path)})
    else:
        ids = [i.strip() for i in ids_arg.split(",") if i.strip()]
    exchanges = []
    for item_id in ids:
        row = answers.get(item_id)
        if row is None:
            sys.exit(f"{item_id} is not in {run_dir / 'answers.jsonl'}")
        exchanges.append(Exchange(item_id, row.get("lang", "en"), row["question"].strip(), row["answer"].strip()))
    if not exchanges:
        sys.exit("No items selected")
    return exchanges, summary


# ---------------------------------------------------------------------------
# drawing helpers (all sizes in 1x units, multiplied by SCALE when drawn)
# ---------------------------------------------------------------------------
def find_font(name: str, explicit_dir: Optional[str]) -> str:
    for directory in ((explicit_dir,) if explicit_dir else ()) + FONT_DIRS:
        path = Path(directory) / name
        if path.is_file():
            return str(path)
    sys.exit(f"Font {name} not found (looked in {', '.join(FONT_DIRS)}); install fonts-dejavu-core or pass --font-dir")


class Fonts:
    def __init__(self, font_dir: Optional[str]):
        regular, bold = find_font("DejaVuSans.ttf", font_dir), find_font("DejaVuSans-Bold.ttf", font_dir)
        self.body = ImageFont.truetype(regular, BODY_SIZE * SCALE)
        self.bold = ImageFont.truetype(bold, BODY_SIZE * SCALE)
        self.small = ImageFont.truetype(regular, SMALL_SIZE * SCALE)
        self.small_bold = ImageFont.truetype(bold, SMALL_SIZE * SCALE)
        self.title = ImageFont.truetype(bold, 16 * SCALE)
        self.footer = ImageFont.truetype(regular, 11 * SCALE)


def s(value: float) -> int:
    return int(round(value * SCALE))


def downsample(img: Image.Image) -> Image.Image:
    return img.resize((max(1, img.width // SCALE), max(1, img.height // SCALE)), Image.LANCZOS)


def split_link(word: str) -> List[Tuple[str, bool]]:
    """A word as (text, is_link) parts: trailing punctuation after a URL is not part of the link."""
    m = URL_RE.match(word)
    if not m:
        return [(word, False)]
    url = m.group(0).rstrip(".,;:!?)'\"")
    return [(url, True)] + ([(word[len(url):], False)] if len(url) < len(word) else [])


def wrap(text: str, font: ImageFont.FreeTypeFont, width: int) -> List[List[Tuple[str, bool, bool]]]:
    """Word-wrap ``text`` into lines of (text, is_link, space_before) runs; over-long words are split."""
    lines: List[List[Tuple[str, bool, bool]]] = []
    space = font.getlength(" ")
    for paragraph in text.split("\n"):
        line: List[Tuple[str, bool, bool]] = []
        line_w = 0.0
        for word in paragraph.split():
            pieces: List[Tuple[str, bool, bool]] = []
            for k, (part, is_link) in enumerate(split_link(word)):
                chunks, cur = [], ""
                for ch in part:  # split anything wider than a line (long URLs) by characters
                    if cur and font.getlength(cur + ch) > width:
                        chunks.append(cur)
                        cur = ""
                    cur += ch
                chunks.append(cur)
                pieces += [(c, is_link, k == 0 and n == 0) for n, c in enumerate(chunks)]
            for piece, is_link, spaced in pieces:
                w = font.getlength(piece)
                gap = space if (line and spaced) else 0.0
                if line and line_w + gap + w > width:
                    lines.append(line)
                    line, line_w, gap = [], 0.0, 0.0
                line_w += gap + w
                line.append((piece, is_link, bool(gap)))
        lines.append(line)
    return lines


def line_width(line, font) -> float:
    return sum(font.getlength(w) + (font.getlength(" ") if sp else 0) for w, _, sp in line)


def draw_runs(draw: ImageDraw.ImageDraw, x: int, y: int, lines, font, color, link_color) -> None:
    space = font.getlength(" ")
    for i, line in enumerate(lines):
        cx = x
        for word, is_link, spaced in line:
            if spaced:
                cx += space
            draw.text((cx, y + s(i * LINE_H)), word, font=font, fill=link_color if is_link else color)
            if is_link:
                w = font.getlength(word)
                uy = y + s(i * LINE_H) + s(BODY_SIZE + 2)
                draw.line((cx, uy, cx + w, uy), fill=link_color, width=max(1, SCALE // 2))
            cx += font.getlength(word)


def ellipsize(text: str, font, width: int) -> str:
    if font.getlength(text) <= width:
        return text
    while text and font.getlength(text + "…") > width:
        text = text[:-1]
    return text.rstrip() + "…"


def thumb_icon(size: int, up: bool, color) -> Image.Image:
    """A simple thumbs-up/down glyph (the vote buttons use emoji, which DejaVu lacks)."""
    k = size / 16.0
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0.5 * k, 7 * k, 3.5 * k, 15 * k), radius=1 * k, fill=color)       # cuff
    d.rounded_rectangle((4.5 * k, 6.5 * k, 15 * k, 15 * k), radius=2 * k, fill=color)       # fist
    d.polygon([(5 * k, 7.5 * k), (8 * k, 1.5 * k), (10.5 * k, 1 * k), (10.5 * k, 7.5 * k)], fill=color)  # thumb
    d.ellipse((7.5 * k, 0.3 * k, 11 * k, 3.6 * k), fill=color)
    return img if up else img.transpose(Image.FLIP_TOP_BOTTOM)


# ---------------------------------------------------------------------------
# message bubbles (pre-rendered RGBA tiles at 1x)
# ---------------------------------------------------------------------------
def render_user(text: str, fonts: Fonts) -> Image.Image:
    inner = BUBBLE_MAX - 2 * PAD
    lines = wrap(text, fonts.body, s(inner))
    text_w = max(line_width(ln, fonts.body) for ln in lines) / SCALE
    w = int(min(BUBBLE_MAX, text_w + 2 * PAD + 2))
    h = len(lines) * LINE_H + 2 * 8 + 2
    img = Image.new("RGBA", (s(w), s(h + 2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, s(1.5), s(w) - 1, s(h + 1.5)), radius=s(14), fill=COLORS["shadow"])
    d.rounded_rectangle((0, 0, s(w) - 1, s(h)), radius=s(14), fill=COLORS["user"])
    draw_runs(d, s(PAD), s(8), lines, fonts.body, COLORS["text"], COLORS["link"])
    return downsample(img)


def render_bot(question: str, answer: str, lang: str, fonts: Fonts) -> Image.Image:
    w = BUBBLE_MAX
    inner = w - 2 * PAD
    lines = wrap(answer, fonts.body, s(inner))
    name_h, quote_h, button_h = 20, 40, 32
    body_h = len(lines) * LINE_H
    bubble_h = 8 + name_h + quote_h + 6 + body_h + 8
    total_h = bubble_h + 4 + button_h
    img = Image.new("RGBA", (s(w), s(total_h + 2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, s(1.5), s(w) - 1, s(bubble_h + 1.5)), radius=s(14), fill=COLORS["shadow"])
    d.rounded_rectangle((0, 0, s(w) - 1, s(bubble_h)), radius=s(14), fill=COLORS["bot"])
    d.text((s(PAD), s(7)), "CengBot", font=fonts.bold, fill=COLORS["name"])
    # reply quote of the student's question (the bot replies to the message)
    qy = 8 + name_h
    d.rounded_rectangle((s(PAD), s(qy), s(w - PAD), s(qy + quote_h - 4)), radius=s(5), fill=COLORS["quote_bg"])
    d.rectangle((s(PAD), s(qy), s(PAD + 3), s(qy + quote_h - 4)), fill=COLORS["quote_bar"])
    d.text((s(PAD + 10), s(qy + 2)), "Öğrenci" if lang == "tr" else "Student", font=fonts.small_bold,
           fill=COLORS["quote_bar"])
    d.text((s(PAD + 10), s(qy + 18)), ellipsize(question, fonts.small, s(inner - 16)), font=fonts.small,
           fill=COLORS["quote_text"])
    draw_runs(d, s(PAD), s(qy + quote_h + 4), lines, fonts.body, COLORS["text"], COLORS["link"])
    # inline vote buttons, like the production bot's 👍 0 / 👎 0 keyboard
    by = bubble_h + 4
    half = (w - 4) / 2
    for i, up in enumerate((True, False)):
        x0 = i * (half + 4)
        d.rounded_rectangle((s(x0), s(by), s(x0 + half), s(by + button_h)), radius=s(9), fill=COLORS["button"])
        icon = thumb_icon(s(15), up, COLORS["button_text"])
        label_w = fonts.small_bold.getlength("0")
        cx = s(x0 + half / 2) - (icon.width + s(6) + label_w) / 2
        img.alpha_composite(icon, (int(cx), s(by + (button_h - 15) / 2)))
        d.text((cx + icon.width + s(6), s(by + 8)), "0", font=fonts.small_bold, fill=COLORS["button_text"])
    return downsample(img)


def render_typing(phase: int) -> Image.Image:
    w, h = 64, 36
    img = Image.new("RGBA", (s(w), s(h + 2)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, s(1.5), s(w) - 1, s(h + 1.5)), radius=s(14), fill=COLORS["shadow"])
    d.rounded_rectangle((0, 0, s(w) - 1, s(h)), radius=s(14), fill=COLORS["bot"])
    for i in range(3):
        active = i == phase % 3
        r = 4.2 if active else 3.2
        cx, cy = 18 + i * 14, h / 2 - (2 if active else 0)
        color = (95, 110, 125) if active else COLORS["dot"]
        d.ellipse((s(cx - r), s(cy - r), s(cx + r), s(cy + r)), fill=color)
    return downsample(img)


def render_chrome(fonts: Fonts, footer_text: str) -> Dict[str, Image.Image]:
    """Header (normal and "typing…" variants) and footer strips."""
    out = {}
    for key, subtitle in (("header", "bot"), ("header_typing", "typing…")):
        img = Image.new("RGB", (s(WIDTH), s(HEADER_H)), COLORS["header"])
        d = ImageDraw.Draw(img)
        d.ellipse((s(14), s(9), s(54), s(49)), fill=COLORS["avatar"])
        cb_w = fonts.title.getlength("CB")
        d.text((s(34) - cb_w / 2, s(19)), "CB", font=fonts.title, fill=(255, 255, 255))
        d.text((s(66), s(10)), "CengBot", font=fonts.title, fill=COLORS["header_text"])
        d.text((s(66), s(32)), subtitle, font=fonts.small, fill=COLORS["header_sub"])
        out[key] = downsample(img)
    img = Image.new("RGB", (s(WIDTH), s(FOOTER_H)), COLORS["footer"])
    d = ImageDraw.Draw(img)
    text = ellipsize(footer_text, fonts.footer, s(WIDTH - 24))
    d.text(((s(WIDTH) - fonts.footer.getlength(text)) / 2, s(8)), text, font=fonts.footer, fill=COLORS["footer_text"])
    out["footer"] = downsample(img)
    return out


# ---------------------------------------------------------------------------
# animation
# ---------------------------------------------------------------------------
@dataclass
class Placed:
    tile: Image.Image
    right: bool


def compose(chrome, placed: Sequence[Placed], typing: Optional[Image.Image], slide: float = 1.0,
            scroll_from: Optional[float] = None) -> Image.Image:
    """One frame. The chat is bottom-aligned like a messenger; the newest element
    slides in (``slide`` 0..1), and ``scroll_from`` animates the scroll position."""
    chat_h = HEIGHT - HEADER_H - FOOTER_H
    items = list(placed) + ([Placed(typing, False)] if typing is not None else [])
    heights = [p.tile.height for p in items]
    content_h = sum(heights) + GAP * (len(items) + 1)
    bottom_target = max(content_h, chat_h)
    top = chat_h - bottom_target  # y of content start
    if scroll_from is not None:
        top = scroll_from + (top - scroll_from) * slide
    chat = Image.new("RGBA", (WIDTH, chat_h), COLORS["chat_bg"] + (255,))
    y = top + GAP
    for idx, p in enumerate(items):
        x = WIDTH - PAD - p.tile.width if p.right else PAD
        tile = p.tile
        newest = idx == len(items) - 1
        if newest and slide < 1.0:
            alpha = tile.getchannel("A").point(lambda a, f=slide: int(a * f))
            tile = tile.copy()
            tile.putalpha(alpha)
        if y + tile.height > 0 and y < chat_h:
            chat.alpha_composite(tile, (int(x), int(round(y))))
        y += tile.height + GAP
    frame = Image.new("RGB", (WIDTH, HEIGHT), COLORS["chat_bg"])
    frame.paste(chrome["header_typing" if typing is not None else "header"], (0, 0))
    frame.paste(chat.convert("RGB"), (0, HEADER_H))
    frame.paste(chrome["footer"], (0, HEIGHT - FOOTER_H))
    return frame


def content_top(placed: Sequence[Placed], typing: Optional[Image.Image]) -> float:
    chat_h = HEIGHT - HEADER_H - FOOTER_H
    n = len(placed) + (1 if typing is not None else 0)
    content_h = sum(p.tile.height for p in placed) + (typing.height if typing is not None else 0) + GAP * (n + 1)
    return chat_h - max(content_h, chat_h)


def build_frames(exchanges: Sequence[Exchange], fonts: Fonts, chrome) -> Tuple[List[Image.Image], List[int]]:
    frames: List[Image.Image] = []
    durations: List[int] = []

    def add(img: Image.Image, ms: int) -> None:
        frames.append(img)
        durations.append(ms)

    typing_tiles = [render_typing(i) for i in range(3)]
    placed: List[Placed] = []
    add(compose(chrome, placed, None), 500)
    for ex in exchanges:
        user = Placed(render_user(ex.question, fonts), True)
        bot = Placed(render_bot(ex.question, ex.answer, ex.lang, fonts), False)
        # the student's question slides in
        start = content_top(placed, None)
        placed.append(user)
        for f in (0.35, 0.7):
            add(compose(chrome, placed, None, slide=f, scroll_from=start), 40)
        add(compose(chrome, placed, None), 350)
        # typing indicator
        start = content_top(placed, None)
        for i in range(6):
            add(compose(chrome, placed, typing_tiles[i % 3], scroll_from=start if i == 0 else None,
                        slide=0.6 if i == 0 else 1.0), 120)
        # the answer replaces the indicator and the chat scrolls to it
        start = content_top(placed, typing_tiles[0])
        placed.append(bot)
        for f in (0.25, 0.5, 0.75):
            add(compose(chrome, placed, None, slide=f, scroll_from=start), 45)
        read_ms = int(min(2300, max(1500, 1000 + 2.2 * len(ex.answer))))
        add(compose(chrome, placed, None), read_ms)
    # fade back to the empty chat so the loop restarts smoothly
    last, first = frames[-1], frames[0]
    for f in (0.33, 0.66):
        add(Image.blend(last, first, f), 70)
    return frames, durations


def save_gif(frames: List[Image.Image], durations: List[int], out: Path, colors: int) -> None:
    # One shared palette for every frame: no colour flicker, and unchanged areas
    # stay byte-identical, so the encoder only stores the changed rectangles.
    sample = Image.new("RGB", (WIDTH, HEIGHT * 3))
    for i, idx in enumerate((len(frames) // 3, 2 * len(frames) // 3, len(frames) - 3)):
        sample.paste(frames[idx], (0, i * HEIGHT))
    palette = sample.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    paletted = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    paletted[0].save(out, save_all=True, append_images=paletted[1:], duration=durations, loop=0,
                     optimize=False, disposal=1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--run-dir", help="benchmark run directory (default: the newest one in benchmark/results/)")
    p.add_argument("--ids", default=",".join(DEFAULT_IDS),
                   help="comma-separated item ids, or 'auto' (default: %(default)s)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output GIF (default: docs/assets/demo.gif)")
    p.add_argument("--font-dir", help="directory with DejaVuSans.ttf and DejaVuSans-Bold.ttf")
    p.add_argument("--colors", type=int, default=96, help="palette size (default: %(default)s)")
    args = p.parse_args(argv)

    run_dir = find_run_dir(args.run_dir)
    exchanges, summary = load_exchanges(run_dir, args.ids)
    version = (summary.get("model") or {}).get("version") or run_dir.name.rsplit("_", 1)[0]
    date = (summary.get("created_at") or "")[:10]
    footer = f"Rendered from real answers of {version}" + (f" (benchmark run, {date})" if date else "")
    fonts = Fonts(args.font_dir)
    frames, durations = build_frames(exchanges, fonts, render_chrome(fonts, footer))
    out = Path(args.out)
    out = out if out.is_absolute() else Path.cwd() / out
    save_gif(frames, durations, out, args.colors)
    print(f"{out}: {len(frames)} frames, {sum(durations) / 1000:.1f} s loop, "
          f"{out.stat().st_size / 1024:.0f} KiB; items: {', '.join(e.item_id for e in exchanges)} from {run_dir.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
