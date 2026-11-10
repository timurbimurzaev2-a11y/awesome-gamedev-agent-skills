#!/usr/bin/env python3
"""«Мейірім шуағы» — сөзсіз, тек мимикамен баяндалатын қысқа мультфильм.

Бәрі процедуралық түрде салынады (PIL), музыка numpy-мен синтезделеді,
ffmpeg арқылы MP4 болып жиналады. Дауыс та, диалог та жоқ.

Usage:  python3 render.py [output.mp4]
"""
import math
import os
import random
import subprocess
import sys
import wave
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1280, 720
SS = 2                      # supersampling factor
FPS = 24
DUR = 82.0
GROUND = 640
OL = (70, 45, 45)           # outline colour
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


# ----------------------------------------------------------------- helpers
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def ease(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def lerp(a, b, u):
    if isinstance(a, dict):
        return {k: lerp(a[k], b[k], u) for k in a}
    if isinstance(a, (tuple, list)):
        return tuple(lerp(x, y, u) for x, y in zip(a, b))
    return a + (b - a) * u


def track(t, keys):
    """Piecewise eased interpolation over [(time, value), ...]."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return lerp(v0, v1, ease((t - t0) / (t1 - t0)) if t1 > t0 else 1)
    return keys[-1][1]


def mix(c1, c2, u):
    return tuple(int(round(a + (b - a) * u)) for a, b in zip(c1, c2))


# expression presets ---------------------------------------------------------
def E(**kw):
    base = dict(smile=0.35, brow=0.0, eye=1.0, mouth=0.0, tear=0.0,
                lx=0.0, ly=0.0, happy=0.0, blush=0.25)
    base.update(kw)
    return base


NEUTRAL = E()
HAPPY = E(smile=1.0, brow=0.15, mouth=0.55, happy=1.0, blush=0.8)
SMILE = E(smile=0.9, brow=0.1, mouth=0.35, blush=0.5)
SURPRISE = E(smile=0.0, brow=1.0, eye=1.25, mouth=0.75, blush=0.1)
SAD = E(smile=-0.75, brow=-1.0, eye=0.85, mouth=0.0, blush=0.1)
CRY = E(smile=-1.0, brow=-1.0, eye=0.35, mouth=0.55, tear=1.0, blush=0.6)
THINK = E(smile=-0.1, brow=-0.35, eye=0.9, mouth=0.0)


def look(e, lx, ly=0.0):
    e = dict(e)
    e["lx"], e["ly"] = lx, ly
    return e


def blink(t, seed):
    """Returns 0..1 eye closure for periodic blinking."""
    period = 3.1 + (seed % 7) * 0.37
    ph = (t + seed * 1.37) % period
    if ph < 0.14:
        return math.sin(ph / 0.14 * math.pi)
    return 0.0


# ----------------------------------------------------------------- drawing
class Pen:
    def __init__(self, img):
        self.d = ImageDraw.Draw(img)

    @staticmethod
    def P(x):
        return x * SS

    def ellipse(self, cx, cy, rx, ry, fill=None, outline=None, width=0):
        s = SS
        self.d.ellipse([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s],
                       fill=fill, outline=outline, width=int(width * s))

    def poly(self, pts, fill=None, outline=None, width=0):
        pts = [(x * SS, y * SS) for x, y in pts]
        self.d.polygon(pts, fill=fill)
        if outline and width:
            self.d.line(pts + [pts[0]], fill=outline, width=int(width * SS), joint="curve")

    def line(self, pts, fill, width):
        pts = [(x * SS, y * SS) for x, y in pts]
        self.d.line(pts, fill=fill, width=max(1, int(width * SS)), joint="curve")
        r = width * SS / 2
        for (x, y) in (pts[0], pts[-1]):
            self.d.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    def chord(self, cx, cy, rx, ry, a0, a1, fill):
        s = SS
        self.d.chord([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], a0, a1, fill=fill)

    def arc(self, cx, cy, rx, ry, a0, a1, fill, width):
        s = SS
        self.d.arc([(cx - rx) * s, (cy - ry) * s, (cx + rx) * s, (cy + ry) * s], a0, a1,
                   fill=fill, width=int(width * s))


def heart(p, x, y, r, col):
    p.ellipse(x - r * 0.5, y - r * 0.25, r * 0.55, r * 0.55, fill=col)
    p.ellipse(x + r * 0.5, y - r * 0.25, r * 0.55, r * 0.55, fill=col)
    p.poly([(x - r * 1.02, y - 0.05 * r), (x + r * 1.02, y - 0.05 * r), (x, y + r * 1.05)], fill=col)


def draw_face(p, cx, cy, R, e, t, seed, skin):
    """Cartoon face. R = head radius. e = expression dict."""
    ex, ey = R * 0.38, cy - R * 0.05
    ew, eh = R * 0.12, R * 0.17
    lx, ly = e["lx"], e["ly"]
    ox, oy = lx * R * 0.08, ly * R * 0.06
    # cheeks
    if e["blush"] > 0.05:
        bc = mix(skin, (255, 120, 130), clamp(e["blush"]) * 0.75)
        for s in (-1, 1):
            p.ellipse(cx + s * R * 0.58 + ox * 0.5, cy + R * 0.22, R * 0.16, R * 0.1, fill=bc)
    # eyes
    cl = blink(t, seed)
    for s in (-1, 1):
        x, y = cx + s * ex + ox, ey + oy
        if e["happy"] > 0.5:
            # closed, smiling eyes  ^ ^
            p.arc(x, y + eh * 0.5, ew * 1.4, eh * 1.0, 200, 340, OL, R * 0.06)
        else:
            hgt = eh * e["eye"] * (1 - cl)
            if hgt < eh * 0.18:
                p.line([(x - ew, y), (x + ew, y)], OL, R * 0.05)
            else:
                p.ellipse(x, y, ew * (1 + 0.1 * (e["eye"] - 1)), hgt, fill=(40, 28, 30))
                p.ellipse(x - ew * 0.35 + lx * ew * 0.3, y - hgt * 0.4, ew * 0.38, ew * 0.38,
                          fill=(255, 255, 255))
    # brows
    b = e["brow"]
    by = ey - eh - R * 0.1 - max(b, 0) * R * 0.12 + oy
    tilt = -min(b, 0) * R * 0.11
    for s in (-1, 1):
        inner = (cx + s * ex * 0.45 + ox, by - tilt)
        outer = (cx + s * ex * 1.45 + ox, by + tilt * 0.6 + (max(b, 0) * -R * 0.02))
        p.line([inner, outer], OL, R * 0.055)
    # mouth
    my = cy + R * 0.42 + oy * 0.5
    sm, op = e["smile"], e["mouth"]
    w = R * 0.32 * (0.55 + 0.45 * min(1, abs(sm) + 0.1)) * (1 + 0.15 * op)
    k = R * 0.14 * sm
    n = 14
    xs = [-w + 2 * w * i / n for i in range(n + 1)]
    base = [(cx + ox * 0.6 + x, my + k * (1 - (x / w) ** 2)) for x in xs]
    if op < 0.06:
        p.line(base, OL, R * 0.055)
    else:
        hh = R * 0.3 * op
        if sm >= 0:
            upper = [(x, y - k * 0.1) for x, y in base]
            lower = [(x, y + hh * math.sqrt(max(0, 1 - ((x - cx - ox * 0.6) / w) ** 2)))
                     for x, y in base]
        else:
            upper = [(x, y - hh * 0.6 * math.sqrt(max(0, 1 - ((x - cx - ox * 0.6) / w) ** 2)))
                     for x, y in base]
            lower = [(x, y + hh * 0.25 * math.sqrt(max(0, 1 - ((x - cx - ox * 0.6) / w) ** 2)))
                     for x, y in base]
        shape = upper + lower[::-1]
        p.poly(shape, fill=(150, 45, 60), outline=OL, width=R * 0.045)
        if op > 0.3 and sm > 0.3:
            # tongue
            tx, ty = cx + ox * 0.6, lower[n // 2][1] - hh * 0.25
            p.ellipse(tx, ty, w * 0.45, hh * 0.22, fill=(240, 120, 130))
    # tears
    if e["tear"] > 0.03:
        for s in (-1, 1):
            for kk in range(3):
                ph = (t * 1.3 + kk / 3 + (0.5 if s > 0 else 0)) % 1.0
                tx = cx + s * (ex + ew * 0.2) + ox
                ty = ey + eh * 0.6 + ph * R * 0.8
                rr = R * 0.07 * e["tear"] * (1 - 0.4 * ph)
                p.ellipse(tx, ty, rr, rr * 1.3, fill=(110, 180, 255))
                p.poly([(tx - rr, ty - rr * 0.3), (tx + rr, ty - rr * 0.3), (tx, ty - rr * 2.2)],
                       fill=(110, 180, 255))


def arm_end(sx, sy, side, ang, L):
    a = math.radians(ang)
    return sx + side * math.sin(a) * L, sy + math.cos(a) * L


def draw_person(p, st, t):
    """st: dict with keys x, kind, e, armL, armR, walk, bounce, sit, nod, seed."""
    sc = st.get("scale", 1.0)
    x = st["x"]
    kind = st["kind"]
    sit = st.get("sit", 0.0)
    walk = st.get("walk", 0.0)
    ph = t * 9.0
    bounce = st.get("bounce", 0.0) + abs(math.sin(ph)) * 4 * walk
    leg_len = (52 - 22 * sit) * sc
    seat_y = 560
    hip_y = lerp(GROUND - leg_len, seat_y, sit) - bounce
    skin = (255, 214, 180)
    # legs
    for s in (-1, 1):
        sw = math.sin(ph + (0 if s < 0 else math.pi)) * 16 * walk * sc
        swing_sit = math.sin(t * 3 + s) * 6 * sit * st.get("dangle", 1.0)
        hx = x + s * 13 * sc
        fx = hx + sw + swing_sit
        fy = hip_y + leg_len
        p.line([(hx, hip_y), (fx, fy)], skin if kind == "boy" else skin, 13 * sc)
        p.ellipse(fx + s * 4 * sc, fy + 2 * sc, 13 * sc, 7 * sc,
                  fill=(200, 60, 70) if kind == "girl" else (80, 70, 160), outline=OL, width=2)
    body_h = 78 * sc
    top_y = hip_y - body_h + 8 * sc
    # body
    if kind == "girl":
        dress = (255, 196, 60)
        p.poly([(x - 22 * sc, top_y), (x + 22 * sc, top_y),
                (x + 48 * sc, hip_y + 12 * sc), (x - 48 * sc, hip_y + 12 * sc)],
               fill=dress, outline=OL, width=3)
        for i in range(-2, 3):
            p.ellipse(x + i * 17 * sc, hip_y - 2 * sc + abs(i) * 2, 4 * sc, 4 * sc, fill=(255, 250, 230))
    else:
        p.poly([(x - 26 * sc, hip_y + 4 * sc), (x + 26 * sc, hip_y + 4 * sc),
                (x + 26 * sc, hip_y - 26 * sc), (x - 26 * sc, hip_y - 26 * sc)],
               fill=(140, 95, 60), outline=OL, width=3)
        p.poly([(x - 28 * sc, top_y), (x + 28 * sc, top_y),
                (x + 30 * sc, hip_y - 22 * sc), (x - 30 * sc, hip_y - 22 * sc)],
               fill=(80, 150, 230), outline=OL, width=3)
        p.ellipse(x, top_y + 28 * sc, 9 * sc, 9 * sc, fill=(255, 230, 90))
    # arms
    hands = {}
    for side, key in ((-1, "armL"), (1, "armR")):
        ang = st.get(key, 12)
        sx, sy = x + side * 22 * sc, top_y + 10 * sc
        hx, hy = arm_end(sx, sy, side, ang, 56 * sc)
        p.line([(sx, sy), (hx, hy)], OL, 15 * sc)
        p.line([(sx, sy), (hx, hy)], skin if kind == "girl" else (80, 150, 230), 11 * sc)
        p.ellipse(hx, hy, 9 * sc, 9 * sc, fill=skin, outline=OL, width=2)
        hands[side] = (hx, hy)
    # head
    R = 50 * sc
    hcx = x + st["e"]["lx"] * 3 * sc
    hcy = top_y - R + 6 * sc + st.get("nod", 0.0) * 6 * sc
    if kind == "girl":
        hair = (95, 55, 40)
        for s in (-1, 1):
            p.ellipse(hcx + s * R * 1.0, hcy + R * 0.05, R * 0.33, R * 0.42, fill=hair, outline=OL, width=3)
            bx, byy = hcx + s * R * 0.92, hcy - R * 0.35
            p.poly([(bx, byy), (bx + s * 18 * sc, byy - 12 * sc), (bx + s * 18 * sc, byy + 12 * sc)],
                   fill=(230, 60, 90), outline=OL, width=2)
            p.poly([(bx, byy), (bx - s * 14 * sc, byy - 12 * sc), (bx - s * 14 * sc, byy + 12 * sc)],
                   fill=(230, 60, 90), outline=OL, width=2)
        p.ellipse(hcx, hcy, R, R, fill=skin, outline=OL, width=3)
        p.chord(hcx, hcy, R + 1, R + 1, 190, 350, fill=hair)
        p.poly([(hcx - R * 0.6, hcy - R * 0.5), (hcx - R * 0.15, hcy - R * 0.25), (hcx + R * 0.2, hcy - R * 0.55)],
               fill=hair)
    else:
        hair = (60, 40, 30)
        p.ellipse(hcx, hcy, R, R, fill=skin, outline=OL, width=3)
        p.chord(hcx, hcy, R + 1, R + 1, 195, 345, fill=hair)
        for i in range(-2, 3):
            bx = hcx + i * R * 0.3
            p.poly([(bx - R * 0.17, hcy - R * 0.8 + abs(i) * 4), (bx + R * 0.17, hcy - R * 0.8 + abs(i) * 4),
                    (bx + i * 4, hcy - R * 1.18 + abs(i) * 6)], fill=hair)
        # ears
        for s in (-1, 1):
            p.ellipse(hcx + s * R * 0.98, hcy + R * 0.08, R * 0.13, R * 0.18, fill=skin, outline=OL, width=2)
    draw_face(p, hcx, hcy, R, st["e"], t, st.get("seed", 1), skin)
    return hands, (hcx, hcy)


def draw_kitten(p, st, t):
    x = st["x"]
    walk = st.get("walk", 0.0)
    ph = t * 10
    bob = abs(math.sin(ph)) * 3 * walk + st.get("bounce", 0.0)
    fur = (245, 160, 70)
    by = GROUND - 26 - bob
    # tail
    sw = math.sin(t * 3.0) * 14 * st.get("tail", 1.0)
    p.line([(x + 26, by), (x + 46, by - 18), (x + 50 + sw, by - 46)], OL, 13)
    p.line([(x + 26, by), (x + 46, by - 18), (x + 50 + sw, by - 46)], fur, 9)
    # legs
    for i, dx in enumerate((-18, -6, 8, 20)):
        lsw = math.sin(ph + i * math.pi) * 5 * walk
        p.ellipse(x + dx + lsw, GROUND - 6, 7, 6, fill=fur, outline=OL, width=2)
    p.ellipse(x, by, 34, 22, fill=fur, outline=OL, width=3)
    p.ellipse(x, by + 6, 18, 12, fill=(255, 225, 190))
    # head
    R = 30
    hx, hy = x - 6 + st["e"]["lx"] * 2, by - 36 + st.get("nod", 0.0) * 4
    for s in (-1, 1):
        p.poly([(hx + s * R * 0.85, hy - R * 0.25), (hx + s * R * 0.25, hy - R * 0.85),
                (hx + s * R * 0.95, hy - R * 1.25)], fill=fur, outline=OL, width=3)
        p.poly([(hx + s * R * 0.75, hy - R * 0.4), (hx + s * R * 0.42, hy - R * 0.75),
                (hx + s * R * 0.85, hy - R * 1.0)], fill=(255, 170, 170))
    p.ellipse(hx, hy, R, R * 0.9, fill=fur, outline=OL, width=3)
    # stripes
    for i in (-1, 0, 1):
        p.line([(hx + i * 8, hy - R * 0.85), (hx + i * 7, hy - R * 0.6)], (210, 120, 50), 3)
    draw_face(p, hx, hy, R, st["e"], t, st.get("seed", 5), fur)
    p.poly([(hx - 4, hy + R * 0.22), (hx + 4, hy + R * 0.22), (hx, hy + R * 0.3)], fill=(230, 100, 120))
    for s in (-1, 1):
        for k in (-1, 1):
            p.line([(hx + s * R * 0.45, hy + R * 0.3 + k * 3), (hx + s * R * 1.15, hy + R * 0.25 + k * 7)],
                   OL, 1.5)
    return (hx, hy)


def draw_balloon(p, bx, by, ax, ay, col, t, seed=0):
    """Balloon centred at (bx,by); string anchored at (ax,ay)."""
    knot = (bx, by + 44)
    mid = ((bx + ax) / 2 + math.sin(t * 2 + seed) * 8, (by + 44 + ay) / 2)
    p.line([knot, mid, (ax, ay)], (90, 80, 80), 1.6)
    p.ellipse(bx, by, 34, 42, fill=col, outline=OL, width=3)
    p.ellipse(bx - 11, by - 15, 7, 11, fill=mix(col, (255, 255, 255), 0.6))
    p.poly([(bx - 6, by + 48), (bx + 6, by + 48), (bx, by + 41)], fill=col, outline=OL, width=2)


def draw_cloud(p, x, y, s, col=(255, 255, 255)):
    for dx, dy, r in ((-50, 8, 34), (-18, -12, 42), (22, -4, 38), (52, 10, 28), (0, 14, 36)):
        p.ellipse(x + dx * s, y + dy * s, r * s, r * s, fill=col)


def draw_sun(p, x, y, e, t, glow):
    for i in range(12):
        a = i / 12 * 2 * math.pi + t * 0.3
        r0, r1 = 72, 92 + 6 * math.sin(t * 3 + i)
        p.line([(x + math.cos(a) * r0, y + math.sin(a) * r0), (x + math.cos(a) * r1, y + math.sin(a) * r1)],
               (255, 200, 60), 7)
    p.ellipse(x, y, 62, 62, fill=mix((255, 225, 90), (255, 240, 150), glow), outline=(240, 170, 50), width=3)
    draw_face(p, x, y + 4, 52, e, t, 9, (255, 225, 90))


def draw_flower(p, x, y, g, col, t):
    if g <= 0:
        return
    sway = math.sin(t * 2 + x) * 3
    h = 34 * ease(g)
    p.line([(x, y), (x + sway, y - h)], (60, 150, 60), 4)
    if g > 0.3:
        p.ellipse(x - 7, y - h * 0.45, 8 * ease((g - 0.3) / 0.4), 4, fill=(80, 170, 70))
    pg = ease((g - 0.45) / 0.55)
    if pg > 0:
        cx, cy = x + sway, y - h
        for i in range(5):
            a = i / 5 * 2 * math.pi + t * 0.2
            p.ellipse(cx + math.cos(a) * 9 * pg, cy + math.sin(a) * 9 * pg, 7 * pg, 7 * pg, fill=col)
        p.ellipse(cx, cy, 5 * pg, 5 * pg, fill=(255, 220, 80))


def draw_bird(p, x, y, t, s=1.0):
    f = math.sin(t * 12) * 8 * s
    p.line([(x - 14 * s, y - f), (x, y), (x + 14 * s, y - f)], (70, 60, 80), 3)


# ------------------------------------------------------------- background
def make_background(dim, rainbow=False):
    top = mix((110, 190, 255), (120, 135, 160), dim)
    bot = mix((210, 240, 255), (175, 185, 200), dim)
    arr = np.zeros((H * SS, W * SS, 3), np.uint8)
    for c in range(3):
        col = np.linspace(top[c], bot[c], H * SS).astype(np.uint8)
        arr[:, :, c] = col[:, None]
    img = Image.fromarray(arr)
    if rainbow:
        rb = make_rainbow()
        img.paste(rb.convert("RGB"), (0, 0), rb.getchannel("A"))
    p = Pen(img)
    g = lambda c: mix(c, (110, 125, 120), dim * 0.8)
    # far hills
    p.ellipse(250, 640, 520, 190, fill=g((150, 210, 130)))
    p.ellipse(1050, 660, 600, 220, fill=g((135, 200, 120)))
    # trees
    for tx, ty, s in ((140, 470, 1.0), (330, 500, 0.75), (1210, 480, 0.95), (640, 515, 0.6)):
        p.poly([(tx - 10 * s, ty + 60 * s), (tx + 10 * s, ty + 60 * s), (tx + 8 * s, ty - 10 * s), (tx - 8 * s, ty - 10 * s)],
               fill=g((140, 95, 60)))
        for dx, dy, r in ((0, -40, 50), (-35, -15, 38), (35, -15, 38), (0, -70, 36)):
            p.ellipse(tx + dx * s, ty + dy * s, r * s, r * s, fill=g((80, 165, 85)))
    # ground
    p.poly([(0, 585), (W, 575), (W, H), (0, H)], fill=g((120, 200, 100)))
    p.poly([(0, 600), (W, 590), (W, H), (0, H)], fill=g((110, 190, 95)))
    # path
    p.poly([(0, 625), (W, 620), (W, 668), (0, 675)], fill=g((235, 215, 170)))
    # grass tufts
    rnd = random.Random(3)
    for _ in range(60):
        gx, gy = rnd.uniform(0, W), rnd.choice([rnd.uniform(600, 618), rnd.uniform(680, 715)])
        p.line([(gx - 4, gy), (gx, gy - 9), (gx + 4, gy)], g((80, 160, 70)), 2)
    # bush on the right (kitten hides behind it - drawn again in foreground)
    # bench
    bx = 900
    wood = g((175, 110, 65))
    for dy in (-62, -44):
        p.poly([(bx - 115, 560 + dy), (bx + 115, 560 + dy), (bx + 115, 560 + dy + 13), (bx - 115, 560 + dy + 13)],
               fill=wood, outline=OL, width=2)
    for s in (-1, 1):
        p.line([(bx + s * 95, 498), (bx + s * 95, 640)], g((90, 80, 85)), 8)
    p.poly([(bx - 120, 556), (bx + 120, 556), (bx + 120, 572), (bx - 120, 572)], fill=wood, outline=OL, width=2)
    return img


def make_rainbow():
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cols = [(255, 90, 90), (255, 165, 70), (255, 230, 90), (110, 210, 110), (90, 170, 255), (150, 110, 220)]
    cx, cy, r = 880 * SS, 640 * SS, 560 * SS
    bw = 22 * SS
    for i, c in enumerate(cols):
        rr = r - i * bw
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], 180, 360, fill=c + (150,), width=bw)
    return img


BG = None


def init_worker():
    global BG
    BG = dict(bright=make_background(0.0), dim=make_background(1.0), rainbow=make_background(0.0, rainbow=True),
              font_t=ImageFont.truetype(FONT_B, 70), font_s=ImageFont.truetype(FONT_R, 34),
              font_e=ImageFont.truetype(FONT_B, 52))


# ----------------------------------------------------------------- script
HERO_X = [(0, -120), (18, -120), (26, 520), (34, 520), (40, 790), (60, 790), (65, 880)]
HERO_WALK = [(17.9, 0), (18, 1), (25.8, 1), (26.1, 0), (34, 0), (34.2, 1), (39.8, 1), (40, 0),
             (63, 0), (63.1, 1), (64.9, 1), (65, 0)]
HERO_E = [(0, HAPPY), (21, look(SMILE, 0.4)), (22.5, HAPPY), (25, look(SMILE, 0.3)),
          (26.3, look(SMILE, 1.0)), (27.0, look(SURPRISE, 1.0)), (28.2, look(SAD, 1.0)),
          (29.6, look(SAD, 1.0)), (30.2, look(THINK, -0.5, -1)), (31.3, look(THINK, -0.5, -1)),
          (31.8, look(THINK, 1.0)), (32.4, look(THINK, 1.0)), (32.9, HAPPY), (34, look(SMILE, 1.0)),
          (40, look(SMILE, 1.0, 0.3)), (44, look(HAPPY, 1.0)), (48, HAPPY), (56.5, look(SMILE, 1.0)),
          (57.5, look(SURPRISE, 1.0)), (58.6, look(SAD, 1.0)), (59.5, look(SMILE, 1.0)),
          (62.5, look(SMILE, 1.0, 0.4)), (63.5, look(HAPPY, 1.0)), (65.5, HAPPY), (76, HAPPY)]
HERO_ARMR = [(0, 25), (40, 25), (41, 70), (42.4, 70), (43.4, 15), (65, 15), (66, 150), (76, 150)]
HERO_ARML = [(0, 12), (48, 12), (49, 150), (55, 150), (56, 12), (76, 12)]

BOY_X = [(0, 900), (48.3, 900), (49.2, 960), (60, 960), (61.5, 1040), (76, 1040)]
BOY_SIT = [(0, 1), (48.2, 1), (49.0, 0)]
BOY_E = [(0, HAPPY), (4, look(SMILE, 0.2, -1)), (6.5, look(HAPPY, 0, 0)), (7.4, look(SURPRISE, 0.4, -1)),
         (10.3, look(SURPRISE, 0.6, -1)), (11.2, look(SAD, 0.3, -0.4)), (12.6, look(CRY, 0, 0.3)),
         (36.5, look(CRY, 0, 0.3)), (37.3, look(CRY, -1.0, 0)), (38.4, look(SAD, -1.0, 0)),
         (41.0, look(SURPRISE, -1.0, -0.2)), (43.0, look(SURPRISE, -0.5, -1)), (44.0, HAPPY),
         (56.5, look(SMILE, 1.0)), (57.5, look(SURPRISE, 1.0)), (58.6, look(SAD, 1.0, 0.5)),
         (60.0, look(SMILE, 1.0, 0.6)), (63.0, look(HAPPY, 1.0, 0.6)), (65.5, HAPPY), (76, HAPPY)]
BOY_ARMR = [(0, 140), (7.3, 140), (7.8, 168), (10.2, 168), (11.5, 10), (60, 10), (61.5, 55), (64.5, 55),
            (65.5, 150), (76, 150)]
BOY_ARML = [(0, 12), (41.5, 12), (42.3, 60), (43.4, 140), (76, 140)]

KIT_X = [(0, 1400), (56, 1400), (59, 1130), (76, 1130)]
KIT_WALK = [(55.9, 0), (56, 1), (58.9, 1), (59, 0)]
KIT_E = [(0, look(SAD, -1.0)), (59, look(SAD, -1.0, 0)), (60.5, look(SAD, -1.0, -0.4)),
         (61.8, E(smile=0.9, mouth=0.2, happy=1.0, blush=0.9)), (64.5, E(smile=1.0, mouth=0.5, happy=1.0, blush=0.9)),
         (66, HAPPY), (76, HAPPY)]

SUN_E = [(0, HAPPY), (8.5, look(SURPRISE, -0.5, 0.5)), (10, look(SAD, -0.5, 0.6)), (44, look(SAD, -0.5, 0.6)),
         (45.5, look(SURPRISE, -0.6, 0.7)), (46.5, look(HAPPY, -0.5, 0.6)), (76, HAPPY)]
CLOUD_SUN = [(0, 1350), (8, 1350), (11, 1085), (44, 1085), (47.5, 1420)]
DIM = [(0, 0), (9, 0), (12.5, 0.55), (43.5, 0.55), (47, 0)]
RAINBOW = [(0, 0), (64, 0), (67.5, 1)]

CAM = [(0, (640, 360, 1.0)), (6.5, (640, 360, 1.0)), (8.5, (880, 380, 1.45)), (11.5, (890, 400, 1.6)),
       (13.5, (900, 440, 2.3)), (17.0, (900, 440, 2.3)), (19.0, (640, 380, 1.08)), (25, (560, 380, 1.15)),
       (27.0, (640, 400, 1.3)), (28.6, (522, 430, 2.3)), (33.6, (522, 430, 2.3)), (35.5, (760, 420, 1.45)),
       (41.5, (850, 430, 1.55)), (43.0, (900, 440, 2.2)), (45.5, (900, 440, 2.2)), (47.5, (760, 340, 1.1)),
       (55.5, (760, 360, 1.1)), (58.5, (980, 440, 1.5)), (61.5, (1080, 520, 1.9)), (62.3, (1124, 556, 2.6)),
       (64.0, (1124, 556, 2.6)), (66.5, (900, 380, 1.25)), (75, (900, 360, 1.15))]

FLOWERS = [(x, y, random.Random(int(x)).choice([(255, 110, 150), (255, 255, 255), (190, 120, 255), (255, 160, 80)]))
           for x, y in [(60, 700), (150, 690), (230, 708), (330, 695), (420, 705), (500, 690), (590, 710),
                        (660, 697), (740, 712), (830, 692), (920, 706), (1010, 694), (1100, 710), (1190, 698),
                        (1250, 708), (40, 612), (260, 608), (470, 612), (690, 606), (1180, 606)]]


def hearts_burst(p, t, t0, t1, ox, oy, n, seed):
    if t < t0:
        return
    rnd = random.Random(seed)
    for i in range(n):
        st = t0 + (t1 - t0) * i / n
        life = t - st
        if 0 <= life < 2.6:
            dx = rnd.uniform(-60, 60)
            sz = rnd.uniform(8, 15)
            x = ox + dx + math.sin(life * 3 + i) * 12
            y = oy - life * 70
            k = 1 - (life / 2.6) ** 2
            heart(p, x, y, sz * min(1, life * 4) * k, (255, 90 + rnd.randint(0, 50), 130))
        else:
            rnd.uniform(-60, 60), rnd.uniform(8, 15)


def render(i):
    t = i / FPS
    dim = track(t, DIM)
    img = Image.blend(BG["bright"], BG["dim"], dim) if dim > 0.001 else BG["bright"].copy()
    rb = track(t, RAINBOW)
    if rb > 0.001:
        img = Image.blend(img, BG["rainbow"], rb)
    p = Pen(img)

    # sky: sun, clouds, birds
    draw_sun(p, 1085, 120, track(t, SUN_E), t, 1 - dim)
    ccol = mix((255, 255, 255), (205, 210, 220), dim)
    for k, (cx0, cy, s, sp) in enumerate(((100, 90, 0.9, 6), (520, 150, 0.7, 9), (820, 70, 0.6, 5))):
        cx = (cx0 + t * sp) % (W + 300) - 150
        draw_cloud(p, cx, cy, s, ccol)
    draw_cloud(p, track(t, CLOUD_SUN), 125, 1.15, mix((255, 255, 255), (185, 190, 205), dim))
    if t > 64:
        for k in range(4):
            bx = -80 + (t - 64) * (60 + k * 8) + k * 50
            draw_bird(p, bx, 200 + k * 22 + math.sin(t * 2 + k) * 6, t + k * 0.3, 0.9)

    # flying-away blue balloon
    boy = dict(x=track(t, BOY_X), kind="boy", e=track(t, BOY_E), armR=track(t, BOY_ARMR), armL=track(t, BOY_ARML),
               sit=track(t, BOY_SIT), seed=3, scale=0.92, dangle=0.3 if 12 < t < 44 else 1.0)
    if 44 < t < 56:
        boy["bounce"] = max(0, math.sin((t - 44) * 5)) * 18 if t > 49 else 0
    if t > 66:
        boy["bounce"] = max(0, math.sin((t - 66) * 4.5)) * 14
    hero = dict(x=track(t, HERO_X), kind="girl", e=track(t, HERO_E), armR=track(t, HERO_ARMR),
                armL=track(t, HERO_ARML), walk=track(t, HERO_WALK), seed=1)
    if 32.9 < t < 34:
        hero["nod"] = math.sin((t - 32.9) * 2 * math.pi * 1.8) * 1.2
    if 49 < t < 56:
        hero["bounce"] = max(0, math.sin((t - 49) * 5 + 1.5)) * 22
    if t > 66:
        hero["armR"] = 150 + math.sin(t * 7) * 22
        hero["bounce"] = max(0, math.sin((t - 66) * 4.5 + 1.5)) * 14
    if t > 66:
        boy["armR"] = 150 + math.sin(t * 7 + 1) * 22
    if 61.5 < t < 64.5:
        boy["armR"] = 55 + math.sin(t * 9) * 8

    hero_hands, hero_head = draw_person(p, hero, t)
    boy_hands, boy_head = draw_person(p, boy, t)
    kit = dict(x=track(t, KIT_X), e=track(t, KIT_E), walk=track(t, KIT_WALK), seed=7)
    if 61.8 < t < 64.5:
        kit["nod"] = math.sin(t * 9) * 0.8
        kit["tail"] = 2.0
    if t > 66:
        kit["bounce"] = max(0, math.sin((t - 66) * 4.5 + 0.7)) * 12
    kit_head = draw_kitten(p, kit, t)

    # foreground bush near the right edge
    for dx, dy, r in ((1235, 640, 50), (1275, 625, 55), (1195, 655, 40)):
        p.ellipse(dx, dy, r, r * 0.8, fill=mix((70, 155, 75), (95, 115, 105), dim * 0.8))

    # balloons
    bob = math.sin(t * 2.2) * 6
    if t < 7.5:
        hx, hy = boy_hands[1]
        draw_balloon(p, hx + 20, hy - 120 + bob, hx, hy, (90, 160, 255), t, 1)
    elif t < 13:
        u = ease((t - 7.5) / 5.0)
        bx = lerp(930, 1240, u) + math.sin(t * 3) * 20
        by = lerp(330, -120, u * u * 0.6 + u * 0.4)
        draw_balloon(p, bx, by, bx - 10 + math.sin(t * 4) * 10, by + 130, (90, 160, 255), t, 1)
    # red balloon: hero -> boy at 42.4..43.4
    hh = hero_hands[1]
    bh = boy_hands[-1]
    v = ease((t - 42.4) / 1.0)
    ax, ay = lerp(hh[0], bh[0], v), lerp(hh[1], bh[1], v)
    hold_off = lerp(48, 15, v)
    draw_balloon(p, ax + hold_off, min(ay, 520) - 140 + bob, ax, ay, (240, 60, 70), t, 2)

    # flowers bloom after kindness
    for k, (fx, fy, col) in enumerate(FLOWERS):
        g = clamp((t - 47.5 - (k % 10) * 0.55) / 1.4)
        draw_flower(p, fx, fy, g, col, t)

    # hearts
    hearts_burst(p, t, 43.5, 47, boy_head[0], boy_head[1] - 40, 7, 11)
    hearts_burst(p, t, 49, 55.5, hero_head[0], hero_head[1] - 50, 9, 12)
    hearts_burst(p, t, 49.5, 55.5, boy_head[0], boy_head[1] - 50, 9, 13)
    hearts_burst(p, t, 62, 65, kit_head[0], kit_head[1] - 30, 7, 14)
    hearts_burst(p, t, 67, 75, 960, 300, 14, 15)

    # leaves in the wind gust (~7s)
    if 6.2 < t < 10:
        rnd = random.Random(4)
        for k in range(10):
            lt = t - 6.2 - k * 0.12
            if lt > 0:
                lx = -50 + lt * 420 + rnd.uniform(0, 300)
                ly = rnd.uniform(380, 560) - lt * 40 + math.sin(lt * 5 + k) * 25
                p.ellipse(lx, ly, 9, 5, fill=(120, 190, 80) if k % 2 else (230, 180, 60))
            else:
                rnd.uniform(0, 300), rnd.uniform(380, 560)

    # camera
    cx, cy, z = track(t, CAM)
    cw, ch = W / z, H / z
    x0 = clamp(cx - cw / 2, 0, W - cw)
    y0 = clamp(cy - ch / 2, 0, H - ch)
    frame = img.resize((W, H), Image.LANCZOS, box=(x0 * SS, y0 * SS, (x0 + cw) * SS, (y0 + ch) * SS))

    # title & ending overlays (screen space)
    title_a = track(t, [(0, 0), (0.8, 1), (4.6, 1), (6.0, 0)])
    if title_a > 0.01:
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        txt = "Мейірім шуағы"
        tw = d.textlength(txt, font=BG["font_t"])
        for o in ((3, 3),):
            d.text(((W - tw) / 2 + o[0], 230 + o[1]), txt, font=BG["font_t"], fill=(120, 60, 60, int(140 * title_a)))
        d.text(((W - tw) / 2, 230), txt, font=BG["font_t"], fill=(255, 255, 255, int(255 * title_a)))
        frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
    end_a = track(t, [(0, 0), (76, 0), (78, 1)])
    if end_a > 0.001:
        ov = Image.new("RGBA", (W, H), (255, 244, 230, int(255 * end_a)))
        d = ImageDraw.Draw(ov)
        ta = track(t, [(0, 0), (77.5, 0), (78.5, 1)])
        if ta > 0:
            hp = Pen.__new__(Pen)
            hp.d = d
            # heart (screen space -> undo SS scaling by drawing directly)
            r = 46 * (1 + 0.06 * math.sin(t * 5))
            hx, hy = W / 2, 250
            col = (240, 80, 110, int(255 * ta))
            d.ellipse([hx - r * 1.05, hy - r * 0.8, hx + r * 0.05, hy + r * 0.3], fill=col)
            d.ellipse([hx - r * 0.05, hy - r * 0.8, hx + r * 1.05, hy + r * 0.3], fill=col)
            d.polygon([(hx - r * 1.02, hy - 0.05 * r), (hx + r * 1.02, hy - 0.05 * r), (hx, hy + r * 1.05)], fill=col)
            txt = "Мейірімділік — бақыттың кілті"
            tw = d.textlength(txt, font=BG["font_e"])
            d.text(((W - tw) / 2, 350), txt, font=BG["font_e"], fill=(150, 70, 80, int(255 * ta)))
        sa = track(t, [(0, 0), (79.8, 0), (80.6, 1)])
        if sa > 0:
            txt = "Соңы"
            tw = d.textlength(txt, font=BG["font_s"])
            d.text(((W - tw) / 2, 450), txt, font=BG["font_s"], fill=(150, 110, 110, int(255 * sa)))
        frame = Image.alpha_composite(frame.convert("RGBA"), ov).convert("RGB")
    fade_in = clamp(t / 0.8)
    if fade_in < 1:
        frame = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), frame, fade_in)
    return frame.tobytes()


# ------------------------------------------------------------------ music
def make_music(path):
    sr = 44100
    n = int(sr * DUR)
    out = np.zeros(n)
    nn = lambda m: 440.0 * 2 ** ((m - 69) / 12)

    def tone(t0, dur, midi, amp, decay=2.5, bright=0.3):
        i0 = int(t0 * sr)
        ln = int(dur * sr)
        if i0 >= n:
            return
        ln = min(ln, n - i0)
        tt = np.arange(ln) / sr
        f = nn(midi)
        env = np.exp(-tt * decay) * np.minimum(1, tt / 0.008)
        env *= np.minimum(1, (ln - np.arange(ln)) / (0.05 * sr))
        w = np.sin(2 * np.pi * f * tt) + bright * np.sin(4 * np.pi * f * tt) + 0.1 * np.sin(6 * np.pi * f * tt)
        out[i0:i0 + ln] += amp * env * w

    # chord = root midi, quality
    C, G, Am, F, Dm, E7, Em = (60, "M"), (55, "M"), (57, "m"), (53, "M"), (50, "m"), (52, "M"), (52, "m")
    sections = [
        # (start, end, chords, chord_len, happy?)
        (0.0, 7.0, [C, G, Am, F], 1.75, True),
        (7.0, 18.0, [Am, F, Dm, E7], 2.75, False),
        (18.0, 26.0, [C, Am, F, G], 2.0, True),
        (26.0, 34.0, [Am, F, Dm, G], 2.0, False),
        (34.0, 44.0, [F, G, Em, Am, F, G], 1.66, None),
        (44.0, 76.0, [C, G, Am, F, C, F, G, C], 2.0, True),
        (76.0, 82.0, [F, G, C], 2.0, True),
    ]
    rnd = random.Random(7)
    for s0, s1, chords, cl, happy in sections:
        t0, ci = s0, 0
        while t0 < s1 - 0.01:
            root, q = chords[ci % len(chords)]
            tones = [root, root + (4 if q == "M" else 3), root + 7]
            dur = min(cl, s1 - t0)
            vol = 0.11 if happy else (0.075 if happy is False else 0.095)
            # bass + pad
            tone(t0, dur + 0.3, root - 12, vol * 1.1, decay=0.9, bright=0.15)
            for m in tones:
                tone(t0, dur + 0.3, m, vol * 0.35, decay=0.6, bright=0.05)
            # arpeggio (music box)
            step = 0.25 if happy else 0.5
            if happy is None:
                step = 0.33
            pat = [0, 1, 2, 1, 2, 3, 2, 1]
            k = 0
            tt = t0
            while tt < t0 + dur - 0.05:
                deg = pat[k % len(pat)]
                m = (tones + [root + 12])[deg] + 12
                tone(tt, 1.2, m, vol * 0.55, decay=3.5, bright=0.4)
                tt += step
                k += 1
            # melody
            tt = t0
            while tt < t0 + dur - 0.05:
                m = rnd.choice(tones + [root + 12]) + 24
                ln = rnd.choice([0.5, 0.5, 1.0]) if happy else 1.0
                tone(tt, ln + 0.6, m, vol * (0.6 if happy else 0.45), decay=2.2, bright=0.25)
                tt += ln
            t0 += cl
            ci += 1
    # fade in/out
    tt = np.arange(n) / sr
    out *= np.minimum(1, tt / 1.5) * np.minimum(1, (DUR - tt) / 3.0)
    # simple echo for warmth
    dly = int(0.23 * sr)
    out[dly:] += 0.28 * out[:-dly]
    out /= max(1e-9, np.abs(out).max()) / 0.8
    st = np.stack([out, np.roll(out, int(0.012 * sr))], 1)
    data = (st * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "meyirim-shuagy.mp4"
    here = os.path.dirname(os.path.abspath(out))
    wav = os.path.join(here, "_music.wav")
    make_music(wav)
    nframes = int(DUR * FPS)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "20",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", out]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(os.cpu_count(), initializer=init_worker) as pool:
        for k, buf in enumerate(pool.imap(render, range(nframes), chunksize=8)):
            ff.stdin.write(buf)
            if k % (FPS * 5) == 0:
                print(f"  {k / FPS:5.1f}s / {DUR:.0f}s", flush=True)
    ff.stdin.close()
    ff.wait()
    os.remove(wav)
    print("done:", out)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--still":
        init_worker()
        for ts in sys.argv[2:]:
            Image.frombytes("RGB", (W, H), render(int(float(ts) * FPS))).save(f"still_{ts}.png")
    else:
        main()
