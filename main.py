"""
Air Drawing System Ultra Pro v3.0 — PySide6 + GPU Edition
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from mediapipe.tasks.python.core import base_options as base_opts

import colorsys
import json
import logging
import math
import os
import sys
import threading
import time
import atexit
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Dict, List, Optional, Tuple

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QStatusBar, QToolBar,
    QFileDialog, QSizePolicy, QGridLayout,
    QFrame, QScrollArea, QToolButton,
)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QAction, QKeySequence,
    QFont, QFontDatabase, QPen, QFontMetrics,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.FileHandler("air_drawing_v3.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("AirDraw")


def _detect_gpu():
    cuda_ok = False
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            info = cv2.cuda.DeviceInfo(0)
            log.info(f"CUDA GPU: {info.name()}")
            cuda_ok = True
    except Exception:
        pass
    cupy_ok = False
    try:
        import cupy
        cupy.cuda.runtime.getDeviceCount()
        log.info("CuPy available")
        cupy_ok = True
    except Exception:
        pass
    if not cuda_ok and not cupy_ok:
        log.info("No GPU — CPU mode")
    return cuda_ok, cupy_ok

_CUDA_OK, _CUPY_OK = _detect_gpu()
_cuda_enabled = _CUDA_OK

# Caches shared across calls — populated lazily
_vig_cache: dict = {}           # (h, w) → float32 vignette array
_cuda_filter_cache: dict = {}   # (ksize, sigma) → cv2.cuda filter object


def _gpu_weighted_add(a: np.ndarray, wa: float, b: np.ndarray, wb: float) -> np.ndarray:
    global _cuda_enabled
    if _cuda_enabled:
        try:
            ga = cv2.cuda_GpuMat(); ga.upload(a)
            gb = cv2.cuda_GpuMat(); gb.upload(b)
            dst = cv2.cuda_GpuMat()
            cv2.cuda.addWeighted(ga, wa, gb, wb, 0, dst)
            return dst.download()
        except Exception:
            _cuda_enabled = False
            log.warning("CUDA weighted-add failed; falling back to CPU permanently")
    return cv2.addWeighted(a, wa, b, wb, 0)


def _gpu_gaussian_blur(img: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
    global _cuda_enabled
    if _cuda_enabled:
        try:
            key = (ksize, round(sigma, 4))
            if key not in _cuda_filter_cache:
                _cuda_filter_cache[key] = cv2.cuda.createGaussianFilter(
                    cv2.CV_8UC3, cv2.CV_8UC3, (ksize, ksize), sigma)
            src = cv2.cuda_GpuMat(); src.upload(img)
            dst = cv2.cuda_GpuMat()
            _cuda_filter_cache[key].apply(src, dst)
            return dst.download()
        except Exception:
            _cuda_enabled = False
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _gpu_flip(img: np.ndarray) -> np.ndarray:
    """Horizontal mirror — uses CUDA when available, falls back to CPU."""
    global _cuda_enabled
    if _cuda_enabled:
        try:
            src = cv2.cuda_GpuMat(); src.upload(img)
            dst = cv2.cuda_GpuMat()
            cv2.cuda.flip(src, 1, dst)
            return dst.download()
        except Exception:
            _cuda_enabled = False
    return cv2.flip(img, 1)


def _gpu_resize(img: np.ndarray, size: tuple, interp=cv2.INTER_LINEAR) -> np.ndarray:
    """Resize — uses CUDA when available, falls back to CPU."""
    global _cuda_enabled
    if _cuda_enabled:
        try:
            src = cv2.cuda_GpuMat(); src.upload(img)
            dst = cv2.cuda.resize(src, size, interpolation=interp)
            return dst.download()
        except Exception:
            _cuda_enabled = False
    return cv2.resize(img, size, interpolation=interp)


def _get_vignette(h: int, w: int) -> np.ndarray:
    """Return a precomputed float32 vignette mask for the given frame size."""
    key = (h, w)
    if key not in _vig_cache:
        kx  = cv2.getGaussianKernel(w, w / 2.8)
        ky  = cv2.getGaussianKernel(h, h / 2.8)
        vig = (ky * kx.T).astype(np.float32)
        vig /= vig.max()
        _vig_cache[key] = np.dstack([vig, vig, vig])
    return _vig_cache[key]


class BrushType(Enum):
    BASIC      = "basic"
    CALLIGRAPHY= "calligraphy"
    SPRAY      = "spray"
    ERASER     = "eraser"
    NEON       = "neon"
    WATERCOLOR = "watercolor"
    RAINBOW    = "rainbow"
    SPARKLE    = "sparkle"
    FIRE       = "fire"
    ICE        = "ice"
    GALAXY     = "galaxy"
    GLITTER    = "glitter"
    SHADOW     = "shadow"
    GRADIENT   = "gradient"
    DOTTED     = "dotted"
    ZIGZAG     = "zigzag"

BRUSH_KEYS = list(BrushType)
TARGET_FPS = 30

BRUSH_LABELS: Dict[str, str] = {
    BrushType.BASIC.value:       "Basic",
    BrushType.CALLIGRAPHY.value: "Calligraphy",
    BrushType.SPRAY.value:       "Spray",
    BrushType.ERASER.value:      "Eraser",
    BrushType.NEON.value:        "Neon",
    BrushType.WATERCOLOR.value:  "Watercolor",
    BrushType.RAINBOW.value:     "Rainbow",
    BrushType.SPARKLE.value:     "Sparkle",
    BrushType.FIRE.value:        "Fire",
    BrushType.ICE.value:         "Ice",
    BrushType.GALAXY.value:      "Galaxy",
    BrushType.GLITTER.value:     "Glitter",
    BrushType.SHADOW.value:      "Shadow",
    BrushType.GRADIENT.value:    "Gradient",
    BrushType.DOTTED.value:      "Dotted",
    BrushType.ZIGZAG.value:      "Zigzag",
}


@dataclass
class BrushStroke:
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int]
    brush_size: int
    timestamp: float
    brush_type: str
    pressure: float = 1.0


@dataclass
class DrawState:
    mode: str = "draw"
    active_brush: str = BrushType.RAINBOW.value
    color: Tuple[int, int, int] = (255, 20, 147)
    brush_size: int = 10
    palette_name: str = "NEON"
    stroke_count: int = 0
    fps: float = 0.0
    quality_level: int = 2
    gpu_active: bool = False


@dataclass
class HandResult:
    gesture: str
    pos: Tuple[int, int]
    pressure: float
    angle: float
    detected: bool


class AirDrawError(Exception): pass
class CameraError(AirDrawError): pass
class MediaPipeError(AirDrawError): pass


@dataclass
class OverlayState:
    cursor_pos:  Optional[Tuple[int, int]]
    raw_pos:     Optional[Tuple[int, int]]
    trail:       List[Tuple[int, int]]
    mode:        str
    gesture:     str
    pressure:    float
    brush_color: Tuple[int, int, int]
    brush_size:  int
    frame_wh:    Tuple[int, int]
    detected:    bool


class Kalman1D:
    __slots__ = ("x", "v", "P", "Q", "R")

    def __init__(self, process_noise=1e-3, meas_noise=1e-1):
        self.x = 0.0; self.v = 0.0
        self.P = np.eye(2)
        self.Q = np.array([[1e-4, 0], [0, process_noise]])
        self.R = meas_noise

    def update(self, z: float) -> float:
        F = np.array([[1, 1], [0, 1]])
        state = F @ np.array([self.x, self.v])
        P     = F @ self.P @ F.T + self.Q
        H     = np.array([[1, 0]])
        K     = P @ H.T / (H @ P @ H.T + self.R)[0, 0]
        state = state + K.flatten() * (z - state[0])
        self.P = (np.eye(2) - np.outer(K.flatten(), H)) @ P
        self.x, self.v = state
        return self.x

    def predict(self, steps=1.0):
        return self.x + self.v * steps


class KalmanCursor:
    def __init__(self):
        self.kx = Kalman1D(5e-3, 8e-2)
        self.ky = Kalman1D(5e-3, 8e-2)

    def update(self, pt):
        return (int(self.kx.update(pt[0])), int(self.ky.update(pt[1])))

    def predict(self, steps=1.0):
        return (int(self.kx.predict(steps)), int(self.ky.predict(steps)))


class VectorParticleSystem:
    MAX = 2000

    def __init__(self):
        n = self.MAX
        self.px   = np.zeros(n, np.float32)
        self.py   = np.zeros(n, np.float32)
        self.vx   = np.zeros(n, np.float32)
        self.vy   = np.zeros(n, np.float32)
        self.life = np.zeros(n, np.float32)
        self.maxl = np.ones(n,  np.float32)
        self.r    = np.zeros(n, np.uint8)
        self.g    = np.zeros(n, np.uint8)
        self.b    = np.zeros(n, np.uint8)
        self.sz   = np.ones(n,  np.int32)
        self._count = 0

    def emit(self, pos, color, count=5, spread=2.0):
        if count <= 0: return
        start = self._count % self.MAX
        idx   = np.arange(start, start + count) % self.MAX
        angles = np.random.uniform(0, 2*np.pi, count).astype(np.float32)
        speeds = np.random.uniform(0.5, spread, count).astype(np.float32)
        lives  = np.random.uniform(0.4, 1.8, count).astype(np.float32)
        self.px[idx]   = pos[0]; self.py[idx] = pos[1]
        self.vx[idx]   = np.cos(angles) * speeds
        self.vy[idx]   = np.sin(angles) * speeds
        self.life[idx] = lives; self.maxl[idx] = lives
        self.r[idx] = min(255, int(color[0] * np.random.uniform(0.8, 1.2)))
        self.g[idx] = min(255, int(color[1] * np.random.uniform(0.8, 1.2)))
        self.b[idx] = min(255, int(color[2] * np.random.uniform(0.8, 1.2)))
        self.sz[idx] = np.random.randint(1, 4, count)
        self._count = min(self._count + count, self.MAX)

    def update(self, dt=1.0/TARGET_FPS):
        if not self._count: return
        n = self._count
        self.px[:n] += self.vx[:n] * dt * 60
        self.py[:n] += self.vy[:n] * dt * 60
        self.life[:n] -= dt
        self.vy[:n]   += 0.05

    def render(self, frame: np.ndarray, quality=2):
        if not self._count: return
        n = self._count
        h, w = frame.shape[:2]
        alive = self.life[:n] > 0
        alpha_v = (self.life[:n] / self.maxl[:n]).clip(0, 1)
        xs = self.px[:n].astype(np.int32)
        ys = self.py[:n].astype(np.int32)
        mask = alive & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        if not mask.any(): return

        # Slice to active range first so mask length == array length
        b_n, g_n, r_n, sz_n = self.b[:n], self.g[:n], self.r[:n], self.sz[:n]

        if quality == 0:
            frame[ys[mask], xs[mask]] = np.stack(
                [b_n[mask], g_n[mask], r_n[mask]], axis=1)
            return

        # Vectorised alpha-blended scatter — no Python loop
        a   = alpha_v[mask].reshape(-1, 1)
        bgr = np.stack([b_n[mask], g_n[mask], r_n[mask]], axis=1).astype(np.float32)
        pix = (bgr * a).clip(0, 255).astype(np.uint8)
        ym, xm, szm = ys[mask], xs[mask], sz_n[mask]

        # Single-pixel particles: direct scatter (the common case)
        s1 = szm <= 1
        frame[ym[s1], xm[s1]] = pix[s1]

        # Larger particles at quality==2 only; quality==1 treats them as 1-px
        if quality == 2:
            for j in np.where(~s1)[0]:
                cv2.circle(frame, (int(xm[j]), int(ym[j])), int(szm[j]),
                           (int(pix[j, 0]), int(pix[j, 1]), int(pix[j, 2])), -1)
        else:
            frame[ym[~s1], xm[~s1]] = pix[~s1]


class ColorPalette:
    CLASSIC  = {"red":(0,0,255),"green":(0,255,0),"blue":(255,0,0),
                "yellow":(0,255,255),"purple":(255,0,255),"black":(0,0,0),
                "white":(255,255,255),"orange":(0,165,255),"pink":(255,192,203),
                "cyan":(255,255,0)}
    NEON     = {"neon_pink":(255,20,147),"neon_green":(57,255,20),
                "neon_blue":(77,77,255),"neon_yellow":(255,255,0),
                "neon_orange":(255,110,0),"neon_purple":(179,0,255),
                "neon_cyan":(0,255,255),"neon_magenta":(255,0,255),
                "electric_lime":(204,255,0),"hot_pink":(255,105,180)}
    PASTEL   = {"pastel_pink":(255,209,220),"pastel_blue":(174,198,255),
                "pastel_green":(204,255,204),"pastel_yellow":(255,255,179),
                "pastel_purple":(210,180,255),"mint":(170,240,209),
                "lavender":(230,230,250),"peach":(255,218,185)}
    METALLIC = {"gold":(255,215,0),"silver":(192,192,192),"bronze":(205,127,50),
                "rose_gold":(183,110,121),"chrome":(220,220,220),"platinum":(229,228,226)}
    OCEAN    = {"deep_blue":(0,105,148),"turquoise":(64,224,208),
                "coral":(255,127,80),"seafoam":(159,226,191),"marine":(20,153,153)}
    SUNSET   = {"sunset_orange":(253,94,83),"sunset_pink":(247,127,127),
                "twilight":(52,73,94),"golden_hour":(248,194,50)}
    PALETTES = ["NEON","PASTEL","METALLIC","OCEAN","SUNSET","CLASSIC"]

    @staticmethod
    def rainbow(steps):
        out = []
        for i in range(steps):
            rgb = colorsys.hsv_to_rgb(i/steps, 1.0, 1.0)
            out.append((int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255)))
        return out

    @staticmethod
    def gradient(c1, c2, n):
        if n <= 1: return [c1]
        return [(int(c1[0]+(c2[0]-c1[0])*i/(n-1)),
                 int(c1[1]+(c2[1]-c1[1])*i/(n-1)),
                 int(c1[2]+(c2[2]-c1[2])*i/(n-1))) for i in range(n)]

    def get_palette(self, name):
        return getattr(self, name.upper(), self.NEON)


class BrushEngine:
    PRESETS = {
        BrushType.BASIC.value:       {"size":5},
        BrushType.CALLIGRAPHY.value: {"size":8,"angle_sensitive":True},
        BrushType.SPRAY.value:       {"size":20,"density":0.35,"scatter":5},
        BrushType.ERASER.value:      {"size":25},
        BrushType.NEON.value:        {"size":10,"glow_r":3,"glow_i":0.7},
        BrushType.WATERCOLOR.value:  {"size":15,"flow":0.5,"bleed":0.3},
        BrushType.RAINBOW.value:     {"size":12,"speed":0.1},
        BrushType.SPARKLE.value:     {"size":8,"count":5},
        BrushType.FIRE.value:        {"size":16,"fh":10,"pc":8},
        BrushType.ICE.value:         {"size":14,"cs":3,"frost":0.5},
        BrushType.GALAXY.value:      {"size":18,"sd":0.3},
        BrushType.GLITTER.value:     {"size":10,"gc":8,"gs":2},
        BrushType.SHADOW.value:      {"size":15,"blur":3,"off":(5,5)},
        BrushType.GRADIENT.value:    {"size":12,"c1":(255,0,0),"c2":(0,0,255)},
        BrushType.DOTTED.value:      {"size":10,"spacing":5},
        BrushType.ZIGZAG.value:      {"size":8,"amp":10,"freq":0.5},
    }
    _FIRE = [(0,0,255),(0,165,255),(0,255,255),(255,255,255)]
    _ICE  = [(255,255,255),(255,255,240),(240,248,255),(176,224,230)]

    def apply(self, canvas, p0, p1, color, btype, pressure=1.0, angle=0.0, ts=0.0, quality=2):
        try:
            b  = self.PRESETS.get(btype, self.PRESETS[BrushType.BASIC.value])
            sz = max(1, int(b["size"] * pressure))
            getattr(self, f"_b_{btype}", self._b_basic)(canvas, p0, p1, color, sz, b, angle, ts, quality)
        except Exception as e:
            log.debug(f"Brush: {e}")

    @staticmethod
    def _interp(p0, p1, n):
        t  = np.linspace(0, 1, max(n, 2))
        return (p0[0]+t*(p1[0]-p0[0])).astype(int), (p0[1]+t*(p1[1]-p0[1])).astype(int)

    @staticmethod
    def _dist(p0, p1):
        return max(1, int(math.hypot(p1[0]-p0[0], p1[1]-p0[1])))

    def _b_basic(self, c, p0, p1, col, sz, b, ang, ts, q):
        cv2.line(c, p0, p1, col, sz, cv2.LINE_AA)

    def _b_eraser(self, c, p0, p1, col, sz, b, ang, ts, q):
        cv2.line(c, p0, p1, (255,255,255), sz, cv2.LINE_AA)

    def _b_calligraphy(self, c, p0, p1, col, sz, b, ang, ts, q):
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        a = math.atan2(dy, dx)
        sx = max(1, int(sz*(0.5+0.5*abs(math.cos(a)))))
        sy = max(1, int(sz*(0.5+0.5*abs(math.sin(a)))))
        xs, ys = self._interp(p0, p1, max(self._dist(p0,p1)//2, 1))
        for x, y in zip(xs, ys):
            cv2.ellipse(c, (int(x),int(y)), (sx,sy), math.degrees(a), 0, 360, col, -1)

    def _b_spray(self, c, p0, p1, col, sz, b, ang, ts, q):
        d, sc = b.get("density", 0.35), b.get("scatter", 5)
        cnt   = max(int(self._dist(p0,p1)*d*3), 4)
        ts_   = np.random.uniform(0, 1, cnt)
        xs    = (p0[0]+ts_*(p1[0]-p0[0]) + np.random.randint(-sc,sc+1,cnt)).astype(int)
        ys    = (p0[1]+ts_*(p1[1]-p0[1]) + np.random.randint(-sc,sc+1,cnt)).astype(int)
        h, w  = c.shape[:2]
        keep  = np.random.random(cnt) < d
        mask  = keep & (xs>=0) & (xs<w) & (ys>=0) & (ys<h)
        if not mask.any(): return
        am   = np.random.uniform(0.4, 1.0, cnt)
        dot  = max(1, sz // 4)
        if dot == 1:
            # Vectorised single-pixel scatter — no Python loop
            pix = (np.array(col, dtype=np.float32) * am[mask, None]).clip(0, 255).astype(np.uint8)
            c[ys[mask], xs[mask]] = pix
        else:
            col_f = np.array(col, dtype=np.float32)
            for x, y, a in zip(xs[mask], ys[mask], am[mask]):
                cv2.circle(c, (x, y), dot, tuple(int(v) for v in (col_f * a).clip(0, 255)), -1)

    def _b_neon(self, c, p0, p1, col, sz, b, ang, ts, q):
        gr, gi = b.get("glow_r",3), b.get("glow_i",0.7)
        for i in range(gr, 0, -1):
            a  = gi/(gr-i+1)
            gc = tuple(int(k*a+255*(1-a)) for k in col)
            cv2.line(c, p0, p1, gc, sz+i*2, cv2.LINE_AA)
        cv2.line(c, p0, p1, col, sz, cv2.LINE_AA)

    def _b_watercolor(self, c, p0, p1, col, sz, b, ang, ts, q):
        bleed = b.get("bleed",0.3)
        xs, ys = self._interp(p0, p1, max(self._dist(p0,p1), 1))
        for x, y in zip(xs, ys):
            for _ in range(max(1, int(bleed*4))):
                bx = x+np.random.randint(-sz//2,sz//2+1)
                by = y+np.random.randint(-sz//2,sz//2+1)
                if 0<=bx<c.shape[1] and 0<=by<c.shape[0]:
                    a = np.random.uniform(0.1, 0.3)
                    cv2.circle(c,(bx,by),max(1,sz//4),tuple(int(k*a) for k in col),-1)
            cv2.circle(c,(x,y),sz//2,tuple(int(k*0.2) for k in col),-1)

    def _b_rainbow(self, c, p0, p1, col, sz, b, ang, ts, q):
        n  = max(self._dist(p0,p1)//2, 1)
        rb = ColorPalette.rainbow(n*2)
        xs, ys = self._interp(p0, p1, n)
        for i, (x, y) in enumerate(zip(xs, ys)):
            ci = int((i + ts*b.get("speed",0.1)*10) % len(rb))
            cv2.circle(c, (int(x),int(y)), sz//2+1, rb[ci], -1)

    def _b_sparkle(self, c, p0, p1, col, sz, b, ang, ts, q):
        cnt = b.get("count",5)
        xs, ys = self._interp(p0, p1, max(self._dist(p0,p1)//3, 1))
        for x, y in zip(xs, ys):
            cv2.circle(c,(int(x),int(y)),sz//2,col,-1)
            for _ in range(cnt):
                a  = np.random.uniform(0, 2*np.pi)
                r  = sz+np.random.randint(2,5)
                sx = int(x+r*math.cos(a)); sy = int(y+r*math.sin(a))
                if 0<=sx<c.shape[1] and 0<=sy<c.shape[0]:
                    cv2.circle(c,(sx,sy),1,(255,255,255),-1)

    def _b_fire(self, c, p0, p1, col, sz, b, ang, ts, q):
        fh, pc = b.get("fh",10), b.get("pc",8)
        xs, ys = self._interp(p0, p1, max(self._dist(p0,p1)//2, 1))
        for x, y in zip(xs, ys):
            for _ in range(pc):
                fx = x+np.random.randint(-fh//2,fh//2+1)
                fy = y-abs(int(np.random.normal(0, fh//3)))
                if 0<=fx<c.shape[1] and 0<=fy<c.shape[0]:
                    ci = min(len(self._FIRE)-1, int(abs(fy-y)/max(fh,1)*len(self._FIRE)))
                    f  = np.random.uniform(0.7, 1.3)
                    cv2.circle(c,(fx,fy),max(1,sz//4),
                               tuple(min(255,int(k*f)) for k in self._FIRE[ci]),-1)

    def _b_ice(self, c, p0, p1, col, sz, b, ang, ts, q):
        cs, fr = b.get("cs",3), b.get("frost",0.5)
        xs, ys = self._interp(p0, p1, max(self._dist(p0,p1)//2, 1))
        for x, y in zip(xs, ys):
            for a in range(0, 360, 45):
                r  = math.radians(a)
                cx2 = int(x+cs*math.cos(r)); cy2 = int(y+cs*math.sin(r))
                cv2.line(c,(int(x),int(y)),(cx2,cy2),
                         self._ICE[np.random.randint(0,len(self._ICE))],1)
            if np.random.random() < fr:
                fx = x+np.random.randint(-sz,sz+1); fy = y+np.random.randint(-sz,sz+1)
                if 0<=fx<c.shape[1] and 0<=fy<c.shape[0]:
                    cv2.circle(c,(fx,fy),1,(255,255,255),-1)

    def _b_galaxy(self, c, p0, p1, col, sz, b, ang, ts, q):
        GC = [(128,0,128),(75,0,130),(0,0,255),(255,255,255),(255,215,0)]
        xs, ys = self._interp(p0, p1, max(self._dist(p0,p1), 1))
        for x, y in zip(xs, ys):
            for _ in range(3):
                nx = x+np.random.randint(-sz,sz+1); ny = y+np.random.randint(-sz,sz+1)
                if 0<=nx<c.shape[1] and 0<=ny<c.shape[0]:
                    a = np.random.uniform(0.1, 0.3)
                    cv2.circle(c,(nx,ny),sz//2,
                               tuple(int(k*a) for k in GC[np.random.randint(0,len(GC)-1)]),-1)
            if np.random.random() < b.get("sd",0.3):
                sx = x+np.random.randint(-sz*2,sz*2+1)
                sy = y+np.random.randint(-sz*2,sz*2+1)
                if 0<=sx<c.shape[1] and 0<=sy<c.shape[0]:
                    cv2.circle(c,(sx,sy),np.random.randint(1,3),(255,255,255),-1)

    def _b_glitter(self, c, p0, p1, col, sz, b, ang, ts, q):
        gc, gs = b.get("gc",8), b.get("gs",2)
        xs, ys = self._interp(p0, p1, max(self._dist(p0,p1)//2, 1))
        for x, y in zip(xs, ys):
            for _ in range(gc):
                a  = np.random.uniform(0, 2*np.pi)
                r  = np.random.uniform(0, sz)
                gx = int(x+r*math.cos(a)); gy = int(y+r*math.sin(a))
                if 0<=gx<c.shape[1] and 0<=gy<c.shape[0]:
                    f = np.random.uniform(1.2, 1.5)
                    cv2.circle(c,(gx,gy),gs,tuple(min(255,int(k*f)) for k in col),-1)

    def _b_shadow(self, c, p0, p1, col, sz, b, ang, ts, q):
        off, blur = b.get("off",(5,5)), b.get("blur",3)
        for i in range(blur):
            cv2.line(c,(p0[0]+off[0]+i,p0[1]+off[1]+i),(p1[0]+off[0]+i,p1[1]+off[1]+i),
                     tuple(int(k*0.25/(i+1)) for k in (0,0,0)),sz+i,cv2.LINE_AA)
        cv2.line(c, p0, p1, col, sz, cv2.LINE_AA)

    def _b_gradient(self, c, p0, p1, col, sz, b, ang, ts, q):
        n    = max(self._dist(p0,p1), 1)
        grad = ColorPalette.gradient(b.get("c1",(255,0,0)), b.get("c2",(0,0,255)), n)
        xs, ys = self._interp(p0, p1, n)
        for i, (x, y) in enumerate(zip(xs, ys)):
            cv2.circle(c,(int(x),int(y)),sz//2+1,grad[i],-1)

    def _b_dotted(self, c, p0, p1, col, sz, b, ang, ts, q):
        sp = b.get("spacing",5)
        xs, ys = self._interp(p0, p1, max(self._dist(p0,p1)//sp, 1))
        for x, y in zip(xs, ys):
            cv2.circle(c,(int(x),int(y)),sz//2,col,-1)

    def _b_zigzag(self, c, p0, p1, col, sz, b, ang, ts, q):
        amp, freq = b.get("amp",10), b.get("freq",0.5)
        n = max(self._dist(p0,p1), 1)
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        pts = []
        for i in range(n+1):
            t = i/n
            x = p0[0]+t*dx; y = p0[1]+t*dy
            if dx!=0 or dy!=0:
                norm = math.hypot(dx, dy)
                off  = amp*math.sin(t*math.pi*2/freq)
                x += int(-dy/norm*off); y += int(dx/norm*off)
            pts.append((int(x),int(y)))
        for i in range(1, len(pts)):
            cv2.line(c, pts[i-1], pts[i], col, sz, cv2.LINE_AA)


class CanvasManager:
    def __init__(self, w, h, max_undo=50):
        self.w, self.h = w, h
        self._canvas  = np.full((h,w,3), 255, np.uint8)
        self._undo:   deque = deque(maxlen=max_undo)
        self._redo:   deque = deque(maxlen=max_undo)
        self.strokes: List[BrushStroke] = []
        self.dirty    = True
        self._snap_cache: Optional[np.ndarray] = None
        self._lock    = threading.RLock()

    def snapshot(self):
        with self._lock:
            if self.dirty or self._snap_cache is None:
                self._snap_cache = self._canvas.copy()
                self.dirty = False
            return self._snap_cache

    def draw(self, p0, p1, color, brush_engine, btype, pressure, angle, ts, quality):
        with self._lock:
            brush_engine.apply(self._canvas, p0, p1, color, btype, pressure, angle, ts, quality)
            self.dirty = True

    def finish_stroke(self, stroke: BrushStroke):
        with self._lock:
            self.strokes.append(stroke)
            self._undo.append(self._canvas.copy())
            self._redo.clear()

    def undo(self):
        with self._lock:
            if not self._undo: return False
            self._redo.append(self._canvas.copy())
            self._canvas = self._undo.pop()
            self.dirty = True
            return True

    def redo(self):
        with self._lock:
            if not self._redo: return False
            self._undo.append(self._canvas.copy())
            self._canvas = self._redo.pop()
            self.dirty = True
            return True

    def clear(self):
        with self._lock:
            self._canvas.fill(255)
            self.strokes.clear()
            self._undo.clear()
            self._redo.clear()
            self.dirty = True

    def save(self, path):
        with self._lock:
            try:
                cv2.imwrite(path, self._canvas)
                return True
            except Exception as e:
                log.error(f"Save failed: {e}")
                return False

    def flood_fill(self, pos, color):
        with self._lock:
            mask = np.zeros((self.h+2, self.w+2), np.uint8)
            cv2.floodFill(self._canvas, mask, pos, color, (10,10,10), (10,10,10),
                          cv2.FLOODFILL_FIXED_RANGE)
            self.dirty = True


MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/"
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"


def _ensure_model():
    if os.path.exists(MODEL_PATH): return
    log.info("Downloading hand_landmarker model (~9 MB)")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        log.info("Model ready.")
    except (urllib.error.URLError, OSError) as e:
        raise MediaPipeError(f"Could not download model: {e}") from e


class _LandmarkWrapper:
    def __init__(self, lms): self._lms = lms
    def __getitem__(self, i): return self._lms[i]


class GestureWorker:
    FINGER = {"thumb":4,"index":8,"middle":12,"ring":16,"pinky":20}

    def __init__(self, max_hands=1, det_conf=0.65, track_conf=0.45, scale=0.6):
        _ensure_model()
        self._scale      = float(scale)
        self._frame_wh   = (1, 1)
        self._frame_lock = threading.Lock()
        self._start      = time.monotonic()
        self._last_ts_ms = -1
        self._ts_lock    = threading.Lock()
        self._out_q: Queue = Queue(maxsize=1)
        opts = HandLandmarkerOptions(
            base_options=base_opts.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.LIVE_STREAM,
            num_hands=max_hands,
            min_hand_detection_confidence=det_conf,
            min_hand_presence_confidence=track_conf,
            min_tracking_confidence=track_conf,
            result_callback=self._on_result,
        )
        self._detector = HandLandmarker.create_from_options(opts)
        log.info("GestureWorker ready")

    def submit(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        with self._frame_lock:
            self._frame_wh = (w, h)
        small = _gpu_resize(frame, (max(1,int(w*self._scale)), max(1,int(h*self._scale))),
                            cv2.INTER_LINEAR) if self._scale != 1.0 else frame
        rgb      = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with self._ts_lock:
            ts_ms = int((time.monotonic()-self._start)*1000)
            if ts_ms <= self._last_ts_ms: ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms
        try:
            self._detector.detect_async(mp_image, ts_ms)
        except Exception as e:
            log.debug(f"detect_async: {e}")

    def get_result(self) -> Optional[HandResult]:
        try: return self._out_q.get_nowait()
        except Empty: return None

    def stop(self):
        try: self._detector.close()
        except Exception: pass

    def _on_result(self, result, image, ts_ms):
        try:
            hr = self._parse(result)
        except Exception as e:
            log.debug(f"result parse error: {e}"); return
        try:
            self._out_q.put_nowait(hr)
        except Full:
            try: self._out_q.get_nowait()
            except Empty: pass
            self._out_q.put_nowait(hr)

    def _parse(self, result) -> HandResult:
        if not result.hand_landmarks:
            return HandResult("none", (0,0), 1.0, 0.0, False)
        hl = _LandmarkWrapper(result.hand_landmarks[0])
        with self._frame_lock:
            w, h = self._frame_wh
        return HandResult(
            gesture  = self._recognise(hl),
            pos      = (int(np.clip(hl[8].x*w, 0, w-1)), int(np.clip(hl[8].y*h, 0, h-1))),
            pressure = float(np.clip(1.5 - math.hypot(hl[8].x-hl[4].x, hl[8].y-hl[4].y)*3, 0.5, 1.5)),
            angle    = math.atan2(hl[8].y-hl[7].y, hl[8].x-hl[7].x),
            detected = True,
        )

    def _is_up(self, hl, idx):
        try: return hl[idx].y < hl[idx-2].y
        except Exception: return False

    def _recognise(self, hl):
        state = {n: self._is_up(hl, i) for n, i in self.FINGER.items() if n != "thumb"}
        state["thumb"] = hl[4].x < hl[3].x
        up = [k for k, v in state.items() if v]
        n  = len(up)
        if n == 1 and state.get("index"):                                   return "draw"
        if n == 2 and state.get("index") and state.get("middle"):           return "hover"
        if n == 0:                                                           return "erase"
        if n == 5:                                                           return "clear"
        if n == 2 and state.get("index") and state.get("pinky"):            return "color"
        if n == 3 and all(state.get(f) for f in ["index","middle","ring"]): return "shape"
        if n == 2 and state.get("thumb") and state.get("index"):            return "pan"
        if n == 4 and not state.get("thumb"):                               return "fill"
        return "unknown"


class QualityGovernor:
    def __init__(self):
        self._hist  = deque(maxlen=30)
        self._level = 2
        self._last_drop = 0.0

    def tick(self, fps):
        self._hist.append(fps)
        if len(self._hist) < 10: return self._level
        avg = sum(self._hist)/len(self._hist)
        now = time.monotonic()
        # Re-check runtime GPU state each tick — _cuda_enabled is set False on failure
        gpu_now = _cuda_enabled
        drop_thresholds = [(12, 1), (7, 0)] if gpu_now else [(20, 1), (12, 0)]
        recover_fps     = 20 if gpu_now else 25
        for threshold, level in drop_thresholds:
            if avg < threshold and self._level > level:
                self._level = level; self._last_drop = now
                log.info(f"Quality → {level} (FPS={avg:.1f})")
                break
        else:
            if self._level < 2 and now-self._last_drop > 3.0 and avg > recover_fps:
                self._level += 1
        return self._level


def _bgr_to_qimage(frame: np.ndarray) -> QImage:
    h, w, ch = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return QImage(rgb.data, w, h, w*ch, QImage.Format_RGB888).copy()


def _ui_font(size: int = 12, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    f = QFont()
    f.setFamilies(["Inter", "SF Pro Text", "Segoe UI", "Helvetica Neue", "Arial"])
    f.setPixelSize(size)
    f.setWeight(weight)
    f.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    f.setStyleStrategy(
        QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality
    )
    return f


class CanvasWidget(QWidget):

    _MODE_COLORS = {
        "draw":  ((100, 220, 100), (80, 200, 80)),
        "hover": ((100, 160, 255), (80, 140, 240)),
        "erase": ((240, 80,  80),  (220, 60, 60)),
        "fill":  ((80,  220, 180), (60, 200, 160)),
        "clear": ((255, 180, 50),  (240, 160, 30)),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._overlay: Optional[OverlayState] = None
        self._ox = self._oy = 0
        self._sx = self._sy = 1.0
        self.setMinimumSize(800, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.BlankCursor)
        self.setAutoFillBackground(False)

    def set_frame(self, frame: np.ndarray):
        self._pixmap = QPixmap.fromImage(_bgr_to_qimage(frame))
        self.update()

    def set_overlay(self, overlay: Optional[OverlayState]):
        self._overlay = overlay
        self.update()

    def _compute_transform(self, pixmap: QPixmap) -> tuple:
        fw, fh = pixmap.width(), pixmap.height()
        ww, wh = self.width(), self.height()
        scale  = min(ww / fw, wh / fh)
        sw, sh = fw * scale, fh * scale
        ox = (ww - sw) / 2
        oy = (wh - sh) / 2
        return ox, oy, scale, scale

    def _ft(self, pt: Tuple[int, int]) -> Tuple[float, float]:
        return self._ox + pt[0] * self._sx, self._oy + pt[1] * self._sy

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHints(
            QPainter.Antialiasing |
            QPainter.SmoothPixmapTransform |
            QPainter.TextAntialiasing
        )

        p.fillRect(self.rect(), QColor(8, 8, 20))

        if self._pixmap:
            ox, oy, sx, sy = self._compute_transform(self._pixmap)
            self._ox, self._oy, self._sx, self._sy = ox, oy, sx, sy
            sw = self._pixmap.width()  * sx
            sh = self._pixmap.height() * sy
            p.drawPixmap(int(ox), int(oy), int(sw), int(sh), self._pixmap)

            p.setPen(QPen(QColor(60, 50, 120, 120), 1.0))
            p.setBrush(Qt.NoBrush)
            p.drawRect(int(ox), int(oy), int(sw) - 1, int(sh) - 1)

            if self._overlay and self._overlay.detected:
                self._paint_trail(p)
                self._paint_cursor(p)
                self._paint_gesture_badge(p, int(ox), int(oy + sh))
        else:
            self._paint_placeholder(p)

        p.end()

    def _paint_cursor(self, p: QPainter):
        ov  = self._overlay
        cx, cy = self._ft(ov.cursor_pos)
        ring, fill = self._MODE_COLORS.get(ov.mode, ((180, 180, 180), (140, 140, 140)))

        sz   = max(4.0, ov.brush_size * ov.pressure * self._sx)
        tick = time.monotonic()
        anim = sz + 4 + 2.5 * math.sin(tick * 7)

        p.setPen(QPen(QColor(*ring, 50), 1.2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(cx - anim, cy - anim, anim * 2, anim * 2)

        p.setPen(QPen(QColor(*ring, 220), 1.5))
        p.drawEllipse(cx - sz, cy - sz, sz * 2, sz * 2)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(*fill, 200))
        dot = max(2.5, sz / 3)
        p.drawEllipse(cx - dot, cy - dot, dot * 2, dot * 2)

        if ov.raw_pos:
            rx, ry = self._ft(ov.raw_pos)
            p.setBrush(QColor(0, 220, 255, 180))
            p.setPen(QPen(QColor(255, 255, 255, 160), 1.0))
            p.drawEllipse(rx - 4, ry - 4, 8, 8)

    def _paint_trail(self, p: QPainter):
        ov = self._overlay
        if len(ov.trail) < 2:
            return
        br, bg, bb = ov.brush_color
        n = len(ov.trail)
        for i in range(1, n):
            alpha = int(255 * i / n)
            width = max(1.0, 1.8 * i / n)
            color = QColor(bb, bg, br, alpha)
            p.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            x0, y0 = self._ft(ov.trail[i - 1])
            x1, y1 = self._ft(ov.trail[i])
            p.drawLine(x0, y0, x1, y1)

    def _paint_gesture_badge(self, p: QPainter, ox: int, bottom: int):
        ov = self._overlay
        ring, _ = self._MODE_COLORS.get(ov.mode, ((180, 180, 180), (140, 140, 140)))
        text = f"{ov.gesture}   p {ov.pressure:.2f}"

        font = _ui_font(13, QFont.Weight.Medium)
        p.setFont(font)

        fm  = p.fontMetrics()
        tw  = fm.horizontalAdvance(text)
        th  = fm.height()
        pad = 7
        bx  = ox + 10
        by  = bottom - th - pad * 2 - 8
        bw  = tw + pad * 2
        bh  = th + pad * 2

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(10, 10, 24, 200))
        p.drawRoundedRect(bx - 1, by - 1, bw + 2, bh + 2, 6, 6)

        p.setBrush(QColor(*ring, 200))
        p.drawRoundedRect(bx - 1, by - 1, 3, bh + 2, 1, 1)

        p.setPen(QColor(*ring))
        p.drawText(bx + pad, by + pad + fm.ascent(), text)

    def _paint_placeholder(self, p: QPainter):
        font = _ui_font(15, QFont.Weight.Normal)
        p.setFont(font)
        p.setPen(QColor(55, 55, 100))
        p.drawText(self.rect(), Qt.AlignCenter, "Camera initializing…")


class SwatchButton(QPushButton):
    SIZE = 36

    def __init__(self, color: tuple, parent=None):
        super().__init__(parent)
        self._color   = color
        self._selected = False
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setToolTip(f"RGB({color[2]},{color[1]},{color[0]})")
        self._refresh()

    def _refresh(self):
        r, g, b    = self._color[2], self._color[1], self._color[0]
        hex_color  = QColor(r, g, b).name()
        if self._selected:
            self.setStyleSheet(
                f"QPushButton {{ background:{hex_color}; border:2px solid #a5a4ff;"
                f" border-radius:{self.SIZE//2}px; }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{ background:{hex_color}; border:2px solid #1e1e36;"
                f" border-radius:{self.SIZE//2}px; }}"
                f"QPushButton:hover {{ border:2px solid #5046e5; }}"
            )

    def set_color(self, color: tuple):
        self._color = color
        self.setToolTip(f"RGB({color[2]},{color[1]},{color[0]})")
        self._refresh()

    def set_selected(self, sel: bool):
        self._selected = sel
        self._refresh()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._selected:
            p   = QPainter(self)
            p.setRenderHint(QPainter.Antialiasing)
            r, g, b = self._color[2], self._color[1], self._color[0]
            lum = 0.299*r + 0.587*g + 0.114*b
            tick_color = QColor(0, 0, 0) if lum > 140 else QColor(255, 255, 255)
            pen = QPen(tick_color, 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            cx, cy = self.width()//2, self.height()//2
            p.drawLine(cx-5, cy, cx-1, cy+4)
            p.drawLine(cx-1, cy+4, cx+5, cy-4)
            p.end()


APP_FONT_SIZE = 12

DARK_STYLE = """
QMainWindow, QDialog { background: #0b0b14; color: #ddddf0; }
QWidget { background: transparent; color: #ddddf0; font-size: 12px; }

QMenuBar {
    background: #0f0f1c;
    border-bottom: 1px solid #1e1e36;
    padding: 2px 4px;
    spacing: 2px;
    font-size: 12px;
}
QMenuBar::item { background: transparent; padding: 5px 12px; border-radius: 4px; }
QMenuBar::item:selected { background: #1e1e36; }
QMenuBar::item:pressed  { background: #252545; }

QMenu {
    background: #131325;
    border: 1px solid #252545;
    border-radius: 6px;
    padding: 4px 0;
    font-size: 12px;
}
QMenu::item { padding: 8px 24px 8px 14px; border-radius: 4px; margin: 1px 4px; }
QMenu::item:selected { background: #1f1f42; color: #a5a4ff; }
QMenu::separator { height: 1px; background: #1e1e36; margin: 4px 10px; }

QToolBar {
    background: #0f0f1c;
    border-bottom: 1px solid #1e1e36;
    spacing: 2px;
    padding: 4px 8px;
    font-size: 12px;
}
QToolBar::separator { width: 1px; background: #252545; margin: 4px 6px; }

QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 5px 14px;
    color: #9090b8;
    font-size: 12px;
}
QToolButton:hover   { background: #1e1e36; border-color: #252545; color: #ddddf0; }
QToolButton:pressed { background: #252545; }
QToolButton:checked { background: #1a1a42; border-color: #5046e5; color: #a5a4ff; }

QPushButton {
    background: #141428;
    border: 1px solid #252545;
    border-radius: 5px;
    padding: 6px 14px;
    color: #b0b0d0;
    font-size: 12px;
}
QPushButton:hover   { background: #1c1c38; border-color: #353565; color: #ddddf0; }
QPushButton:pressed { background: #0e0e22; }
QPushButton:checked { background: #1a1a42; border-color: #5046e5; color: #a5a4ff; }
QPushButton:disabled { color: #404060; border-color: #1a1a28; }

QPushButton[accent="true"] {
    background: #2d2880;
    border: 1px solid #5046e5;
    color: #c4c2ff;
    border-radius: 5px;
    padding: 6px 14px;
    font-size: 12px;
}
QPushButton[accent="true"]:hover   { background: #3730a3; border-color: #6457f5; }
QPushButton[accent="true"]:pressed { background: #1e1a60; }

QPushButton[danger="true"] {
    background: #2a0f0f;
    border: 1px solid #7f1d1d;
    color: #fca5a5;
    font-size: 12px;
}
QPushButton[danger="true"]:hover { background: #3b1111; border-color: #991b1b; }

QPushButton[brush="true"] {
    background: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0;
    text-align: left;
    padding: 8px 12px 8px 10px;
    color: #7878a0;
    font-size: 12px;
}
QPushButton[brush="true"]:hover {
    background: #141428;
    color: #ddddf0;
    border-left-color: #353560;
}
QPushButton[brush="true"]:checked {
    background: #1a1a3a;
    color: #a5a4ff;
    border-left: 3px solid #5046e5;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #1e1e36;
    border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #5046e5;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    background: #a5a4ff;
    border-radius: 7px;
    margin: -5px 0;
    border: 2px solid #5046e5;
}
QSlider::handle:horizontal:hover { background: #c4c3ff; }

QSpinBox {
    background: #131325;
    border: 1px solid #252545;
    border-radius: 5px;
    padding: 4px 6px;
    color: #ddddf0;
    font-size: 12px;
}
QSpinBox:hover { border-color: #353565; }
QSpinBox::up-button, QSpinBox::down-button {
    background: #1e1e36;
    border: none;
    width: 16px;
    border-radius: 3px;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: #252550; }

QGroupBox {
    background: #0f0f1c;
    border: 1px solid #1e1e36;
    border-radius: 8px;
    margin-top: 20px;
    padding: 12px 8px 8px 8px;
    font-size: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    top: -1px;
    color: #555588;
    font-size: 10px;
    letter-spacing: 1.2px;
    padding: 0 6px;
    background: #0f0f1c;
}

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: #0f0f1c;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #252545;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #353565; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QLabel { color: #b0b0d0; background: transparent; font-size: 12px; }
QLabel[heading="true"] { color: #ddddf0; font-size: 14px; font-weight: bold; }
QLabel[muted="true"]   { color: #505078; font-size: 10px; }

QStatusBar {
    background: #080810;
    border-top: 1px solid #1a1a2e;
    color: #6060a0;
    padding: 0 8px;
    font-size: 11px;
}
QStatusBar::item { border: none; }

QFrame[frameShape="4"], QFrame[frameShape="5"] { color: #1e1e36; }

QFileDialog { background: #131325; color: #ddddf0; }
"""


class AirDrawWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state     = DrawState(gpu_active=_CUDA_OK or _CUPY_OK)
        self.pal       = ColorPalette()
        self.brush_eng = BrushEngine()
        self.gov       = QualityGovernor()
        self.kalman    = KalmanCursor()
        self.particles = VectorParticleSystem()
        self.trail:    deque = deque(maxlen=10)
        self.canvas:   Optional[CanvasManager] = None
        self.gesture:  Optional[GestureWorker] = None
        self.cap:      Optional[cv2.VideoCapture] = None

        self.prev_pt     = None
        self.cur_stroke: List[Tuple[int,int]] = []
        self.is_drawing  = False
        self.last_result: Optional[HandResult] = None
        self.frames_since_result = 0

        # Gesture debounce — prevents fill/clear firing on a single mis-classified frame
        self._last_gesture   = ""
        self._gesture_streak = 0
        self._gesture_fired  = False   # True once a one-shot gesture has acted

        # Palette cooldown — after any palette selection (UI click or gesture),
        # the "color" gesture cannot cycle the palette again for this many
        # seconds.  Prevents the camera picking up the clicking hand and
        # immediately advancing to the next palette.
        self._last_palette_change = 0.0
        self._PALETTE_COOLDOWN    = 1.5

        self._raw_q: Queue = Queue(maxsize=2)
        self._running = False
        self._cap_thread: Optional[threading.Thread] = None
        self._fps_times: deque = deque(maxlen=60)
        self._last_frame_t = time.perf_counter()
        self._swatch_buttons: List[SwatchButton] = []
        self._theme_btns:    dict = {}

        self._build_ui()
        self.setStyleSheet(DARK_STYLE)
        self.setWindowTitle("Air Drawing System  ·  Ultra Pro v3")
        self.setMinimumSize(900, 600)
        self.showMaximized()
        atexit.register(self._cleanup)

    def _build_ui(self):
        self._build_menu()
        self._build_top_toolbar()

        central = QWidget()
        central.setStyleSheet("background: #0b0b14;")
        self.setCentralWidget(central)

        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._left_panel = self._build_brush_panel()
        root.addWidget(self._left_panel)
        self._left_sep = self._make_vline()
        root.addWidget(self._left_sep)

        self._canvas_w = CanvasWidget()
        root.addWidget(self._canvas_w, stretch=1)

        self._right_sep = self._make_vline()
        root.addWidget(self._right_sep)
        self._right_panel = self._build_side_panel()
        root.addWidget(self._right_panel)

        self._status = QStatusBar()
        self._status.setSizeGripEnabled(False)
        self.setStatusBar(self._status)

        self._lbl_fps     = QLabel("FPS  —")
        self._lbl_mode    = QLabel("MODE  —")
        self._lbl_strokes = QLabel("0 strokes")
        self._sep_lbl     = QLabel("  ·  ")
        self._sep_lbl2    = QLabel("  ·  ")
        for w in [self._lbl_fps, self._sep_lbl, self._lbl_mode,
                  self._sep_lbl2, self._lbl_strokes]:
            self._status.addWidget(w)
            w.setStyleSheet("color:#3a3a60; font-size:11px;")

        gpu_active = _CUDA_OK or _CUPY_OK
        self._lbl_gpu = QLabel("  GPU ✓  " if gpu_active else "  CPU  ")
        self._lbl_gpu.setStyleSheet(
            ("color:#22c55e;" if gpu_active else "color:#404060;") +
            "background:#0f0f1c; border-left:1px solid #1a1a2e; padding:0 10px;"
            "font-size:11px;"
        )
        self._status.addPermanentWidget(self._lbl_gpu)

    @staticmethod
    def _make_vline() -> QFrame:
        f = QFrame()
        f.setFrameShape(QFrame.VLine)
        f.setStyleSheet("color: #15152a; max-width:1px;")
        return f

    def _build_menu(self):
        mb = self.menuBar()

        file_m = mb.addMenu("File")
        for label, shortcut, slot in [
            ("Save Canvas", "S", self._save_canvas),
            ("Quit",        "Q", self.close),
        ]:
            a = QAction(label, self, shortcut=QKeySequence(shortcut))
            a.triggered.connect(slot)
            if label == "Save Canvas":
                file_m.addAction(a)
                file_m.addSeparator()
            else:
                file_m.addAction(a)

        edit_m = mb.addMenu("Edit")
        for label, shortcut, slot in [
            ("Undo",         "Ctrl+Z", lambda: self.canvas.undo() if self.canvas else None),
            ("Redo",         "Ctrl+Y", lambda: self.canvas.redo() if self.canvas else None),
            ("Clear Canvas", "C",      self._clear_canvas),
        ]:
            a = QAction(label, self, shortcut=QKeySequence(shortcut))
            a.triggered.connect(slot)
            edit_m.addAction(a)

    def _build_top_toolbar(self):
        tb = QToolBar("Actions")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(tb)

        for label, tip, shortcut, slot, danger in [
            ("Save",  "Save artwork",  "S",      self._save_canvas,                                    False),
            ("Undo",  "Undo stroke",   "Ctrl+Z", lambda: self.canvas.undo() if self.canvas else None,  False),
            ("Redo",  "Redo stroke",   "Ctrl+Y", lambda: self.canvas.redo() if self.canvas else None,  False),
            ("Clear", "Clear canvas",  "C",      self._clear_canvas,                                   True),
        ]:
            btn = QToolButton()
            btn.setText(label)
            btn.setToolTip(f"{tip}  [{shortcut}]")
            btn.clicked.connect(slot)
            if danger:
                tb.addSeparator()
                btn.setStyleSheet(
                    "QToolButton { color:#f87171; }"
                    "QToolButton:hover { color:#fca5a5; background:#2a0f0f;"
                    " border:1px solid #7f1d1d; border-radius:5px; }"
                )
            tb.addWidget(btn)

        tb.addSeparator()
        panel_toggle_style = (
            "QToolButton { color:#404068; padding:0 6px; }"
            "QToolButton:hover { color:#a5a4ff; background:#1a1a36;"
            " border:1px solid #303060; border-radius:5px; }"
            "QToolButton:checked { color:#a5a4ff; background:#141432; }"
        )
        self._left_toggle = QToolButton()
        self._left_toggle.setText("‹ Brushes")
        self._left_toggle.setToolTip("Toggle brush panel  [[]")
        self._left_toggle.setCheckable(True)
        self._left_toggle.setChecked(True)
        self._left_toggle.setStyleSheet(panel_toggle_style)
        self._left_toggle.clicked.connect(self._toggle_left_panel)
        tb.addWidget(self._left_toggle)

        self._right_toggle = QToolButton()
        self._right_toggle.setText("Colors ›")
        self._right_toggle.setToolTip("Toggle colour panel  []]")
        self._right_toggle.setCheckable(True)
        self._right_toggle.setChecked(True)
        self._right_toggle.setStyleSheet(panel_toggle_style)
        self._right_toggle.clicked.connect(self._toggle_right_panel)
        tb.addWidget(self._right_toggle)

    def _build_brush_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(155)
        panel.setStyleSheet("background: #0f0f1c;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 10, 0, 10)
        layout.setSpacing(0)

        hdr = QLabel("  BRUSHES")
        hdr.setFixedHeight(26)
        hdr.setStyleSheet(
            "font-size:10px; letter-spacing:1.2px; color:#303060;"
            " padding:0 0 4px 10px; background:transparent;"
        )
        layout.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setFrameShape(QFrame.NoFrame)

        brush_container = QWidget()
        brush_container.setStyleSheet("background: transparent;")
        bl = QVBoxLayout(brush_container)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(0)

        self._brush_btns: Dict[str, QPushButton] = {}
        for i, bt in enumerate(BRUSH_KEYS):
            key_hint = str(i + 1) if i < 9 else ("0" if i == 9 else "")
            label = f"  {BRUSH_LABELS[bt.value]}"
            if key_hint:
                label = f"  {BRUSH_LABELS[bt.value]}  [{key_hint}]"
            btn = QPushButton(label)
            btn.setProperty("brush", "true")
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            if bt.value == self.state.active_brush:
                btn.setChecked(True)
            btn.clicked.connect(lambda _, b=bt.value, button=btn: self._select_brush(b, button))
            bl.addWidget(btn)
            self._brush_btns[bt.value] = btn

        bl.addStretch()
        scroll.setWidget(brush_container)
        layout.addWidget(scroll, stretch=1)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#15152a; max-height:1px; margin:0;")
        layout.addWidget(sep)

        size_area = QWidget()
        size_area.setStyleSheet("background:transparent;")
        sa = QVBoxLayout(size_area)
        sa.setContentsMargins(12, 10, 12, 4)
        sa.setSpacing(6)

        size_row = QHBoxLayout()
        size_hdr = QLabel("SIZE")
        size_hdr.setStyleSheet(
            "font-size:10px; letter-spacing:1.2px; color:#303060; background:transparent;"
        )
        self._size_val_lbl = QLabel(str(self.state.brush_size))
        self._size_val_lbl.setStyleSheet(
            "color:#8880f8; font-weight:bold; font-size:12px; background:transparent;"
        )
        self._size_val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        size_row.addWidget(size_hdr)
        size_row.addStretch()
        size_row.addWidget(self._size_val_lbl)
        sa.addLayout(size_row)

        self._size_slider = QSlider(Qt.Horizontal)
        self._size_slider.setRange(1, 60)
        self._size_slider.setValue(self.state.brush_size)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        sa.addWidget(self._size_slider)
        layout.addWidget(size_area)
        return panel

    def _build_side_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(200)
        panel.setStyleSheet("background: #0f0f1c;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        color_lbl_row = QHBoxLayout()
        color_hdr = QLabel("CURRENT COLOR")
        color_hdr.setStyleSheet(
            "font-size:10px; letter-spacing:1.2px; color:#303060; background:transparent;"
        )
        color_lbl_row.addWidget(color_hdr)
        color_lbl_row.addStretch()
        layout.addLayout(color_lbl_row)

        self._color_preview = QFrame()
        self._color_preview.setFixedHeight(44)
        self._color_preview.setFrameShape(QFrame.NoFrame)
        self._refresh_color_preview()
        layout.addWidget(self._color_preview)

        pal_card = QGroupBox("COLOR THEME")
        pal_v = QVBoxLayout(pal_card)
        pal_v.setSpacing(8)

        theme_grid = QGridLayout()
        theme_grid.setSpacing(4)
        theme_btn_style = (
            "QPushButton{background:#111128;border:1px solid #202048;border-radius:4px;"
            "color:#6060a0;font-size:10px;letter-spacing:0.6px;padding:3px 2px;}"
            "QPushButton:hover{background:#1c1c44;color:#a5a4ff;border-color:#4040a0;}"
            "QPushButton:checked{background:#1a1a42;border:1px solid #5046e5;"
            "color:#a5a4ff;font-weight:600;}"
        )
        for i, name in enumerate(ColorPalette.PALETTES):
            btn = QPushButton(name.capitalize())
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.setStyleSheet(theme_btn_style)
            btn.setChecked(name == self.state.palette_name)
            btn.clicked.connect(lambda _, n=name: self._select_palette(n))
            theme_grid.addWidget(btn, i // 3, i % 3)
            self._theme_btns[name] = btn
        pal_v.addLayout(theme_grid)

        self._swatch_grid = QGridLayout()
        self._swatch_grid.setSpacing(6)
        self._swatch_grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        pal_v.addLayout(self._swatch_grid)

        # Pre-create a fixed pool of swatch buttons (max palette size = 10).
        # _rebuild_swatches updates their colours in-place so no widgets are
        # ever destroyed — avoids the main-thread stall that paused drawing.
        _dummy = (0, 0, 0)
        for i in range(10):
            btn = SwatchButton(_dummy)
            btn.setVisible(False)
            btn.clicked.connect(lambda _, b=btn: self._pick_color(b._color))
            self._swatch_grid.addWidget(btn, i // 5, i % 5)
            self._swatch_buttons.append(btn)
        layout.addWidget(pal_card)

        act_card = QGroupBox("ACTIONS")
        act_v = QVBoxLayout(act_card)
        act_v.setSpacing(5)

        save_btn = QPushButton("Save Canvas")
        save_btn.setProperty("accent", "true")
        save_btn.clicked.connect(self._save_canvas)
        act_v.addWidget(save_btn)

        btn_row = QHBoxLayout()
        undo_btn = QPushButton("Undo")
        undo_btn.clicked.connect(lambda: self.canvas.undo() if self.canvas else None)
        redo_btn = QPushButton("Redo")
        redo_btn.clicked.connect(lambda: self.canvas.redo() if self.canvas else None)
        btn_row.addWidget(undo_btn)
        btn_row.addWidget(redo_btn)
        act_v.addLayout(btn_row)

        clear_btn = QPushButton("Clear Canvas")
        clear_btn.setProperty("danger", "true")
        clear_btn.clicked.connect(self._clear_canvas)
        act_v.addWidget(clear_btn)

        layout.addWidget(act_card)

        gest_card = QGroupBox("GESTURES")
        g_v = QVBoxLayout(gest_card)
        g_v.setSpacing(5)
        for icon, trigger, action in [
            ("☝",    "1 finger",    "Draw"),
            ("✌",    "2 fingers",   "Hover"),
            ("✊",    "Fist",        "Erase"),
            ("🖐",    "5 fingers",   "Clear"),
            ("☝🤙",  "Index+Pinky", "Palette"),
            ("✋",    "4 fingers",   "Fill"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(6)
            ic = QLabel(icon)
            ic.setFixedWidth(22)
            ic.setStyleSheet("font-size:14px; background:transparent;")
            trig = QLabel(trigger)
            trig.setStyleSheet("color:#404068; font-size:11px; background:transparent;")
            act_l = QLabel(action)
            act_l.setStyleSheet("color:#606090; font-size:11px; background:transparent;")
            act_l.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(ic)
            row.addWidget(trig)
            row.addStretch()
            row.addWidget(act_l)
            g_v.addLayout(row)

        layout.addWidget(gest_card)
        layout.addStretch()

        self._rebuild_swatches()
        return panel

    def _refresh_color_preview(self):
        r, g, b = self.state.color[2], self.state.color[1], self.state.color[0]
        hex_c   = QColor(r, g, b).name()
        self._color_preview.setStyleSheet(
            f"background:{hex_c}; border-radius:8px; border:1px solid #202040;"
        )
        self._color_preview.setToolTip(f"{hex_c.upper()}  (R{r} G{g} B{b})")

    def _rebuild_swatches(self):
        # Sync theme-selector checked states.
        for name, btn in self._theme_btns.items():
            btn.setChecked(name == self.state.palette_name)

        # Update the pre-created swatch pool in-place — no widget churn,
        # so the main thread is never stalled while drawing is running.
        pal    = self.pal.get_palette(self.state.palette_name)
        colors = list(pal.values())[:10]
        for i, btn in enumerate(self._swatch_buttons):
            if i < len(colors):
                col = colors[i]
                btn.set_color(col)
                btn.set_selected(col == self.state.color)
                btn.setVisible(True)
            else:
                btn.setVisible(False)

    def _pick_color(self, color):
        self.state.color = color
        for btn in self._swatch_buttons:
            btn.set_selected(btn._color == color)
        self._refresh_color_preview()

    def _select_brush(self, btype, checked_btn=None):
        for btn in self._brush_btns.values():
            btn.setChecked(False)
        if checked_btn:
            checked_btn.setChecked(True)
        self.state.active_brush = btype
        sz = self.brush_eng.PRESETS[btype]["size"]
        self.state.brush_size = sz
        self._size_slider.blockSignals(True)
        self._size_slider.setValue(min(sz, self._size_slider.maximum()))
        self._size_slider.blockSignals(False)
        self._size_val_lbl.setText(str(sz))
        if btype == BrushType.ERASER.value:
            self.state.mode = "erase"

    def _on_size_changed(self, val):
        self.brush_eng.PRESETS[self.state.active_brush]["size"] = val
        self.state.brush_size = val
        self._size_val_lbl.setText(str(val))

    def _select_palette(self, name: str):
        self._last_palette_change = time.time()   # start cooldown
        self.state.palette_name   = name
        pal = self.pal.get_palette(name)
        self.state.color = list(pal.values())[0]
        self._rebuild_swatches()
        self._refresh_color_preview()
        if self.canvas:
            self.particles.emit((self.canvas.w//2, self.canvas.h//2), self.state.color, 20, 8)

    def _cycle_palette(self):
        # Block gesture-driven cycling for _PALETTE_COOLDOWN seconds after any
        # selection — prevents the hand used to click a UI button from
        # immediately advancing the palette one step further.
        if time.time() - self._last_palette_change < self._PALETTE_COOLDOWN:
            return
        pals = ColorPalette.PALETTES
        idx  = (pals.index(self.state.palette_name)+1) % len(pals) \
               if self.state.palette_name in pals else 0
        self._select_palette(pals[idx])

    def _save_canvas(self):
        if not self.canvas: return
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path, _ = QFileDialog.getSaveFileName(self, "Save Canvas",
                                              f"airdraw_{ts}.png", "PNG Files (*.png)")
        if path:
            if self.canvas.save(path):
                self._status.showMessage(f"Saved → {path}", 3000)
                self.particles.emit((self.canvas.w//2, self.canvas.h//2), (0,255,0), 40, 15)
            else:
                self._status.showMessage("Save failed", 3000)

    def _clear_canvas(self):
        if not self.canvas: return
        self.canvas.clear()
        w, h = self.canvas.w, self.canvas.h
        for _ in range(30):
            x = np.random.randint(0, w); y = np.random.randint(0, h)
            self.particles.emit((x,y), (255,255,255), 3, 8)

    def start(self):
        if not self._open_camera():
            self._status.showMessage("No camera found", 5000)
            return False
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas = CanvasManager(w, h)
        try:
            self.gesture = GestureWorker()
        except MediaPipeError as e:
            log.error(str(e))
            self._status.showMessage(str(e), 5000)
            return False
        self._running = True
        self._cap_thread = threading.Thread(target=self._capture_loop, daemon=True, name="Capture")
        self._cap_thread.start()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(int(1000/TARGET_FPS))
        log.info("AirDraw started")
        return True

    def _open_camera(self):
        if sys.platform == "darwin":   backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif os.name == "nt":          backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
        else:                          backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
        for be in backends:
            for idx in range(3):
                try:
                    cap = cv2.VideoCapture(idx, be)
                    if not cap.isOpened(): continue
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, fr = cap.read()
                    if ret and fr is not None:
                        log.info(f"Camera {idx}: {int(cap.get(3))}×{int(cap.get(4))}")
                        self.cap = cap; return True
                    cap.release()
                except Exception as e:
                    log.debug(f"Camera {idx}/{be}: {e}")
        return False

    def _capture_loop(self):
        while self._running:
            ret, fr = self.cap.read()
            if not ret or fr is None:
                time.sleep(0.01); continue
            flipped = _gpu_flip(fr)
            try:
                self._raw_q.put_nowait(flipped)
            except Full:
                try: self._raw_q.get_nowait()
                except Empty: pass
                self._raw_q.put_nowait(flipped)
            self.gesture.submit(flipped)

    def _tick(self):
        try:
            frame = self._raw_q.get_nowait()
        except Empty:
            return

        result = self.gesture.get_result()
        if result:
            self.last_result = result; self.frames_since_result = 0
        else:
            self.frames_since_result += 1

        display, overlay = self._compose(frame, self.last_result)
        self._canvas_w.set_frame(display)
        self._canvas_w.set_overlay(overlay)
        self._update_fps()

    def _compose(self, frame: np.ndarray, result: Optional[HandResult]):
        q     = self.state.quality_level
        stale = result is not None and self.frames_since_result > 4
        overlay: Optional[OverlayState] = None

        if result and result.detected and not stale:
            smooth = self.kalman.update(result.pos)
            self.trail.append(smooth)
            self._handle_gesture(result.gesture, smooth, result.pressure, result.angle)
            bt  = self.brush_eng.PRESETS.get(self.state.active_brush, {"size": 8})
            overlay = OverlayState(
                cursor_pos  = smooth,
                raw_pos     = result.pos,
                trail       = list(self.trail),
                mode        = self.state.mode,
                gesture     = result.gesture,
                pressure    = result.pressure,
                brush_color = self.state.color,
                brush_size  = max(4, int(bt["size"] * result.pressure)),
                frame_wh    = (frame.shape[1], frame.shape[0]),
                detected    = True,
            )
        else:
            self._reset_drawing()
            self.trail.clear()
            # Reset gesture debounce so the same gesture can fire again after a
            # tracking dropout — without this, _gesture_fired stays True and
            # fill/clear never trigger a second time unless a different gesture
            # is seen first.
            self._last_gesture   = ""
            self._gesture_streak = 0
            self._gesture_fired  = False

        if q > 0:
            fps_for_dt = self.state.fps if self.state.fps > 0 else TARGET_FPS
            self.particles.update(1.0 / fps_for_dt)
            self.particles.render(frame, q)

        snap  = self.canvas.snapshot()
        frame = _gpu_weighted_add(frame, 0.3, snap, 0.7)

        if q == 2:
            vig   = _get_vignette(frame.shape[0], frame.shape[1])
            frame = (frame.astype(np.float32) * vig).clip(0, 255).astype(np.uint8)

        self.state.stroke_count = len(self.canvas.strokes)
        return frame, overlay

    # Frames a destructive gesture must be held before it fires.
    _GESTURE_DEBOUNCE = 8

    def _handle_gesture(self, gesture, pos, pressure, angle):
        # Track how many consecutive frames this gesture has been seen.
        if gesture == self._last_gesture:
            self._gesture_streak += 1
        else:
            self._last_gesture   = gesture
            self._gesture_streak = 0
            self._gesture_fired  = False   # new gesture → reset one-shot guard

        if gesture == "clear":
            self.state.mode = "clear"
            # Require the gesture to be held for _GESTURE_DEBOUNCE frames and
            # only fire once per onset — prevents accidental full-canvas wipes.
            if self._gesture_streak >= self._GESTURE_DEBOUNCE and not self._gesture_fired:
                self._gesture_fired = True
                self._clear_canvas()
        elif gesture == "color":
            # Fire on the very first frame of a new "color" gesture only.
            if self._gesture_streak == 0:
                self._cycle_palette()
        elif gesture == "erase":
            self.state.mode = "erase"
            self.state.active_brush = BrushType.ERASER.value
            self._do_draw(pos, pressure, angle)
        elif gesture == "hover":
            self._finish_stroke()
            self.state.mode = "hover"; self.prev_pt = None; self.is_drawing = False
        elif gesture == "draw":
            self.state.mode = "draw"
            if self.state.active_brush == BrushType.ERASER.value:
                self.state.active_brush = BrushType.RAINBOW.value
            self._do_draw(pos, pressure, angle)
        elif gesture == "fill":
            self.state.mode = "fill"
            # Same debounce + one-shot protection: fill floods the whole canvas
            # if the background is untouched white, so accidental triggers are
            # especially destructive.
            if self._gesture_streak >= self._GESTURE_DEBOUNCE and not self._gesture_fired:
                self._gesture_fired = True
                self.canvas.flood_fill(pos, self.state.color)
                self.particles.emit(pos, self.state.color, 30, 10)
        mode_colors = {"draw": "#a5a4ff", "hover": "#60a0ff", "erase": "#f87171",
                       "fill": "#34d399", "clear": "#f59e0b"}
        mc = mode_colors.get(self.state.mode, "#606090")
        self._lbl_mode.setText(
            f"<span style='color:{mc};font-weight:600;'>{self.state.mode.upper()}</span>"
            f"<span style='color:#303060;'>  {BRUSH_LABELS.get(self.state.active_brush,'')}</span>"
        )

    def _do_draw(self, pos, pressure, angle):
        q = self.state.quality_level
        if self.prev_pt is not None:
            self.canvas.draw(self.prev_pt, pos, self.state.color,
                             self.brush_eng, self.state.active_brush,
                             pressure, angle, time.time(), q)
            self.cur_stroke.append(pos)
            pc = max(1, int(pressure*4*(q+1)//2))
            self.particles.emit(pos, self.state.color, pc, pressure*2)
        self.prev_pt = pos; self.is_drawing = True

    def _finish_stroke(self):
        if self.cur_stroke and len(self.cur_stroke) > 1:
            self.canvas.finish_stroke(BrushStroke(
                points=self.cur_stroke.copy(), color=self.state.color,
                brush_size=self.state.brush_size, timestamp=time.time(),
                brush_type=self.state.active_brush,
            ))
        self.cur_stroke = []; self.is_drawing = False

    def _toggle_left_panel(self):
        visible = not self._left_panel.isVisible()
        self._left_panel.setVisible(visible)
        self._left_sep.setVisible(visible)
        self._left_toggle.setChecked(visible)
        self._left_toggle.setText("‹ Brushes" if visible else "› Brushes")

    def _toggle_right_panel(self):
        visible = not self._right_panel.isVisible()
        self._right_panel.setVisible(visible)
        self._right_sep.setVisible(visible)
        self._right_toggle.setChecked(visible)
        self._right_toggle.setText("Colors ›" if visible else "Colors ‹")

    def _reset_drawing(self):
        if self.is_drawing: self._finish_stroke()
        self.prev_pt = None; self.kalman = KalmanCursor()

    def _update_fps(self):
        now = time.perf_counter()
        self._fps_times.append(now - self._last_frame_t)
        self._last_frame_t = now
        if self._fps_times:
            self.state.fps = 1.0 / (sum(self._fps_times) / len(self._fps_times))
        self.state.quality_level = self.gov.tick(self.state.fps)

        if self.state.fps > 24:
            fps_color = "#22c55e"
        elif self.state.fps > 14:
            fps_color = "#f59e0b"
        else:
            fps_color = "#ef4444"

        self._lbl_fps.setText(
            f"<span style='color:{fps_color};font-weight:600;'>"
            f"{self.state.fps:.0f}</span>"
            f"<span style='color:#303060;'> fps</span>"
            f"<span style='color:#252550;'>  Q{self.state.quality_level}</span>"
        )
        strokes = len(self.canvas.strokes) if self.canvas else 0
        self._lbl_strokes.setText(
            f"<span style='color:#404068;'>{strokes} stroke{'s' if strokes != 1 else ''}</span>"
        )

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_S and not (event.modifiers() & Qt.ControlModifier):
            self._save_canvas()
        elif k == Qt.Key_C:
            self._clear_canvas()
        elif k in (Qt.Key_U,) or (k == Qt.Key_Z and event.modifiers() & Qt.ControlModifier):
            self.canvas.undo() if self.canvas else None
        elif k in (Qt.Key_R,) or (k == Qt.Key_Y and event.modifiers() & Qt.ControlModifier):
            self.canvas.redo() if self.canvas else None
        elif k == Qt.Key_P:
            self._cycle_palette()
        elif Qt.Key_1 <= k <= Qt.Key_9:
            idx = k - Qt.Key_1
            if idx < len(BRUSH_KEYS):
                self._select_brush(BRUSH_KEYS[idx].value, self._brush_btns.get(BRUSH_KEYS[idx].value))
        elif k == Qt.Key_0:
            if len(BRUSH_KEYS) > 9:
                b = BRUSH_KEYS[9].value
                self._select_brush(b, self._brush_btns.get(b))
        elif k in (Qt.Key_Equal, Qt.Key_Plus):
            self._size_slider.setValue(min(self._size_slider.value() + 2, self._size_slider.maximum()))
        elif k == Qt.Key_Minus:
            self._size_slider.setValue(max(1, self._size_slider.value() - 2))
        elif k == Qt.Key_BracketLeft:
            self._toggle_left_panel()
        elif k == Qt.Key_BracketRight:
            self._toggle_right_panel()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._cleanup(); event.accept()

    def _cleanup(self):
        if not self._running: return
        self._running = False
        if hasattr(self, "_timer"): self._timer.stop()
        if self._cap_thread:
            self._cap_thread.join(timeout=2.0)
        if self.gesture:
            self.gesture.stop()
        if self.canvas and self.canvas.strokes:
            ts = time.strftime("%Y%m%d_%H%M%S")
            fn = f"autosave_{ts}.png"
            self.canvas.save(fn)
            log.info(f"Autosaved → {fn}")
        if self.cap:
            self.cap.release()
        log.info("Shutdown complete")


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("AirDraw Pro v3")
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    font = QFont()
    font.setFamilies(["Inter", "SF Pro Text", "Segoe UI", "Helvetica Neue", "Arial"])
    font.setPixelSize(APP_FONT_SIZE)
    font.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    font.setStyleStrategy(
        QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.PreferQuality
    )
    app.setFont(font)

    win = AirDrawWindow()
    win.show()

    if not win.start():
        log.error("Startup failed — check camera and MediaPipe model")
        sys.exit(1)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

