#!/usr/bin/env python3
"""
Electric Guitar Real-time Visual System
GPU-accelerated particle visualizer — moderngl + pygame + sounddevice + aubio

Controls:
  ESC    – quit
  SPACE  – pause / resume visuals  (audio keeps running)
  F      – toggle fullscreen
  C      – toggle color mode  (pitch-mapped hue  ↔  random hue per onset)

Usage:
  python guitar_viz.py              # fullscreen at native resolution (auto-detected)
  python guitar_viz.py --windowed   # 1280x720 window  (easier for dev)
"""

from __future__ import annotations
import sys
import queue
import time
import argparse
import numpy as np

# ── dependency check ────────────────────────────────────────────────────────────
_missing = []
try:
    import sounddevice as sd
except ImportError:
    _missing.append("sounddevice")
try:
    import aubio
except ImportError:
    _missing.append("aubio")
try:
    import pygame
    from pygame.locals import (
        OPENGL, DOUBLEBUF, FULLSCREEN, RESIZABLE,
        QUIT, KEYDOWN, K_ESCAPE, K_SPACE, K_f, K_c,
    )
except ImportError:
    _missing.append("pygame")
try:
    import moderngl
except ImportError:
    _missing.append("moderngl")

if _missing:
    print(f"Missing packages: {', '.join(_missing)}")
    print("Install with:  pip install " + " ".join(_missing))
    sys.exit(1)


# ── config ──────────────────────────────────────────────────────────────────────

SR       = 44100
HOP      = 512         # audio block size → latency ≈ HOP/SR × 1000 ms ≈ 11.6 ms
WIN_P    = 2048        # YIN pitch analysis window
WIN_O    = 1024        # HFC onset analysis window

W, H     = 1920, 1080  # overridden by --windowed
MAX_P    = 100_000     # particle pool — dense micro-smoke cloud
FPS      = 144

# waveform oscilloscope
WAVE_N   = 2048        # samples in display window  (~46 ms at 44100 Hz)
WAVE_CY  = -0.08       # NDC y centre  (0 = exact centre, negative = lower)

# particle turbulence — random per-frame nudge that makes particles wander
TURB     = 160.0       # px/s²  — gentle smoke drift

# post-processing
TRAIL_ALPHA_SHORT = 0.12   # fade speed at sustain=0  (short trail)
TRAIL_ALPHA_LONG  = 0.035  # fade speed at sustain=1  (long trail → "延音控拖尾")
BLOOM_STR         = 1.0    # glow intensity (2 blur passes = wider, so lower value)
TRICKLE_SPR       = 120    # trickle spread (was 380, narrowed — cx tracks pitch now)

# physics
GRAVITY      = 20.0    # px / s²  (lighter → more fluid float)
DAMP_LOW     = 0.80    # velocity × DAMP_LOW^dt  when no sustain  (short trail)
DAMP_HIGH    = 0.985   # velocity × DAMP_HIGH^dt when max sustain (long trail)
LIFE_MIN     = 0.5     # seconds
LIFE_MAX     = 5.0     # seconds

# pitch → hue  (log scale)
HZ_LO        = 80.0    # E2  → HUE_LO  (deep blue, 240°)
HZ_HI        = 1400.0  # ~F6 → HUE_HI  (orange-red, 25°)
HUE_LO       = 240.0
HUE_HI_WRAP  = 385.0   # 25° reached via 240→270→300→360→25 (through purple/pink)

# audio thresholds  (calibrated: noise≈0.001, fingerpick≈0.004, strum≈0.010~0.015)
CONF_MIN     = 0.55
RMS_FLOOR    = 0.004   # was 0.008 — lowered to catch fingerpicking
ONSET_THOLD  = 0.5

# sustain envelope → trail_factor
SUS_ATK      = 2.0     # sustain_time growth per playback second
SUS_REL      = 1.5     # sustain_time decay per silence second
SUS_MAX      = 3.0     # sustain_time → trail_factor = 1.0

# spawn counts  (calibrated: noise≈0.001, fingerpick≈0.004, strum≈0.010~0.015)
SPAWN_BASE   = 100
SPAWN_RMS_K  = 8000    # at RMS=0.005 → +40,  at RMS=0.015 → +120
SPAWN_STR_K  = 180
SPAWN_MAX    = 900     # more particles OK now that they're tiny sparks
TRICKLE_K    = 12
TRICKLE_MAX  = 18
TRICKLE_SPR  = 120    # trickle spread — narrowed, cx now tracks pitch

# RMS gate for onset — prevents noise spikes from triggering explosion
ONSET_RMS_GATE = 0.003

# ring shockwave system
RING_SPEED  = 520.0   # px/s expansion rate
RING_R_BASE = 60      # base r_max (px)
RING_R_RMS  = 2000    # RMS → r_max contribution
RING_R_STR  = 60      # onset strength → r_max
RING_R_CAP  = 450     # cap r_max
MAX_R       = 30      # ring pool size
RING_SEGS   = 96      # circle segments


# ── shaders ─────────────────────────────────────────────────────────────────────

# Particle shader — point sprites with sine envelope + cloud shadow depth
VERT = """
#version 410 core

in vec2  in_pos;
in float in_ratio;
in float in_hue;
in float in_size;
in float in_alpha;   // per-particle alpha scale (0.55 normal, 0.025 volumetric)

out float v_ratio;
out float v_hue;
out float v_size;
out float v_alpha;

uniform vec2 u_res;
uniform vec2 u_shake;   // screen-shake offset in pixels

void main() {
    vec2 ndc     = (in_pos / u_res) * 2.0 - 1.0;
    ndc.y        = -ndc.y;
    ndc         += u_shake * 2.0 / u_res;
    gl_Position  = vec4(ndc, 0.0, 1.0);
    float t_size = sin((1.0 - in_ratio) * 3.14159265);
    gl_PointSize = max(in_size * t_size, 1.0);
    v_ratio = in_ratio;
    v_hue   = in_hue;
    v_size  = in_size;
    v_alpha = in_alpha;
}
"""

FRAG = """
#version 410 core

in  float v_ratio;
in  float v_hue;
in  float v_size;
in  float v_alpha;
out vec4  fragColor;

vec3 hsv2rgb(float h, float s, float v) {
    h = mod(h, 360.0) / 60.0;
    int   i = int(h);
    float f = h - float(i);
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));
    if (i == 0) return vec3(v, t, p);
    if (i == 1) return vec3(q, v, p);
    if (i == 2) return vec3(p, v, t);
    if (i == 3) return vec3(p, q, v);
    if (i == 4) return vec3(t, p, v);
    return vec3(v, p, q);
}

void main() {
    vec2  d    = gl_PointCoord - 0.5;
    float dist = length(d);
    if (dist > 0.5) discard;

    // Sine lifecycle: born invisible → peak at mid-life → fade
    float t = sin((1.0 - v_ratio) * 3.14159265);

    // Cloud shadow depth: outer ring of large particles is darker & more saturated
    // giving a "hot core / coloured shadow fringe" look
    float cloud = smoothstep(3.5, 8.0, v_size);   // 0=spark  1=cloud
    float depth = smoothstep(0.0, 0.46, dist);    // 0=center 1=edge
    float sat   = (0.0 + 0.82 * depth) + cloud * 0.18 * depth;
    float val   = 1.0  - depth * (0.30 + cloud * 0.25);

    float alpha = (1.0 - smoothstep(0.12, 0.5, dist)) * t * v_alpha;

    fragColor = vec4(hsv2rgb(v_hue, sat, val), alpha);
}
"""

# Post-processing shaders ─────────────────────────────────────────────────────

QUAD_VERT = """
#version 410 core
in  vec2 in_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

FADE_FRAG = """
#version 410 core
out vec4 fragColor;
uniform float u_alpha;
uniform vec3  u_bg;
void main() { fragColor = vec4(u_bg, u_alpha); }
"""

BLUR_FRAG = """
#version 410 core
in  vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform vec2      u_dir;
// 9-tap Gaussian sigma≈2, step×1.5 for wider glow
const float W[5] = float[5](0.204, 0.180, 0.124, 0.066, 0.028);
void main() {
    vec4 c = texture(u_tex, v_uv) * W[0];
    for (int i = 1; i <= 4; i++) {
        c += texture(u_tex, v_uv + u_dir * float(i) * 1.5) * W[i];
        c += texture(u_tex, v_uv - u_dir * float(i) * 1.5) * W[i];
    }
    fragColor = c;
}
"""

COMP_FRAG = """
#version 410 core
in  vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float     u_bloom_str;

vec3 reinhard(vec3 hdr) { return hdr / (hdr + vec3(1.0)); }

void main() {
    vec3 scene = texture(u_scene, v_uv).rgb;
    vec3 bloom = texture(u_bloom, v_uv).rgb;

    // HDR composite + Reinhard tone map + sRGB gamma
    vec3 hdr = scene + bloom * u_bloom_str;
    vec3 ldr = pow(clamp(reinhard(hdr), 0.0, 1.0), vec3(1.0 / 2.2));

    // Vignette: darkens edges, makes particle cloud feel centred and deep
    vec2  vc  = v_uv - 0.5;
    float vig = 1.0 - dot(vc, vc) * 1.6;
    ldr      *= clamp(vig, 0.0, 1.0);

    fragColor = vec4(ldr, 1.0);
}
"""


# Ring shockwave shaders ──────────────────────────────────────────────────────

RING_VERT = """
#version 410 core
in  vec2  in_dir;          // unit-circle direction (cos θ, sin θ)
uniform vec2  u_center;    // screen-space centre (pixels)
uniform float u_radius;    // current radius (pixels)
uniform vec2  u_res;
uniform vec2  u_shake;
void main() {
    vec2 world  = u_center + in_dir * u_radius;
    vec2 ndc    = (world / u_res) * 2.0 - 1.0;
    ndc.y       = -ndc.y;
    ndc        += u_shake * 2.0 / u_res;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

RING_FRAG = """
#version 410 core
out vec4 fragColor;
uniform float u_hue;
uniform float u_alpha;

vec3 hsv2rgb(float h, float s, float v) {
    h = mod(h, 360.0) / 60.0;
    int   i = int(h);
    float f = h - float(i);
    float p = v*(1.0-s);
    float q = v*(1.0-s*f);
    float t = v*(1.0-s*(1.0-f));
    if (i==0) return vec3(v,t,p);
    if (i==1) return vec3(q,v,p);
    if (i==2) return vec3(p,v,t);
    if (i==3) return vec3(p,q,v);
    if (i==4) return vec3(t,p,v);
    return vec3(v,p,q);
}

void main() {
    vec3 col = hsv2rgb(u_hue, 0.70, 1.0);
    fragColor = vec4(col, u_alpha);
}
"""


# Band-ring shader — centre hardcoded at NDC (0,0) = screen centre ───────────

BAND_RING_VERT = """
#version 410 core
in vec2   in_dir;           // unit-circle point (cos θ, sin θ)
uniform float u_radius;     // current radius in pixels
uniform vec2  u_res;        // (W, H) screen resolution
uniform vec2  u_shake;      // shake offset in pixels

void main() {
    // Expand from NDC (0,0) — no u_center needed, centre always exact
    vec2 ndc  = in_dir * u_radius / (u_res * 0.5);
    ndc.y     = -ndc.y;
    ndc      += u_shake * 2.0 / u_res;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

# Lightning bolt shaders ──────────────────────────────────────────────────────

BOLT_VERT = """
#version 410 core
in  vec2  in_pos;
uniform vec2  u_res;
uniform vec2  u_shake;
void main() {
    vec2 ndc  = (in_pos / u_res) * 2.0 - 1.0;
    ndc.y     = -ndc.y;
    ndc      += u_shake * 2.0 / u_res;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
"""

BOLT_FRAG = """
#version 410 core
out vec4 fragColor;
uniform float u_hue;
uniform float u_alpha;

vec3 hsv2rgb(float h, float s, float v) {
    h = mod(h, 360.0) / 60.0;
    int   i = int(h);
    float f = h - float(i);
    float p = v*(1.0-s); float q = v*(1.0-s*f); float t = v*(1.0-s*(1.0-f));
    if (i==0) return vec3(v,t,p); if (i==1) return vec3(q,v,p);
    if (i==2) return vec3(p,v,t); if (i==3) return vec3(p,q,v);
    if (i==4) return vec3(t,p,v); return vec3(v,p,q);
}

void main() {
    // near-white core with slight colour tint — like real lightning
    vec3 col = hsv2rgb(u_hue, 0.28, 1.0);
    fragColor = vec4(col, u_alpha);
}
"""


# Waveform oscilloscope shaders ───────────────────────────────────────────────

WAVE_VERT = """
#version 410 core
in float in_x;      // pre-computed NDC x, linspace(-1,1,WAVE_N)
in float in_amp;    // audio sample value
uniform float u_scale;  // amplitude → NDC multiplier
uniform float u_cy;     // waveform centre NDC y
uniform vec2  u_shake;
uniform vec2  u_res;
out float v_amp;
void main() {
    float y = u_cy + in_amp * u_scale;
    gl_Position = vec4(
        in_x + u_shake.x * 2.0 / u_res.x,
        y    + u_shake.y * 2.0 / u_res.y,
        0.0, 1.0);
    v_amp = abs(in_amp);
}
"""

WAVE_FRAG = """
#version 410 core
in  float v_amp;
out vec4  fragColor;
uniform float u_hue;

vec3 hsv2rgb(float h, float s, float v) {
    h = mod(h, 360.0) / 60.0;
    int   i = int(h);
    float f = h - float(i);
    float p = v*(1.0-s);
    float q = v*(1.0-s*f);
    float t = v*(1.0-s*(1.0-f));
    if (i==0) return vec3(v,t,p);
    if (i==1) return vec3(q,v,p);
    if (i==2) return vec3(p,v,t);
    if (i==3) return vec3(p,q,v);
    if (i==4) return vec3(t,p,v);
    return vec3(v,p,q);
}

void main() {
    // purple glow — fixed hue, high saturation for vivid neon look
    float a   = clamp(0.08 + v_amp * 12.0, 0.0, 0.55);
    vec3  col = hsv2rgb(u_hue, 0.88, 1.0);
    fragColor = vec4(col, a);
}
"""


# ── audio ───────────────────────────────────────────────────────────────────────

def select_device() -> int:
    devs = sd.query_devices()
    input_devs = [(i, d) for i, d in enumerate(devs) if d["max_input_channels"] > 0]

    if not input_devs:
        _fatal_dialog("No audio input devices found.\n\nPlease connect a microphone or audio interface and restart.")
        sys.exit(1)

    default_idx = sd.default.device[0]
    if default_idx < 0 or default_idx >= len(devs):
        default_idx = input_devs[0][0]

    # ── terminal mode ────────────────────────────────────────────────────────
    if sys.stdin and sys.stdin.isatty():
        print("\n═══ Audio Input Devices ════════════════════════════════════════")
        for i, d in input_devs:
            tag = " ◀" if i == default_idx else "  "
            print(f"{tag} [{i:3d}]  {d['name']:<46}"
                  f"ch:{d['max_input_channels']}  sr:{int(d['default_samplerate'])}")
        print("════════════════════════════════════════════════════════════════\n")
        try:
            ans = input(f"Device index [{default_idx}]: ").strip()
            idx = int(ans) if ans else default_idx
        except (ValueError, EOFError):
            idx = default_idx
        print(f"→ [{idx}]  {sd.query_devices(idx)['name']}\n")
        return idx

    # ── packaged / no-console mode → tkinter dialog ──────────────────────────
    return _select_device_gui(input_devs, default_idx)


def _fatal_dialog(msg: str):
    """Show error in a tkinter popup (works without a console)."""
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("GuitarViz — Error", msg)
        root.destroy()
    except Exception:
        print(msg)


def _select_device_gui(input_devs: list, default_idx: int) -> int:
    """Tkinter device-picker dialog for packaged (no-console) mode."""
    try:
        import tkinter as tk
        from tkinter import ttk

        chosen = [default_idx]

        root = tk.Tk()
        root.title("GuitarViz — Select Audio Input")
        root.resizable(False, False)
        root.configure(bg="#1a1a2e")

        tk.Label(root, text="Select audio input device:",
                 bg="#1a1a2e", fg="white",
                 font=("Segoe UI", 11)).pack(padx=20, pady=(16, 6))

        var = tk.StringVar()
        labels = [f"{i:3d}  {d['name']}" for i, d in input_devs]
        default_label = next((l for l in labels if l.startswith(f"{default_idx:3d}")), labels[0])
        var.set(default_label)

        cb = ttk.Combobox(root, textvariable=var, values=labels,
                          state="readonly", width=52, font=("Consolas", 10))
        cb.pack(padx=20, pady=4)

        def on_ok():
            sel = var.get()
            chosen[0] = int(sel.split()[0])
            root.destroy()

        tk.Button(root, text="  Start  ", command=on_ok,
                  bg="#6a0dad", fg="white",
                  font=("Segoe UI", 11, "bold"),
                  relief="flat", padx=12, pady=6).pack(pady=(10, 16))

        root.bind("<Return>", lambda _: on_ok())
        root.eval("tk::PlaceWindow . center")
        root.mainloop()
        return chosen[0]

    except Exception:
        # tkinter unavailable — silently fall back to default
        return default_idx


class Audio:
    """Real-time audio processor: pitch (YIN) + onset (HFC) via aubio."""

    def __init__(self, dev: int):
        self.q: queue.Queue = queue.Queue(maxsize=64)

        self._pitch = aubio.pitch("yin", WIN_P, HOP, SR)
        self._pitch.set_unit("Hz")
        self._pitch.set_tolerance(0.8)

        self._onset = aubio.onset("hfc", WIN_O, HOP, SR)
        self._onset.set_threshold(ONSET_THOLD)

        self._stream = sd.InputStream(
            device=dev, channels=1, samplerate=SR,
            blocksize=HOP, dtype="float32",
            callback=self._cb, latency="low",
        )

    def _cb(self, indata, frames, t, status):
        if status:
            print(f"[sd] {status}", file=sys.stderr)

        mono  = indata[:, 0]
        rms   = float(np.sqrt(np.mean(mono * mono)))
        pitch = float(self._pitch(mono)[0])
        conf  = float(self._pitch.get_confidence())
        onset = bool(self._onset(mono))
        ostr  = float(self._onset.get_descriptor()) if onset else 0.0

        # three frequency bands — tuned to guitar range (80 Hz–5000 Hz)
        wave_hi  = band_filter(mono, SR, 1500, 5000)   # harmonics / pick attack
        wave_mid = band_filter(mono, SR,  300, 1500)   # main guitar fundamentals
        wave_low = band_filter(mono, SR,   80,  300)   # bass strings / body resonance
        rms_hi  = float(np.sqrt(np.mean(wave_hi  * wave_hi)))
        rms_mid = float(np.sqrt(np.mean(wave_mid * wave_mid)))
        rms_low = float(np.sqrt(np.mean(wave_low * wave_low)))

        try:
            self.q.put_nowait({
                "rms":      rms,
                "pitch":    pitch if conf >= CONF_MIN else 0.0,
                "onset":    onset,
                "ostr":     ostr,
                "wave":     wave_hi,
                "wave_mid": wave_mid,
                "wave_low": wave_low,
                "rms_hi":   rms_hi,
                "rms_mid":  rms_mid,
                "rms_low":  rms_low,
            })
        except queue.Full:
            pass

    def start(self): self._stream.start()
    def stop(self):  self._stream.stop(); self._stream.close()


# ── particle system ─────────────────────────────────────────────────────────────

class Particles:
    """
    Ring-buffer particle pool.  Pre-allocated numpy arrays; zero per-frame
    heap allocations in steady state.

    VBO layout per vertex  (5 × float32):
      [px, py, life_ratio, hue_deg, size_px]
    """

    def __init__(self, cap: int):
        self.cap        = cap
        self.px         = np.zeros(cap, np.float32)
        self.py         = np.zeros(cap, np.float32)
        self.vx         = np.zeros(cap, np.float32)
        self.vy         = np.zeros(cap, np.float32)
        self.life       = np.zeros(cap, np.float32)
        self.mlife      = np.zeros(cap, np.float32)
        self.hue        = np.zeros(cap, np.float32)
        self.size       = np.zeros(cap, np.float32)
        self.alpha_scale= np.full(cap, 0.55, np.float32)  # per-particle alpha
        self._ptr       = 0
        self._t         = 0.0   # time accumulator for curl-noise field
        # pre-alloc output buffer to avoid per-frame malloc
        self._buf = np.zeros((cap, 6), np.float32)

    # ── alloc ──────────────────────────────────────────────────────────────────
    def _alloc(self, n: int) -> np.ndarray:
        n  = min(n, self.cap)
        i0 = self._ptr
        i1 = i0 + n
        if i1 <= self.cap:
            idx = np.arange(i0, i1, dtype=np.int32)
        else:
            idx = np.concatenate([
                np.arange(i0,          self.cap, dtype=np.int32),
                np.arange(0, i1 - self.cap, dtype=np.int32),
            ])
        self._ptr = int(i1 % self.cap)
        return idx

    # ── spawn ──────────────────────────────────────────────────────────────────
    def spawn(self,
              cx: float, cy: float, n: int,
              hue: float, rms: float, ostr: float,
              trail: float, spread: float = 0.0,
              ambient: bool = False, macro: bool = False,
              volume: bool = False,
              rms_hi: float = 0.0, rms_mid: float = 0.0, rms_low: float = 0.0):
        idx = self._alloc(n)
        k   = len(idx)

        if volume:
            # Volumetric haze layer — large, ultra-transparent, very slow drift
            # Creates soft cloud outline / foggy volume around micro-particle cloud
            self.px[idx] = (cx + np.random.normal(0, spread, k)).astype(np.float32)
            self.py[idx] = (cy + np.random.normal(0, spread, k)).astype(np.float32)
            # almost no velocity — just gentle upward float with tiny sway
            energy = float(np.clip(rms * 200.0, 0.0, 1.0))
            self.vx[idx] = np.random.normal(0, 12.0 + energy * 40.0, k).astype(np.float32)
            self.vy[idx] = np.random.uniform(-(4.0 + energy * 20.0), -0.5, k).astype(np.float32)
            self.mlife[idx] = np.random.uniform(3.0, 7.0, k).astype(np.float32)
            self.size[idx]  = np.random.uniform(8.0, 20.0, k).astype(np.float32)
            self.alpha_scale[idx] = 0.028   # barely visible — just a wisp
            self.life[idx] = self.mlife[idx]
            self.hue[idx]  = (hue + np.random.uniform(-80, 80, k)).astype(np.float32)
            return

        if macro:
            # Each particle randomly follows one frequency band's energy:
            #   band 0 = low  → large, slow, wide drift   (bass weight)
            #   band 1 = mid  → medium normal dynamics
            #   band 2 = high → tiny, fast, erratic sparks (treble snap)
            band_rms = np.array([rms_low, rms_mid, rms_hi], np.float32)
            band     = np.random.randint(0, 3, k)           # per-particle band choice
            p_energy = np.clip(band_rms[band] * 320.0, 0.0, 1.0)

            # sway: reduced so curl noise guides the path instead
            sway = (6.0 + band * 10.0 + p_energy * 50.0).astype(np.float32)
            # gentle upward seed velocity — curl field takes over after birth
            up_base = np.array([1.0, 2.0, 4.0], np.float32)
            up_mag  = (up_base[band] + p_energy * 28.0).astype(np.float32)

            # size: low=bigger blobs, high=tiny sparks
            s_lo = np.array([0.25, 0.12, 0.06], np.float32)
            s_hi = np.array([0.80, 0.50, 0.22], np.float32)
            t_s  = np.random.uniform(0.0, 1.0, k)
            self.size[idx] = (s_lo[band] + t_s * (s_hi[band] - s_lo[band])).astype(np.float32)

            # life: low-band lingers, high-band fades fast
            l_lo = np.array([2.0, 1.2, 0.6], np.float32)
            l_hi = np.array([5.0, 4.0, 2.0], np.float32)
            t_l  = np.random.uniform(0.0, 1.0, k)
            self.mlife[idx] = (l_lo[band] + t_l * (l_hi[band] - l_lo[band])).astype(np.float32)

            self.px[idx] = (cx + np.random.normal(0, spread, k)).astype(np.float32)
            self.py[idx] = (cy + np.random.normal(0, spread, k)).astype(np.float32)
            self.vx[idx] = (np.random.normal(0, 1, k) * sway).astype(np.float32)
            t_v = np.random.uniform(0.2, 1.5, k)
            self.vy[idx] = (-up_mag * t_v).astype(np.float32)

            self.life[idx]        = self.mlife[idx]
            band_hues = np.array([210.0, 120.0, 272.0], np.float32)  # low=blue, mid=green, hi=purple
            self.hue[idx]         = (band_hues[band] + np.random.uniform(-15, 15, k)).astype(np.float32)
            self.alpha_scale[idx] = 0.38
            return   # skip generic assignment below

        elif ambient:
            # Tight point emission — like smoke rising from a candle tip
            self.px[idx] = (cx + np.random.normal(0, 3, k)).astype(np.float32)
            self.py[idx] = (cy + np.random.normal(0, 3, k)).astype(np.float32)
            # gentle upward rise with slight lateral sway
            self.vx[idx] = np.random.normal(0, 13, k).astype(np.float32)
            self.vy[idx] = np.random.uniform(-40, -5, k).astype(np.float32)
            self.mlife[idx] = np.random.uniform(0.15, 0.65, k).astype(np.float32)
            # tiny soft blobs only
            tiers = np.random.choice([0, 1], size=k, p=[0.65, 0.35])
            base  = np.where(tiers == 0,
                        np.random.uniform(0.5, 2.0, k),
                        np.random.uniform(2.0, 5.0, k)).astype(np.float32)
            self.size[idx] = base

        else:
            # Smoke puff from emitter: origin near cx/cy, upward-dominant velocity
            if spread > 0.0:
                self.px[idx] = (cx + np.random.normal(0, spread, k)).astype(np.float32)
                self.py[idx] = (cy + np.random.normal(0, spread * 0.5, k)).astype(np.float32)
            else:
                self.px[idx] = cx
                self.py[idx] = cy
            # mostly upward, gentle sway — not radial explosion
            spd = float(np.clip(55.0 + ostr * 140.0 + rms * 280.0, 18, 400))
            self.vx[idx] = np.random.normal(0, spd * 0.48, k).astype(np.float32)
            self.vy[idx] = -(np.abs(np.random.normal(spd * 0.55, spd * 0.28, k))).astype(np.float32)
            base_life = LIFE_MIN + (LIFE_MAX - LIFE_MIN) * trail
            self.mlife[idx] = (base_life * np.random.uniform(0.55, 1.45, k)).astype(np.float32)
            # small-to-medium smoke blobs — no giant explosions
            tiers = np.random.choice([0, 1, 2], size=k, p=[0.55, 0.35, 0.10])
            base  = np.where(tiers == 0, np.random.uniform(0.5,  2.0, k),
                    np.where(tiers == 1, np.random.uniform(2.0,  5.0, k),
                                         np.random.uniform(5.0,  8.0, k))).astype(np.float32)
            self.size[idx] = (base * float(1.0 + rms * 0.5 + ostr * 0.05)).astype(np.float32)

        self.life[idx]        = self.mlife[idx]
        self.hue[idx]         = (hue + np.random.uniform(-15, 15, k)).astype(np.float32)
        self.alpha_scale[idx] = 0.55

    # ── update (fully vectorised) ───────────────────────────────────────────────
    def update(self, dt: float, trail: float):
        a = self.life > 0.0
        if not a.any():
            return

        self._t += dt

        # gravity (screen +y = down)
        self.vy[a] += GRAVITY * dt

        # curl-noise flow field — divergence-free → smooth swirling fluid motion
        # potential φ = cos(kx·x + sx·t) · cos(ky·y + sy·t)
        # curl: vx = ky·cos(kx·x+sx·t)·sin(ky·y+sy·t)
        #        vy = kx·sin(kx·x+sx·t)·cos(ky·y+sy·t)  (negated for upward bias)
        t   = self._t
        kx, ky   = 0.0016, 0.0013
        sx, sy   = 0.38,   0.29
        px = self.px[a];  py = self.py[a]
        cx = np.cos(kx * px + sx * t);  sx_ = np.sin(kx * px + sx * t)
        cy = np.cos(ky * py + sy * t);  sy_ = np.sin(ky * py + sy * t)
        curl_vx =  ky * cx * sy_
        curl_vy = -kx * sx_ * cy
        n_a = int(a.sum())
        noise = np.random.normal(0, 0.18, (n_a, 2)).astype(np.float32)
        self.vx[a] += (curl_vx * TURB + noise[:, 0] * TURB) * dt
        self.vy[a] += (curl_vy * TURB + noise[:, 1] * TURB) * dt

        # damping: lerp between low-sustain and high-sustain, applied per-frame
        dps = DAMP_LOW + (DAMP_HIGH - DAMP_LOW) * trail
        dpf = float(dps ** dt)
        self.vx[a] *= dpf
        self.vy[a] *= dpf

        self.px[a] += self.vx[a] * dt
        self.py[a] += self.vy[a] * dt
        self.life[a] -= dt

        # cull off-screen
        oob = (
            (self.px[a] < -250) | (self.px[a] > W + 250) |
            (self.py[a] < -250) | (self.py[a] > H + 250)
        )
        if oob.any():
            alive_idx = np.where(a)[0]
            self.life[alive_idx[oob]] = 0.0

    # ── pack VBO data ───────────────────────────────────────────────────────────
    def vbo_data(self) -> tuple[bytes | None, int]:
        a = self.life > 0.0
        if not a.any():
            return None, 0
        idx   = np.where(a)[0]
        n     = len(idx)
        ratio = self.life[idx] / np.maximum(self.mlife[idx], 1e-6)

        buf = self._buf[:n]
        buf[:, 0] = self.px[idx]
        buf[:, 1] = self.py[idx]
        buf[:, 2] = ratio
        buf[:, 3] = self.hue[idx]
        buf[:, 4] = self.size[idx]
        buf[:, 5] = self.alpha_scale[idx]
        return buf.ravel().tobytes(), n


# ── helpers ──────────────────────────────────────────────────────────────────────

def pitch_to_hue(freq: float) -> float:
    """
    Log-map Hz → hue °.
    E2 (80 Hz) → 240° (deep blue)  …through purple/pink…  →  25° (orange-red)
    """
    if freq <= 0.0:
        return HUE_LO
    lo = np.log2(HZ_LO)
    hi = np.log2(HZ_HI)
    t  = float(np.clip((np.log2(max(freq, HZ_LO)) - lo) / (hi - lo), 0.0, 1.0))
    return (HUE_LO + t * (HUE_HI_WRAP - HUE_LO)) % 360.0


def band_filter(signal: np.ndarray, sr: int,
                lo_hz: float, hi_hz: float) -> np.ndarray:
    """Zero-phase FFT bandpass.  lo_hz=0 → low-pass, hi_hz≥sr/2 → high-pass."""
    spec  = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    if lo_hz > 0:
        spec[freqs < lo_hz] = 0.0
    if hi_hz < sr / 2:
        spec[freqs > hi_hz] = 0.0
    return np.fft.irfft(spec, n=len(signal)).astype(np.float32)


# ── ring shockwave system ────────────────────────────────────────────────────────

class RingSystem:
    """
    Pool of expanding circular shockwaves triggered on each onset.
    Rendered as LINE_LOOP into scene_fbo → receives bloom glow.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx

        # Pre-compute unit circle (RING_SEGS directions)
        angles = np.linspace(0, 2 * np.pi, RING_SEGS, endpoint=False, dtype=np.float32)
        circle = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
        self.circle_vbo = ctx.buffer(circle.tobytes())

        self.prog = ctx.program(vertex_shader=RING_VERT, fragment_shader=RING_FRAG)
        self.prog['u_res'].value = (float(W), float(H))
        self.vao = ctx.vertex_array(
            self.prog, [(self.circle_vbo, '2f', 'in_dir')])

        # Ring state pool
        self.cx    = np.zeros(MAX_R, np.float32)
        self.cy    = np.zeros(MAX_R, np.float32)
        self.r     = np.zeros(MAX_R, np.float32)
        self.r_max = np.ones(MAX_R,  np.float32)   # avoid div-by-zero
        self.hue   = np.zeros(MAX_R, np.float32)
        self.alive = np.zeros(MAX_R, dtype=bool)
        self._ptr  = 0

    def spawn(self, cx: float, cy: float, r_max: float, hue: float):
        i = self._ptr % MAX_R
        self._ptr += 1
        self.cx[i]    = cx
        self.cy[i]    = cy
        self.r[i]     = 0.0
        self.r_max[i] = max(r_max, 1.0)
        self.hue[i]   = hue
        self.alive[i] = True

    def update(self, dt: float):
        a = self.alive
        if not a.any():
            return
        self.r[a] += RING_SPEED * dt
        # kill rings that have fully expanded
        self.alive[a & (self.r >= self.r_max)] = False

    def render(self, shake_px: float):
        alive = np.where(self.alive)[0]
        if not alive.size:
            return
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.line_width  = 2.0
        sx = float(np.random.uniform(-1, 1) * shake_px * 0.4)
        sy = float(np.random.uniform(-1, 1) * shake_px * 0.4)
        self.prog['u_shake'].value = (sx, sy)
        for i in alive:
            ratio = float(self.r[i]) / float(self.r_max[i])
            alpha = (1.0 - ratio) ** 0.6 * 0.85
            self.prog['u_center'].value = (float(self.cx[i]), float(self.cy[i]))
            self.prog['u_radius'].value = float(self.r[i])
            self.prog['u_hue'].value    = float(self.hue[i])
            self.prog['u_alpha'].value  = float(alpha)
            self.vao.render(moderngl.LINE_LOOP)


# ── band-driven pulse rings (screen centre) ──────────────────────────────────────

class BandRingSystem:
    """
    Continuous concentric rings expanding from screen centre.
    Each frequency band has its own colour and dynamics:
      band 0 = high  → purple  272°  — fast, small-medium rings
      band 1 = mid   → green   120°  — moderate
      band 2 = low   → blue    210°  — slow, large rings
    """

    MAX_PER_BAND = 24
    HUES       = (272.0, 120.0, 210.0)
    SPEEDS     = (440.0, 520.0, 190.0)   # px/s expansion speed
    R_FRACS    = (0.38,  1.10,  0.88)    # r_max as fraction of half-diagonal
    CLIP_MINS  = (0.15,  0.55,  0.15)   # minimum size fraction per band
    #             hi     mid    low
    INTERVALS  = (0.09,  1.05,  0.18)    # min seconds between spawns (green=1s+)
    FADE_POWS  = (0.45,  2.8,   0.45)    # fade curve exponent; higher = faster fade
    BAND_SEGS  = (96,    96,    96)      # all smooth circles
    BAND_WIDTHS= (1.0,   2.5,   3.5)    # line widths px
    THRESH    = 0.0008                  # per-band RMS gate

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._W  = float(W)
        self._H  = float(H)
        self._cx = self._W * 0.5
        self._cy = self._H * 0.52
        # Use the same proven RING_VERT shader as RingSystem with explicit pixel center
        self.prog = ctx.program(vertex_shader=RING_VERT, fragment_shader=RING_FRAG)
        self.prog['u_res'].value    = (self._W, self._H)
        self.prog['u_center'].value = (self._cx, self._cy)
        self.prog['u_shake'].value  = (0.0, 0.0)
        # One VAO per band — different segment counts for distinct visual styles
        self._vaos = []
        for segs in self.BAND_SEGS:
            angles = np.linspace(0, 2 * np.pi, segs, endpoint=False, dtype=np.float32)
            circle = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float32)
            vbo = ctx.buffer(circle.tobytes())
            self._vaos.append(ctx.vertex_array(self.prog, [(vbo, '2f', 'in_dir')]))

        N = self.MAX_PER_BAND * 3
        self.r        = np.zeros(N, np.float32)
        self.r_max    = np.ones(N,  np.float32)
        self.spd      = np.zeros(N, np.float32)
        self.hue      = np.zeros(N, np.float32)
        self.fade_pow = np.full(N, 0.45, np.float32)
        self.band_idx = np.zeros(N, dtype=np.int8)
        self.alive    = np.zeros(N, dtype=bool)
        self._ptrs = [0, 0, 0]
        self._cd   = [0.0, 0.0, 0.0]   # per-band cooldown timers

        half_diag = float(np.hypot(W / 2, H / 2))
        self._rmaxes = [half_diag * f for f in self.R_FRACS]

        # Blue breathing ring — single persistent circle, radius driven by rms_low
        self._breath_r    = 60.0          # smoothed radius (px)
        self._breath_idle = 55.0          # idle centre radius
        self._breath_amp  = 12.0          # idle oscillation amplitude (px)
        self._breath_freq = 0.28          # breaths per second (~17 bpm, resting)
        self._breath_t    = 0.0           # phase accumulator
        self._breath_max  = half_diag * 0.70

    def set_breath(self, rms_low: float, dt: float):
        """Update blue breathing ring: gentle idle sine + rms-driven expansion."""
        self._breath_t += dt
        idle = self._breath_idle + self._breath_amp * np.sin(2 * np.pi * self._breath_freq * self._breath_t)
        rms_drive = float(np.clip(rms_low * 2200.0, 0.0, 1.0)) * (self._breath_max - self._breath_idle)
        target = idle + rms_drive
        k = 1.0 - np.exp(-dt * 8.0)      # ~8 Hz smoothing
        self._breath_r += (target - self._breath_r) * k

    def try_spawn(self, band: int, rms_band: float, dt: float):
        if band == 2:   # blue handled by breathing ring, not spawning
            return
        self._cd[band] = max(0.0, self._cd[band] - dt)
        if rms_band < self.THRESH or self._cd[band] > 0.0:
            return
        # louder → shorter interval → more rings
        interval = float(np.clip(
            self.INTERVALS[band] / (1.0 + rms_band * 45.0),
            self.INTERVALS[band] * 0.22, self.INTERVALS[band]))
        self._cd[band] = interval

        base = self.MAX_PER_BAND * band
        i    = base + (self._ptrs[band] % self.MAX_PER_BAND)
        self._ptrs[band] += 1

        c_min = self.CLIP_MINS[band]
        r_max = self._rmaxes[band] * float(np.clip(c_min + rms_band * 30.0, c_min, 1.0))
        self.r[i]        = 0.0
        self.r_max[i]    = max(r_max, 15.0)
        self.spd[i]      = self.SPEEDS[band] * float(np.clip(0.5 + rms_band * 22.0, 0.5, 3.0))
        self.hue[i]      = self.HUES[band]
        self.fade_pow[i] = self.FADE_POWS[band]
        self.band_idx[i] = band
        self.alive[i]    = True

    def update(self, dt: float):
        a = self.alive
        if not a.any():
            return
        self.r[a] += self.spd[a] * dt
        self.alive[a & (self.r >= self.r_max)] = False

    def render(self, shake_px: float):
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        # set u_res + u_center every frame — guarantees pixel-exact centre
        self.prog['u_res'].value    = (self._W, self._H)
        self.prog['u_center'].value = (self._cx, self._cy)
        sx = float(np.random.uniform(-1, 1) * shake_px * 0.3)
        sy = float(np.random.uniform(-1, 1) * shake_px * 0.3)
        self.prog['u_shake'].value = (sx, sy)

        # Blue breathing ring — always drawn first, never disappears
        self.ctx.line_width = self.BAND_WIDTHS[2]
        self.prog['u_radius'].value = float(self._breath_r)
        self.prog['u_hue'].value    = self.HUES[2]
        self.prog['u_alpha'].value  = 0.55
        self._vaos[2].render(moderngl.LINE_LOOP)

        # Purple / green expanding rings
        alive = np.where(self.alive)[0]
        for i in alive:
            ratio = float(self.r[i]) / float(self.r_max[i])
            alpha = (1.0 - ratio) ** float(self.fade_pow[i]) * 0.70
            b = int(self.band_idx[i])
            self.ctx.line_width = self.BAND_WIDTHS[b]
            self.prog['u_radius'].value = float(self.r[i])
            self.prog['u_hue'].value    = float(self.hue[i])
            self.prog['u_alpha'].value  = float(alpha)
            self._vaos[b].render(moderngl.LINE_LOOP)


# ── waveform display ─────────────────────────────────────────────────────────────

class WaveformDisplay:
    """
    Three-band oscilloscope waveform:
      purple (272°) — high freq  2500 Hz+
      green  (120°) — mid  freq  250–2500 Hz
      blue   (210°) — low  freq  0–250 Hz
    Drawn into scene_fbo → bloom gives neon glow.
    """

    # (hue°, per-band amplitude boost, line_width)
    BANDS = [
        (272.0, 3.5, 1.5),   # high — purple, boost weak highs
        (120.0, 1.8, 1.5),   # mid  — green
        (210.0, 1.0, 2.0),   # low  — blue,  thicker for bass weight
    ]

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx

        # Static X positions shared by all three bands
        xs = np.linspace(-1.0, 1.0, WAVE_N, dtype=np.float32)
        self.x_vbo = ctx.buffer(xs.tobytes())

        self.prog = ctx.program(vertex_shader=WAVE_VERT, fragment_shader=WAVE_FRAG)
        self.prog['u_res'].value   = (float(W), float(H))
        self.prog['u_cy'].value    = float(WAVE_CY)
        self.prog['u_scale'].value = 50.0
        self.prog['u_shake'].value = (0.0, 0.0)

        # Per-band circular buffers, dynamic VBOs, VAOs
        self._bufs  = [np.zeros(WAVE_N, np.float32) for _ in self.BANDS]
        self._vbos  = [
            ctx.buffer(np.zeros(WAVE_N, np.float32).tobytes(), dynamic=True)
            for _ in self.BANDS
        ]
        self._vaos  = [
            ctx.vertex_array(self.prog, [
                (self.x_vbo, '1f', 'in_x'),
                (vbo,        '1f', 'in_amp'),
            ])
            for vbo in self._vbos
        ]

    def _push_buf(self, buf: np.ndarray, samples: np.ndarray):
        n = len(samples)
        if n >= WAVE_N:
            buf[:] = samples[-WAVE_N:]
        else:
            buf[:-n] = buf[n:]
            buf[-n:] = samples

    def push(self, wave_hi: np.ndarray,
             wave_mid: np.ndarray, wave_low: np.ndarray):
        self._push_buf(self._bufs[0], wave_hi)
        self._push_buf(self._bufs[1], wave_mid)
        self._push_buf(self._bufs[2], wave_low)

    def render(self, shake_px: float, rms: float):
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        sx = float(np.random.uniform(-1, 1) * shake_px * 0.3)
        sy = float(np.random.uniform(-1, 1) * shake_px * 0.3)
        self.prog['u_shake'].value = (sx, sy)
        base_scale = float(np.clip(50.0 + rms * 2000.0, 50.0, 150.0))

        for i, (hue, boost, lw) in enumerate(self.BANDS):
            self._vbos[i].write(self._bufs[i].tobytes())
            self.ctx.line_width        = lw
            self.prog['u_hue'].value   = hue
            self.prog['u_scale'].value = base_scale * boost
            self._vaos[i].render(moderngl.LINE_STRIP)


# ── lightning system ─────────────────────────────────────────────────────────────

class LightningSystem:
    """
    Electric arc bolts triggered on each onset.
    Each bolt is a CPU-generated zigzag LINE_STRIP rendered into scene_fbo → bloom.
    """
    MAX_BOLTS = 12
    BOLT_SEGS = 10   # line segments per bolt

    def __init__(self, ctx: moderngl.Context):
        self.ctx  = ctx
        # VBO holds one bolt at a time (11 points × 2 floats)
        self.pts_vbo = ctx.buffer(
            np.zeros((self.BOLT_SEGS + 1) * 2, np.float32).tobytes(), dynamic=True)
        self.prog = ctx.program(vertex_shader=BOLT_VERT, fragment_shader=BOLT_FRAG)
        self.prog['u_res'].value   = (float(W), float(H))
        self.prog['u_shake'].value = (0.0, 0.0)
        self.vao = ctx.vertex_array(self.prog, [(self.pts_vbo, '2f', 'in_pos')])

        # bolt pool
        self._pts  = np.zeros((self.MAX_BOLTS, self.BOLT_SEGS + 1, 2), np.float32)
        self.hue   = np.zeros(self.MAX_BOLTS, np.float32)
        self.life  = np.zeros(self.MAX_BOLTS, np.float32)
        self.mlife = np.ones (self.MAX_BOLTS, np.float32) * 0.12
        self._ptr  = 0

    def _gen(self, i: int, cx: float, cy: float, length: float):
        """Generate a jagged zigzag path outward from (cx, cy)."""
        angle     = float(np.random.uniform(0, 2 * np.pi))
        roughness = length * 0.20
        dx = np.cos(angle) * length / self.BOLT_SEGS
        dy = np.sin(angle) * length / self.BOLT_SEGS
        px = np.cos(angle + np.pi / 2)
        py = np.sin(angle + np.pi / 2)
        pts = self._pts[i]
        pts[0] = [cx, cy]
        for s in range(1, self.BOLT_SEGS + 1):
            perp = float(np.random.uniform(-roughness, roughness))
            pts[s, 0] = pts[s - 1, 0] + dx + px * perp
            pts[s, 1] = pts[s - 1, 1] + dy + py * perp

    def spawn(self, cx: float, cy: float, length: float, hue: float):
        n = np.random.randint(2, 4)   # 2-3 bolts per onset
        for _ in range(n):
            i = self._ptr % self.MAX_BOLTS
            self._ptr += 1
            self._gen(i, cx, cy, length)
            self.hue[i]   = float(np.random.uniform(0, 360))   # always vivid random
            self.mlife[i] = float(np.random.uniform(0.06, 0.18))
            self.life[i]  = self.mlife[i]

    def update(self, dt: float):
        a = self.life > 0
        if not a.any():
            return
        self.life[a] -= dt
        self.life[self.life < 0] = 0.0

    def render(self, shake_px: float):
        alive = np.where(self.life > 0)[0]
        if not alive.size:
            return
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.ctx.line_width  = 1.5
        sx = float(np.random.uniform(-1, 1) * shake_px * 0.5)
        sy = float(np.random.uniform(-1, 1) * shake_px * 0.5)
        self.prog['u_shake'].value = (sx, sy)
        for i in alive:
            ratio = float(self.life[i]) / float(self.mlife[i])
            self.pts_vbo.write(self._pts[i].ravel().tobytes())
            self.prog['u_hue'].value   = float(self.hue[i])
            self.prog['u_alpha'].value = float(ratio * 0.92)
            self.vao.render(moderngl.LINE_STRIP)


# ── main visualizer ──────────────────────────────────────────────────────────────

class Visualizer:

    def __init__(self, audio: Audio, windowed: bool = False):
        global W, H
        if windowed:
            W, H = 1280, 720

        pygame.init()

        if not windowed:
            # Auto-detect native desktop resolution so fullscreen fills the screen
            sizes = pygame.display.get_desktop_sizes()
            if sizes:
                W, H = sizes[0]
                print(f"[display] native resolution detected: {W}×{H}")

        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

        flags = OPENGL | DOUBLEBUF | (RESIZABLE if windowed else FULLSCREEN)
        pygame.display.set_mode((W, H), flags)
        pygame.display.set_caption("Guitar Visualizer  —  ESC quit  SPACE pause")
        pygame.mouse.set_visible(False)

        ctx = moderngl.create_context()
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        ctx.enable(moderngl.BLEND)
        self.ctx = ctx

        # ── DPI-scaling fix ────────────────────────────────────────────────────
        # pygame reports logical pixels; the actual GL framebuffer uses physical
        # pixels (e.g. 125 % DPI: pygame says 2048×1152 but GL is 2560×1440).
        # We MUST use the GL physical size for all coordinate maths.
        gl_W, gl_H = ctx.screen.size
        if (gl_W, gl_H) != (W, H):
            print(f"[display] DPI scaling: pygame={W}×{H}  GL={gl_W}×{gl_H}  → using GL size")
            W, H = gl_W, gl_H
        # ───────────────────────────────────────────────────────────────────────

        prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        prog["u_res"].value   = (float(W), float(H))
        prog["u_shake"].value = (0.0, 0.0)
        self.prog = prog

        self.vbo = ctx.buffer(
            np.zeros(MAX_P * 6, np.float32).tobytes(), dynamic=True)
        self.vao = ctx.vertex_array(
            prog,
            [(self.vbo, "2f 1f 1f 1f 1f", "in_pos", "in_ratio", "in_hue", "in_size", "in_alpha")],
        )

        self._init_post()

        self.parts    = Particles(MAX_P)
        self.audio    = audio
        self.paused   = False
        self.hue      = HUE_LO
        self.rms      = 0.0
        self.sustain  = 0.0
        self.trail    = 0.0
        self.last_ostr = 0.0
        self.shake_px  = 0.0   # current shake magnitude in pixels
        self.clock    = pygame.time.Clock()
        # wandering emitter — drifts around screen like a floating leaf
        self.emit_x  = float(W * 0.5)
        self.emit_y  = float(H * 0.5)
        self.emit_vx = 0.0
        self.emit_vy = 0.0
        self.rings      = RingSystem(ctx)
        self.band_rings = BandRingSystem(ctx)
        self.wave       = WaveformDisplay(ctx)
        self.lightning  = LightningSystem(ctx)
        self.color_mode = 'pitch'   # 'pitch' or 'random'  (C key toggles)
        self.rms_hi  = 0.0
        self.rms_mid = 0.0
        self.rms_low = 0.0

    # ── post-processing setup ───────────────────────────────────────────────────
    def _init_post(self):
        ctx = self.ctx
        BW, BH = W // 2, H // 2

        # Scene FBO — HDR float32, persistent between frames for trail effect
        self.scene_tex = ctx.texture((W, H), 4, dtype='f4')
        self.scene_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self.scene_fbo = ctx.framebuffer(color_attachments=[self.scene_tex])
        self.scene_fbo.use();  ctx.clear(0, 0, 0, 1)

        # Bloom FBOs — half resolution
        self.bloom_tex_h = ctx.texture((BW, BH), 4, dtype='f4')
        self.bloom_fbo_h = ctx.framebuffer(color_attachments=[self.bloom_tex_h])
        self.bloom_tex_v = ctx.texture((BW, BH), 4, dtype='f4')
        self.bloom_fbo_v = ctx.framebuffer(color_attachments=[self.bloom_tex_v])

        # Full-screen quad (TRIANGLE_STRIP)
        quad = np.array([-1, -1,  1, -1,  -1,  1,  1,  1], dtype=np.float32)
        self.quad_vbo = ctx.buffer(quad.tobytes())

        # Fade program — trail effect (darkens scene FBO each frame)
        fp = ctx.program(vertex_shader=QUAD_VERT, fragment_shader=FADE_FRAG)
        fp['u_bg'].value = (0.0, 0.0, 0.0)
        self.fade_prog = fp
        self.fade_vao  = ctx.vertex_array(fp, [(self.quad_vbo, '2f', 'in_pos')])

        # Blur program — separable Gaussian (reused for H and V passes)
        bp = ctx.program(vertex_shader=QUAD_VERT, fragment_shader=BLUR_FRAG)
        self.blur_prog  = bp
        self.blur_vao   = ctx.vertex_array(bp, [(self.quad_vbo, '2f', 'in_pos')])

        # Composite program — scene + bloom + tone map + vignette → screen
        cp = ctx.program(vertex_shader=QUAD_VERT, fragment_shader=COMP_FRAG)
        cp['u_scene'].value     = 0
        cp['u_bloom'].value     = 1
        cp['u_bloom_str'].value = BLOOM_STR
        self.comp_prog = cp
        self.comp_vao  = ctx.vertex_array(cp, [(self.quad_vbo, '2f', 'in_pos')])

    # ── helpers ─────────────────────────────────────────────────────────────────
    def _pitch_t(self) -> float:
        """Normalised pitch position [0=low/left … 1=high/right] from current hue."""
        h = self.hue
        # hue was mapped 240→385(=25°) so unwrap values below HUE_LO
        h_uw = h if h >= HUE_LO else h + 360.0
        return float(np.clip((h_uw - HUE_LO) / (HUE_HI_WRAP - HUE_LO), 0.0, 1.0))

    # ── audio → particle events ─────────────────────────────────────────────────
    def _drain_audio(self, dt: float):
        rms     = self.rms
        hue     = self.hue
        rms_hi  = 0.0
        rms_mid = 0.0
        rms_low = 0.0
        onsets: list[tuple[float, float]] = []

        try:
            while True:
                m       = self.audio.q.get_nowait()
                rms     = m["rms"]
                rms_hi  = m["rms_hi"]
                rms_mid = m["rms_mid"]
                rms_low = m["rms_low"]
                if m["pitch"] > 0.0:
                    hue = pitch_to_hue(m["pitch"])
                if m["onset"]:
                    onsets.append((m["ostr"], m["rms"]))
                self.wave.push(m["wave"], m["wave_mid"], m["wave_low"])
        except queue.Empty:
            pass

        self.rms     = rms
        self.hue     = hue
        self.rms_hi  = rms_hi
        self.rms_mid = rms_mid
        self.rms_low = rms_low

        # sustain envelope — lower gate so fingerpicking registers
        sus_gate = RMS_FLOOR * 0.6
        if rms > sus_gate:
            self.sustain = min(self.sustain + dt * SUS_ATK, SUS_MAX)
        else:
            self.sustain = max(self.sustain - dt * SUS_REL, 0.0)
        self.trail = self.sustain / SUS_MAX

        # ── wandering emitter: drifts like a floating leaf ───────────────────
        self.emit_vx += float(np.random.uniform(-40, 40))
        self.emit_vy += float(np.random.uniform(-28, 28))
        self.emit_vx *= 0.97
        self.emit_vy *= 0.97
        self.emit_x = float(np.clip(self.emit_x + self.emit_vx * dt, W * 0.06, W * 0.94))
        self.emit_y = float(np.clip(self.emit_y + self.emit_vy * dt, H * 0.06, H * 0.94))
        # soft bounce off edges
        if self.emit_x < W * 0.12: self.emit_vx += 80
        if self.emit_x > W * 0.88: self.emit_vx -= 80
        if self.emit_y < H * 0.12: self.emit_vy += 55
        if self.emit_y > H * 0.88: self.emit_vy -= 55

        # ── macro smoke layer — per-particle band dynamics
        macro_n = int(np.clip(80 + rms * 7000, 80, 200))
        self.parts.spawn(self.emit_x, self.emit_y, macro_n, hue,
                         rms, 0.0, 0.0, spread=260.0, macro=True,
                         rms_hi=rms_hi, rms_mid=rms_mid, rms_low=rms_low)

        # ── band-driven pulse rings from screen centre
        self.band_rings.try_spawn(0, rms_hi,  dt)
        self.band_rings.try_spawn(1, rms_mid, dt)
        self.band_rings.set_breath(rms_low, dt)

        # pitch-mapped position for energetic spawns
        pt = self._pitch_t()
        cx = W * (0.12 + pt * 0.76)
        cy = H * (0.60 - pt * 0.18)

        # onset: screen shake + rings + lightning only (no particle burst)
        for ostr, m_rms in onsets:
            if m_rms < ONSET_RMS_GATE:
                continue
            self.last_ostr = ostr
            burst_hue = (float(np.random.uniform(0, 360))
                         if self.color_mode == 'random' else hue)
            self.shake_px = min(self.shake_px + (m_rms * 80 + ostr * 4), 12.0)
            r_max = float(min(RING_R_BASE + m_rms * RING_R_RMS + ostr * RING_R_STR,
                              RING_R_CAP))
            self.rings.spawn(W * 0.5, H * 0.52, r_max, burst_hue)
            bolt_len = float(np.clip(120 + m_rms * 2500 + ostr * 150, 120, 520))
            self.lightning.spawn(W * 0.5, H * 0.52, bolt_len, burst_hue)

        # trickle when playing — also from emitter, tight, smoke-like
        if rms > RMS_FLOOR and not self.paused:
            n = min(int(rms * TRICKLE_K), TRICKLE_MAX)
            if n > 0:
                self.parts.spawn(self.emit_x, self.emit_y, n, hue, rms, 0.0,
                                 self.trail, spread=5.0)

    # ── render ──────────────────────────────────────────────────────────────────
    def _render(self):
        BW, BH = W // 2, H // 2

        # ① Scene FBO — fade (trail) then particles ───────────────────────────
        self.scene_fbo.use()
        self.ctx.viewport = (0, 0, W, H)

        # Trail fade: normal blend darkens existing HDR content each frame
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        fade_a = TRAIL_ALPHA_SHORT - self.trail * (TRAIL_ALPHA_SHORT - TRAIL_ALPHA_LONG)
        self.fade_prog['u_alpha'].value = float(fade_a)
        self.fade_vao.render(moderngl.TRIANGLE_STRIP)

        # Band pulse rings (from screen centre) — drawn first, sits under everything
        self.band_rings.render(self.shake_px)

        # Particles: additive blend into HDR scene
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        data, n = self.parts.vbo_data()
        if n > 0:
            sx = float(np.random.uniform(-1, 1) * self.shake_px)
            sy = float(np.random.uniform(-1, 1) * self.shake_px)
            self.prog['u_shake'].value = (sx, sy)
            self.vbo.write(data)
            self.vao.render(moderngl.POINTS, vertices=n)
        else:
            self.prog['u_shake'].value = (0.0, 0.0)

        # Onset shockwave rings (from emitter)
        self.rings.render(self.shake_px)

        # Lightning: electric arcs (very short life, bloom makes them glow white)
        self.lightning.render(self.shake_px)

        # Waveform: 3-band oscilloscope — purple hi / green mid / blue low (top layer)
        self.wave.render(self.shake_px, self.rms)

        # ② Bloom pass 1 — tight glow (H then V at half res) ─────────────────
        self.bloom_fbo_h.use()
        self.ctx.viewport = (0, 0, BW, BH)
        self.ctx.clear(0, 0, 0, 1)
        self.ctx.blend_func = moderngl.ONE, moderngl.ZERO
        self.scene_tex.use(0)
        self.blur_prog['u_tex'].value = 0
        self.blur_prog['u_dir'].value = (1.0 / BW, 0.0)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        self.bloom_fbo_v.use()
        self.ctx.clear(0, 0, 0, 1)
        self.bloom_tex_h.use(0)
        self.blur_prog['u_dir'].value = (0.0, 1.0 / BH)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        # ③ Bloom pass 2 — wider glow (blur the blurred result again) ─────────
        self.bloom_fbo_h.use()
        self.ctx.clear(0, 0, 0, 1)
        self.bloom_tex_v.use(0)
        self.blur_prog['u_dir'].value = (1.0 / BW, 0.0)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        self.bloom_fbo_v.use()
        self.ctx.clear(0, 0, 0, 1)
        self.bloom_tex_h.use(0)
        self.blur_prog['u_dir'].value = (0.0, 1.0 / BH)
        self.blur_vao.render(moderngl.TRIANGLE_STRIP)

        # ④ Composite → screen: scene + bloom + tone map + vignette ──────────
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.clear(0, 0, 0, 1)
        self.scene_tex.use(0)
        self.bloom_tex_v.use(1)
        self.comp_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()

    # ── main loop ───────────────────────────────────────────────────────────────
    def run(self):
        t0        = time.perf_counter()
        run       = True
        debug_t   = 0.0   # accumulator for debug print

        print(f"Running  —  {W}×{H}  |  buffer {HOP}/{SR} = {1000*HOP/SR:.1f} ms  "
              f"|  pool {MAX_P:,}  |  FPS cap {FPS}")
        print("ESC = quit   SPACE = pause/resume   C = color mode (pitch/random)\n")

        while run:
            now = time.perf_counter()
            dt  = min(now - t0, 0.05)   # cap at 50 ms (protects spiral-of-death)
            t0  = now

            for ev in pygame.event.get():
                if ev.type == QUIT:
                    run = False
                elif ev.type == KEYDOWN:
                    if ev.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit(0)
                    elif ev.key == K_SPACE:
                        self.paused = not self.paused
                        print("[vis]", "paused" if self.paused else "resumed")
                    elif ev.key == K_f:
                        pygame.display.toggle_fullscreen()
                    elif ev.key == K_c:
                        self.color_mode = ('random' if self.color_mode == 'pitch'
                                           else 'pitch')
                        print(f"[vis] color mode: {self.color_mode}")

            self._drain_audio(dt)

            if not self.paused:
                self.parts.update(dt, self.trail)
                self.rings.update(dt)
                self.band_rings.update(dt)
                self.lightning.update(dt)
                # decay screen shake
                self.shake_px = max(0.0, self.shake_px - self.shake_px * 10.0 * dt)

            self._render()

            # debug print every 1 s
            debug_t += dt
            if debug_t >= 1.0:
                debug_t = 0.0
                alive = int((self.parts.life > 0).sum())
                fps   = self.clock.get_fps()
                # also show last onset strength if any fired recently
                rings_alive = int(self.rings.alive.sum())
                print(f"RMS={self.rms:.4f}  pitch={self.hue:.0f}°hue  "
                      f"sustain={self.trail:.2f}  alive={alive:,}  rings={rings_alive}  "
                      f"fps={fps:.0f}  ostr={self.last_ostr:.2f}  "
                      f"color={self.color_mode}  "
                      f"[gate={'OPEN' if self.rms >= RMS_FLOOR else 'closed'}]")

            self.clock.tick(FPS)

        pygame.quit()
        sys.exit(0)


# ── entry point ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Electric guitar real-time visualizer")
    parser.add_argument("--windowed", action="store_true",
                        help="Run in 1280×720 window (default: fullscreen at native res)")
    args = parser.parse_args()

    print("═══ Guitar Visual System ══════════════════════════════════════════")
    dev   = select_device()
    audio = Audio(dev)
    audio.start()
    try:
        Visualizer(audio, windowed=args.windowed).run()
    finally:
        audio.stop()
        print("Bye!")


if __name__ == "__main__":
    main()
