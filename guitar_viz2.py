#!/usr/bin/env python3
"""
guitar_viz2.py  —  Fluid Guitar Visualizer
Real-time Navier-Stokes GPU fluid simulation driven by guitar audio.
Low (80-300 Hz) → blue dye  |  Mid (300-1500 Hz) → green  |  High (1500-5000 Hz) → purple
ESC=quit  Space=pause  F=fullscreen
"""

import sys, argparse, queue, time
import numpy as np

_miss = []
try:
    import pygame
    from pygame.locals import (OPENGL, DOUBLEBUF, FULLSCREEN, QUIT,
                                KEYDOWN, K_ESCAPE, K_SPACE, K_f)
except ImportError: _miss.append("pygame")
try:    import moderngl
except ImportError: _miss.append("moderngl")
try:    import sounddevice as sd
except ImportError: _miss.append("sounddevice")
try:    import aubio
except ImportError: _miss.append("aubio")

if _miss:
    print("Missing:", ", ".join(_miss), "\npip install", " ".join(_miss))
    sys.exit(1)

# ── config ────────────────────────────────────────────────────────────────────

SR, HOP, WIN_P, WIN_O = 44100, 512, 2048, 1024
W, H         = 1920, 1080          # overridden by DPI fix at runtime
SIM_W, SIM_H = 512, 288            # fluid grid (16:9)
FPS          = 144

PRESSURE_ITERS  = 25
VEL_DISSIPATION = 0.998            # velocity persists
DYE_DISSIPATION = 0.992            # dye fades slowly
VORTICITY_EPS   = 22.0             # swirl confinement strength
SPLAT_RADIUS    = 0.08             # UV-space radius
SPLAT_FORCE     = 2200.0
DYE_AMOUNT      = 5.0
ONSET_FORCE     = 5200.0
ONSET_RADIUS    = 0.20
BLOOM_STR       = 0.90

RMS_FLOOR      = 0.003
ONSET_THOLD    = 0.5
ONSET_RMS_GATE = 0.003
CONF_MIN       = 0.55
HZ_LO, HZ_HI  = 80.0, 1400.0
MARGIN         = 0.12

WAVE_N  = 2048
WAVE_CY = -0.08   # NDC y of waveform centre

# ── shaders ───────────────────────────────────────────────────────────────────

QUAD_VERT = """
#version 410 core
in vec2 in_pos;
out vec2 vUV;
void main() { vUV = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

ADVECT_FRAG = """
#version 410 core
uniform sampler2D u_vel;
uniform sampler2D u_src;
uniform float u_dt;
uniform float u_diss;
in vec2 vUV; out vec4 f;
void main() {
    vec2 v = texture(u_vel, vUV).xy;
    vec2 c = clamp(vUV - v * u_dt, 0.001, 0.999);
    f = vec4(texture(u_src, c).rgb * u_diss, 1.0);
}
"""

SPLAT_FRAG = """
#version 410 core
uniform sampler2D u_src;
uniform vec2  u_pt;
uniform vec3  u_val;
uniform float u_r;
uniform float u_asp;
in vec2 vUV; out vec4 f;
void main() {
    vec2 d = vUV - u_pt; d.x *= u_asp;
    float sp = exp(-dot(d, d) / (u_r * u_r));
    f = vec4(texture(u_src, vUV).rgb + u_val * sp, 1.0);
}
"""

DIVERGENCE_FRAG = """
#version 410 core
uniform sampler2D u_vel;
uniform float u_tx; uniform float u_ty;
in vec2 vUV; out vec4 f;
void main() {
    float L = texture(u_vel, vUV - vec2(u_tx, 0)).x;
    float R = texture(u_vel, vUV + vec2(u_tx, 0)).x;
    float B = texture(u_vel, vUV - vec2(0, u_ty)).y;
    float T = texture(u_vel, vUV + vec2(0, u_ty)).y;
    f = vec4(0.5 * (R - L + T - B), 0, 0, 1);
}
"""

CURL_FRAG = """
#version 410 core
uniform sampler2D u_vel;
uniform float u_tx; uniform float u_ty;
in vec2 vUV; out vec4 f;
void main() {
    float L = texture(u_vel, vUV - vec2(u_tx, 0)).y;
    float R = texture(u_vel, vUV + vec2(u_tx, 0)).y;
    float B = texture(u_vel, vUV - vec2(0, u_ty)).x;
    float T = texture(u_vel, vUV + vec2(0, u_ty)).x;
    f = vec4((R - L - T + B) * 0.5, 0, 0, 1);
}
"""

VORTICITY_FRAG = """
#version 410 core
uniform sampler2D u_vel;
uniform sampler2D u_curl;
uniform float u_tx; uniform float u_ty;
uniform float u_eps; uniform float u_dt;
in vec2 vUV; out vec4 f;
void main() {
    float L = texture(u_curl, vUV - vec2(u_tx, 0)).r;
    float R = texture(u_curl, vUV + vec2(u_tx, 0)).r;
    float B = texture(u_curl, vUV - vec2(0, u_ty)).r;
    float T = texture(u_curl, vUV + vec2(0, u_ty)).r;
    float c = texture(u_curl, vUV).r;
    vec2 force = 0.5 * vec2(abs(T) - abs(B), abs(R) - abs(L));
    force = normalize(force + 1e-5) * c * u_eps;
    f = vec4(texture(u_vel, vUV).xy + force * u_dt, 0, 1);
}
"""

PRESSURE_FRAG = """
#version 410 core
uniform sampler2D u_p;
uniform sampler2D u_div;
uniform float u_tx; uniform float u_ty;
in vec2 vUV; out vec4 f;
void main() {
    float L = texture(u_p, vUV - vec2(u_tx, 0)).r;
    float R = texture(u_p, vUV + vec2(u_tx, 0)).r;
    float B = texture(u_p, vUV - vec2(0, u_ty)).r;
    float T = texture(u_p, vUV + vec2(0, u_ty)).r;
    f = vec4((L + R + B + T - texture(u_div, vUV).r) * 0.25, 0, 0, 1);
}
"""

GRADIENT_SUB_FRAG = """
#version 410 core
uniform sampler2D u_p;
uniform sampler2D u_vel;
uniform float u_tx; uniform float u_ty;
in vec2 vUV; out vec4 f;
void main() {
    float L = texture(u_p, vUV - vec2(u_tx, 0)).r;
    float R = texture(u_p, vUV + vec2(u_tx, 0)).r;
    float B = texture(u_p, vUV - vec2(0, u_ty)).r;
    float T = texture(u_p, vUV + vec2(0, u_ty)).r;
    f = vec4(texture(u_vel, vUV).xy - 0.5 * vec2(R - L, T - B), 0, 1);
}
"""

# Dye → screen colours: R=low→blue  G=mid→green  B=high→purple
DISPLAY_FRAG = """
#version 410 core
uniform sampler2D u_dye;
in vec2 vUV; out vec4 f;
void main() {
    vec3 d = max(texture(u_dye, vUV).rgb, 0.0);
    vec3 col = d.r * vec3(0.05, 0.38, 1.00)
             + d.g * vec3(0.05, 1.00, 0.25)
             + d.b * vec3(0.68, 0.05, 1.00);
    col = col / (col + 1.0);
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
    f = vec4(col, 1.0);
}
"""

BLUR_FRAG = """
#version 410 core
uniform sampler2D u_tex;
uniform vec2 u_dir;
in vec2 vUV; out vec4 f;
void main() {
    float k[9] = float[](0.0625,0.0625,0.125,0.125,0.25,0.125,0.125,0.0625,0.0625);
    vec3 c = vec3(0.0);
    for (int i = 0; i < 9; i++)
        c += k[i] * texture(u_tex, vUV + u_dir * float(i - 4)).rgb;
    f = vec4(c, 1.0);
}
"""

COMP_FRAG = """
#version 410 core
uniform sampler2D u_base;
uniform sampler2D u_bloom;
uniform float u_str;
in vec2 vUV; out vec4 f;
void main() {
    vec3 col = texture(u_base, vUV).rgb + texture(u_bloom, vUV).rgb * u_str;
    vec2 uv   = vUV * 2.0 - 1.0;
    float vig = 1.0 - dot(uv, uv) * 0.22;
    f = vec4(col * vig, 1.0);
}
"""

WAVE_VERT = """
#version 410 core
in float in_y;
uniform float u_cy;
uniform float u_yscale;
uniform float u_xstep;
uniform float u_shake;
out float vR;
void main() {
    float x = float(gl_VertexID) * u_xstep * 2.0 - 1.0;
    float y = u_cy + in_y * u_yscale + u_shake;
    vR = float(gl_VertexID) / float(2047);
    gl_Position = vec4(x, y, 0.0, 1.0);
}
"""

WAVE_FRAG = """
#version 410 core
uniform float u_hue;
uniform float u_alpha;
in float vR; out vec4 f;
vec3 hsl2rgb(float h, float s, float l) {
    h = mod(h, 360.0) / 60.0;
    float c = (1.0 - abs(2.0*l - 1.0)) * s;
    float x = c * (1.0 - abs(mod(h, 2.0) - 1.0));
    vec3 r;
    if      (h < 1.0) r = vec3(c, x, 0);
    else if (h < 2.0) r = vec3(x, c, 0);
    else if (h < 3.0) r = vec3(0, c, x);
    else if (h < 4.0) r = vec3(0, x, c);
    else if (h < 5.0) r = vec3(x, 0, c);
    else              r = vec3(c, 0, x);
    return r + (l - c * 0.5);
}
void main() {
    float e = sin(vR * 3.14159) * 0.35 + 0.65;
    f = vec4(hsl2rgb(u_hue, 0.90, 0.60) * e, u_alpha * e);
}
"""

# ── helpers ───────────────────────────────────────────────────────────────────

def band_filter(signal: np.ndarray, sr: int, lo_hz: float, hi_hz: float) -> np.ndarray:
    spec  = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), 1.0 / sr)
    if lo_hz  > 0:    spec[freqs < lo_hz]  = 0.0
    if hi_hz  < sr/2: spec[freqs > hi_hz]  = 0.0
    return np.fft.irfft(spec, n=len(signal)).astype(np.float32)

def pitch_to_x(freq: float) -> float:
    if freq <= 0: return 0.5
    t = float(np.clip(np.log(freq / HZ_LO) / np.log(HZ_HI / HZ_LO), 0.0, 1.0))
    return MARGIN + t * (1.0 - 2 * MARGIN)

def _fatal_dialog(msg: str):
    try:
        import tkinter as tk; from tkinter import messagebox
        r = tk.Tk(); r.withdraw()
        messagebox.showerror("GuitarViz2 — Error", msg); r.destroy()
    except Exception: print(msg)

def _select_device_gui(input_devs, default_idx):
    try:
        import tkinter as tk; from tkinter import ttk
        chosen = [default_idx]
        root = tk.Tk(); root.title("GuitarViz2 — Select Audio Input")
        root.resizable(False, False); root.configure(bg="#0d0d1a")
        tk.Label(root, text="Select audio input device:", bg="#0d0d1a", fg="white",
                 font=("Segoe UI", 11)).pack(padx=20, pady=(16, 6))
        var = tk.StringVar()
        labels = [f"{i:3d}  {d['name']}" for i, d in input_devs]
        var.set(next((l for l in labels if l.startswith(f"{default_idx:3d}")), labels[0]))
        ttk.Combobox(root, textvariable=var, values=labels,
                     state="readonly", width=52, font=("Consolas", 10)).pack(padx=20, pady=4)
        def ok():
            chosen[0] = int(var.get().split()[0]); root.destroy()
        tk.Button(root, text="  Start  ", command=ok, bg="#4b0082", fg="white",
                  font=("Segoe UI", 11, "bold"), relief="flat", padx=12, pady=6).pack(pady=(10, 16))
        root.bind("<Return>", lambda _: ok())
        root.eval("tk::PlaceWindow . center"); root.mainloop()
        return chosen[0]
    except Exception: return default_idx

def select_device() -> int:
    devs = sd.query_devices()
    input_devs = [(i, d) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
    if not input_devs:
        _fatal_dialog("No audio input devices found."); sys.exit(1)
    default_idx = sd.default.device[0]
    if default_idx < 0 or default_idx >= len(devs):
        default_idx = input_devs[0][0]
    if sys.stdin and sys.stdin.isatty():
        print("\n═══ Audio Devices ═══")
        for i, d in input_devs:
            print(f"{'◀' if i==default_idx else ' '} [{i:3d}]  {d['name']}")
        try:
            ans = input(f"Device [{default_idx}]: ").strip()
            return int(ans) if ans else default_idx
        except (ValueError, EOFError): return default_idx
    return _select_device_gui(input_devs, default_idx)

# ── ping-pong FBO pair ────────────────────────────────────────────────────────

class PingPong:
    def __init__(self, ctx, w, h, components):
        self._texs, self._fbos = [], []
        for _ in range(2):
            t = ctx.texture((w, h), components, dtype='f4')
            t.filter = moderngl.LINEAR, moderngl.LINEAR
            t.repeat_x = t.repeat_y = False
            self._texs.append(t)
            self._fbos.append(ctx.framebuffer(color_attachments=[t]))
        self._i = 0

    @property
    def read(self):       return self._texs[self._i]
    @property
    def write_fbo(self):  return self._fbos[1 - self._i]
    def swap(self):       self._i = 1 - self._i

# ── GPU fluid simulation ──────────────────────────────────────────────────────

class FluidSim:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        tx, ty = 1.0 / SIM_W, 1.0 / SIM_H
        self._tx, self._ty = tx, ty
        self._asp = SIM_W / SIM_H

        quad = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32)
        self._qvbo = ctx.buffer(quad)

        def prog(frag):
            return ctx.program(vertex_shader=QUAD_VERT, fragment_shader=frag)

        self._p = {
            'advect':   prog(ADVECT_FRAG),
            'splat':    prog(SPLAT_FRAG),
            'div':      prog(DIVERGENCE_FRAG),
            'curl':     prog(CURL_FRAG),
            'vort':     prog(VORTICITY_FRAG),
            'pressure': prog(PRESSURE_FRAG),
            'grad':     prog(GRADIENT_SUB_FRAG),
            'display':  prog(DISPLAY_FRAG),
        }
        self._vaos = {
            name: ctx.vertex_array(p, [(self._qvbo, '2f', 'in_pos')])
            for name, p in self._p.items()
        }

        self.vel_pp  = PingPong(ctx, SIM_W, SIM_H, 2)
        self.dye_pp  = PingPong(ctx, SIM_W, SIM_H, 3)
        self.pres_pp = PingPong(ctx, SIM_W, SIM_H, 1)

        def single(c):
            t = ctx.texture((SIM_W, SIM_H), c, dtype='f4')
            t.filter = moderngl.LINEAR, moderngl.LINEAR
            t.repeat_x = t.repeat_y = False
            return ctx.framebuffer(color_attachments=[t]), t

        self._div_fbo,  self._div_tex  = single(1)
        self._curl_fbo, self._curl_tex = single(1)

        self._vel_splats = []
        self._dye_splats = []

    def _run(self, name, fbo, tex_map, uniforms):
        prog = self._p[name]
        fbo.use()
        self.ctx.viewport = (0, 0, SIM_W, SIM_H)
        for slot, tex in tex_map.items():
            tex.use(location=slot)
        for k, v in uniforms.items():
            if k in prog:
                prog[k].value = v
        self._vaos[name].render(moderngl.TRIANGLE_STRIP)

    def add_velocity(self, xu, yu, vx, vy, r=None):
        self._vel_splats.append((xu, yu, vx, vy, r or SPLAT_RADIUS))

    def add_dye(self, xu, yu, dr, dg, db, r=None):
        self._dye_splats.append((xu, yu, dr, dg, db, r or SPLAT_RADIUS))

    def step(self, dt):
        tx, ty, asp = self._tx, self._ty, self._asp

        # 1. Advect velocity
        self._run('advect', self.vel_pp.write_fbo,
                  {0: self.vel_pp.read, 1: self.vel_pp.read},
                  {'u_vel': 0, 'u_src': 1, 'u_dt': dt, 'u_diss': VEL_DISSIPATION})
        self.vel_pp.swap()

        # 2. Velocity splats
        for xu, yu, vx, vy, r in self._vel_splats:
            self._run('splat', self.vel_pp.write_fbo,
                      {0: self.vel_pp.read},
                      {'u_src': 0, 'u_pt': (xu, yu), 'u_val': (vx, vy, 0.0),
                       'u_r': r, 'u_asp': asp})
            self.vel_pp.swap()
        self._vel_splats.clear()

        # 3. Curl
        self._run('curl', self._curl_fbo,
                  {0: self.vel_pp.read},
                  {'u_vel': 0, 'u_tx': tx, 'u_ty': ty})

        # 4. Vorticity confinement
        self._run('vort', self.vel_pp.write_fbo,
                  {0: self.vel_pp.read, 1: self._curl_tex},
                  {'u_vel': 0, 'u_curl': 1, 'u_tx': tx, 'u_ty': ty,
                   'u_eps': VORTICITY_EPS, 'u_dt': dt})
        self.vel_pp.swap()

        # 5. Divergence
        self._run('div', self._div_fbo,
                  {0: self.vel_pp.read},
                  {'u_vel': 0, 'u_tx': tx, 'u_ty': ty})

        # 6. Pressure (Jacobi)
        for _ in range(PRESSURE_ITERS):
            self._run('pressure', self.pres_pp.write_fbo,
                      {0: self.pres_pp.read, 1: self._div_tex},
                      {'u_p': 0, 'u_div': 1, 'u_tx': tx, 'u_ty': ty})
            self.pres_pp.swap()

        # 7. Gradient subtraction
        self._run('grad', self.vel_pp.write_fbo,
                  {0: self.pres_pp.read, 1: self.vel_pp.read},
                  {'u_p': 0, 'u_vel': 1, 'u_tx': tx, 'u_ty': ty})
        self.vel_pp.swap()

        # 8. Advect dye
        self._run('advect', self.dye_pp.write_fbo,
                  {0: self.vel_pp.read, 1: self.dye_pp.read},
                  {'u_vel': 0, 'u_src': 1, 'u_dt': dt, 'u_diss': DYE_DISSIPATION})
        self.dye_pp.swap()

        # 9. Dye splats
        for xu, yu, dr, dg, db, r in self._dye_splats:
            self._run('splat', self.dye_pp.write_fbo,
                      {0: self.dye_pp.read},
                      {'u_src': 0, 'u_pt': (xu, yu), 'u_val': (dr, dg, db),
                       'u_r': r, 'u_asp': asp})
            self.dye_pp.swap()
        self._dye_splats.clear()

    def blit(self, target_fbo):
        """Render dye field to target_fbo."""
        target_fbo.use()
        self.ctx.viewport = (0, 0, target_fbo.width, target_fbo.height)
        self.dye_pp.read.use(location=0)
        prog = self._p['display']
        if 'u_dye' in prog:
            prog['u_dye'].value = 0
        self._vaos['display'].render(moderngl.TRIANGLE_STRIP)

# ── audio ─────────────────────────────────────────────────────────────────────

class Audio:
    def __init__(self, dev: int):
        self.q = queue.Queue(maxsize=64)
        self._pitch = aubio.pitch("yin", WIN_P, HOP, SR)
        self._pitch.set_unit("Hz"); self._pitch.set_silence(-40)
        self._onset = aubio.onset("hfc", WIN_O, HOP, SR)
        self._onset.set_threshold(ONSET_THOLD)
        self._stream = sd.InputStream(
            device=dev, samplerate=SR, blocksize=HOP,
            channels=1, dtype='float32', latency='low', callback=self._cb)

    def _cb(self, indata, frames, time_info, status):
        mono = indata[:, 0].copy()
        pitch = float(self._pitch(mono)[0])
        conf  = float(self._pitch.get_confidence())
        onset = bool(self._onset(mono)[0])
        ostr  = float(self._onset.get_descriptor()) if onset else 0.0
        wave_hi  = band_filter(mono, SR, 1500, 5000)
        wave_mid = band_filter(mono, SR,  300, 1500)
        wave_low = band_filter(mono, SR,   80,  300)
        rms_hi   = float(np.sqrt(np.mean(wave_hi  ** 2)))
        rms_mid  = float(np.sqrt(np.mean(wave_mid ** 2)))
        rms_low  = float(np.sqrt(np.mean(wave_low ** 2)))
        rms      = float(np.sqrt(np.mean(mono ** 2)))
        if conf < CONF_MIN: pitch = 0.0
        try:
            self.q.put_nowait({'rms': rms, 'pitch': pitch, 'onset': onset, 'ostr': ostr,
                               'wave': wave_hi, 'wave_mid': wave_mid, 'wave_low': wave_low,
                               'rms_hi': rms_hi, 'rms_mid': rms_mid, 'rms_low': rms_low})
        except queue.Full: pass

    def start(self): self._stream.start()
    def stop(self):  self._stream.stop(); self._stream.close()

# ── waveform overlay ──────────────────────────────────────────────────────────

class WaveformDisplay:
    # (hue°, amplitude boost, line width px)
    BANDS = [(272.0, 2.5, 1.5), (120.0, 1.5, 1.5), (210.0, 1.0, 2.0)]

    def __init__(self, ctx: moderngl.Context):
        self.ctx  = ctx
        self._buf = [np.zeros(WAVE_N, np.float32) for _ in range(3)]
        self._vbo = [ctx.buffer(b) for b in self._buf]
        self._prg = ctx.program(vertex_shader=WAVE_VERT, fragment_shader=WAVE_FRAG)
        self._vao = [ctx.vertex_array(self._prg, [(v, '1f', 'in_y')]) for v in self._vbo]

    def update(self, hi, mid, low):
        for i, d in enumerate([hi, mid, low]):
            n = min(len(d), WAVE_N)
            self._buf[i][:n] = d[:n]
            if n < WAVE_N: self._buf[i][n:] = 0.0
            self._vbo[i].write(self._buf[i].tobytes())

    def render(self, shake_px: float, rms: float):
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        xstep = 1.0 / WAVE_N
        for i, (hue, boost, lw) in enumerate(self.BANDS):
            scale = float(np.clip(50.0 + rms * 2000.0, 50.0, 150.0)) * boost / H
            self.ctx.line_width = lw
            self._prg['u_cy'].value     = WAVE_CY
            self._prg['u_yscale'].value = scale
            self._prg['u_xstep'].value  = xstep
            self._prg['u_shake'].value  = float(np.random.uniform(-1, 1) * shake_px / W)
            self._prg['u_hue'].value    = hue
            self._prg['u_alpha'].value  = 0.50
            self._vao[i].render(moderngl.LINE_STRIP, vertices=WAVE_N)

# ── main visualizer ───────────────────────────────────────────────────────────

class Visualizer2:
    def __init__(self, audio: Audio, windowed: bool = False):
        global W, H
        self.audio = audio

        pygame.init()
        if windowed:
            W, H = 1280, 720
            screen = pygame.display.set_mode((W, H), OPENGL | DOUBLEBUF)
        else:
            info = pygame.display.Info()
            W, H = info.current_w, info.current_h
            screen = pygame.display.set_mode((W, H), OPENGL | DOUBLEBUF | FULLSCREEN)
        pygame.display.set_caption("GuitarViz2 — Fluid")

        ctx = moderngl.create_context()
        ctx.enable(moderngl.BLEND)

        # DPI fix
        gl_W, gl_H = ctx.screen.size
        if (gl_W, gl_H) != (W, H):
            print(f"[display] DPI: pygame={W}×{H}  GL={gl_W}×{gl_H}")
            W, H = gl_W, gl_H
        self.ctx = ctx

        # Bloom FBOs (half res)
        BW, BH = W // 2, H // 2
        def ftex(w, h, c=3):
            t = ctx.texture((w, h), c, dtype='f4')
            t.filter = moderngl.LINEAR, moderngl.LINEAR
            t.repeat_x = t.repeat_y = False
            return t
        self._scene_tex = ftex(W, H)
        self._scene_fbo = ctx.framebuffer(color_attachments=[self._scene_tex])
        self._blur_h_tex = ftex(BW, BH)
        self._blur_h_fbo = ctx.framebuffer(color_attachments=[self._blur_h_tex])
        self._blur_v_tex = ftex(BW, BH)
        self._blur_v_fbo = ctx.framebuffer(color_attachments=[self._blur_v_tex])

        # Post-processing programs
        quad = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype=np.float32)
        qvbo = ctx.buffer(quad)
        def pp(frag):
            p = ctx.program(vertex_shader=QUAD_VERT, fragment_shader=frag)
            return p, ctx.vertex_array(p, [(qvbo, '2f', 'in_pos')])
        self._blur_p, self._blur_vao = pp(BLUR_FRAG)
        self._comp_p, self._comp_vao = pp(COMP_FRAG)

        self.fluid = FluidSim(ctx)
        self.wave  = WaveformDisplay(ctx)
        self.clock = pygame.time.Clock()

        self._pitch_x  = 0.5
        self._shake    = 0.0
        self._paused   = False
        self.rms       = 0.0

    def _drain_audio(self, dt: float):
        while not self.audio.q.empty():
            try: d = self.audio.q.get_nowait()
            except queue.Empty: break

            rms     = d['rms']
            rms_hi  = d['rms_hi']
            rms_mid = d['rms_mid']
            rms_low = d['rms_low']
            pitch   = d['pitch']
            ostr    = d['ostr']

            self.wave.update(d['wave'], d['wave_mid'], d['wave_low'])

            if rms < RMS_FLOOR: continue
            self.rms = rms

            # Smooth pitch → x position
            if pitch > 0:
                tx = pitch_to_x(pitch)
                k  = 1.0 - np.exp(-dt * 14.0)
                self._pitch_x += (tx - self._pitch_x) * k
            px = float(self._pitch_x)

            asp = SIM_W / SIM_H

            # ── low band: blue dye injected from lower region, rises
            if rms_low > RMS_FLOOR * 0.5:
                e = float(np.clip(rms_low * 10.0, 0.0, 1.0))
                self.fluid.add_velocity(px, 0.72, 0.0, -SPLAT_FORCE * e,
                                        SPLAT_RADIUS * 1.2)
                self.fluid.add_dye(px, 0.72,
                                   rms_low * DYE_AMOUNT * 140, 0.0, 0.0,
                                   SPLAT_RADIUS * 1.2)

            # ── mid band: green dye from centre
            if rms_mid > RMS_FLOOR * 0.5:
                e = float(np.clip(rms_mid * 10.0, 0.0, 1.0))
                self.fluid.add_velocity(px, 0.50, 0.0, -SPLAT_FORCE * e * 0.8,
                                        SPLAT_RADIUS)
                self.fluid.add_dye(px, 0.50,
                                   0.0, rms_mid * DYE_AMOUNT * 110, 0.0,
                                   SPLAT_RADIUS)

            # ── high band: purple dye from upper region, erratic direction
            if rms_hi > RMS_FLOOR * 0.3:
                e = float(np.clip(rms_hi * 12.0, 0.0, 1.0))
                ang = float(np.random.uniform(0, 2 * np.pi))
                vx  = float(np.cos(ang) * SPLAT_FORCE * e * 0.4)
                vy  = float(-abs(np.sin(ang)) * SPLAT_FORCE * e * 0.6)
                self.fluid.add_velocity(px, 0.30, vx, vy, SPLAT_RADIUS * 0.75)
                self.fluid.add_dye(px, 0.30,
                                   0.0, 0.0, rms_hi * DYE_AMOUNT * 100,
                                   SPLAT_RADIUS * 0.75)

            # ── onset: radial burst from pitch position
            if d['onset'] and ostr > ONSET_THOLD and rms > ONSET_RMS_GATE:
                self._shake = min(18.0, self._shake + ostr * 10.0)
                burst = float(np.clip(ostr * 1.8, 0.5, 3.0))
                for ang in np.linspace(0, 2 * np.pi, 10, endpoint=False):
                    vx = float(np.cos(ang) * ONSET_FORCE * burst * 0.25)
                    vy = float(np.sin(ang) * ONSET_FORCE * burst * 0.25)
                    self.fluid.add_velocity(px, 0.50, vx, vy, ONSET_RADIUS)
                self.fluid.add_dye(px, 0.50,
                                   burst * 0.8, burst * 0.7, burst * 1.0,
                                   ONSET_RADIUS)

    def _render(self, dt: float):
        BW, BH = W // 2, H // 2

        # ── fluid step
        self.fluid.step(dt)

        # ── render dye to scene FBO
        self._scene_fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.ctx.clear(0.0, 0.0, 0.0)
        self.fluid.blit(self._scene_fbo)

        # ── waveform on top (additive, same scene FBO still bound)
        self._scene_fbo.use()
        self.ctx.viewport = (0, 0, W, H)
        self.wave.render(self._shake, self.rms)

        # ── bloom: blur_h (scene → half-res H)
        self._blur_h_fbo.use()
        self.ctx.viewport = (0, 0, BW, BH)
        self._scene_tex.use(location=0)
        self._blur_p['u_tex'].value = 0
        self._blur_p['u_dir'].value = (1.0 / BW, 0.0)
        self._blur_vao.render(moderngl.TRIANGLE_STRIP)

        # ── bloom: blur_v (blur_h → half-res V)
        self._blur_v_fbo.use()
        self.ctx.viewport = (0, 0, BW, BH)
        self._blur_h_tex.use(location=0)
        self._blur_p['u_dir'].value = (0.0, 1.0 / BH)
        self._blur_vao.render(moderngl.TRIANGLE_STRIP)

        # ── composite to screen
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, W, H)
        self._scene_tex.use(location=0)
        self._blur_v_tex.use(location=1)
        self._comp_p['u_base'].value  = 0
        self._comp_p['u_bloom'].value = 1
        self._comp_p['u_str'].value   = BLOOM_STR
        self._comp_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()

    def run(self):
        run = True
        t0  = time.perf_counter()
        while run:
            now = time.perf_counter()
            dt  = min(now - t0, 1.0 / 30.0)
            t0  = now

            for ev in pygame.event.get():
                if ev.type == QUIT:
                    pygame.quit(); sys.exit(0)
                elif ev.type == KEYDOWN:
                    if ev.key == K_ESCAPE:
                        pygame.quit(); sys.exit(0)
                    elif ev.key == K_SPACE:
                        self._paused = not self._paused
                    elif ev.key == K_f:
                        pygame.display.toggle_fullscreen()

            self._drain_audio(dt)
            self._shake = max(0.0, self._shake - self._shake * 8.0 * dt)

            if not self._paused:
                self._render(dt)

            self.clock.tick(FPS)

# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GuitarViz2 — Fluid Visualizer")
    parser.add_argument("--windowed", action="store_true",
                        help="Run in 1280×720 window")
    args = parser.parse_args()
    print("═══ GuitarViz2 — Fluid Mode ═══")
    dev   = select_device()
    audio = Audio(dev)
    audio.start()
    try:
        Visualizer2(audio, windowed=args.windowed).run()
    finally:
        audio.stop()
        print("Bye!")

if __name__ == "__main__":
    main()
