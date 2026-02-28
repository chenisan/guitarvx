# GuitarViz

Real-time electric guitar visualizer with GPU-accelerated particle effects.

## ⬇️ Download

[![Download](https://img.shields.io/github/v/release/chenisan/guitarvx?label=Download%20Installer&style=for-the-badge&color=6a0dad)](https://github.com/chenisan/guitarvx/releases/latest/download/GuitarViz_Setup.exe)

> Windows 10/11 x64 only — double-click to install, no other software required.

---

## Features

- Real-time pitch and onset detection via aubio (YIN + HFC)
- GPU-accelerated particle system with fluid curl-noise motion
- Three-band frequency visualization — purple (high) / green (mid) / blue (low)
- Bloom glow, trail effects and screen shake on note attacks
- Breathing blue ring that pulses with low-frequency energy
- Fullscreen at native resolution (tested at 2560×1440, 144 fps)

## System Requirements

| | |
|---|---|
| OS | Windows 10 / 11 x64 |
| GPU | OpenGL 4.1+ (any modern discrete or integrated GPU) |
| Audio | Microphone or audio interface |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `ESC` | Quit |
| `Space` | Pause / Resume |
| `F` | Toggle fullscreen |
| `C` | Toggle color mode (pitch / random) |

## License

MIT License — see [LICENSE](LICENSE)
