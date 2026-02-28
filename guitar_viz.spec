# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for GuitarViz
# Run from the project folder:  pyinstaller guitar_viz.spec

from PyInstaller.utils.hooks import collect_all, collect_dynamic_libs

block_cipher = None

# Collect all submodules + data + binaries for each dependency
pygame_d,     pygame_b,     pygame_h     = collect_all('pygame')
moderngl_d,   moderngl_b,   moderngl_h   = collect_all('moderngl')
aubio_d,      aubio_b,      aubio_h      = collect_all('aubio')
sounddev_d,   sounddev_b,   sounddev_h   = collect_all('sounddevice')
sddata_d,     sddata_b,     sddata_h     = collect_all('_sounddevice_data')
numpy_d,      numpy_b,      numpy_h      = collect_all('numpy')

a = Analysis(
    ['guitar_viz.py'],
    pathex=['.'],
    binaries=(
        pygame_b + moderngl_b + aubio_b + sounddev_b + sddata_b + numpy_b
    ),
    datas=(
        pygame_d + moderngl_d + aubio_d + sounddev_d + sddata_d + numpy_d
    ),
    hiddenimports=(
        pygame_h + moderngl_h + aubio_h + sounddev_h + sddata_h + numpy_h + [
            'moderngl.mgl',
            'numpy.core._multiarray_umath',
            'numpy.core._multiarray_tests',
        ]
    ),
    hookspath=[],
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'PIL', 'cv2'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GuitarViz',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX can break some DLLs — safer off
    console=False,      # no black terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    icon=None,          # replace with 'icon.ico' if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name='GuitarViz',
)
