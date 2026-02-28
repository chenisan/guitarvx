# GuitarViz Windows Build Script
# Run in PowerShell as Administrator from the project folder:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\build_windows.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir    = Join-Path $ProjectDir ".venv_build"
$DistDir    = Join-Path $ProjectDir "dist\GuitarViz"

Write-Host "=== GuitarViz Builder ===" -ForegroundColor Cyan
Write-Host "Project: $ProjectDir"

# ── 1. Create isolated venv ──────────────────────────────────────────────────
if (-not (Test-Path $VenvDir)) {
    Write-Host "`n[1/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvDir
} else {
    Write-Host "`n[1/5] Reusing existing venv." -ForegroundColor Green
}

$pip = Join-Path $VenvDir "Scripts\pip.exe"
$py  = Join-Path $VenvDir "Scripts\python.exe"

# ── 2. Install dependencies ──────────────────────────────────────────────────
Write-Host "`n[2/5] Installing dependencies..." -ForegroundColor Yellow
& $pip install --upgrade pip | Out-Null
& $pip install `
    numpy `
    pygame `
    moderngl `
    "sounddevice>=0.4.6" `
    aubio `
    pyinstaller

# ── 3. Run PyInstaller ───────────────────────────────────────────────────────
Write-Host "`n[3/5] Running PyInstaller..." -ForegroundColor Yellow
Set-Location $ProjectDir
& $py -m PyInstaller guitar_viz.spec --clean --noconfirm

# ── 4. Verify output ─────────────────────────────────────────────────────────
Write-Host "`n[4/5] Checking output..." -ForegroundColor Yellow
$exe = Join-Path $DistDir "GuitarViz.exe"
if (Test-Path $exe) {
    $size = [math]::Round((Get-Item $exe).Length / 1MB, 1)
    Write-Host "  GuitarViz.exe  ($size MB)" -ForegroundColor Green
} else {
    Write-Host "  ERROR: exe not found!" -ForegroundColor Red
    exit 1
}

# ── 5. Optional: compile Inno Setup installer ────────────────────────────────
Write-Host "`n[5/5] Looking for Inno Setup compiler..." -ForegroundColor Yellow
$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (Test-Path $iscc) {
    Write-Host "  Compiling installer..." -ForegroundColor Yellow
    & $iscc (Join-Path $ProjectDir "installer.iss")
    Write-Host "  Installer created in: $ProjectDir\Output\" -ForegroundColor Green
} else {
    Write-Host "  Inno Setup not found — skipping installer." -ForegroundColor DarkGray
    Write-Host "  Install from: https://jrsoftware.org/isdl.php" -ForegroundColor DarkGray
    Write-Host "  Then re-run this script to also get a .exe installer." -ForegroundColor DarkGray
}

Write-Host "`n=== Done! ===`nDistribution folder: $DistDir" -ForegroundColor Cyan
