# Guitar Visualizer — Windows install script
# Run from PowerShell:  .\install_windows.ps1

$python = "C:\Program Files\Python310\python.exe"
if (-not (Test-Path $python)) {
    # fallback to PATH python
    $python = "python"
}

Write-Host "Using Python: $python" -ForegroundColor Cyan
& $python -m pip install --upgrade pip
& $python -m pip install pygame moderngl sounddevice aubio numpy

Write-Host ""
Write-Host "Done! Run the visualizer with:" -ForegroundColor Green
Write-Host "  $python guitar_viz.py            # fullscreen at native resolution"
Write-Host "  $python guitar_viz.py --windowed  # 1280x720 window"
