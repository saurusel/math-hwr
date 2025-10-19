
    param(
      [string]$VenvPath = ".\.venv",
      [string]$PyCmd = "python"
    )

    $ErrorActionPreference = "Stop"

    if (Test-Path $VenvPath) {
        Write-Host "==> Venv already exists at $VenvPath (skipping creation)"
    } else {
        Write-Host "==> Creating venv at $VenvPath"
        & $PyCmd -m venv $VenvPath
    }

    $Activate = Join-Path $VenvPath "Scripts\Activate.ps1"
    . $Activate

    Write-Host "==> Upgrading pip"
    python -m pip install --upgrade pip

    Write-Host "==> Installing PyTorch (CUDA 12.1)"
    python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

    Write-Host "==> Installing project requirements"
    python -m pip install -r (Join-Path "backend" "requirements.txt")

    $code = @"
import torch
print('torch:', torch.__version__)
print('cuda_available:', torch.cuda.is_available())
print('cuda_device_count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu_name:', torch.cuda.get_device_name(0))
"@
    $code | python -

    Write-Host "==> Done."
