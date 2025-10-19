
param(
  [string]$VenvPath = ".\.venv"
)
$ErrorActionPreference = "Stop"

# Ensure we run from repo root
$Root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $Root

$Activate = Join-Path $VenvPath "Scripts\Activate.ps1"
. $Activate
$env:PYTHONUNBUFFERED="1"
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
