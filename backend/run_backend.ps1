param(
  [switch]$Reload = $false,
  [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

# перейти в корень репозитория
$Root = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $Root

# venv
$Activate = Join-Path ".\.venv" "Scripts\Activate.ps1"
if (Test-Path $Activate) { . $Activate } else { Write-Error "Venv not found at $Activate"; exit 1 }

# PYTHONPATH на корень (на всякий)
$env:PYTHONPATH = $Root

# подчистить кеш байткода
Get-ChildItem -Recurse -Include *.pyc,__pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue | Out-Null

# аргументы uvicorn — как МАССИВ (важно!)
$common = @("--host","127.0.0.1","--port",$Port.ToString(),"--log-level","debug")

# если uvicorn не резолвится, используй .\.venv\Scripts\uvicorn.exe ниже
if ($Reload) {
  Write-Host "==> Starting Uvicorn with reload"
  uvicorn "backend.app:app" @common "--reload"
} else {
  Write-Host "==> Starting Uvicorn without reload"
  uvicorn "backend.app:app" @common
}
