# NEXUS-Engine launcher for Windows PowerShell.
#
#   .\nexus.ps1 setup        install venv + dependencies + self-check
#   .\nexus.ps1 quickstart   data -> tokenizer -> training -> acceptance gate
#   .\nexus.ps1 serve        HTTP API on 0.0.0.0:8000
#   .\nexus.ps1 all          setup + quickstart + serve
#   .\nexus.ps1 demo | test | doctor | analyze FILE.scad | cli ...
#
# If PowerShell blocks scripts:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   or:  .\nexus.cmd setup

param(
  [Parameter(Position = 0)]
  [string]$Command = "help",
  [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
  [string[]]$Rest = @()
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
$env:PYTHONUTF8 = "1"

$Venv = $env:NEXUS_VENV
if (-not $Venv) { $Venv = Join-Path $PSScriptRoot ".venv" }
$Py = Join-Path $Venv "Scripts\python.exe"
$Pip = Join-Path $Venv "Scripts\pip.exe"

$Port = $env:NEXUS_PORT
if (-not $Port) { $Port = "8000" }
$Scale = $env:NEXUS_SCALE
if (-not $Scale) { $Scale = "small" }

function Write-Info([string]$Text) { Write-Host ("-> " + $Text) -ForegroundColor Cyan }
function Write-Ok([string]$Text) { Write-Host ("OK " + $Text) -ForegroundColor Green }
function Write-Warn([string]$Text) { Write-Host ("!  " + $Text) -ForegroundColor Yellow }

function Stop-WithError([string]$Text) {
  Write-Host ("X  " + $Text) -ForegroundColor Red
  exit 1
}

function Get-PythonExe {
  $candidates = @("python", "python3", "py")
  foreach ($name in $candidates) {
    $found = Get-Command $name -ErrorAction SilentlyContinue
    if ($null -eq $found) { continue }
    $version = ""
    try {
      $version = & $name -c "import sys; print(str(sys.version_info[0]) + '.' + str(sys.version_info[1]))"
    } catch {
      $version = ""
    }
    if ($version) {
      $parts = $version.Trim().Split(".")
      $major = [int]$parts[0]
      $minor = [int]$parts[1]
      if ($major -eq 3 -and $minor -ge 10 -and $minor -le 13) { return $name }
      Write-Warn ("skipping " + $name + " (Python " + $version.Trim() + "); need 3.10-3.13")
    }
  }
  Stop-WithError "Python 3.10-3.13 not found. Install from python.org or use: conda create -n nexus python=3.11"
}

function Test-Environment {
  if (-not (Test-Path $Py)) {
    Stop-WithError "environment is not ready - run:  .\nexus.ps1 setup"
  }
}

function Invoke-NexusCli([string[]]$CliArgs) {
  Test-Environment
  & $Py -m nexus.cli @CliArgs
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Get-CudaWheel {
  # NEXUS_CUDA overrides autodetection: cu128 | cu126 | cpu
  if ($env:NEXUS_CUDA) { return $env:NEXUS_CUDA }
  $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($null -eq $smi) { return "cpu" }
  $name = ""
  try { $name = (& nvidia-smi --query-gpu=name --format=csv,noheader) -join " " } catch { }
  $name = $name.ToUpper()
  if ($name -match "RTX 50") { return "cu128" }     # Blackwell sm_120
  if ($name -match "RTX 40" -or $name -match "RTX 30") { return "cu126" }
  return "cu128"
}

function Install-Torch {
  $wheel = Get-CudaWheel
  if ($wheel -eq "cpu") {
    Write-Info "no NVIDIA GPU detected - installing CPU build of torch"
    & $Pip install --quiet torch
    return
  }
  Write-Info ("NVIDIA GPU detected - installing torch build " + $wheel)
  & $Pip uninstall -y -q torch 2>$null | Out-Null
  & $Pip install torch --index-url ("https://download.pytorch.org/whl/" + $wheel)
  if ($LASTEXITCODE -ne 0) {
    Write-Warn "CUDA wheel failed, falling back to CPU build"
    & $Pip install --quiet torch
  }
}

function Test-Cuda {
  $probe = "import torch, json; print(json.dumps({'v': torch.__version__, 'cuda': torch.cuda.is_available()}))"
  $raw = & $Py -c $probe
  Write-Info ("torch: " + $raw)
  if ($raw -match '"cuda": false' -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Warn "GPU is present but torch does not see CUDA. Fix:  .\nexus.ps1 gpu"
  }
}

function Invoke-InstallGpu {
  Test-Environment
  $wheel = $env:NEXUS_CUDA
  if (-not $wheel) { $wheel = Get-CudaWheel }
  if ($wheel -eq "cpu") { $wheel = "cu128" }
  Write-Info ("reinstalling torch with CUDA build " + $wheel)
  & $Pip uninstall -y torch
  & $Pip install torch --index-url ("https://download.pytorch.org/whl/" + $wheel)
  if ($LASTEXITCODE -ne 0) { Stop-WithError "failed to install CUDA wheel" }
  Test-Cuda
  Invoke-NexusCli @("doctor")
}

function Invoke-Setup {
  if (-not (Test-Path $Py)) {
    $python = Get-PythonExe
    Write-Info ("creating virtual environment: " + $Venv)
    & $python -m venv $Venv
    if (-not (Test-Path $Py)) { Stop-WithError ("failed to create venv at " + $Venv) }
    & $Py -m pip install --quiet --upgrade pip
  }
  Write-Info "installing dependencies (torch is ~2-3 GB for CUDA builds)"
  Install-Torch
  & $Pip install --quiet -e ".[dev]"
  if ($LASTEXITCODE -ne 0) { Stop-WithError "package installation failed" }
  Write-Ok "dependencies installed"
  Test-Cuda

  Write-Info "environment self-check"
  Invoke-NexusCli @("doctor")

  Write-Info "fast tests (~30 s)"
  & $Py -m pytest -q tests/test_core.py tests/test_geometry_scad.py
  if ($LASTEXITCODE -ne 0) { Stop-WithError "tests failed" }
  Write-Ok "ready. Next:  .\nexus.ps1 quickstart"
}

function Invoke-Quickstart {
  $cliArgs = @("quickstart", "--scale", $Scale) + $Rest
  Invoke-NexusCli $cliArgs
}

function Invoke-Serve {
  $cliArgs = @("serve", "--host", "0.0.0.0", "--port", $Port,
               "--registry", "artifacts/registry", "--model-name", "core",
               "--ref", "production", "--tokenizer", "artifacts/tokenizer/bpe.json") + $Rest
  Invoke-NexusCli $cliArgs
}

function Show-Usage {
  $lines = @(
    "NEXUS-Engine launcher (Windows)",
    "",
    "  .\nexus.ps1 setup                install environment and check it",
    "  .\nexus.ps1 quickstart           data -> tokenizer -> training -> gate",
    "  .\nexus.ps1 serve                HTTP API on 0.0.0.0:" + $Port,
    "  .\nexus.ps1 all                  setup + quickstart + serve",
    "  .\nexus.ps1 demo                 architecture demo (all three levels)",
    "  .\nexus.ps1 analyze FILE.scad    mass, audit, FEM for a part",
    "  .\nexus.ps1 test                 test suite",
    "  .\nexus.ps1 gpu                  reinstall torch with CUDA (RTX 50xx -> cu128)",
    "  .\nexus.ps1 doctor               environment diagnostics",
    "  .\nexus.ps1 cli ...              any CLI command (nexus --help)",
    "",
    "Env vars: NEXUS_VENV, NEXUS_PORT, NEXUS_SCALE (nano|small|medium|gpu),",
    "          NEXUS_CUDA (cu128|cu126|cpu), NEXUS_API_KEY"
  )
  foreach ($line in $lines) { Write-Host $line }
}

$action = $Command.ToLower()
if ($action -eq "setup") {
  Invoke-Setup
} elseif ($action -eq "quickstart") {
  Invoke-Quickstart
} elseif ($action -eq "serve") {
  Invoke-Serve
} elseif ($action -eq "all") {
  Invoke-Setup
  Invoke-Quickstart
  Write-Info ("starting API on port " + $Port)
  Invoke-Serve
} elseif ($action -eq "demo") {
  Invoke-NexusCli (@("demo") + $Rest)
} elseif ($action -eq "analyze") {
  Invoke-NexusCli (@("analyze") + $Rest)
} elseif ($action -eq "gpu") {
  Invoke-InstallGpu
} elseif ($action -eq "doctor") {
  Invoke-NexusCli @("doctor")
} elseif ($action -eq "test") {
  Test-Environment
  & $Py -m pytest -q @Rest
} elseif ($action -eq "cli") {
  Invoke-NexusCli $Rest
} elseif ($action -eq "help" -or $action -eq "-h" -or $action -eq "--help") {
  Show-Usage
} else {
  Write-Warn ("unknown command: " + $Command)
  Show-Usage
  exit 1
}
