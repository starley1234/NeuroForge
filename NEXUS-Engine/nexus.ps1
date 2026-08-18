<#
NEXUS-Engine — запуск под Windows (PowerShell).

  .\nexus.ps1 setup        поставить окружение (venv + зависимости) и самопроверка
  .\nexus.ps1 quickstart   данные -> токенизатор -> обучение -> приёмка
  .\nexus.ps1 serve        HTTP API на 0.0.0.0:8000
  .\nexus.ps1 all          setup + quickstart + serve
  .\nexus.ps1 demo | test | doctor | analyze FILE.scad | cli ...

Если PowerShell блокирует выполнение скриптов:
  powershell -ExecutionPolicy Bypass -File .\nexus.ps1 setup
#>
param(
  [Parameter(Position = 0)][string]$Command = "help",
  [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$Venv  = if ($env:NEXUS_VENV)  { $env:NEXUS_VENV }  else { Join-Path $PSScriptRoot ".venv" }
$Py    = Join-Path $Venv "Scripts\python.exe"
$Pip   = Join-Path $Venv "Scripts\pip.exe"
$Port  = if ($env:NEXUS_PORT)  { $env:NEXUS_PORT }  else { "8000" }
$Scale = if ($env:NEXUS_SCALE) { $env:NEXUS_SCALE } else { "small" }
$env:PYTHONUTF8 = "1"          # русский текст и рамки в консоли Windows

function Info($m) { Write-Host "> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "OK $m" -ForegroundColor Green }
function Warn($m) { Write-Host "! $m"  -ForegroundColor Yellow }
function Die($m)  { Write-Host "X $m"  -ForegroundColor Red; exit 1 }

function Find-Python {
  foreach ($cand in @("python", "python3", "py")) {
    $exe = Get-Command $cand -ErrorAction SilentlyContinue
    if ($exe) {
      $ver = & $cand -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
      if ($ver -and [version]$ver -ge [version]"3.10") { return $cand }
    }
  }
  Die "не найден Python >= 3.10. Поставьте python.org/downloads или используйте conda."
}

function Require-Env {
  if (-not (Test-Path $Py)) { Die "окружение не готово — запустите: .\nexus.ps1 setup" }
}

function Invoke-Nexus([string[]]$CliArgs) {
  Require-Env
  & $Py -m nexus.cli @CliArgs
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Cmd-Setup {
  if (-not (Test-Path $Py)) {
    $python = Find-Python
    Info "создаю виртуальное окружение: $Venv"
    & $python -m venv $Venv
    if (-not (Test-Path $Py)) { Die "не удалось создать venv ($Venv)" }
    & $Pip install --quiet --upgrade pip
  }
  Info "устанавливаю зависимости (torch ~800 МБ, займёт несколько минут)"
  if ($env:NEXUS_GPU -eq "1") {
    & $Pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
    if ($LASTEXITCODE -ne 0) { Warn "CUDA-сборка не поставилась, ставлю CPU-версию" }
  }
  & $Pip install --quiet -e ".[dev]"
  if ($LASTEXITCODE -ne 0) { Die "ошибка установки пакета" }
  Ok "зависимости установлены"
  Info "самопроверка окружения"
  Invoke-Nexus @("doctor")
  Info "быстрые тесты (~30 с)"
  & $Py -m pytest -q tests/test_core.py tests/test_geometry_scad.py
  if ($LASTEXITCODE -ne 0) { Die "тесты не прошли" }
  Ok "готово. Дальше: .\nexus.ps1 quickstart"
}

function Cmd-Quickstart { Invoke-Nexus (@("quickstart", "--scale", $Scale) + $Rest) }
function Cmd-Serve {
  Invoke-Nexus (@("serve", "--host", "0.0.0.0", "--port", $Port,
                  "--registry", "artifacts/registry", "--model-name", "core",
                  "--ref", "production", "--tokenizer", "artifacts/tokenizer/bpe.json") + $Rest)
}
function Cmd-All { Cmd-Setup; Cmd-Quickstart; Info "поднимаю API на порту $Port"; Cmd-Serve }

function Show-Usage {
@"
NEXUS-Engine — запуск (Windows)

  .\nexus.ps1 setup                поставить окружение и проверить его
  .\nexus.ps1 quickstart           данные -> токенизатор -> обучение -> приёмка
                                   (масштаб: `$env:NEXUS_SCALE = "nano|small|medium|gpu")
  .\nexus.ps1 serve                HTTP API на 0.0.0.0:$Port
  .\nexus.ps1 all                  setup + quickstart + serve
  .\nexus.ps1 demo                 демонстрация всех трёх уровней архитектуры
  .\nexus.ps1 analyze FILE.scad    масса, аудит, FEM для детали
  .\nexus.ps1 test                 тесты
  .\nexus.ps1 doctor               диагностика окружения
  .\nexus.ps1 cli ...              любая команда CLI (nexus --help)

Переменные: NEXUS_VENV, NEXUS_PORT, NEXUS_SCALE, NEXUS_GPU=1, NEXUS_API_KEY
"@ | Write-Host
}

switch ($Command.ToLower()) {
  "setup"      { Cmd-Setup }
  "quickstart" { Cmd-Quickstart }
  "serve"      { Cmd-Serve }
  "all"        { Cmd-All }
  "demo"       { Invoke-Nexus (@("demo") + $Rest) }
  "analyze"    { Invoke-Nexus (@("analyze") + $Rest) }
  "doctor"     { Invoke-Nexus @("doctor") }
  "test"       { Require-Env; & $Py -m pytest -q @Rest }
  "cli"        { Invoke-Nexus $Rest }
  { $_ -in @("help", "-h", "--help") } { Show-Usage }
  default      { Warn "неизвестная команда: $Command"; Show-Usage; exit 1 }
}
