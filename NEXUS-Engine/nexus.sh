#!/usr/bin/env bash
# NEXUS-Engine — единая точка запуска.
#   ./nexus.sh setup        поставить окружение (venv + зависимости) и самопроверка
#   ./nexus.sh quickstart   данные → токенизатор → обучение → приёмка (одна команда)
#   ./nexus.sh serve        HTTP API на 0.0.0.0:8000
#   ./nexus.sh demo|test|doctor|analyze FILE|cli ...
# Если файл пришёл из Windows-чекаута с CRLF, bash выдаст
#   env: $'bash\r': No such file or directory
# Лечится один раз:  sed -i 's/\r$//' nexus.sh   (см. docs/QUICKSTART.md)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
VENV="${NEXUS_VENV:-$ROOT/.venv}"
PY="$VENV/bin/python"
PORT="${NEXUS_PORT:-8000}"
SCALE="${NEXUS_SCALE:-small}"

c()  { printf '\033[%sm%s\033[0m\n' "$1" "$2"; }
ok() { c "32" "✓ $1"; }
info(){ c "36" "▸ $1"; }
warn(){ c "33" "! $1"; }
die(){ c "31" "✗ $1"; exit 1; }

need_python() {
  for cand in python3.12 python3.11 python3.10 python3; do
    if command -v "$cand" >/dev/null 2>&1; then echo "$cand"; return; fi
  done
  die "не найден python3 ≥ 3.10"
}

ensure_venv() {
  if [ ! -x "$PY" ]; then
    info "создаю виртуальное окружение: $VENV"
    "$(need_python)" -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
  fi
}

cuda_wheel() {            # cu128 для 50-й серии, иначе cu126; cpu если нет GPU
  if [ -n "${NEXUS_CUDA:-}" ]; then echo "$NEXUS_CUDA"; return; fi
  command -v nvidia-smi >/dev/null 2>&1 || { echo cpu; return; }
  local name
  name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr 'a-z' 'A-Z')"
  case "$name" in
    *"RTX 50"*) echo cu128 ;;
    *"RTX 40"*|*"RTX 30"*) echo cu126 ;;
    "") echo cpu ;;
    *) echo cu128 ;;
  esac
}

install_torch() {
  local wheel; wheel="$(cuda_wheel)"
  if [ "$wheel" = "cpu" ]; then
    info "видеокарта NVIDIA не найдена — ставлю CPU-сборку torch"
    "$VENV/bin/pip" install --quiet torch
    return
  fi
  info "обнаружен GPU — ставлю torch сборки $wheel"
  "$VENV/bin/pip" uninstall -y -q torch >/dev/null 2>&1 || true
  "$VENV/bin/pip" install torch --index-url "https://download.pytorch.org/whl/$wheel" || {
    warn "CUDA-колесо не поставилось, ставлю CPU-сборку"
    "$VENV/bin/pip" install --quiet torch
  }
}

check_cuda() {
  "$PY" - <<'PYCHK' || true
import torch
print(f"[nexus] torch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
PYCHK
}

cmd_gpu() {
  require_env
  local wheel; wheel="$(cuda_wheel)"; [ "$wheel" = "cpu" ] && wheel=cu128
  info "переустанавливаю torch со сборкой $wheel"
  "$VENV/bin/pip" uninstall -y torch
  "$VENV/bin/pip" install torch --index-url "https://download.pytorch.org/whl/$wheel" || die "не удалось поставить CUDA-колесо"
  check_cuda
  "$PY" -m nexus.cli doctor
}

cmd_setup() {
  ensure_venv
  info "устанавливаю зависимости (CUDA-сборка torch — 2–3 ГБ)"
  install_torch
  "$VENV/bin/pip" install --quiet -e ".[dev]" || die "ошибка установки пакета"
  ok "зависимости установлены"
  check_cuda
  info "самопроверка окружения"
  "$PY" -m nexus.cli doctor
  info "быстрые тесты (~30 с)"
  "$PY" -m pytest -q tests/test_core.py tests/test_geometry_scad.py || die "тесты не прошли"
  ok "готово. Дальше: ./nexus.sh quickstart"
}

require_env() {
  [ -x "$PY" ] || die "окружение не готово — запустите ./nexus.sh setup"
}

cmd_quickstart() { require_env; "$PY" -m nexus.cli quickstart --scale "$SCALE" "$@"; }
cmd_serve()      { require_env; "$PY" -m nexus.cli serve --host 0.0.0.0 --port "$PORT" \
                     --registry artifacts/registry --model-name core --ref production \
                     --tokenizer artifacts/tokenizer/bpe.json "$@"; }
cmd_demo()       { require_env; "$PY" -m nexus.cli demo "$@"; }
cmd_doctor()     { require_env; "$PY" -m nexus.cli doctor; }
cmd_test()       { require_env; "$PY" -m pytest -q "$@"; }
cmd_analyze()    { require_env; "$PY" -m nexus.cli analyze "$@"; }
cmd_cli()        { require_env; "$PY" -m nexus.cli "$@"; }

cmd_all() {   # setup + quickstart + serve — «одна кнопка»
  cmd_setup
  cmd_quickstart
  info "поднимаю API на порту $PORT"
  cmd_serve
}

usage() {
  cat <<TXT
NEXUS-Engine — запуск

  ./nexus.sh setup                поставить окружение и проверить его
  ./nexus.sh quickstart           данные → токенизатор → обучение → приёмка
                                  (масштаб: NEXUS_SCALE=nano|small|medium|gpu)
  ./nexus.sh serve                HTTP API на 0.0.0.0:$PORT
  ./nexus.sh all                  setup + quickstart + serve
  ./nexus.sh demo                 демонстрация всех трёх уровней архитектуры
  ./nexus.sh analyze FILE.scad    масса, аудит, FEM для детали
  ./nexus.sh test                 тесты
  ./nexus.sh gpu                  переустановить torch с CUDA (RTX 50xx → cu128)
  ./nexus.sh doctor               диагностика окружения
  ./nexus.sh cli ...              любая команда CLI (nexus --help)

Переменные: NEXUS_VENV, NEXUS_PORT, NEXUS_SCALE, NEXUS_CUDA=cu128|cu126|cpu, NEXUS_API_KEY
TXT
}

case "${1:-help}" in
  setup)      shift; cmd_setup "$@" ;;
  quickstart) shift; cmd_quickstart "$@" ;;
  serve)      shift; cmd_serve "$@" ;;
  all)        shift; cmd_all "$@" ;;
  demo)       shift; cmd_demo "$@" ;;
  analyze)    shift; cmd_analyze "$@" ;;
  test)       shift; cmd_test "$@" ;;
  gpu)        shift; cmd_gpu "$@" ;;
  doctor)     shift; cmd_doctor "$@" ;;
  cli)        shift; cmd_cli "$@" ;;
  help|-h|--help) usage ;;
  *) warn "неизвестная команда: $1"; usage; exit 1 ;;
esac
