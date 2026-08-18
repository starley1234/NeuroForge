#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  NEXUS-Capital — HTTP API + веб-UI
═══════════════════════════════════════════════════════════════════════════

Запуск:
    python serve.py                 # http://127.0.0.1:8000
    PORT=9000 python serve.py       # другой порт

Эндпоинты:
    GET  /              — веб-UI (HTML-страница)
    GET  /api/health    — статус и метаданные
    POST /api/risk      — расчёт риска по тексту/числам
    POST /api/pricing   — оптимальная цена (Бертран)
    POST /api/negotiate — делёж излишка (Nash bargaining)
    POST /api/fields    — извлечение полей отчётности из текста (EN/RU)

Сервер однократно загружает модель при старте. Первый запуск ~5–15 c.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NEXUS_OFFLINE_TOKENIZER", "1")

from nexus_capital import small_config, build_model  # noqa: E402
from nexus_capital.core.tokenizer import NexusTokenizer  # noqa: E402
from nexus_capital.data.edgar_parser import (  # noqa: E402
    parse_text_filing, fields_to_tensor, DEFAULT_UNIT_FIELDS,
)

# ── Загрузка модели один раз ──────────────────────────────────────────
print("⏳ Загрузка модели NEXUS-Capital…", flush=True)
_t0 = time.time()
TOKENIZER = NexusTokenizer.default(prefer_offline=True)
CFG = small_config()
CFG.mc_paths = 256
CFG.mc_horizon = 8
CFG.orderbook_levels = 16
CFG.tabular_features = 16
MODEL = build_model(CFG, tokenizer=TOKENIZER)
MODEL.eval()
# Калибруем SDE на синтетических «типичных» рыночных возвратах,
# чтобы VaR не был случайным при cold-start.
_synth = torch.from_numpy(
    np.random.default_rng(0).standard_t(5, size=2000).astype(np.float32)
    * 0.015).unsqueeze(-1)
MODEL.workspace.mc.sde.calibrate_to_returns(_synth)
print(f"✅ Модель загружена за {time.time()-_t0:.1f}c "
      f"(vocab={TOKENIZER.vocab_size}, params={MODEL.count_parameters()/1e6:.1f}M)",
      flush=True)

app = FastAPI(title="NEXUS-Capital API", version="1.0")


# ── Схемы запросов ───────────────────────────────────────────────────
class RiskRequest(BaseModel):
    text: str | None = Field(None, description="Финансовый текст (EN/RU)")
    fields: dict[str, float] | None = Field(
        None, description="Поля отчётности как словарь")
    competitor_price: float | None = None


class PricingRequest(BaseModel):
    cost: float = Field(..., description="себестоимость")
    competitor_price: float = Field(..., description="цена конкурента")
    elasticity: float = Field(2.0, description="эластичность спроса")
    market_size: float = Field(1000.0, description="размер рынка")


class NegotiateRequest(BaseModel):
    reservation_a: float
    reservation_b: float
    surplus: float
    power_a: float = 0.5


# ── Вспомогательное ──────────────────────────────────────────────────
def _encode_inputs(text: str | None, fields: dict[str, float] | None):
    """Готовит тензоры для forward из текста и/или полей отчётности."""
    kwargs: dict[str, Any] = {}

    if text:
        enc = TOKENIZER.encode(text, max_length=CFG.text_n_pos)
        ids = torch.tensor([enc.ids], dtype=torch.long)
        nv = torch.tensor([enc.number_values], dtype=torch.float32)
        nm = torch.tensor([enc.number_mask], dtype=torch.bool)
        am = torch.tensor([enc.attention_mask], dtype=torch.long)
        kwargs["tokens"] = ids
        kwargs["number_values"] = nv
        kwargs["number_mask"] = nm
        kwargs["attention_mask"] = am.bool()

    if fields:
        order = DEFAULT_UNIT_FIELDS[:CFG.tabular_features]
        vals = np.zeros(CFG.tabular_features, dtype=np.float32)
        mask = np.zeros(CFG.tabular_features, dtype=np.float32)
        for i, name in enumerate(order):
            if name in fields:
                vals[i] = float(fields[name])
                mask[i] = 1.0
        # Если ничего не распарсилось — но поля переданы, всё равно прокидываем
        kwargs["fields"] = torch.from_numpy(vals).unsqueeze(0)
        kwargs["field_mask"] = torch.from_numpy(mask).unsqueeze(0)

    # Если ничего нет — синтетический рынок для демонстрации
    if not kwargs:
        from nexus_capital.data.orderbook_stream import collate_book_ticks
        bt = collate_book_ticks(1, n_levels=16, T=16, n_features=8)
        kwargs["book"] = bt["book"]
        kwargs["ticks"] = bt["ticks"]

    return kwargs


# ── Эндпоинты ────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": "NEXUS-Capital",
        "vocab_size": TOKENIZER.vocab_size,
        "tokenizer": TOKENIZER.name,
        "params_M": round(MODEL.count_parameters() / 1e6, 2),
        "d_value": CFG.d_value,
        "mc_paths": CFG.mc_paths,
        "sde_calibrated": bool(MODEL.workspace.mc.sde.calibrated.item()),
    }


@app.post("/api/risk")
def api_risk(req: RiskRequest):
    t0 = time.time()
    with torch.no_grad():
        kwargs = _encode_inputs(req.text, req.fields)
        mode = "pricing" if req.competitor_price else "risk"
        cp = (torch.tensor([req.competitor_price])
              if req.competitor_price else None)
        out = MODEL(**kwargs, workspace_mode=mode, competitor_price=cp)

    risk = out["workspace"]["risk"]
    result = {
        "expected_pnl": float(risk["expected_pnl"].mean()),
        "std_pnl": float(risk["std_pnl"].mean()),
        "var_5pct": float(risk["var"].mean()),
        "expected_shortfall_5pct": float(risk["expected_shortfall"].mean()),
        "default_prob": float(risk["default_prob"].mean()),
        "sde_calibrated": bool(risk["calibrated"]),
        "economic": {
            "price": float(out["economic"]["price"].mean()),
            "default_prob": float(out["economic"]["default_prob"].mean()),
            "spread": float(out["economic"]["spread"].mean()),
            "portfolio_weights_sum":
                float(out["economic"]["portfolio_weights"].sum()),
            "utility": float(out["economic"]["utility"]["utility"].mean()),
        },
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
    if mode == "pricing":
        result["optimal_price"] = float(out["workspace"]["optimal_price"][0])
    return result


@app.post("/api/pricing")
def api_pricing(req: PricingRequest):
    from nexus_capital.workspace.game_theory import bertrand_price
    cost = torch.tensor([req.cost])
    comp = torch.tensor([req.competitor_price])
    elas = torch.tensor([req.elasticity])
    size = torch.tensor([req.market_size])
    with torch.no_grad():
        p = bertrand_price(cost, comp, elas, size)
        demand = size * torch.sigmoid(elas * (comp - p))
        profit = (p - cost) * demand
    return {
        "optimal_price": float(p[0]),
        "expected_demand": float(demand[0]),
        "expected_profit": float(profit[0]),
        "margin_pct": float((p[0] - cost[0]) / p[0]),
    }


@app.post("/api/negotiate")
def api_negotiate(req: NegotiateRequest):
    from nexus_capital.workspace.game_theory import nash_bargaining
    ra = torch.tensor([req.reservation_a])
    rb = torch.tensor([req.reservation_b])
    s = torch.tensor([req.surplus])
    with torch.no_grad():
        a, b = nash_bargaining(ra, rb, s,
                               bargaining_power_a=req.power_a)
    return {
        "share_a": float(a[0]),
        "share_b": float(b[0]),
        "total_welfare": float(a[0] + b[0]),
    }


@app.post("/api/fields")
def api_fields(text: str):
    """Извлекает известные финансовые поля из текста 10-K/RU-отчёта."""
    parsed = parse_text_filing(text)
    return {"fields": parsed, "count": len(parsed)}


# ── Веб-UI (один HTML-файл, без сборки) ─────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return WEB_UI


WEB_UI = r"""<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NEXUS-Capital</title>
<style>
  :root{--bg:#0b0e14;--card:#141a24;--line:#222b3a;--fg:#e6edf3;--muted:#8b98a9;
        --acc:#4c8bf5;--good:#3fb950;--warn:#d29922;--bad:#f85149}
  *{box-sizing:border-box} body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
    background:var(--bg);color:var(--fg)}
  header{padding:20px 28px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
  h1{font-size:18px;margin:0;letter-spacing:.3px} .dot{width:10px;height:10px;border-radius:50%;background:var(--good)}
  .sub{color:var(--muted);font-size:12px}
  main{padding:24px;max-width:1100px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:20px}
  @media(max-width:820px){main{grid-template-columns:1fr}}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:18px}
  .card h2{font-size:14px;margin:0 0 14px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted)}
  textarea,input{width:100%;background:#0d1117;color:var(--fg);border:1px solid var(--line);
    border-radius:8px;padding:10px;font:13px monospace;resize:vertical}
  textarea{min-height:120px} label{display:block;font-size:12px;color:var(--muted);margin:10px 0 4px}
  button{margin-top:14px;background:var(--acc);color:#fff;border:0;border-radius:8px;
    padding:10px 16px;font-weight:600;cursor:pointer}
  button:hover{filter:brightness(1.1)} button:disabled{opacity:.5;cursor:default}
  .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
  .metric{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px dashed var(--line)}
  .metric:last-child{border:0} .metric .v{font-variant-numeric:tabular-nums;font-weight:600}
  .good{color:var(--good)} .warn{color:var(--warn)} .bad{color:var(--bad)}
  .tabs{display:flex;gap:6px;margin-bottom:14px} .tab{padding:6px 12px;border-radius:6px;
    background:#0d1117;border:1px solid var(--line);cursor:pointer;font-size:12px;color:var(--muted)}
  .tab.active{background:var(--acc);color:#fff;border-color:var(--acc)}
  pre{background:#0d1117;border:1px solid var(--line);border-radius:8px;padding:12px;
    overflow:auto;font:12px monospace;white-space:pre-wrap;word-break:break-word;min-height:120px;color:#9ecbff}
  .pill{display:inline-block;padding:2px 8px;border-radius:10px;background:#0d1117;
    border:1px solid var(--line);font-size:11px;color:var(--muted);margin-left:8px}
</style></head>
<body>
<header>
  <div class="dot" id="status"></div>
  <div>
    <h1>NEXUS-Capital <span class="pill" id="ver">загрузка…</span></h1>
    <div class="sub">экономический движок: риск · ценообразование · переговоры · поля отчётности</div>
  </div>
</header>
<main>
  <section class="card">
    <h2>Ввод</h2>
    <div class="tabs">
      <div class="tab active" data-tab="risk">Риск</div>
      <div class="tab" data-tab="pricing">Цена</div>
      <div class="tab" data-tab="neg">Переговоры</div>
    </div>

    <div id="tab-risk">
      <label>Финансовый текст (EN/RU), напр. выдержка из 10-K или новость</label>
      <textarea id="risk-text">Revenue grew 12% to $1,420.5 million. Net income was $210.0 million.
Выручка выросла на 12 процентов, чистая прибыль 1,2 млрд руб.</textarea>
      <label>Поля отчётности (необязательно, JSON)</label>
      <textarea id="risk-fields" style="min-height:60px">{"revenue": 1420.5, "net_income": 210.0, "cogs": 880.0}</textarea>
      <button onclick="runRisk()">Рассчитать риск</button>
    </div>

    <div id="tab-pricing" style="display:none">
      <div class="row">
        <div><label>Себестоимость</label><input type="number" id="p-cost" value="50"></div>
        <div><label>Цена конкурента</label><input type="number" id="p-comp" value="99"></div>
      </div>
      <div class="row">
        <div><label>Эластичность</label><input type="number" id="p-elas" value="2" step="0.1"></div>
        <div><label>Размер рынка</label><input type="number" id="p-size" value="1000"></div>
      </div>
      <button onclick="runPricing()">Найти оптимальную цену</button>
    </div>

    <div id="tab-neg" style="display:none">
      <div class="row">
        <div><label>Резерв продавца</label><input type="number" id="n-ra" value="80"></div>
        <div><label>Резерв покупателя</label><input type="number" id="n-rb" value="140"></div>
      </div>
      <div class="row">
        <div><label>Излишек (surplus)</label><input type="number" id="n-s" value="60"></div>
        <div><label>Сила продавца (0..1)</label><input type="number" id="n-p" value="0.5" step="0.1"></div>
      </div>
      <button onclick="runNeg()">Разделить по Нэшу</button>
    </div>
  </section>

  <section class="card">
    <h2>Результат</h2>
    <div id="metrics"><pre>// нажмите кнопку слева</pre></div>
  </section>
</main>

<script>
const $=s=>document.querySelector(s);
document.querySelectorAll('.tab').forEach(t=>t.onclick=()=>{
  document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
  t.classList.add('active');
  ['risk','pricing','neg'].forEach(id=>$('#tab-'+id).style.display = id===t.dataset.tab?'block':'none');
});
function setMetrics(obj){ $('#metrics').innerHTML = Object.entries(obj)
  .map(([k,v])=>{const cls=typeof v==='number'?(v>0?'good':v<0?'bad':''):'';
    return `<div class="metric"><span>${k}</span><span class="v ${cls}">${
      typeof v==='number'?v.toFixed(4):String(v)}</span></div>`}).join(''); }
function setJson(obj){ $('#metrics').textContent = JSON.stringify(obj,null,2); }
async function post(path,body){
  const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok) throw new Error(await r.text()); return r.json(); }
async function runRisk(){
  const b=$('button'); b.disabled=true; b.textContent='Считаю…';
  try{ let fields=null;
    try{fields=JSON.parse($('#risk-fields').value||'null')}catch(e){}
    const res=await post('/api/risk',{text:$('#risk-text').value, fields});
    setMetrics({
      'E[P&L]':res.expected_pnl, 'σ(P&L)':res.std_pnl,
      'VaR 5%':res.var_5pct, 'ES 5%':res.expected_shortfall_5pct,
      'P(default)':res.default_prob, 'utility':res.economic.utility,
      'цена':res.economic.price, 'спред':res.economic.spread,
      'мс':res.latency_ms, 'SDE калиброван':res.sde_calibrated});
    if(res.optimal_price) setMetrics({...{'оптимальная цена':res.optimal_price},
      'E[P&L]':res.expected_pnl,'мс':res.latency_ms});
  }catch(e){setJson({error:e.message})} finally{b.disabled=false;b.textContent='Рассчитать риск'} }
async function runPricing(){
  const res=await post('/api/pricing',{cost:+$('#p-cost').value,competitor_price:+$('#p-comp').value,
    elasticity:+$('#p-elas').value,market_size:+$('#p-size').value});
  setMetrics({'оптимальная цена':res.optimal_price,'спрос':res.expected_demand,
    'прибыль':res.expected_profit,'маржа, %':res.margin_pct*100}); }
async function runNeg(){
  const res=await post('/api/negotiate',{reservation_a:+$('#n-ra').value,reservation_b:+$('#n-rb').value,
    surplus:+$('#n-s').value,power_a:+$('#n-p').value});
  setMetrics({'доля продавца':res.share_a,'доля покупателя':res.share_b,'общее благо':res.total_welfare}); }

fetch('/api/health').then(r=>r.json()).then(h=>{
  $('#status').style.background=h.status==='ok'?'var(--good)':'var(--bad)';
  $('#ver').textContent=`${h.params_M}M · d=${h.d_value} · ${h.tokenizer} · ${h.mc_paths} MC путей`;
}).catch(e=>{ $('#status').style.background='var(--bad)'; $('#ver').textContent='API недоступен'; });
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"\n🌐 Веб-UI:  http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
