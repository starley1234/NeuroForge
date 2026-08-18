"""
Загрузчик реальной корпоративной отчётности из SEC EDGAR.

Использует официальные публичные bulk-выгрузки Financial Statement Data
Sets (XBRL) — public domain, без ключа. Каждый квартал это ZIP с файлами:
    sub.txt — заголовки компаний/подач (CIK, форма, период)
    num.txt — числовые факты (тег, значение, дата, durp и т.д.)
    pre.txt — представление (какая строка в какой отчёт)

URL:
    https://www.sec.gov/files/dera/data/financial-statement-data-sets/{YYYY}q{Q}.zip

Требуется вежливый User-Agent (SEC его требует).

Из num.txt извлекаются стандартные теги и маппятся на поля
юнит-экономики StructuredEncoder (revenue, cogs, gross_profit, opex,
net_income, assets, liabilities, equity, cash_flow, debt, inventory...).
"""
from __future__ import annotations

import io
import zipfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_UA = "NEXUS-Capital/0.1 research@neuroforge.local"

# XBRL-теги US-GAAP -> внутренние поля. Берём наиболее часто встречаемые
# варианты (annual/quarterly и разные исторические имена).
TAG_MAP: dict[str, list[str]] = {
    "revenue": [
        "Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cogs": [
        "CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold",
    ],
    "gross_profit": ["GrossProfit"],
    "opex": [
        "OperatingExpenses", "GeneralAndAdministrativeExpense",
        "SellingGeneralAndAdministrativeExpense",
    ],
    "ebitda": ["EBITDA"],
    "net_income": ["NetIncomeLoss"],
    "cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "assets": ["Assets"],
    "liabilities": ["Liabilities", "LiabilitiesCurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "debt": ["LongTermDebt", "LongTermDebtAndCapitalLeaseObligations", "DebtCurrent"],
    "inventory": ["InventoryNet"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "payables": ["AccountsPayableCurrent"],
}

DEFAULT_FIELDS = list(TAG_MAP.keys())


def _http_get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={
        "User-Agent": _UA,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def quarter_url(year: int, q: int) -> str:
    return (f"https://www.sec.gov/files/dera/data/"
            f"financial-statement-data-sets/{year}q{q}.zip")


@dataclass
class EdgarConfig:
    years: list[int] = field(default_factory=lambda: [2023, 2024])
    quarters: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    forms: tuple[str, ...] = ("10-K", "10-Q")
    cache_dir: str = "data/raw/sec"
    unit: str = "USD"


def download_quarter(year: int, q: int, cache_dir: str = "data/raw/sec",
                     force: bool = False) -> dict[str, pd.DataFrame]:
    """Скачивает и парсит один квартальный ZIP в {sub, num, pre}."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    sub_f = cache / f"{year}q{q}_sub.parquet"
    num_f = cache / f"{year}q{q}_num.parquet"
    if sub_f.exists() and num_f.exists() and not force:
        return {"sub": pd.read_parquet(sub_f), "num": pd.read_parquet(num_f)}

    url = quarter_url(year, q)
    print(f"[sec] качаю {url}")
    raw = _http_get(url)
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        with zf.open("sub.txt") as f:
            sub = pd.read_csv(f, sep="\t", low_memory=False)
        with zf.open("num.txt") as f:
            num = pd.read_csv(f, sep="\t", low_memory=False,
                              usecols=["adsh", "tag", "value", "ddate",
                                       "qtrs", "uom", "coreg"])
    sub.to_parquet(sub_f)
    num.to_parquet(num_f)
    return {"sub": sub, "num": num}


def load_filings(cfg: EdgarConfig) -> pd.DataFrame:
    """
    Собирает длинную таблицу financial-полей по всем запрошенным
    кварталам. Возвращает DataFrame с колонками:
        adsh, cik, name, form, fy, fp, filed, field, value
    """
    sub_parts, num_parts = [], []
    for y in cfg.years:
        for q in cfg.quarters:
            try:
                d = download_quarter(y, q, cfg.cache_dir)
            except Exception as e:
                print(f"[sec] пропуск {y}q{q}: {e}")
                continue
            sub = d["sub"]
            sub = sub[sub["form"].isin(cfg.forms)]
            sub_parts.append(sub[["adsh", "cik", "name", "form",
                                  "fy", "fp", "filed"]])
            num_parts.append(d["num"])

    if not sub_parts:
        raise RuntimeError("Ни один квартал SEC не загружен")

    sub_all = pd.concat(sub_parts, ignore_index=True).drop_duplicates("adsh")
    num_all = pd.concat(num_parts, ignore_index=True)
    num_all = num_all[num_all["uom"] == cfg.unit]
    # Самое свежее значение по (adsh, tag)
    num_all = (num_all.sort_values("ddate")
               .groupby(["adsh", "tag"], as_index=False)["value"].last())

    # Разворачиваем нужные теги в колонки
    records = []
    for field, tags in TAG_MAP.items():
        sub_num = num_all[num_all["tag"].isin(tags)].copy()
        sub_num["field"] = field
        records.append(sub_num[["adsh", "field", "value"]])
    long = pd.concat(records, ignore_index=True)
    # Если по полю несколько тегов — берём максимум доступных
    wide = (long.groupby(["adsh", "field"], as_index=False)["value"].mean()
            .pivot(index="adsh", columns="field", values="value")
            .reset_index())
    out = sub_all.merge(wide, on="adsh", how="inner")
    # Балансовое тождество контроля
    if {"assets", "liabilities", "equity"}.issubset(out.columns):
        out["balance_residual"] = (
            out["assets"] - out["liabilities"].fillna(0)
            - out["equity"].fillna(0))
    return out


def to_field_tensor(
    df: pd.DataFrame,
    field_order: list[str] | None = None,
    n_fields: int = 16,
):
    """
    Конвертирует DataFrame с полями в (fields, mask) numpy-массивы,
    совместимые со StructuredEncoder.
    """
    order = (field_order or DEFAULT_FIELDS)[:n_fields]
    n = len(df)
    fields = np.zeros((n, len(order)), dtype=np.float32)
    mask = np.zeros((n, len(order)), dtype=np.float32)
    for i, f in enumerate(order):
        if f in df.columns:
            vals = pd.to_numeric(df[f], errors="coerce").to_numpy(np.float32)
            ok = ~np.isnan(vals)
            fields[ok, i] = vals[ok]
            mask[ok, i] = 1.0
    return fields, mask, order
