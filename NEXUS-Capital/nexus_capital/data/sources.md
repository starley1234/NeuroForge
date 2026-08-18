# Рекомендуемые открытые реальные датасеты

Сводка источников под 5 модальностей NEXUS-Capital. Приоритет отдан данным,
которые скачиваются без платной подписки и без VPN: прямые HTTPS-ссылки,
официальные API и зеркала.

## 1. Микроструктура рынка (тики/стаканы/OHLCV)

| Источник | Что даёт | Доступ | Лицензия |
|---|---|---|---|
| **Binance Public Data** `data.binance.vision` | Spot/futures klines (1s..1mo), aggTrades, trades, bookDepth snapshots, fundingRate | прямой HTTPS, без ключа [1](https://data.binance.vision) | открытые данные биржи |
| **Tardis.dev** | L2/L3 крипто-стаканы, тики, опционы с десятков бирж | бесплатные сэмплы/CSV; полное — платно [2](https://tardis.dev/) | согласно биржам |
| **Dukascopy** | тиковая история FX (bid/ask) | бесплатный экспорт [1](https://www.dukascopy.com/swiss/english/marketwatch/historical/) | бесплатное некоммерческое |
| **HF Data Library** | 1-min OHLCV по 1391 акции/ETF США, 2002–н.в., ~1.5 млрд баров, Parquet | Zenodo bulk, CC-BY 4.0 [3](https://hfdatalibrary.com/) | CC-BY 4.0 |
| **LOBSTER** | реконструированный NASDAQ LOB для академических исследований | бесплатная регистрация для академиков | academic |

**Рекомендация для старта:** Binance 1m/1s klines + aggTrades — проще всего
автоматизируется, без ключей, покрывает и «цены», и «объёмы/направление сделок».

## 2. Корпоративная отчётность (10-K/10-Q, юнит-экономика)

| Источник | Что даёт | Доступ |
|---|---|---|
| **SEC EDGAR Financial Statement Data Sets** | XBRL-выгрузки `num.txt`/`sub.txt`/`pre.txt` по всем компаниям, поквартально | прямой HTTPS `efts.sec.gov/...`, без ключа [4](https://www.sec.gov/dera/data/financial-statement-data-sets.html), public domain |
| **SEC companyfacts API** | JSON всех XBRL-фактов по CIK | `data.sec.gov/api/xbrl/companyfacts/CIK##########.json`, User-Agent обязателен |
| **Kaggle: SEC Financial Statement Extracts** | те же выгрузки 2015–2017 одним архивом ~350 МБ | бесплатный аккаунт Kaggle [5](https://www.kaggle.com/securities-exchange-commission/financial-statement-extracts), CC0 |
| **EDGAR-CRAWLER** | тула для выкачки 10-K/10-Q/8-K и парсинга в JSON | open-source [6](https://github.com/lefterisloukas/edgar-crawler) |

**Рекомендация:** официальные SEC bulk ZIP — public domain, качаются `urllib`.
Датасет Kaggle удобен для первого оффлайн-эксперимента.

## 3. Макроэкономика и цепочки поставок

| Источник | Что даёт | Доступ |
|---|---|---|
| **FRED** (St. Louis Fed) | 800k+ рядов: ставки (DFF, DGS10), CPIAUCSL, GDP, UNRATE, VIXCLS, кредитные спреды, FX | бесплатный API-ключ (32 символа), [7](https://fred.stlouisfed.org) |
| **U.S. Treasury / BLS / BEA** | кривая доходности, занятость, ВВП | прямые открытые данные |
| **Supply-explorer / Open Supply Hub** | графы поставщиков (частично открытые) | открытый API |

**Рекомендация:** FRED для ставки/инфляции/VIX/спредов — один ключ на все ряды.

## 4. Сентимент и новости

| Источник | Что даёт | Доступ |
|---|---|---|
| **SEC 8-K + 10-K текст** | корпоративные события/риск-факторы | EDGAR, без ключа |
| **Kaggle финансовых новостей** | FinancialPhraseBank, Reuters/CNBC новости с разметкой сентимента | бесплатный аккаунт Kaggle |
| **Reddit pushshift-архивы** | посты r/wallstreetbets и т.п. | открытые архивы |

## 5. Поведенческие данные и ценообразование

| Источник | Что даёт |
|---|---|
| **Online Retail / E-commerce datasets (UCI/Kaggle)** | чеки, корзины, скидки, A/B — для эластичности спроса |
| **Instacart / H&M / Retailrocket (Kaggle)** | кликстримы, воронки, история покупок |

---

## Что реализовано в коде

`nexus_capital/data/real/` содержит загрузчики без внешних ключей для
моментального старта:

- `binance.py` — klines/aggTrades со `data.binance.vision` (zipped CSV),
  конвертация в тиковые фичи и синтетический L2-стакан из OHLCV.
- `sec_edgar.py` — поквартальные XBRL ZIP с SEC, парсинг `num.txt` в
  юнит-экономику (revenue, cogs, net_income, assets, liabilities, equity...).
- `fred.py` — загрузка макро-рядов (ставка, CPI, VIX, доходность трежерис)
  через FRED API (опциональный ключ) **или** оффлайн-генерация из реальных
  средних для воспроизводимости без ключа.

`scripts/train_real_data.py` — единая процедура: скачать → подготовить →
обучить ядро с `L_total` → оценить VaR/полезность.
