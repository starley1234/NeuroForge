"""
Мультиязычный токенизатор NEXUS-Capital (русский + английский).

Основной путь — **предобученный SentencePiece BPE из XLM-RoBERTa**
(`xlm-roberta-base/sentencepiece.bpe.model`), словарь ~250 000 токенов,
который покрывает и русский, и английский с общим алфавитом. Благодаря
этому словарь НЕ обучается заново — мы переиспользуем готовую
мультиязычную сегментацию, а обучается только embedding-матрица под наши
экономические задачи.

Если сеть недоступна (изолированная среда/CI), автоматически строится
компактный BPE-фоллбэк из 8k токенов на встроенном RU+EN финансовом
корпусе — только чтобы пайплайн запускался. Для настоящей работы нужен
XLM-R (скачивается один раз и кэшируется huggingface_hub).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

# Специальные токены
PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"
UNK = "<unk>"
MASK = "<mask>"
NUM = "<num>"  # плейсхолдер для числа в тексте

_SPECIALS = [PAD, UNK, BOS, EOS, MASK, NUM]

# Встроенный RU+EN финансовый корпус для оффлайн-фоллбэка.
# Реальные данные для обучения подаются отдельно (см. data/real/corpus.py).
_BUNDED_FINANCIAL_CORPUS = [
    # English
    "Revenue grew by twenty percent year over year in the first quarter.",
    "Gross profit margin expanded to forty two percent as cost of goods sold fell.",
    "The company reported net income of one hundred million dollars.",
    "Operating cash flow increased while capital expenditures remained stable.",
    "Customer acquisition cost decreased and lifetime value improved.",
    "Value at risk at the five percent confidence level is within limits.",
    "The Federal Reserve raised the federal funds rate by fifty basis points.",
    "Inflation accelerated to three point four percent and treasury yields rose.",
    "Earnings before interest taxes depreciation and amortization beat consensus.",
    "Inventory turnover improved and accounts receivable days declined.",
    "The board approved a share buyback and increased the dividend.",
    "Credit spreads widened as high yield bonds sold off on risk aversion.",
    # Русский
    "Выручка компании выросла на двадцать процентов в первом квартале.",
    "Валовая рентабельность достигла сорока двух процентов за счёт снижения себестоимости.",
    "Чистая прибыль составила сто миллионов долларов по итогам года.",
    "Операционный денежный поток увеличился, капитальные затраты стабильны.",
    "Стоимость привлечения клиента снизилась, а пожизненная ценность выросла.",
    "Стоимость под риском на уровне доверия девяносто пять процентов в норме.",
    "Центральный банк поднял ключевую ставку на пятьдесят базисных пунктов.",
    "Инфляция ускорилась до трёх целых четырёх десятых процента, доходность выросла.",
    "Прибыль до уплаты процентов налогов и амортизации превысила прогноз.",
    "Оборачиваемость запасов улучшилась, дни дебиторской задолженности сократились.",
    "Совет директоров одобрил обратный выкуп акций и рост дивидендов.",
    "Кредитные спреды расширились на фоне бегства от риска по высокодоходным облигациям.",
    "Спрос эластичен по цене, скидка приводит к росту объёма продаж.",
    "Дебиторская и кредиторская задолженность отражены в бухгалтерском балансе.",
    "Активы равны обязательствам плюс капитал, бухгалтерский баланс сходится.",
]


# Числа в финансовом тексте (включая $, %, скобки, минус, рус. пробел/запятая)
_NUMBER_RE = re.compile(
    r"[-−–]?\s?\$?\u2212?\d{1,3}(?:[ ,.]?\d{3})*(?:[.,]\d+)?\s?%?"
)


@dataclass
class EncodingResult:
    ids: list[int]
    tokens: list[str]
    attention_mask: list[int]
    number_values: list[float]
    number_mask: list[bool]


class NexusTokenizer:
    """
    Обёртка над предобученным XLM-RoBERTa SentencePiece (или оффлайн BPE).

    Особенность: числа в тексте детектируются ДО токенизации, заменяются
    специальным токеном <num> и их вещественные значения возвращаются
    отдельно — это позволяет FinancialTextEncoder внедрять числа как
    непрерывные тензоры стоимости вместо разрезания на BPE-куски.
    """

    def __init__(self, backend, vocab_size: int, name: str):
        self._backend = backend
        self.vocab_size = vocab_size
        self.name = name
        # id специальных токенов
        self.pad_id = self._special_id(PAD, default=0)
        self.unk_id = self._special_id(UNK, default=1)
        self.bos_id = self._special_id(BOS, default=2)
        self.eos_id = self._special_id(EOS, default=3)
        self.mask_id = self._special_id(MASK, default=4)
        self.num_id = self._special_id(NUM, default=5)

    def _special_id(self, tok: str, default: int) -> int:
        try:
            return self._backend.token_to_id(tok)
        except Exception:
            return default

    # ── фабрики ──────────────────────────────────────────────────
    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "xlm-roberta-base",
        filename: str = "sentencepiece.bpe.model",
        cache_dir: str | None = None,
    ) -> "NexusTokenizer":
        """
        Скачивает и загружает предобученный SentencePiece XLM-R.
        Бросает исключение, если нет сети — вызывающий код может
        переключиться на .offline().
        """
        from huggingface_hub import hf_hub_download
        from tokenizers import SentencePieceUnigramTokenizer

        path = hf_hub_download(model_id, filename, cache_dir=cache_dir)
        backend = SentencePieceUnigramTokenizer.from_spm(path)
        # Добавляем спец-токены (PAD и т.д.), если их ещё нет. XLM-R уже
        # содержит <s>,</s>,<unk>,<pad>,<mask>; добавляем <num>.
        existing = set(backend.get_vocab())
        if NUM not in existing:
            backend.add_special_tokens([NUM])
        vocab_size = backend.get_vocab_size()
        return cls(backend, vocab_size, f"xlm-r:{model_id}")

    @classmethod
    def offline(cls, vocab_size: int = 8000,
                corpus: Iterable[str] | None = None) -> "NexusTokenizer":
        """
        Оффлайн BPE-фоллбэк: обучается на встроенном RU+EN корпусе.
        Используется только когда предобученный XLM-R недоступен.
        """
        from tokenizers import Tokenizer, models, trainers, pre_tokenizers
        from tokenizers.processors import TemplateProcessing

        backend = Tokenizer(models.BPE(unk_token=UNK))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=_SPECIALS,  # порядок задаёт id 0..5
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        texts = list(corpus) if corpus is not None else list(_BUNDED_FINANCIAL_CORPUS)
        backend.train_from_iterator(texts, trainer=trainer)
        bos_id = backend.token_to_id(BOS)
        eos_id = backend.token_to_id(EOS)
        backend.post_processor = TemplateProcessing(
            single=f"{BOS} $A {EOS}",
            special_tokens=[(BOS, bos_id), (EOS, eos_id)],
        )
        vs = backend.get_vocab_size()
        return cls(backend, vs, "offline-bpe")

    @classmethod
    def default(cls, cache_dir: str | None = None,
                prefer_offline: bool | None = None) -> "NexusTokenizer":
        """
        Пытается загрузить предобученный XLM-R; при неудаче — оффлайн BPE.
        Установите NEXUS_OFFLINE_TOKENIZER=1, чтобы принудительно использовать
        оффлайн-режим (без сети).
        """
        if prefer_offline is None:
            prefer_offline = os.environ.get(
                "NEXUS_OFFLINE_TOKENIZER", "0") == "1"
        if prefer_offline:
            tok = cls.offline()
            print(f"[tokenizer] оффлайн-режим: {tok.name}, "
                  f"vocab={tok.vocab_size}")
            return tok
        try:
            tok = cls.from_pretrained(cache_dir=cache_dir)
            print(f"[tokenizer] загружен предобученный XLM-R: "
                  f"vocab={tok.vocab_size}")
            return tok
        except Exception as e:
            print(f"[tokenizer] предобученный XLM-R недоступен ({e}); "
                  f"использую оффлайн BPE.")
            tok = cls.offline()
            print(f"[tokenizer] оффлайн: vocab={tok.vocab_size}")
            return tok

    # ── кодирование ──────────────────────────────────────────────
    def _extract_numbers(self, text: str) -> tuple[str, list[float]]:
        values: list[float] = []

        def repl(m: re.Match) -> str:
            raw = m.group(0)
            v = _parse_number(raw)
            values.append(v)
            return f" {NUM} "

        masked = _NUMBER_RE.sub(repl, text)
        return masked, values

    def encode(
        self,
        text: str,
        max_length: int | None = None,
        add_special_tokens: bool = True,
    ) -> EncodingResult:
        masked, values = self._extract_numbers(text)
        enc = self._backend.encode(masked, add_special_tokens=add_special_tokens)
        ids = list(enc.ids)
        tokens = list(enc.tokens)
        number_mask = [tid == self.num_id for tid in ids]
        number_values: list[float] = []
        vi = 0
        for is_num in number_mask:
            if is_num and vi < len(values):
                number_values.append(values[vi])
                vi += 1
            else:
                number_values.append(0.0)
        attn = [1] * len(ids)
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
            tokens = tokens[:max_length]
            attn = attn[:max_length]
            number_mask = number_mask[:max_length]
            number_values = number_values[:max_length]
        return EncodingResult(
            ids=ids, tokens=tokens, attention_mask=attn,
            number_values=number_values, number_mask=number_mask,
        )

    def encode_batch(
        self,
        texts: list[str],
        max_length: int = 256,
    ) -> dict[str, torch.Tensor]:
        encs = [self.encode(t, max_length=max_length) for t in texts]
        max_len = max(len(e.ids) for e in encs)
        B = len(encs)
        ids = torch.full((B, max_len), self.pad_id, dtype=torch.long)
        attn = torch.zeros(B, max_len, dtype=torch.long)
        num_vals = torch.zeros(B, max_len, dtype=torch.float32)
        num_mask = torch.zeros(B, max_len, dtype=torch.bool)
        for i, e in enumerate(encs):
            n = len(e.ids)
            ids[i, :n] = torch.tensor(e.ids, dtype=torch.long)
            attn[i, :n] = torch.tensor(e.attention_mask, dtype=torch.long)
            num_vals[i, :n] = torch.tensor(e.number_values, dtype=torch.float32)
            num_mask[i, :n] = torch.tensor(e.number_mask, dtype=torch.bool)
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "number_values": num_vals,
            "number_mask": num_mask,
        }

    def decode(self, ids: list[int] | torch.Tensor,
               skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._backend.decode(ids, skip_special_tokens=skip_special_tokens)

    def id_to_token(self, idx: int) -> str | None:
        return self._backend.id_to_token(idx)

    def token_to_id(self, token: str) -> int | None:
        return self._backend.token_to_id(token)

    def get_vocab(self) -> dict[str, int]:
        return self._backend.get_vocab()


def _parse_number(s: str) -> float:
    """Парсит число из финансового формата, включая $, %, скобки, запятые,
    а также денежные суффиксы K/M/B (тыс./млн/млрд)."""
    neg = s.strip().startswith("(") and s.strip().endswith(")")
    t = s.strip().replace("$", "").replace("%", "").replace(" ", "")
    t = t.replace("\u2212", "-").replace("−", "-").replace("–", "-")
    t = t.replace("(", "").replace(")", "")
    # Денежные суффиксы
    mult = 1.0
    if t and t[-1].upper() in ("K", "M", "B", "T"):
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[t[-1].upper()]
        t = t[:-1]
    # Нормализация десятичных/тысячных разделителей
    has_comma = "," in t
    has_dot = "." in t
    if has_comma and has_dot:
        if t.rfind(",") > t.rfind("."):
            t = t.replace(".", "").replace(",", ".")  # евро-формат 1.234,56
        else:
            t = t.replace(",", "")                    # US 1,234.56
    elif has_comma and not has_dot:
        # запятая как разделитель тысяч (если 3 цифры после) или как запятая
        parts = t.split(",")
        if all(len(p) == 3 for p in parts[1:]):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
    sign = -1.0 if neg or t.startswith("-") else 1.0
    t = t.lstrip("-")
    try:
        return sign * mult * float(t)
    except ValueError:
        return 0.0
