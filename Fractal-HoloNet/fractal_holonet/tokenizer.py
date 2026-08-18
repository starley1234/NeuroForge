import os
import json
from typing import List, Dict, Any, Optional, Union
import torch


class SimpleProductionTokenizer:
    """
    Byte-level UTF-8 production tokenizer with support for Cyrillic and multilingual text.
    """
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        if vocab is not None:
            self.vocab = vocab
        else:
            self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
            for b in range(256):
                token_key = f"<byte_{b}>"
                if token_key not in self.vocab:
                    self.vocab[token_key] = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    # ---- словарь как единый источник истины ------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def pad_token_id(self) -> int:
        return self.vocab["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.vocab["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.vocab["<eos>"]

    @property
    def unk_token_id(self) -> int:
        return self.vocab["<unk>"]

    def encode(self, text: str, add_bos: bool = False) -> List[int]:
        tokens = [self.bos_token_id] if add_bos else []
        byte_data = text.encode("utf-8")
        for b in byte_data:
            token_key = f"<byte_{b}>"
            tokens.append(self.vocab.get(token_key, self.unk_token_id))
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        byte_list = []
        special_ids = {0, 1, 2, 3}
        for tid in token_ids:
            if skip_special and tid in special_ids:
                continue
            token_str = self.inv_vocab.get(tid, "")
            if token_str.startswith("<byte_") and token_str.endswith(">"):
                try:
                    b = int(token_str[6:-1])
                    byte_list.append(b)
                except ValueError:
                    pass
        return bytes(byte_list).decode("utf-8", errors="replace")

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleProductionTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)


class BpeTokenizer:
    """
    Byte-level BPE токенизатор (обучается на корпусах проекта, scripts/train_tokenizer.py).

    Обёртка над `tokenizers` (HuggingFace): подсловные токены дают в 2-4 раза
    более короткие последовательности для кириллицы, чем байтовый словарь, и
    сохраняют байтовое покрытие для любых символов (byte-level fallback).
    Спец-токены: <pad>, <bos>, <eos>, <unk>.
    """

    def __init__(self, hf_tokenizer):
        self._tok = hf_tokenizer

    # ---- словарь ----------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        return self._tok.token_to_id("<pad>")

    @property
    def bos_token_id(self) -> int:
        return self._tok.token_to_id("<bos>")

    @property
    def eos_token_id(self) -> int:
        return self._tok.token_to_id("<eos>")

    @property
    def unk_token_id(self) -> int:
        return self._tok.token_to_id("<unk>")

    def encode(self, text: str, add_bos: bool = False) -> List[int]:
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        if skip_special:
            special = {self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id}
            token_ids = [t for t in token_ids if t not in special]
        return self._tok.decode(token_ids)

    def save(self, path: str):
        self._tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BpeTokenizer":
        from tokenizers import Tokenizer as HFTokenizer

        return cls(HFTokenizer.from_file(path))

    @classmethod
    def train_from_files(
        cls,
        files: List[str],
        vocab_size: int = 8192,
        path: Optional[str] = None,
        encoding: str = "utf-8",
        min_frequency: int = 2,
    ) -> "BpeTokenizer":
        """
        Обучает byte-level BPE на списке файлов и опционально сохраняет.
        `encoding` позволяет подавать корпуса в cp1251 и т.п.
        """
        from tokenizers import Tokenizer as HFTokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.trainers import BpeTrainer

        tok = HFTokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            min_frequency=min_frequency,
            initial_alphabet=ByteLevel.alphabet(),
        )
        tok.train(files, trainer=trainer)
        obj = cls(tok)
        if path is not None:
            obj.save(path)
        return obj


def load_tokenizer(model_dir: str) -> Union[SimpleProductionTokenizer, BpeTokenizer]:
    """
    Автоопределение типа токенизатора по каталогу модели:
      * tokenizer.json со словарём "<pad>"/"<byte_N>" -> SimpleProductionTokenizer
      * tokenizer.json в HF-схеме -> BpeTokenizer
      * файла нет -> SimpleProductionTokenizer по умолчанию
    """
    path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(path):
        return SimpleProductionTokenizer()
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1024)
    if '"<pad>"' in head and '"<byte_' in head:
        return SimpleProductionTokenizer.load(path)
    return BpeTokenizer.load(path)


def build_config_for_tokenizer(
    tokenizer: Union[SimpleProductionTokenizer, BpeTokenizer],
    vocab_size: Optional[int] = None,
    **arch_kwargs,
):
    """
    Строит FractalHoloNetConfig, согласованный с токенизатором:
    vocab_size берётся из токенизатора (если не задан явно), pad/bos/eos —
    тоже. Устраняет классический рассинхрон (конфиг 300 vs словарь 260).
    """
    from fractal_holonet.core import FractalHoloNetConfig

    kwargs = dict(
        vocab_size=vocab_size or tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    kwargs.update(arch_kwargs)
    return FractalHoloNetConfig(**kwargs)


class FractalHoloNetInferencePipeline:
    """
    High-level production inference pipeline for text completion and embedding extraction.
    """
    def __init__(self, model_dir: str, device: str = "cpu"):
        from fractal_holonet.core import ProductionFractalHoloNet
        self.device = torch.device(device)
        self.model = ProductionFractalHoloNet.from_pretrained(model_dir, map_location=device)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = load_tokenizer(model_dir)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None
    ) -> Dict[str, Any]:
        input_ids = torch.tensor([self.tokenizer.encode(prompt, add_bos=False)], dtype=torch.long, device=self.device)
        prompt_len = input_ids.size(1)
        
        output_ids = self.model.generate(
            prompt_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=eos_token_id
        )
        
        generated_ids = output_ids[0].tolist()
        new_ids = generated_ids[prompt_len:]
        full_text = self.tokenizer.decode(generated_ids, skip_special=True)
        new_text = self.tokenizer.decode(new_ids, skip_special=True)
        
        return {
            "prompt": prompt,
            "generated_text": new_text,
            "full_text": full_text,
            "prompt_tokens": prompt_len,
            "generated_tokens": len(new_ids),
        }

    def get_embeddings(self, text: str) -> List[float]:
        input_ids = torch.tensor([self.tokenizer.encode(text, add_bos=True)], dtype=torch.long, device=self.device)
        with torch.no_grad():
            x = self.model.token_emb(input_ids)
            for block in self.model.blocks:
                x, _ = block(x)
            x = self.model.ln_f(x)
            emb = x.mean(dim=1).squeeze(0).cpu().tolist()
        return emb
