"""
Fractal-HoloNet / ELAST-HOLO.

Пакет с ядром архитектуры v1 (core), мультимодальной веткой (multimodal),
токенизаторами (tokenizer: байтовый + byte-level BPE), дистилляцией
(distillation), автономным самообучением (self_train), реестром обученных
моделей (registry) и REST-сервисом (serve).

Пример:
    from fractal_holonet.core import ProductionFractalHoloNet
    from fractal_holonet.tokenizer import load_tokenizer, build_config_for_tokenizer
    from fractal_holonet import registry
"""

__version__ = "2.3.0"
