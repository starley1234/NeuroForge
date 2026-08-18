"""Local test console: 0.0.0.0 HTTP + JSON API."""

from __future__ import annotations

import json
import math
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch

from aura_micro.config import CLASS_NAMES, MODIFIER_NAMES, AuraConfig
from aura_micro.engine import load_pipeline
from aura_micro.paths import WEB_DIR
from aura_micro.pipeline import AuraPipeline
from aura_micro.synth import render_scene


def _json(handler: SimpleHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def make_handler(pipe: AuraPipeline):
    cfg = pipe.cfg

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(WEB_DIR), **kwargs)

        def log_message(self, fmt: str, *args) -> None:
            print("[http]", args[0] if args else fmt)

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/health":
                return _json(self, 200, {"ok": True, "classes": list(CLASS_NAMES)})
            if path == "/api/scene":
                return self._scene()
            if path in ("/", "/index.html"):
                self.path = "/index.html"
            return super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/infer":
                return self._infer()
            return _json(self, 404, {"error": "not found"})

        def _scene(self) -> None:
            rng = np.random.default_rng()
            wav, lab = render_scene(cfg, 0.4, rng)
            with torch.no_grad():
                out = pipe(torch.from_numpy(wav).unsqueeze(0))
            dec = pipe.decode(out)[0]
            payload = {
                "truth": {
                    "class": CLASS_NAMES[lab.class_id],
                    "azimuth_deg": round(math.degrees(lab.theta), 1),
                    "elevation_deg": round(math.degrees(lab.phi), 1),
                    "range_m": round(lab.range_m, 2),
                    "vr_mps": round(lab.vr, 2),
                    "modifiers": [
                        MODIFIER_NAMES[i] for i, v in enumerate(lab.modifiers.tolist()) if v > 0.5
                    ],
                },
                "pred": dec,
            }
            _json(self, 200, payload)

        def _infer(self) -> None:
            n = int(self.headers.get("Content-Length", "0"))
            raw = json.loads(self.rfile.read(n) or b"{}")
            class_id = int(raw.get("class_id", 0))
            class_id = max(0, min(class_id, len(CLASS_NAMES) - 1))
            rng = np.random.default_rng()
            wav, lab = render_scene(cfg, 0.4, rng, class_id=class_id)
            with torch.no_grad():
                out = pipe(torch.from_numpy(wav).unsqueeze(0))
            _json(
                self,
                200,
                {
                    "truth": {"class": CLASS_NAMES[lab.class_id], "range_m": lab.range_m},
                    "pred": pipe.decode(out)[0],
                },
            )

    return Handler


def run_server(host: str = "0.0.0.0", port: int = 8080, ckpt: Path | None = None) -> None:
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    pipe = load_pipeline(ckpt)
    httpd = ThreadingHTTPServer((host, port), make_handler(pipe))
    print(f"AURA-Micro console  http://{host}:{port}")
    httpd.serve_forever()
