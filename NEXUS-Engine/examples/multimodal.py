"""Пять классов модальностей на одной латентной шине с общим временем."""
import torch

from nexus import LatentPacket, NexusConfig, NexusEngine
from nexus.encoders import (AudioSSMEncoder, BiophysicsEncoder, BRepGNOEncoder,
                            EventODEEncoder, PointSSMEncoder)
from nexus.scad import render

cfg = NexusConfig.tiny()
model = NexusEngine(cfg)
d = cfg.d_latent

brep = BRepGNOEncoder(d, width=64, layers=2)
audio = AudioSSMEncoder(d, width=32)
points = PointSSMEncoder(d, width=32)
events = EventODEEncoder(d, width=32)
bio = BiophysicsEncoder(d, channels=8, width=32)
for name, enc in [("brep", brep), ("audio", audio), ("gaussian", points),
                  ("event", events), ("ecg", bio)]:
    model.register_encoder(name, enc)

res = render("difference(){cube([30,30,6],center=true); cylinder(h=20,r=4,center=true);}", 16)
nodes, adj, mask = brep.encode_graphs([res.graph])
ts = torch.cumsum(torch.rand(1, 24) * 2e-3, dim=1)

packets = [
    brep(nodes, adj, mask),                        # инженерия: топология B-Rep
    audio(torch.randn(1, 4800)),                   # акустика работы механизма
    points(torch.randn(1, 32, 10)),                # 3D Gaussian Splatting
    events(torch.rand(1, 24, 3), ts),              # DVS-камера
    bio(torch.randn(1, 24, 8), ts),                # биофизика оператора
]
tokens = model.text_encoder.encode_text(["<task>Проверь деталь под нагрузкой 300 Н"])
out = model(tokens=tokens, packets=packets, reason=True, continuous=True)

print({p.modality: p.shape for p in packets})
print("latent-шагов рассуждения:", out.reasoning.steps)
print("непрерывный выход:", tuple(out.actions.shape), tuple(out.field.shape))
