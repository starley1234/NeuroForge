"""Мини-пример: спроектировали → проверили → получили физическую награду."""
from nexus.fem import solve
from nexus.scad import render
from nexus.training import score_scad

CODE = """
difference() {
  cube([60, 40, 6], center=true);
  cylinder(h=20, r=6, center=true);
}
"""

res = render(CODE, resolution=32, material="alu6061")
print("масса, г:", round(res.mass.mass_g, 2))
print("аудит:", res.audit_report.to_dict())

fem = solve(res.voxels, force_n=(0, 0, -450.0), fixture="base", material="alu6061")
print("σmax, МПа:", round(fem.max_stress_pa / 1e6, 2), "| запас:", round(fem.safety_factor, 2))

print("награда RL:", score_scad(CODE, (0, 0, -450.0), material="alu6061").to_dict())
