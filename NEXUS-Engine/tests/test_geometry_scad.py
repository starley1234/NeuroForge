import numpy as np
import pytest

from nexus.fem.solver import solve_loadpath
from nexus.geometry import Cube, Cylinder, audit, build_graph, mass_properties, voxelize
from nexus.geometry.csg import difference
from nexus.scad import compile_scad, generate, parse_scad, render


def test_cube_volume_and_mass():
    node = Cube((20.0, 20.0, 20.0))
    vox = voxelize(node, resolution=40)
    props = mass_properties(vox, "alu6061")
    assert props.volume_mm3 == pytest.approx(8000.0, rel=0.08)
    # 8 см³ алюминия ≈ 21.6 г
    assert props.mass_g == pytest.approx(21.6, rel=0.1)
    assert np.allclose(props.centroid_mm, [0, 0, 0], atol=0.5)


def test_difference_creates_hole():
    solid = difference(Cube((20.0, 20.0, 10.0)), Cylinder(h=40.0, r1=4.0))
    vox = voxelize(solid, resolution=36)
    props = mass_properties(vox)
    full = mass_properties(voxelize(Cube((20.0, 20.0, 10.0)), resolution=36))
    assert props.volume_mm3 < full.volume_mm3 * 0.95


def test_parse_scad_roundtrip():
    tree = parse_scad("""
        w = 10; t = 2;
        difference() {
          cube([w, w, t], center=true);
          cylinder(h=t*3, r=2, center=true);
        }
    """)
    vox = voxelize(tree, resolution=32)
    assert vox.occupancy.any()
    graph = build_graph(tree)
    assert graph.n_nodes >= 3 and graph.n_edges > 0


def test_compile_error_detected():
    assert compile_scad("cube([1,1,1)").ok is False
    assert compile_scad("cube([1,1,1]);").ok is True


def test_audit_flags_two_components():
    two = Cube((5.0, 5.0, 5.0))
    from nexus.geometry.csg import Transform, union
    model = union(two, Transform(Cube((5.0, 5.0, 5.0)), translate=(30.0, 0.0, 0.0)))
    report = audit(voxelize(model, resolution=32))
    assert report.n_solid_components == 2
    assert report.manifold is False


@pytest.mark.parametrize("template", list(__import__("nexus.scad.generator", fromlist=["TEMPLATES"]).TEMPLATES))
def test_all_templates_compile_and_render(template):
    import random
    from nexus.scad.generator import sample
    s = sample(template, random.Random(11))
    res = render(s.code, resolution=16, material=s.material)
    assert res.ok, res.error
    assert res.mass.mass_g > 0
    assert res.graph.n_nodes > 1


def test_fem_stress_scales_with_force():
    vox = voxelize(Cube((10.0, 10.0, 30.0)), resolution=20)
    low = solve_loadpath(vox, (0, 0, -100.0), iterations=80)
    high = solve_loadpath(vox, (0, 0, -1000.0), iterations=80)
    assert high.max_stress_pa > low.max_stress_pa * 5
    assert high.safety_factor < low.safety_factor


def test_generator_produces_valid_parts():
    ok = 0
    for s in generate(6, seed=5):
        res = render(s.code, resolution=14, material=s.material)
        ok += int(res.ok)
    assert ok == 6


def test_wall_thickness_is_measured_in_mm():
    from nexus.geometry.voxel import min_wall_thickness
    plate = voxelize(Cube((100.0, 60.0, 3.0)), resolution=32)
    assert min_wall_thickness(plate) == pytest.approx(3.0, rel=0.15)
    tube = parse_scad("difference(){cylinder(h=20,r=15); cylinder(h=60,r=13,center=true);}")
    assert min_wall_thickness(voxelize(tube, resolution=32)) == pytest.approx(2.0, rel=0.25)


def test_thin_plate_survives_anisotropic_voxelization():
    res = render("difference(){cube([100,60,3],center=true); cube([50,30,9],center=true);}",
                 resolution=24)
    assert res.mass.mass_g > 0
    # объём рамки: 100*60*3 − 50*30*3 = 13500 мм³
    assert res.mass.volume_mm3 == pytest.approx(13500, rel=0.2)
