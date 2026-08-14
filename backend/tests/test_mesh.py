import numpy as np

from tidescout.config import load_fishery
from tidescout.pipeline import mesh

from . import synth
from .test_features_pipeline import _fake_bathy


def test_domain_mask_is_single_connected_component():
    z = np.full((200, 200), -5.0, dtype="float32")
    z[0:20, :] = 5.0                     # land strip
    z[150:160, 150:160] = -5.0           # isolated pond, must be dropped
    z[140:170, 60:70] = 5.0
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []  # synthetic raster has no overlap with real Winyah km
    m = mesh.domain_mask(z, synth.TRANSFORM, f, polygon=None)
    from scipy import ndimage
    _, n = ndimage.label(m)
    assert n == 1, "mesh domain must be exactly one connected water body"


def test_domain_polygon_simplifies_hard_but_keeps_area():
    z = np.full((200, 200), -5.0, dtype="float32")
    # rough up the shoreline so simplification has something to do
    z[0:30, :] = 5.0
    z[30, ::2] = 5.0
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []  # synthetic raster has no overlap with real Winyah km
    m = mesh.domain_mask(z, synth.TRANSFORM, f, polygon=None)
    poly = mesh.domain_polygon(m, synth.TRANSFORM, simplify_m=25.0)
    assert poly.is_valid
    assert len(poly.exterior.coords) < 400
    assert poly.area > 0.9 * m.sum() * 100.0   # 10 m cells = 100 m2 each


def test_build_mesh_sets_elevation_on_every_centroid(tmp_path, monkeypatch):
    z = np.full((300, 300), -5.0, dtype="float32")
    z[0:40, :] = 5.0
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []      # empty => use the whole water mask
    d = mesh.build_mesh("winyah-bay", f)
    elev = d.get_quantity("elevation").get_values(location="centroids")
    assert len(elev) == len(d.triangles)
    assert np.isfinite(elev).all(), "no NaN may reach the solver"
    assert elev.min() < 0.0


def test_friction_field_has_no_zero_values(tmp_path, monkeypatch):
    z = np.full((300, 300), -5.0, dtype="float32")
    z[0:40, :] = 5.0
    _fake_bathy(tmp_path, monkeypatch, z)
    f = load_fishery("winyah-bay")
    f.model_domain.polygon_utm_km = []
    from tidescout.pipeline.derivatives import build_derivatives
    build_derivatives("winyah-bay", f)
    d = mesh.build_mesh("winyah-bay", f)
    n = mesh.friction_field(d, "winyah-bay", f)
    assert len(n) == len(d.triangles)
    assert (n > 0).all(), "a zero Manning n is frictionless, not a default"
    assert n.max() <= 0.1
