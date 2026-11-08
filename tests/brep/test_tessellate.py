"""brepcam.tessellate: triangles within an absolute tolerance of the exact surfaces."""

import numpy as np
import pytest

from brepcam import tessellate as T


def _dense_points(tris, n=20, seed=1):
    """Points spread over every triangle (random barycentric + centroid + edge midpoints)."""
    w = np.vstack([np.random.default_rng(seed).dirichlet([1, 1, 1], n), T._BARY])
    return np.einsum("kj,mjc->mkc", w, tris).reshape(-1, 3)


def sphere_error(tris, center, radius):
    Q = _dense_points(tris)
    return np.abs(np.linalg.norm(Q - center, axis=1) - radius).max()


def cylinder_side_error(tris, center, radius, height):
    """Max radial error on the side wall (caps excluded)."""
    Q = _dense_points(tris)
    d = np.linalg.norm(Q[:, :2] - center[:2], axis=1)
    side = (Q[:, 2] > center[2] + 1e-9) & (Q[:, 2] < center[2] + height - 1e-9) & (d > radius / 2)
    return np.abs(d[side] - radius).max()


@pytest.fixture(scope="module")
def solids(kernel):
    return kernel.module("ops.solids")


@pytest.mark.parametrize("tol", [1e-4, 1e-5])
def test_sphere_within_tolerance(kernel, solids, tol):
    R = 0.02
    m = T.cam_mesh(kernel, solids.sphere((0, 0, 0), R), None, tol)
    assert m.within_tolerance and m.max_deviation <= tol
    err = sphere_error(m.triangles, np.zeros(3), R)
    assert err <= tol
    # the measurement finds the real worst case, not a loose bound
    assert m.max_deviation == pytest.approx(err, rel=0.05)


@pytest.mark.parametrize("tol", [1e-4, 1e-5])
def test_cylinder_within_tolerance(kernel, solids, tol):
    R, H = 0.01, 0.03
    m = T.cam_mesh(kernel, solids.cylinder((0, 0, 0), (0, 0, 1), R, H), None, tol)
    assert m.within_tolerance
    assert cylinder_side_error(m.triangles, np.zeros(3), R, H) <= tol


def test_trimmed_body_within_tolerance(kernel, solids):
    boolean = kernel.module("ops.boolean").boolean
    body, _ = boolean(
        solids.box((0, 0, 0), (0.05, 0.05, 0.02)),
        solids.cylinder((0.005, 0.0, -0.02), (0, 0, 1), 0.008, 0.04),
        "difference",
    )
    tol = 1e-5
    m = T.cam_mesh(kernel, body, None, tol)
    assert m.within_tolerance
    assert set(np.unique(m.face_ids)) == set(body.faces)  # every face meshed
    # the drilled hole's wall is a cylinder: check it directly
    Q = _dense_points(m.triangles)
    d = np.linalg.norm(Q[:, :2] - np.array([0.005, 0.0]), axis=1)
    wall = (d < 0.0085) & (np.abs(Q[:, 2]) < 0.0099)
    assert wall.sum() > 100
    assert np.abs(d[wall] - 0.008).max() <= tol


def test_matrix_maps_to_world_and_scales_tolerance(kernel, solids):
    R, tol = 0.01, 1e-5
    M = np.diag([2.0, 2.0, 2.0, 1.0])
    M[:3, 3] = (0.1, -0.2, 0.3)
    m = T.cam_mesh(kernel, solids.sphere((0, 0, 0), R), M, tol)
    center = np.array([0.1, -0.2, 0.3])
    assert m.within_tolerance
    # tolerance holds in world units, on the scaled sphere
    assert sphere_error(m.triangles, center, 2 * R) <= tol
    lo, hi = m.triangles.reshape(-1, 3).min(0), m.triangles.reshape(-1, 3).max(0)
    assert np.allclose(lo, center - 2 * R, atol=tol)
    assert np.allclose(hi, center + 2 * R, atol=tol)


def test_finer_tolerance_gives_more_triangles(kernel, solids):
    body = solids.sphere((0, 0, 0), 0.02)
    coarse = T.cam_mesh(kernel, body, None, 1e-4)
    fine = T.cam_mesh(kernel, body, None, 1e-5)
    assert len(fine.triangles) > len(coarse.triangles)


def test_triangulate_splits_every_polygon(kernel, solids):
    TessOptions = kernel.module("tessellate.grid").TessOptions
    tess = kernel.module("tessellate.body").tessellate_body(
        solids.box((0, 0, 0), (1, 1, 1)), TessOptions()
    )
    corners, fids = T.triangulate(tess)
    sizes = np.asarray(tess.poly_sizes)
    assert len(corners) == int(np.sum(sizes - 2))
    assert len(fids) == len(corners)
    # triangles reference corners of their own polygon's face
    assert set(np.unique(fids)) == set(np.unique(tess.face_ids))


def test_unverified_mesh_reports_no_deviation(kernel, solids):
    m = T.cam_mesh(kernel, solids.sphere((0, 0, 0), 0.02), None, 1e-4, verify=False)
    assert np.isnan(m.max_deviation)
    assert not m.within_tolerance
    assert m.stats()["max_deviation"] is None


def test_stats_are_plain_types(kernel, solids):
    s = T.cam_mesh(kernel, solids.sphere((0, 0, 0), 0.02), None, 1e-4).stats()
    assert isinstance(s["triangles"], int) and isinstance(s["max_deviation"], float)
    assert isinstance(s["bbox_min"], list)


def test_bad_tolerance(kernel, solids):
    with pytest.raises(ValueError):
        T.cam_mesh(kernel, solids.sphere((0, 0, 0), 0.02), None, 0.0)


def test_linear_scale():
    assert T.linear_scale(None) == 1.0
    assert T.linear_scale(np.diag([1.0, 3.0, 2.0, 1.0])) == pytest.approx(3.0)


def _drilled_box(kernel, solids, s=1.0):
    boolean = kernel.module("ops.boolean").boolean
    body, _ = boolean(
        solids.box((0, 0, 0), (0.05 * s, 0.05 * s, 0.02 * s)),
        solids.cylinder((0.005 * s, 0.0, -0.02 * s), (0, 0, 1), 0.008 * s, 0.04 * s),
        "difference",
    )
    return body


def test_chunked_trim_tests_give_the_same_mesh_and_are_restored(kernel, solids):
    body = _drilled_box(kernel, solids)
    TessOptions = kernel.module("tessellate.grid").TessOptions
    tessellate_body = kernel.module("tessellate.body").tessellate_body
    trimmed = kernel.module("tessellate.trimmed")
    originals = (trimmed.points_in_region, trimmed.distance_to_polylines)
    opts = TessOptions(chord_tol=5e-5)
    plain = tessellate_body(body, opts)
    with T.bounded_trim_tests(kernel, chunk=7):  # tiny chunks: many concatenations
        assert trimmed.points_in_region is not originals[0]
        with T.bounded_trim_tests(kernel):  # nesting doesn't double-wrap
            assert not hasattr(trimmed.points_in_region, "__wrapped__")
        chunked = tessellate_body(body, opts)
    assert (trimmed.points_in_region, trimmed.distance_to_polylines) == originals
    assert np.array_equal(plain.points, chunked.points)
    assert np.array_equal(plain.corner_verts, chunked.corner_verts)


def test_large_part_at_fine_tolerance_is_bounded(kernel, solids):
    body = _drilled_box(kernel, solids, s=40.0)  # 2 m x 2 m x 0.8 m
    m = T.cam_mesh(kernel, body, None, 1e-5)
    assert m.chord_tol == T.MIN_CHORD_TOL
    assert any("relative limit" in w for w in m.warnings)
    assert m.passes == 1  # no pointless refinement at the limit
    assert np.isfinite(m.max_deviation) and m.max_deviation > 0
    assert m.within_tolerance == (m.max_deviation <= 1e-5)
