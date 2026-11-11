"""A B-rep object as the planner's part model (camcore/part.py), in world space.

From the classification (brep/info.py, cached) and the exact geometry:
- level faces facing up: their trimmed outline (every loop, sampled within the tolerance);
- faces that are neither level nor exactly vertical (fillets, chamfers, domes, slopes,
  undersides, drafted walls): their XY footprint from the exact triangle mesh (obstacles);
- exactly vertical faces: the XY trace of their outer loop (walls);
- the holes, with their bottoms.
"""

import numpy as np
import shapely

from ..brep import bridge, info
from ..brep.brepcam import edges as brep_edges
from ..brep.ocl_mesh import cam_mesh_for
from .camcore.part import Hole, LevelFace, Obstacle, PartModel, Wall, as_geometry

VERTICAL_NZ = 1e-6  # |nz| below this everywhere: a vertical wall


def _rings(face, tol, M) -> list:
    loops = range(len(face.loops)) if face.loops is not None else [0]
    rings = []
    for li in loops:
        P = brep_edges.sample_loop(face, li, tol, M)[0]
        rings.append(np.asarray(P)[:, :2].tolist())
    return rings


def _footprint(tris: np.ndarray):
    """Union of the XY projections of triangles (n, 3, 3)."""
    if not len(tris):
        return shapely.Polygon()
    xy = tris[:, :, :2]
    a = (xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (
        xy[:, 1, 1] - xy[:, 0, 1]
    )
    xy = xy[np.abs(a) > 1e-14]
    if not len(xy):
        return shapely.Polygon()
    polys = shapely.polygons(np.concatenate([xy, xy[:, :1]], axis=1))
    return shapely.union_all(polys).buffer(0)


def part_model(ob, tolerance: float = 1e-5):
    """(PartModel, Classification) of a B-rep object in world space."""
    cl, _ = info.classification(ob, tolerance)
    body = bridge.read_body(ob)
    M = bridge.world_matrix(ob)
    local_tol = tolerance / brep_edges.linear_scale(M)
    mesh = None
    levels, obstacles, walls, faces, warnings = [], [], [], [], list(cl.warnings)
    for f in cl.faces:
        faces.append(
            {
                "face": f.face,
                "role": f.role,
                "area": f.area,
                "z_min": f.z_min,
                "z_max": f.z_max,
                "hole": f.hole,
            }
        )
        level_up = f.nz_min >= info.classify.LEVEL_COS
        level_down = f.nz_max <= -info.classify.LEVEL_COS
        vertical = max(abs(f.nz_min), abs(f.nz_max)) <= VERTICAL_NZ
        face = body.faces[f.face]
        if level_up:
            poly = as_geometry([_rings(face, local_tol, M)])
            if poly.is_empty:
                warnings.append(f"Face {f.face}: no outline")
                continue
            levels.append(LevelFace(f.face, f.z_max, poly, f.role))
        elif vertical:
            P = brep_edges.sample_loop(face, 0, local_tol * 10, M)[0]
            walls.append(Wall(f.face, f.role, f.z_min, f.z_max, np.asarray(P)[:, :2], f.hole))
        elif level_down and f.role == "bottom":
            continue
        else:
            if mesh is None:
                mesh, _ = cam_mesh_for(ob, tolerance)
            tris = np.asarray(mesh.triangles)[np.asarray(mesh.face_ids) == f.face]
            obstacles.append(Obstacle(f.face, f.role, f.z_max, f.z_min, _footprint(tris)))
    holes = []
    for h in cl.holes:
        holes.append(
            Hole(
                index=int(h["index"]),
                center=[float(x) for x in h["center"]],
                diameter=h["diameter"],
                z_top=float(h["z_top"]),
                z_bottom=float(h["z_bottom"]),
                kind=h["kind"],
                bottom=h.get("bottom", "other"),
                tip_angle=h.get("tip_angle"),
                circular=bool(h["circular"]),
                vertical=bool(h["vertical"]),
                prismatic=bool(h.get("prismatic")),
                faces=[int(x) for x in h["faces"]],
                walls=[int(x) for x in h["walls"]],
            )
        )
    part = PartModel(
        name=ob.name,
        bbox_min=list(cl.bbox_min),
        bbox_max=list(cl.bbox_max),
        levels=levels,
        obstacles=obstacles,
        walls=walls,
        holes=holes,
        faces=faces,
        tolerance=tolerance,
        warnings=warnings,
    )
    return part, cl


def part_triangles(ob, tolerance: float = 1e-5):
    """(triangles (n, 3, 3), face ids) of the exact mesh in world space (m)."""
    mesh, _ = cam_mesh_for(ob, tolerance)
    return np.asarray(mesh.triangles, float), np.asarray(mesh.face_ids, dtype=np.int64)
