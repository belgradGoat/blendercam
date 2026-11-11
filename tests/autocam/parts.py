"""Synthetic prismatic parts for the planner and verifier tests: a PartModel and a triangle
mesh (for verify) from an outline, pockets and round holes. Lengths in metres."""

import numpy as np
import shapely
from shapely.geometry import Point, Polygon, box

from camcore.part import Hole, LevelFace, Obstacle, PartModel, Wall


class Builder:
    def __init__(self, name, outline, top=0.0, bottom=-0.01):
        self.name = name
        self.outline = outline
        self.top, self.bottom = top, bottom
        self.pockets = []  # (polygon, floor z)
        self.holes = []  # (x, y, d, z_bottom or None (through), bottom kind)
        self.fillets = []  # (polygon, z_min, z_max): obstacles

    def pocket(self, poly, floor):
        self.pockets.append((poly, floor))
        return self

    def hole(self, x, y, d, z_bottom=None, bottom="flat", tip_angle=None, z_top=None):
        self.holes.append((x, y, d, z_bottom, bottom, tip_angle, z_top))
        return self

    def obstacle(self, poly, z_min, z_max):
        self.fillets.append((poly, z_min, z_max))
        return self

    def build(self):
        fid = iter(range(1000))
        levels, walls, faces, holes, tris, tri_faces = [], [], [], [], [], []

        def add_face(role, area, z_min, z_max, hole=None):
            f = next(fid)
            faces.append({"face": f, "role": role, "area": area, "z_min": z_min,
                          "z_max": z_max, "hole": hole})  # fmt: skip
            return f

        def wall_ring(ring_xy, z0, z1, role, hole=None):
            P = np.asarray(ring_xy)[:-1, :2]
            f = add_face(role, 0.0, z0, z1, hole)
            walls.append(Wall(f, role, z0, z1, P, hole))
            Q = np.roll(P, -1, axis=0)
            for a, b in zip(P, Q):
                tris.append([[*a, z0], [*b, z0], [*b, z1]])
                tris.append([[*a, z0], [*b, z1], [*a, z1]])
                tri_faces.extend([f, f])

        def level(poly, z, role):
            if poly.is_empty:
                return
            f = add_face(role, poly.area, z, z)
            levels.append(LevelFace(f, z, poly, role))
            dense = poly.segmentize(0.0005)
            for t in shapely.get_parts(shapely.delaunay_triangles(dense)):
                if dense.contains(t.centroid):
                    c = np.asarray(t.exterior.coords)[:3]
                    tris.append([[x, y, z] for x, y in c])
                    tri_faces.append(f)

        openings_top = []
        for poly, floor in self.pockets:
            openings_top.append(poly)
        hole_discs = []
        for i, (x, y, d, zb, bottom, tip, zt) in enumerate(self.holes):
            disc = Point(x, y).buffer(d / 2, quad_segs=64)
            hole_discs.append(disc)
            if zt is None:
                openings_top.append(disc)
        top_face = (
            self.outline.difference(shapely.union_all(openings_top))
            if openings_top
            else self.outline
        )
        level(top_face, self.top, "top")
        wall_ring(self.outline.exterior.coords, self.bottom, self.top, "wall")
        index = 0
        for poly, floor in self.pockets:
            inner = [hole_discs[i] for i, h in enumerate(self.holes) if h[6] is not None
                     and abs(h[6] - floor) < 1e-9 and poly.contains(hole_discs[i])]  # fmt: skip
            level(poly.difference(shapely.union_all(inner)) if inner else poly, floor, "floor")
            wall_ring(poly.exterior.coords, floor, self.top, "wall", index)
            holes.append(Hole(index, [poly.centroid.x, poly.centroid.y, self.top], None, self.top,
                              floor, "blind", "flat", None, False, True, True,
                              [faces[-1]["face"], faces[-2]["face"]], [faces[-1]["face"]]))  # fmt: skip
            index += 1
        for (x, y, d, zb, bottom, tip, zt), disc in zip(self.holes, hole_discs):
            z_top = self.top if zt is None else zt
            through = zb is None
            z_bottom = self.bottom if through else zb
            wall_ring(disc.exterior.coords, z_bottom, z_top, "hole_wall", index)
            fs = [faces[-1]["face"]]
            if not through and bottom == "flat":
                level(disc, z_bottom, "floor")
                fs.append(faces[-1]["face"])
            elif not through and bottom == "cone":
                f = add_face(
                    "curved", disc.area, z_bottom, z_bottom + d / 2 / np.tan(np.radians(tip / 2))
                )
                fs.append(f)
            holes.append(Hole(index, [x, y, z_top], d, z_top, z_bottom,
                              "through" if through else "blind",
                              "through" if through else bottom, tip, True, True, False,
                              fs, [fs[0]]))  # fmt: skip
            index += 1
        obstacles = []
        for poly, z0, z1 in self.fillets:
            f = add_face("fillet", poly.area, z0, z1)
            obstacles.append(Obstacle(f, "fillet", z1, z0, poly))
        bottom_face = add_face("bottom", self.outline.area, self.bottom, self.bottom)
        lo = list(self.outline.bounds[:2]) + [self.bottom]
        hi = list(self.outline.bounds[2:]) + [self.top]
        part = PartModel(self.name, lo, hi, levels, obstacles, walls, holes, faces)
        return part, np.asarray(tris, float), np.asarray(tri_faces, int)


def bracket(x0=0.1, y0=0.05, z_top=0.0):
    """The M8 bracket: 60 x 40 x 10, 2 x 5 mm through holes, a 4 mm blind (flat) hole 4 mm
    deep and a 16 x 8 x 3 mm pocket (sharp corners)."""
    b = Builder("Bracket", box(x0 - 0.03, y0 - 0.02, x0 + 0.03, y0 + 0.02), z_top, z_top - 0.01)
    b.pocket(box(x0 - 0.008, y0 + 0.006, x0 + 0.008, y0 + 0.014), z_top - 0.003)
    b.hole(x0 - 0.02, y0 + 0.01, 0.005)
    b.hole(x0 + 0.02, y0 + 0.01, 0.005)
    b.hole(x0, y0 - 0.012, 0.004, z_bottom=z_top - 0.004, bottom="flat")
    return b.build()
