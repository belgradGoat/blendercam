"""What the planner knows about a part: its faces as seen from above (+Z).

Made from a B-rep by the Blender glue (fabex/autocam/partmodel.py), or by hand in tests.
Lengths in metres, XY polygons as shapely geometry.

- levels: level faces facing up (tops and floors), each with its exact outline;
- obstacles: faces that are neither level nor vertical (fillets, chamfers, domes, slopes,
  undersides). 2.5D machining treats each as solid from its highest point down, so it is
  never cut into; 3D finishing (a later milestone) takes them;
- walls: vertical faces, with the XY trace of their boundary (for coverage);
- holes: the openings found in the B-rep (round holes with their diameter and bottom);
- faces: every face's role, for reporting what is machined and what is not.

    section(z) -> the XY area that is material just below height z, as far as 2.5D
                  machining from above is concerned.
"""

from dataclasses import dataclass, field

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely import affinity

from .errors import CamcoreError

Z_TOL = 1e-6  # m: level faces this close are at the same height


def as_geometry(rings: list):
    """[[shell, hole, ...], ...] (lists of [x, y]) -> valid (Multi)Polygon."""
    polys = []
    for p in rings or []:
        if not p:
            continue
        shell, holes = p[0], p[1:]
        if len(shell) < 3:
            continue
        poly = Polygon(shell, [h for h in holes if len(h) >= 3])
        if not poly.is_valid:
            poly = shapely.make_valid(poly)
        polys.append(poly)
    if not polys:
        return Polygon()
    g = shapely.union_all(polys)
    return _polygonal(g)


def _polygonal(g):
    """Only the polygons of a geometry (make_valid / unions can add lines and points)."""
    if g.is_empty:
        return Polygon()
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    parts = [p for p in getattr(g, "geoms", [g]) if isinstance(p, (Polygon, MultiPolygon))]
    return shapely.union_all(parts) if parts else Polygon()


def polygons(g) -> list:
    """The polygons of a (Multi)Polygon as a list."""
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    return [p for p in g.geoms if isinstance(p, Polygon) and not p.is_empty]


def as_rings(g, digits: int = 7) -> list:
    """(Multi)Polygon -> [[shell, hole, ...], ...] with rounded coordinates (JSON)."""
    out = []
    for p in polygons(g):
        rings = [p.exterior] + list(p.interiors)
        out.append([np.round(np.asarray(r.coords)[:-1, :2], digits).tolist() for r in rings])
    return out


@dataclass
class LevelFace:
    face: int
    z: float
    polygon: object  # shapely (Multi)Polygon
    role: str = "floor"


@dataclass
class Obstacle:
    face: int
    role: str
    z_max: float
    z_min: float
    polygon: object


@dataclass
class Wall:
    face: int
    role: str
    z_min: float
    z_max: float
    points: np.ndarray  # (n, 2) XY trace
    hole: int | None = None


@dataclass
class Hole:
    index: int
    center: list  # [x, y, z] (entry)
    diameter: float | None  # None: not round
    z_top: float
    z_bottom: float
    kind: str  # through / blind / open
    bottom: str  # through / flat / cone / other
    tip_angle: float | None = None  # degrees, cone bottoms
    circular: bool = True
    vertical: bool = True
    prismatic: bool = False
    faces: list = field(default_factory=list)
    walls: list = field(default_factory=list)

    def disc(self, grow: float = 0.0):
        r = self.diameter / 2 + grow
        return shapely.Point(self.center[0], self.center[1]).buffer(r, quad_segs=64)


@dataclass
class PartModel:
    name: str
    bbox_min: list
    bbox_max: list
    levels: list  # [LevelFace]
    obstacles: list  # [Obstacle]
    walls: list  # [Wall]
    holes: list  # [Hole]
    faces: list  # [{"face", "role", "area", "z_min", "z_max", "hole"}]
    tolerance: float = 1e-5
    warnings: list = field(default_factory=list)

    @property
    def top(self) -> float:
        return float(self.bbox_max[2])

    @property
    def bottom(self) -> float:
        return float(self.bbox_min[2])

    def outline(self):
        """The part's footprint (every level face, filled)."""
        g = shapely.union_all([lv.polygon for lv in self.levels] + [Polygon()])
        return _polygonal(g)

    def level_heights(self) -> list:
        """Distinct heights of the level faces, highest first."""
        zs = sorted((lv.z for lv in self.levels), reverse=True)
        out = []
        for z in zs:
            if not out or out[-1] - z > Z_TOL:
                out.append(z)
        return out

    def section(self, z: float, drilled=()):
        """XY area that is material just below `z` for machining from above: level faces at
        or above z, obstacles reaching above z (solid from their top down) and the discs of
        `drilled` holes whose top is above z (drilling clears them, not milling)."""
        parts = [lv.polygon for lv in self.levels if lv.z >= z - Z_TOL]
        parts += [ob.polygon for ob in self.obstacles if ob.z_max > z + Z_TOL]
        parts += [h.disc() for h in drilled if h.z_top > z + Z_TOL]
        if not parts:
            return Polygon()
        return _polygonal(shapely.union_all(parts))

    def translated(self, dx: float, dy: float, dz: float) -> "PartModel":
        def mv(g):
            return affinity.translate(g, dx, dy)

        return PartModel(
            name=self.name,
            bbox_min=[self.bbox_min[0] + dx, self.bbox_min[1] + dy, self.bbox_min[2] + dz],
            bbox_max=[self.bbox_max[0] + dx, self.bbox_max[1] + dy, self.bbox_max[2] + dz],
            levels=[LevelFace(lv.face, lv.z + dz, mv(lv.polygon), lv.role) for lv in self.levels],
            obstacles=[
                Obstacle(o.face, o.role, o.z_max + dz, o.z_min + dz, mv(o.polygon))
                for o in self.obstacles
            ],
            walls=[
                Wall(w.face, w.role, w.z_min + dz, w.z_max + dz, w.points + [dx, dy], w.hole)
                for w in self.walls
            ],
            holes=[
                Hole(
                    h.index,
                    [h.center[0] + dx, h.center[1] + dy, h.center[2] + dz],
                    h.diameter,
                    h.z_top + dz,
                    h.z_bottom + dz,
                    h.kind,
                    h.bottom,
                    h.tip_angle,
                    h.circular,
                    h.vertical,
                    h.prismatic,
                    list(h.faces),
                    list(h.walls),
                )
                for h in self.holes
            ],
            faces=[dict(f, z_min=f["z_min"] + dz, z_max=f["z_max"] + dz) for f in self.faces],
            tolerance=self.tolerance,
            warnings=list(self.warnings),
        )

    # --- JSON (tests, debugging) ------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bbox_min": list(self.bbox_min),
            "bbox_max": list(self.bbox_max),
            "levels": [
                {"face": lv.face, "z": lv.z, "role": lv.role, "polygon": as_rings(lv.polygon)}
                for lv in self.levels
            ],
            "obstacles": [
                {
                    "face": o.face,
                    "role": o.role,
                    "z_max": o.z_max,
                    "z_min": o.z_min,
                    "polygon": as_rings(o.polygon),
                }
                for o in self.obstacles
            ],
            "walls": [
                {
                    "face": w.face,
                    "role": w.role,
                    "z_min": w.z_min,
                    "z_max": w.z_max,
                    "points": np.round(w.points, 7).tolist(),
                    "hole": w.hole,
                }
                for w in self.walls
            ],
            "holes": [dict(h.__dict__) for h in self.holes],
            "faces": self.faces,
            "tolerance": self.tolerance,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PartModel":
        try:
            return cls(
                name=d.get("name", "part"),
                bbox_min=list(d["bbox_min"]),
                bbox_max=list(d["bbox_max"]),
                levels=[
                    LevelFace(int(x["face"]), float(x["z"]), as_geometry(x["polygon"]),
                              x.get("role", "floor"))
                    for x in d.get("levels", [])
                ],  # fmt: skip
                obstacles=[
                    Obstacle(int(x["face"]), x.get("role", "curved"), float(x["z_max"]),
                             float(x.get("z_min", x["z_max"])), as_geometry(x["polygon"]))
                    for x in d.get("obstacles", [])
                ],  # fmt: skip
                walls=[
                    Wall(int(x["face"]), x.get("role", "wall"), float(x["z_min"]),
                         float(x["z_max"]), np.asarray(x["points"], dtype=float).reshape(-1, 2),
                         x.get("hole"))
                    for x in d.get("walls", [])
                ],  # fmt: skip
                holes=[Hole(**x) for x in d.get("holes", [])],
                faces=list(d.get("faces", [])),
                tolerance=float(d.get("tolerance", 1e-5)),
                warnings=list(d.get("warnings", [])),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CamcoreError("INVALID_VALUE", f"Bad part model: {e}") from e
