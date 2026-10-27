#!/usr/bin/env python3
"""Body-fitted blockMesh for a car STL (no snappyHexMesh).

The STL location is the one chosen in the GUI (Geometry panel, persisted in
system/geometry.json); BLOCKMESH_STL overrides it.

OPTIMUM SMOOTH BLOCKING STRATEGY
--------------------------------
Cartesian cut-cell (omit-solid) stair-steps a curvy sedan -> "minecraft".
Instead:
  A. Terrain-following base: displace the whole background grid so y-planes
     drape over roofline roof(x) and z-planes neck with plan taper W(x),
     decaying to far-field (exp, L=1.5m). Shifts are 0 at ground/far-field,
     gradients <0.6 so no inversions. Omission then follows draped grid.
  B. SDF projection: surface vertices and wall-edge intermediates project to
     the sampled whole-car signed-distance zero surface.
  C. Volume relief: Laplacian smoothing of fluid interior (wall-fixed, 5 it),
     edge-collapse (>35% L0, 8mm floor) + block volume + cage guards.
This keeps every wall edge ON the surface while subdivided cells stay valid.

Run:  python3 make_car_blockMeshDict.py  (run from case dir)
      -> writes system/blockMeshDict
"""
import json
import struct
import sys
import os
import numpy as np

CASE = os.environ.get("BLOCKMESH_CASE", "/home/garcfd/working/blockMesh-car")


def geometry_stl(case):
    """STL selected in the GUI (system/geometry.json), else the old default."""
    try:
        with open(os.path.join(case, "system", "geometry.json")) as f:
            rel = json.load(f).get("stl")
        if isinstance(rel, str) and rel.strip():
            rel = rel.strip()
            return rel if os.path.isabs(rel) else os.path.join(case, rel)
    except (OSError, ValueError, AttributeError):
        pass
    return os.path.join(case, "constant", "geometry", "car_simplified.stl")


STL = os.environ.get("BLOCKMESH_STL") or geometry_stl(CASE)
OUT = os.path.join(CASE, "system", "blockMeshDict")
SDF_PATH = os.path.join(CASE, "system", "car_sdf.npy")
sys.path.insert(0, os.path.dirname(__file__))

from sdf import load_field
SDF = load_field(SDF_PATH)

def backup_dict():
    """Keep the previous blockMeshDict as blockMeshDict.bak before overwriting.
    Guards against a bad/empty export wiping the only copy (as happened once)."""
    try:
        if not os.path.exists(OUT):
            return
        with open(OUT) as _f:
            _txt = _f.read()
        # only back up something that actually looks like a mesh dict
        if "vertices" in _txt and "blocks" in _txt and _txt.count("hex (") > 1:
            with open(OUT + ".bak", "w") as _f:
                _f.write(_txt)
    except OSError:
        pass

# ---------------------------------------------------------------- STL load
print("loading STL %s ..." % STL, file=sys.stderr)
with open(STL, "rb") as f:
    f.read(80)
    ntri = struct.unpack("<I", f.read(4))[0]
    raw = f.read()
print(f"  tris: {ntri}", file=sys.stderr)
T0 = np.empty((ntri, 3)); T1 = np.empty((ntri, 3)); T2 = np.empty((ntri, 3))
for i in range(ntri):
    v = struct.unpack_from("<12f", raw, i * 50)
    T0[i] = v[3:6]; T1[i] = v[6:9]; T2[i] = v[9:12]
CENT = (T0 + T1 + T2) / 3.0

# ------------------------------------------------- analytic profiles (from STL)
# roof centreline (x, roofY)
ROOF = np.array([
    [-2.267, -0.118], [-2.10, 0.008], [-1.90, 0.090], [-1.70, 0.129],
    [-1.50, 0.143], [-1.30, 0.154], [-1.10, 0.164], [-0.90, 0.218],
    [-0.70, 0.334], [-0.50, 0.465], [-0.30, 0.558], [-0.10, 0.600],
    [0.10, 0.610], [0.30, 0.610], [0.50, 0.588], [0.70, 0.560],
    [0.90, 0.511], [1.10, 0.451], [1.30, 0.380], [1.50, 0.303],
    [1.70, 0.230], [1.90, 0.158], [2.10, 0.080], [2.267, 0.003],
])
# plan half-width at y~0 (x, halfW)
HALFW = np.array([
    [-2.267, 0.45], [-2.10, 0.90], [-1.90, 0.986], [-1.70, 1.012],
    [-1.50, 1.012], [-1.30, 1.012], [-1.10, 1.008], [-0.90, 0.996],
    [-0.70, 1.000], [-0.50, 1.000], [-0.30, 1.000], [-0.10, 1.000],
    [0.10, 1.000], [0.30, 0.999], [0.50, 0.990], [0.70, 0.988],
    [0.90, 1.012], [1.10, 1.012], [1.30, 1.012], [1.50, 0.996],
    [1.70, 0.972], [1.90, 0.908], [2.10, 0.732], [2.267, 0.44],
])
def _interp(table, x):
    if x <= table[0, 0]:
        return table[0, 1]
    if x >= table[-1, 0]:
        return table[-1, 1]
    i = np.searchsorted(table[:, 0], x) - 1
    x0, y0 = table[i]; x1, y1 = table[i + 1]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def roof_center(x):
    return _interp(ROOF, x)
def half_width(x):
    return _interp(HALFW, x)
BELT_Y = 0.17
def roof_yz(x, z):
    """roof height at (x,z): centre minus shoulder drop."""
    R = roof_center(x); W = half_width(x)
    if W <= 1e-9:
        return R
    az = min(abs(z) / W, 1.0)
    drop = max(0.0, R - BELT_Y) * (az ** 2.5)
    return R - drop

UNDER = -0.41
GROUND = -0.59
# wheels (tire boxes, from STL low-point clusters)
FX0, FX1 = -1.653, -1.171
RX0, RX1 = 0.879, 1.375
WZ_IN, WZ_OUT = 0.74, 1.012
WY_TOP = -0.10
WHEEL_R = 0.205
WHEEL_FCX = 0.5 * (FX0 + FX1)
WHEEL_RCX = 0.5 * (RX0 + RX1)
WHEEL_CY = GROUND + WHEEL_R

# ---------------------------------------------------------------- STL ray-cast
# Intelligent inside/outside test: even/odd parity of ray crossings against
# the ACTUAL car STL at each deformed block centre (majority vote +x/-x/+y),
# instead of the analytic lookup-table proxy (which over-filled shoulders by
# centimetres). STL is verified watertight (0 boundary edges), so parity is
# exact; deterministic transverse jitter dodges vertex-exact hits.
_TRI = {}


def _load_triangles():
    if "t" not in _TRI:
        import struct as _st
        with open(STL, "rb") as _f:
            _f.read(80)
            _n = _st.unpack("<I", _f.read(4))[0]
            _raw = _f.read()
        _T0 = np.empty((_n, 3))
        _T1 = np.empty((_n, 3))
        _T2 = np.empty((_n, 3))
        for _i in range(_n):
            _v = _st.unpack_from("<12f", _raw, _i * 50)
            _T0[_i] = _v[3:6]
            _T1[_i] = _v[6:9]
            _T2[_i] = _v[9:12]
        _lo = np.minimum(np.minimum(_T0, _T1), _T2)
        _hi = np.maximum(np.maximum(_T0, _T1), _T2)
        _TRI["t"] = (_T0, _T1, _T2, _lo, _hi)
        print(f"  STL for ray-cast: {_n} tris", file=sys.stderr)
    return _TRI["t"]


def _ray_hits_inside(pts, direction):
    _T0, _T1, _T2, _lo, _hi = _load_triangles()
    _d = int(np.argmax(np.abs(direction)))
    _sgn = 1.0 if direction[_d] >= 0 else -1.0
    _t1 = (_d + 1) % 3
    _t2 = (_d + 2) % 3
    _P = np.asarray(pts, float) + np.array([1e-7, 2e-7, 3e-7])[:3]
    _inside = np.zeros(len(_P), dtype=bool)
    _E1 = _T1 - _T0
    _E2 = _T2 - _T0
    _dv = np.array([1.0 if _d == 0 else 0.0, 1.0 if _d == 1 else 0.0,
                    1.0 if _d == 2 else 0.0])
    if _sgn < 0:
        _dv = -_dv
    for _n in range(len(_P)):
        _o = _P[_n]
        if _sgn > 0:
            _m = (_hi[:, _d] > _o[_d]) & (_lo[:, _t1] <= _o[_t1] + 1e-9) \
                & (_hi[:, _t1] >= _o[_t1] - 1e-9) & (_lo[:, _t2] <= _o[_t2] + 1e-9) \
                & (_hi[:, _t2] >= _o[_t2] - 1e-9)
        else:
            _m = (_lo[:, _d] < _o[_d]) & (_lo[:, _t1] <= _o[_t1] + 1e-9) \
                & (_hi[:, _t1] >= _o[_t1] - 1e-9) & (_lo[:, _t2] <= _o[_t2] + 1e-9) \
                & (_hi[:, _t2] >= _o[_t2] - 1e-9)
        _idx = np.nonzero(_m)[0]
        if len(_idx) == 0:
            continue
        _a = _T0[_idx]
        _e1 = _E1[_idx]
        _e2 = _E2[_idx]
        _pvec = np.cross(_dv, _e2)
        _det = np.einsum("ij,ij->i", _e1, _pvec)
        _ok = np.abs(_det) > 1e-12
        _inv = np.zeros_like(_det)
        _inv[_ok] = 1.0 / _det[_ok]
        _tvec = _o - _a
        _u = np.einsum("ij,ij->i", _tvec, _pvec) * _inv
        _ok &= (_u >= -1e-9) & (_u <= 1.0 + 1e-9)
        _qvec = np.cross(_tvec, _e1)
        _v = np.einsum("ij,ij->i", _qvec, np.broadcast_to(_dv, _qvec.shape)) * _inv
        _ok &= (_v >= -1e-9) & (_u + _v <= 1.0 + 1e-9)
        _t = np.einsum("ij,ij->i", _e2, _qvec) * _inv
        _ok &= _t > 1e-7
        _inside[_n] = (int(np.count_nonzero(_ok)) % 2) == 1
    return _inside


def _classify_raycast(centres):
    _C = np.asarray(centres, float)
    _v0 = _ray_hits_inside(_C, (1.0, 0.0, 0.0))
    _v1 = _ray_hits_inside(_C, (-1.0, 0.0, 0.0))
    _solid = _v0 & _v1
    _tied = _v0 ^ _v1
    _nv = int(_tied.sum())
    if _nv:
        _v2 = _ray_hits_inside(_C[_tied], (0.0, 1.0, 0.0))
        _s = _solid.copy()
        _s[_tied] = (_v0[_tied].astype(int) + _v1[_tied].astype(int)
                     + _v2.astype(int)) >= 2
        print(f"  ray-cast votes: {len(_C)-_nv} unanimous, {_nv} majority-voted",
              file=sys.stderr)
        return _s
    print(f"  ray-cast votes: {len(_C)} unanimous, 0 voted", file=sys.stderr)
    return _solid


def is_solid(cx, cy, cz):
    # Legacy analytic proxy (kept for reference; classification now uses
    # _classify_raycast against the STL — see solid fill below).
    az = abs(cz)
    # True tyre boxes (0.7 planes sit just inboard of tyre inner 0.74):
    # centre interval (-0.7..0.7) stays fluid underbody gap, outboard
    # (-1.012..-0.7 / 0.7..1.012, centre |z|=0.856) captures tyres.
    if ((FX0 < cx < FX1) or (RX0 < cx < RX1)) and (WZ_IN < az < WZ_OUT) and (GROUND < cy < WY_TOP):
        return True
    if cx < -2.267 or cx > 2.267:
        return False
    if cy < UNDER or cy > 0.65:
        return False
    W = half_width(cx)
    if az > W:
        return False
    if cy > roof_yz(cx, cz):
        return False
    return True

# ---------------------------------------------------------------- grid lines
# OPTIMUM SMOOTH BLOCKING: terrain-following (see header). Cartesian base
# (xs,ys,zs) is deformed FIRST by roof/width taper so grid planes hug the
# car (no stair-steps), THEN SDF-projected for fine detail. Deformation
# gradients are <0.6 so cells stay positive; guards below verify.
# (coord, nCells to reach it from prev, grading last/first)
x_lines = [
    (-8.0, 0, 1),
    (-2.267, 20, 0.15),
    (-2.10, 3, 1),
    (-1.90, 3, 1),
    (-1.65, 3, 1),
    (-1.41, 4, 1),
    (-1.17, 4, 1),
    (-0.90, 3, 1),
    (-0.70, 3, 1),
    (-0.50, 3, 1),
    (-0.30, 3, 1),
    (-0.10, 3, 1),
    (0.10, 3, 1),
    (0.30, 3, 1),
    (0.50, 3, 1),
    (0.70, 3, 1),
    (0.88, 3, 1),
    (1.125, 4, 1),
    (1.37, 4, 1),
    (1.50, 2, 1),
    (1.70, 3, 1),
    (1.90, 3, 1),
    (2.10, 3, 1),
    (2.267, 3, 1),
    (12.0, 30, 8),
]
y_lines = [
    (-0.59, 0, 1),
    (-0.41, 6, 1),
    (-0.15, 6, 1),
    (0.05, 6, 1),
    (0.17, 4, 1),
    (0.30, 4, 1),
    (0.45, 4, 1),
    (0.55, 3, 1),
    (0.61, 2, 1),
    (4.0, 20, 8),
]
z_lines = [
    (-4.0, 0, 1),
    (-1.012, 18, 0.15),
    (-0.70, 4, 1),
    (0.70, 12, 1),
    (1.012, 4, 1),
    (4.0, 18, 6.0),
]
# ------------------------------------------------- GUI override (splits.json)
# The blocksplit GUI writes system/splits.json with manual split positions:
#   {"x": [...], "y": [...], "z": [...]}  (coords only -> auto cells/grading)
# or {"x_lines": [[c,n,g],...], "y_lines": ..., "z_lines": ...} (full control).
# Auto rule: L>1.5 far-field (18-30 cells, graded outward/inward); L<=0.5
# near-car (GX cells/m, uniform); else mid. Keeps Mesh-OK guards valid.
def _auto_lines(coords, axis):
    coords = sorted(set(float(c) for c in coords))
    out = [(coords[0], 0, 1)]
    for prev, cur in zip(coords[:-1], coords[1:]):
        L = cur - prev
        if L <= 1e-9:
            continue
        last = (cur == coords[-1])
        first = (prev == coords[0])
        if L > 1.5:
            if first:
                n, g = 18, 0.15
            elif last:
                n, g = (30, 8.0) if (axis == "x" and L > 5) else (20, 6.0)
                if axis == "z" and last:
                    n, g = 18, 6.0
            else:
                n, g = max(6, int(round(L / 0.15))), 1
        elif L <= 0.5:
            n, g = max(2, int(round(L / 0.08))), 1
        else:
            n, g = max(3, int(round(L / 0.12))), 1
        out.append((cur, n, g))
    return out
# GUI solid/fluid overrides: explicit per-block inside/outside classification
# from splits.json {"solid": [[i,j,k],...], "fluid": [[i,j,k],...]} indices
# into the (nx,ny,nz) block grid. Applied AFTER the analytic test, so manual
# selection always wins. Out-of-range entries are ignored with a warning.
_OVERRIDES = {"solid": [], "fluid": []}
_PINS = {}
# Fit-first mode: splits.json {"snap_all": true} snaps EVERY near-car surface
# vertex to its closest STL point regardless of distance, and skips all
# revert guards + edge auto-drop. Fit wins, checkMesh may fail.
_SNAP_ALL = False
# Flat mode: splits.json {"flat": true} disables terrain-following (planes stay
# flat so every block in a row shares ymin/ymax etc). Snapping still applies.
_FLAT = False
try:
    import json as _json
    _sp = os.path.join(CASE, "system", "splits.json")
    _d = None
    if os.path.exists(_sp):
        _d = _json.load(open(_sp))
        # a stripped/clobbered splits.json has no interior splits — treat as missing
        if not any(len(_d.get(ax, [])) > 2 for ax in ("x", "y", "z")):
            _d = None
    if _d is not None:
        if "x_lines" in _d:
            x_lines = [(float(c), int(n), float(g)) for c, n, g in _d["x_lines"]]
        elif "x" in _d:
            x_lines = _auto_lines(_d["x"], "x")
        if "y_lines" in _d:
            y_lines = [(float(c), int(n), float(g)) for c, n, g in _d["y_lines"]]
        elif "y" in _d:
            y_lines = _auto_lines(_d["y"], "y")
        if "z_lines" in _d:
            z_lines = [(float(c), int(n), float(g)) for c, n, g in _d["z_lines"]]
        elif "z" in _d:
            z_lines = _auto_lines(_d["z"], "z")
        for _k in ("solid", "fluid"):
            _v = _d.get(_k, [])
            if isinstance(_v, list):
                _OVERRIDES[_k] = [tuple(int(a) for a in t) for t in _v
                                  if isinstance(t, (list, tuple)) and len(t) == 3]
        print(f"using splits.json override: {len(x_lines)-1}x{len(y_lines)-1}x{len(z_lines)-1} intervals", file=sys.stderr)
        _SNAP_ALL = bool(_d.get("snap_all", False))
        if _SNAP_ALL:
            print("SNAP_ALL: fit-first, guards off", file=sys.stderr)
        _FLAT = bool(_d.get("flat", False))
        if _FLAT:
            print("FLAT: Cartesian planes, no drape", file=sys.stderr)
        _pv = _d.get("pins", {})
        if isinstance(_pv, dict):
            # pins keyed "x|y|z" of base grid planes (survive splits edits);
            # resolved to vertex ids at apply time. Legacy int vids accepted.
            for _k, _p in _pv.items():
                try:
                    _pp = [float(c) for c in _p]
                    if len(_pp) == 3 and all(np.isfinite(_pp)):
                        _PINS[str(_k)] = _pp
                except (TypeError, ValueError):
                    pass
            if _PINS:
                print(f"pins: {len(_PINS)} hand-placed vertexes", file=sys.stderr)
except Exception as _e:
    print(f"splits.json ignored ({_e}), using defaults", file=sys.stderr)
xs = [p[0] for p in x_lines]; ys = [p[0] for p in y_lines]; zs = [p[0] for p in z_lines]
nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
print(f"grid intervals: {nx} x {ny} x {nz} = {nx*ny*nz} blocks (pre-cut)", file=sys.stderr)

def vid(i, j, k):
    return i + (nx + 1) * (j + (ny + 1) * k)

# Cartesian vertices
V = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3))
for k in range(nz + 1):
    for j in range(ny + 1):
        for i in range(nx + 1):
            V[vid(i, j, k)] = (xs[i], ys[j], zs[k])

# ------------------------------------------------- terrain-following deform
# OPTIMUM SMOOTH STRATEGY (replaces stair-step cut-cell): displace the whole
# background grid so constant-j (y) planes drape over the roofline and
# constant-k (z) planes neck in with plan taper -- like terrain-following
# coordinates. All shifts -> 0 far from the car, gradients < 0.6 so no
# inversions. Solid omission below then follows the draped grid (smooth
# interface), and the later STL snap only needs millimetres.
RREF, WREF = 0.61, 1.012
# Smoothed half-width table FOR THE DEFORM ONLY (surface definition keeps raw
# half_width). Raw W(x) drops 0.732->0.44 over 0.167 m at the tail tip
# (|dW/dx|=1.75>1), which folds wake blocks. Low-pass (sigma 0.25 m) plus a
# hard slope limit |dW/dx|<=0.5 keeps deform gradients <1; STL snapping still
# resolves the true tip shape on top.
def _build_smooth_W():
    _xs = np.arange(-3.0, 3.001, 0.05)
    _raw = np.array([half_width(_x) if -2.267 <= _x <= 2.267 else WREF
                     for _x in _xs])
    _g = np.arange(-3 * 5, 3 * 5 + 1) * 0.05
    _ker = np.exp(-0.5 * (_g / 0.25) ** 2)
    _ker /= _ker.sum()
    _s = np.convolve(np.pad(_raw, len(_ker), mode="edge"), _ker, mode="valid")
    _s = _s[len(_ker) // 2:len(_ker) // 2 + len(_xs)]
    _dx = 0.05
    for _ in range(2):
        for _i in range(1, len(_s)):
            _s[_i] = min(max(_s[_i], _s[_i - 1] - 0.5 * _dx),
                         _s[_i - 1] + 0.5 * _dx)
        for _i in range(len(_s) - 2, -1, -1):
            _s[_i] = min(max(_s[_i], _s[_i + 1] - 0.5 * _dx),
                         _s[_i + 1] + 0.5 * _dx)
    return _xs, _s


_SW_XS, _SW_W = _build_smooth_W()


def _smooth_W(x):
    if x <= _SW_XS[0]:
        return float(_SW_W[0])
    if x >= _SW_XS[-1]:
        return float(_SW_W[-1])
    _i = int(np.searchsorted(_SW_XS, x)) - 1
    _x0, _y0 = _SW_XS[_i], _SW_W[_i]
    _x1, _y1 = _SW_XS[_i + 1], _SW_W[_i + 1]
    return float(_y0 + (_y1 - _y0) * (x - _x0) / (_x1 - _x0))
# Shoulder reference (x=0 section): (yn, az_norm) yn=(y-BELT)/(Rx-BELT), az=|z|/Wx.
# Measured: belt (0.17,1.0) -> (0.386,0.8) -> (0.848,0.6) -> (0.973,0.4) -> (1.0,0.2)
_SH_YN = np.array([0.0, 0.386, 0.848, 0.973, 1.0])
_SH_AZ = np.array([1.0, 0.8, 0.6, 0.4, 0.2])
def _shoulder_az(yn):
    yn = min(max(yn, 0.0), 1.0)
    return float(np.interp(yn, _SH_YN, _SH_AZ))
def _deform(x, y, z):
    if _FLAT:
        return y, z
    if -2.267 <= x <= 2.267:
        Rx = roof_center(x)
    else:
        Rx = RREF
    Wx = _smooth_W(x)
    # Local roof at this (x,z): shoulder drops to belt at sides (kills overhang)
    Rz = roof_yz(x, z) if -2.267 <= x <= 2.267 else RREF
    if y <= UNDER:
        dy = 0.0
    elif y <= RREF:
        dy = (Rz - RREF) * ((y - UNDER) / (RREF - UNDER))
    else:
        dy = (Rz - RREF) * float(np.exp(-(y - RREF) / 1.5))
    y2 = y + dy
    az = abs(z)
    s = 1.0 if z >= 0 else -1.0
    if az <= WREF:
        dz = s * (Wx - WREF) * (az / WREF)
        z2 = z + dz
    else:
        dz = s * (Wx - WREF) * float(np.exp(-(az - WREF) / 1.5))
        z2 = z + dz
    # Shoulder drape: pull z-planes inward with height so they follow the
    # shoulder instead of cutting it (removes flat z-cuts on the sides).
    # BLEND 0.6 (not 1.0): full pull folds 8 blocks (shear gradient 1.04>1);
    # 60% halves the stair-step (13cm->5cm) with gradient 0.62<1, safe.
    # Only inside car x-range, belt..roof heights, near-car |z|<=Wx.
    if -2.267 <= x <= 2.267 and BELT_Y < y2 < Rx and Rx > BELT_Y + 0.05 and abs(z2) <= Wx:
        yn = (y2 - BELT_Y) / (Rx - BELT_Y)
        az_ref = _shoulder_az(yn)          # 1.0 at belt -> 0.2 at roof
        Ws = az_ref * Wx                   # surface width at this height
        # scale z toward surface width (centre stays, side maps to Ws)
        if Wx > 1e-9:
            z_scaled = s * abs(z2) * (Ws / Wx)
            z2 = z2 + 0.6 * (z_scaled - z2)
    return y2, z2
for v in range(len(V)):
    x, y, z = V[v]
    y2, z2 = _deform(x, y, z)
    V[v] = (x, y2, z2)
print("terrain-following deform done", file=sys.stderr)

# ---------------------------------------------------------------- blocks
# NOTE: solid test on DEFORMED centres (grid already drapes the car)
solid = np.zeros((nx, ny, nz), dtype=bool)
_centres = []
_keys = []
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            # deformed block centre ~ deform(Cartesian centre)
            cx = 0.5 * (xs[i] + xs[i + 1]); cy = 0.5 * (ys[j] + ys[j + 1]); cz = 0.5 * (zs[k] + zs[k + 1])
            cy2, cz2 = _deform(cx, cy, cz)
            _centres.append([cx, cy2, cz2])
            _keys.append((i, j, k))
print("classifying blocks against STL (ray-cast)...", file=sys.stderr)
_arr = _classify_raycast(_centres)
for (_i, _j, _k), _s in zip(_keys, _arr):
    solid[_i, _j, _k] = bool(_s)
# Manual inside/outside selection wins over the analytic test.
_n_ov = 0
for (_i, _j, _k) in _OVERRIDES["solid"]:
    if 0 <= _i < nx and 0 <= _j < ny and 0 <= _k < nz:
        if not solid[_i, _j, _k]:
            solid[_i, _j, _k] = True
            _n_ov += 1
    else:
        print(f"splits.json solid override out of range: {(_i,_j,_k)}",
              file=sys.stderr)
for (_i, _j, _k) in _OVERRIDES["fluid"]:
    if 0 <= _i < nx and 0 <= _j < ny and 0 <= _k < nz:
        if solid[_i, _j, _k]:
            solid[_i, _j, _k] = False
            _n_ov += 1
    else:
        print(f"splits.json fluid override out of range: {(_i,_j,_k)}",
              file=sys.stderr)
if _n_ov:
    print(f"applied {_n_ov} manual solid/fluid overrides", file=sys.stderr)

FACES = [
    ((-1, 0, 0), (0, 4, 7, 3)),
    ((+1, 0, 0), (1, 2, 6, 5)),
    ((0, -1, 0), (0, 1, 5, 4)),
    ((0, +1, 0), (3, 7, 6, 2)),
    ((0, 0, -1), (0, 3, 2, 1)),
    ((0, 0, +1), (4, 5, 6, 7)),
]
def domain_patch(d):
    return {(-1, 0, 0): "inlet", (+1, 0, 0): "outlet", (0, -1, 0): "sides",
            (0, +1, 0): "sides", (0, 0, -1): "ground", (0, 0, +1): "top"}[d]

def solid_or_none(i, j, k):
    if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
        return solid[i, j, k]
    return None

blocks = []  # (hv, nc, g)
block_ijk = {}
patches = {"inlet": [], "outlet": [], "sides": [], "top": [], "ground": [], "body": []}
vert_out = {}  # vertex -> summed fluid->solid neighbour dirs (points inward)
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            if solid[i, j, k]:
                continue
            hv = [vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i, j + 1, k),
                  vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)]
            nc = (x_lines[i + 1][1], y_lines[j + 1][1], z_lines[k + 1][1])
            g = (x_lines[i + 1][2], y_lines[j + 1][2], z_lines[k + 1][2])
            block_ijk[len(blocks)] = (i, j, k)
            blocks.append((hv, nc, g))
            for d, loc in FACES:
                nb = solid_or_none(i + d[0], j + d[1], k + d[2])
                face = tuple(hv[n] for n in loc)
                if nb is None:
                    patches[domain_patch(d)].append(face)
                elif nb:
                    for _vv in face:
                        _o = vert_out.get(_vv)
                        vert_out[_vv] = [(_o[0] if _o else 0) + d[0],
                                         (_o[1] if _o else 0) + d[1],
                                         (_o[2] if _o else 0) + d[2]]
                    patches["body"].append(face)
nfluid = len(blocks)
print(f"fluid blocks: {nfluid}, body faces: {len(patches['body'])}", file=sys.stderr)

# ------------------------------------------------- STL closest-point query
# candidate prefilter via centroids, exact point-triangle on top candidates
try:
    from scipy.spatial import cKDTree
    tree = cKDTree(CENT)
    HAVE_KD = True
    print("using scipy cKDTree", file=sys.stderr)
except Exception as e:
    tree = None
    HAVE_KD = False
    print(f"no scipy ({e}), using brute-force centroids", file=sys.stderr)

def point_tri_closest(p, a, b, c):
    """closest point on triangle abc to p (single). returns (q, dist2)."""
    ab = b - a; ac = c - a; ap = p - a
    d1 = np.dot(ab, ap); d2 = np.dot(ac, ap)
    if d1 <= 0 and d2 <= 0:
        return a, np.dot(ap, ap)
    bp = p - b
    d3 = np.dot(ab, bp); d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return b, np.dot(bp, bp)
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3)
        q = a + v * ab
        return q, np.dot(p - q, p - q)
    cp = p - c
    d5 = np.dot(ab, cp); d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return c, np.dot(cp, cp)
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6)
        q = a + w * ac
        return q, np.dot(p - q, p - q)
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        q = b + w * (c - b)
        return q, np.dot(p - q, p - q)
    denom = 1.0 / (va + vb + vc)
    v = vb * denom; w = vc * denom
    q = a + ab * v + ac * w
    return q, np.dot(p - q, p - q)

def closest_on_stl_for_audit(p, ncand=120):
    if HAVE_KD:
        _, idx = tree.query(p, k=ncand)
    else:
        d2 = np.sum((CENT - p) ** 2, axis=1)
        idx = np.argpartition(d2, ncand)[:ncand]
    best_q = None; best_d2 = 1e100
    for t in np.atleast_1d(idx):
        q, d2 = point_tri_closest(p, T0[t], T1[t], T2[t])
        if d2 < best_d2:
            best_d2 = d2; best_q = q
    return best_q, np.sqrt(best_d2)

# ------------------------------------------------- vertex snapping w/ guards
snapped = V.copy()
snap_flag = np.zeros((len(V), 2), dtype=bool)  # [:,0]=y-snapped, [:,1]=z-snapped
is_domain_v = np.zeros(len(V), dtype=bool)
for k in range(nz + 1):
    for j in range(ny + 1):
        for i in range(nx + 1):
            if i == 0 or i == nx or j == 0 or j == ny or k == 0 or k == nz:
                is_domain_v[vid(i, j, k)] = True

# surface vertices = incident to a wall face
surf_verts = set()
for f in patches["body"]:
    surf_verts.update(f)
print(f"surface vertices: {len(surf_verts)}", file=sys.stderr)

def local_cell(i, j, k):
    # DEFORMED spacing (terrain-following compresses near car) -- use actual V
    v = vid(i, j, k)
    ds = []
    for ii, jj, kk in ((i-1,j,k),(i+1,j,k),(i,j-1,k),(i,j+1,k),(i,j,k-1),(i,j,k+1)):
        if 0 <= ii <= nx and 0 <= jj <= ny and 0 <= kk <= nz:
            ds.append(float(np.linalg.norm(V[vid(ii,jj,kk)] - V[v])))
    return min(ds) if ds else 0.05

n_snapped_sdf = 0; n_kept = 0
# reverse map vertex -> (i,j,k)
vijk = {}
for k in range(nz + 1):
    for j in range(ny + 1):
        for i in range(nx + 1):
            vijk[vid(i, j, k)] = (i, j, k)

for v in sorted(surf_verts):
    i, j, k = vijk[v]
    if is_domain_v[v]:
        n_kept += 1
        continue
    x, y, z = V[v]
    if not (-2.6 < x < 2.6 and -0.65 < y < 0.85 and abs(z) < 1.4):
        n_kept += 1
        continue
    tol = 0.5 * local_cell(i, j, k)
    tol = min(tol, 0.12)
    # neighbour cage with 15% margin on DEFORMED neighbours (no plane crossing)
    def _nbr(ii, jj, kk):
        return V[vid(ii, jj, kk)]
    xlo = float(min(_nbr(i-1,j,k)[0], x) + 0.15*abs(x-_nbr(i-1,j,k)[0])) if i > 0 else x - tol
    xhi = float(max(_nbr(i+1,j,k)[0], x) - 0.15*abs(_nbr(i+1,j,k)[0]-x)) if i < nx else x + tol
    ylo = float(min(_nbr(i,j-1,k)[1], y) + 0.15*abs(y-_nbr(i,j-1,k)[1])) if j > 0 else y - tol
    yhi = float(max(_nbr(i,j+1,k)[1], y) - 0.15*abs(_nbr(i,j+1,k)[1]-y)) if j < ny else y + tol
    zlo = float(min(_nbr(i,j,k-1)[2], z) + 0.15*abs(z-_nbr(i,j,k-1)[2])) if k > 0 else z - tol
    zhi = float(max(_nbr(i,j,k+1)[2], z) - 0.15*abs(_nbr(i,j,k+1)[2]-z)) if k < nz else z + tol
    q = SDF.project(V[v])
    d = abs(float(SDF.sample_gradient(q)[0][0]))
    moved = False
    if _SNAP_ALL or (d <= tol and (xlo < q[0] < xhi) and (ylo < q[1] < yhi) and (zlo < q[2] < zhi)):
        snapped[v] = q
        n_snapped_sdf += 1
        moved = True
        # record snap direction for edge building
        if abs(q[1] - y) > 0.25 * abs(q[2] - z) + 1e-9:
            snap_flag[v, 0] = True
        if abs(q[2] - z) > 0.25 * abs(q[1] - y) + 1e-9:
            snap_flag[v, 1] = True
        if abs(q[1] - y) < 1e-6 and abs(q[2] - z) < 1e-6:
            # pure x snap (nose/tail) - mark none, edge stays straight
            pass
    else:
        n_kept += 1
print(f"snapped to whole-car SDF: {n_snapped_sdf}, kept: {n_kept}", file=sys.stderr)

# ------------------------------------------------- edge collapse guard
# A vertex snap must not collapse any incident block edge: after snapping,
# every fluid-block edge must retain >35% of its original Cartesian length
# and stay above an absolute 8 mm floor. Otherwise the subdivided cells
# fold (zero-area faces, aspect 1e101, negative volumes in checkMesh).
# Revert the larger-displacement endpoint of offending edges.
def edge_list_of_block(b):
    return [(b[0], b[1]), (b[1], b[2]), (b[2], b[3]), (b[3], b[0]),
            (b[4], b[5]), (b[5], b[6]), (b[6], b[7]), (b[7], b[4]),
            (b[0], b[4]), (b[1], b[5]), (b[2], b[6]), (b[3], b[7])]
collapsed = 0
for _pass in range(0 if _SNAP_ALL else 3):
    collapsed = 0
    for b, _nc, _g in blocks:
        for a, c in edge_list_of_block(b):
            l0 = float(np.linalg.norm(V[a] - V[c]))
            l = float(np.linalg.norm(snapped[a] - snapped[c]))
            if l0 < 1e-12:
                continue
            if l < 0.35 * l0 or l < 0.012:
                da = float(np.linalg.norm(snapped[a] - V[a]))
                dc = float(np.linalg.norm(snapped[c] - V[c]))
                v = a if da >= dc else c
                if np.linalg.norm(snapped[v] - V[v]) > 1e-12:
                    snapped[v] = V[v]
                    snap_flag[v] = False
                    collapsed += 1
    if collapsed == 0:
        break
print(f"edge-collapse reverts: {collapsed} (last pass)", file=sys.stderr)

# ------------------------------------------------- interior smoothing
# Full-projection needs volume relief: wall vertices sit ON the STL while
# neighbours stay Cartesian -> wedge cells fold when subdivided. Diffuse wall
# displacement into the fluid interior (Laplacian, wall-fixed) so curvature is
# shared over 2-3 cell layers instead of crushing the first cell.
USED = set(v for b, _nc, _g in blocks for v in b)
INTERIOR = [v for v in USED if v not in surf_verts and not bool(is_domain_v[v])]
print(f"smoothing interior verts: {len(INTERIOR)} (used {len(USED)})", file=sys.stderr)


def _local_spacing(v):
    i, j, k = vijk[v]
    _ds = []
    for ii, jj, kk in ((i-1,j,k),(i+1,j,k),(i,j-1,k),(i,j+1,k),(i,j,k-1),(i,j,k+1)):
        if 0 <= ii <= nx and 0 <= jj <= ny and 0 <= kk <= nz:
            _ds.append(float(np.linalg.norm(V[vid(ii,jj,kk)] - V[v])))
    return min(_ds) if _ds else 0.05


for _it in range(0 if _FLAT else 5):
    newpos = {}
    for v in INTERIOR:
        i, j, k = vijk[v]
        _num = np.zeros(3)
        _den = 0.0
        for ii, jj, kk in ((i-1,j,k),(i+1,j,k),(i,j-1,k),(i,j+1,k),(i,j,k-1),(i,j,k+1)):
            if 0 <= ii <= nx and 0 <= jj <= ny and 0 <= kk <= nz:
                u = vid(ii, jj, kk)
                if u in USED:
                    # inverse-distance weight: far-field neighbours (metres
                    # away across graded outlet/inlet/top/side intervals) must
                    # not drag this vertex after them (that flung a tail vertex
                    # 1.39 m and folded 86 cells).
                    _w = 1.0 / max(float(np.linalg.norm(V[u] - V[v])) ** 2, 1e-12)
                    _num += _w * snapped[u]
                    _den += _w
        if _den <= 0:
            continue
        newpos[v] = 0.5 * snapped[v] + 0.5 * _num / _den
    for v, p in newpos.items():
        # cage on DEFORMED neighbours (5% margin) AND absolute cap at half
        # the local deformed spacing (same budget as vertex snapping)
        if float(np.linalg.norm(np.asarray(p, float) - V[v])) > 0.5 * _local_spacing(v):
            continue
        i, j, k = vijk[v]
        def _q(ii, jj, kk):
            return V[vid(ii, jj, kk)]
        xlo = float(min(_q(i-1,j,k)[0], V[v][0]) + 0.05*abs(V[v][0]-_q(i-1,j,k)[0])) if i > 0 else p[0]-1
        xhi = float(max(_q(i+1,j,k)[0], V[v][0]) - 0.05*abs(_q(i+1,j,k)[0]-V[v][0])) if i < nx else p[0]+1
        ylo = float(min(_q(i,j-1,k)[1], V[v][1]) + 0.05*abs(V[v][1]-_q(i,j-1,k)[1])) if j > 0 else p[1]-1
        yhi = float(max(_q(i,j+1,k)[1], V[v][1]) - 0.05*abs(_q(i,j+1,k)[1]-V[v][1])) if j < ny else p[1]+1
        zlo = float(min(_q(i,j,k-1)[2], V[v][2]) + 0.05*abs(V[v][2]-_q(i,j,k-1)[2])) if k > 0 else p[2]-1
        zhi = float(max(_q(i,j,k+1)[2], V[v][2]) - 0.05*abs(_q(i,j,k+1)[2]-V[v][2])) if k < nz else p[2]+1
        if xlo < p[0] < xhi and ylo < p[1] < yhi and zlo < p[2] < zhi:
            snapped[v] = p
print("smoothing done", file=sys.stderr)

# ------------------------------------------------- exact z-symmetry
# The STL is ~4mm wider on +z; closest-point snapping copies that into the
# mesh. Aero meshes must be exactly symmetric, so mirror the +z side onto -z
# (keeps the conservative wider side, matches the symmetric HALFW tables).
# No z=0 plane exists, so every vertex pairs with a distinct mirror.
def _mirror(v):
    i, j, k = vijk[v]
    return vid(i, j, nz - k)
for k in range(nz + 1):
    kk = nz - k
    if kk <= k:
        continue
    for j in range(ny + 1):
        for i in range(nx + 1):
            a, b = vid(i, j, k), vid(i, j, kk)
            # b is the +z side (V is sign-preserving deformed base)
            if V[b][2] < V[a][2]:
                a, b = b, a
            pb = np.asarray(snapped[b], dtype=float)
            x, y, az = float(pb[0]), float(pb[1]), abs(float(pb[2]))
            snapped[a] = (x, y, -az)
            snapped[b] = (x, y, az)
print("symmetry enforced", file=sys.stderr)

# ------------------------------------------------- frozen vertices
# Surgical reverts for mesh-level skew faces traced back to dict blocks
# (system/freeze.json {"vertices": [...]}): pin them to the deformed base so
# neither snapping, smoothing, edges nor symmetry can warp them. Used only
# when checkMesh -writeSets localises an issue to specific vertices.
_FZ_N = 0
try:
    _fzpath = os.path.join(CASE, "system", "freeze.json")
    if os.path.exists(_fzpath):
        for _v in _json.load(open(_fzpath)).get("vertices", []):
            _v = int(_v)
            if 0 <= _v < len(snapped):
                if float(np.linalg.norm(snapped[_v] - V[_v])) > 1e-9:
                    _FZ_N += 1
                snapped[_v] = V[_v]
                snap_flag[_v] = False
    if _FZ_N or os.path.exists(_fzpath):
        print(f"froze {len(_json.load(open(_fzpath)).get('vertices', []))} vertices "
              f"({_FZ_N} had moved) from freeze.json", file=sys.stderr)
except Exception as _e:
    print(f"freeze.json ignored ({_e})", file=sys.stderr)

# ------------------------------------------------- hand-placed pins
# User-dragged vertexes (splits.json {"pins": {"x|y|z": [x,y,z]}}): grid coords
# resolve to vertex ids here, so splits edits elsewhere can't orphan them.
# Applied last so they survive every Write. Fatal guards below may still
# revert a pin that folds its block (a mesh that builds beats a pin).
_PIN_N = 0
presnap = snapped.copy()
_pin_rev = set()
_resolved = {}
_dead_v = set()
for _dk in range(nz):
    for _dj in range(ny):
        for _di in range(nx):
            if solid[_di, _dj, _dk]:
                _dead_v.update([vid(_di, _dj, _dk), vid(_di + 1, _dj, _dk),
                                vid(_di + 1, _dj + 1, _dk), vid(_di, _dj + 1, _dk),
                                vid(_di, _dj, _dk + 1), vid(_di + 1, _dj, _dk + 1),
                                vid(_di + 1, _dj + 1, _dk + 1), vid(_di, _dj + 1, _dk + 1)])
for _k, _pp in _PINS.items():
    try:
        _parts = str(_k).split("|")
        if len(_parts) == 3:
            _cx, _cy, _cz = (float(c) for c in _parts)
            _ii = next(i for i, v in enumerate(xs) if abs(v - _cx) < 1e-6)
            _jj = next(j for j, v in enumerate(ys) if abs(v - _cy) < 1e-6)
            _kk = next(k for k, v in enumerate(zs) if abs(v - _cz) < 1e-6)
            _res = _ii + (nx + 1) * (_jj + (ny + 1) * _kk)
        else:
            _res = int(_k)  # legacy plain vertex id
    except (ValueError, StopIteration):
        continue
    if 0 <= _res < len(snapped):
        # Reject a boundary pin only if it lands on NO dead block. The blockMesh
        # outer walls over live (fluid) blocks must stay flat -- a stray pin
        # there (e.g. a far-field ground vertex) folds its column -> negative
        # cells. But the car touches the ground boundary (wheel base), and those
        # boundary vertices may be shared with live blocks, so a boundary vertex
        # that is on any dead block is allowed.
        if is_domain_v[_res] and _res not in _dead_v:
            continue
        _resolved[_res] = _pp
_PINS = _resolved
_PIN_N = 0
if _PINS:
    print(f"pins: {len(_PINS)} hand-placed vertexes queued", file=sys.stderr)

# ------------------------------------------------- face-outwardness guard
# Same check blockMesh itself applies (blockDescriptor::check): every block
# face normal must point outward. Centre-fan volume can stay positive while a
# single twisted face winds inward (typical when two snapped vertexes cross a
# face diagonal) — blockMesh fatals on those, so catch them here by reverting
# the largest-displacement vertexes of offending blocks.
_OUT_FACES = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
              (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]


def _newell(quad):
    n = np.zeros(3)
    m = len(quad)
    for s in range(m):
        a, b = quad[s], quad[(s + 1) % m]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    return n


def _block_faces_ok(hv):
    pts = snapped[np.array(hv)]
    bc = pts.mean(axis=0)
    for f in _OUT_FACES:
        q = pts[np.array(f)]
        if float(np.dot(_newell(q), q.mean(axis=0) - bc)) <= 1e-12:
            return False
    return True


_n_face_bad = sum(1 for b, _nc, _g in blocks if not _block_faces_ok(b))
print(f"blocks failing face-outwardness: {_n_face_bad}/{len(blocks)}",
      file=sys.stderr)
if _n_face_bad:
    _disp = np.linalg.norm(snapped - V, axis=1)
    _order = np.argsort(-_disp)
    _rev = set()
    for _v in _order:
        if _disp[_v] < 1e-9 or _v in _rev:
            continue
        if any(((_v in b) and not _block_faces_ok(b)) for b, _nc, _g in blocks):
            for _w in (_v, _mirror(_v)):
                snapped[_w] = presnap[_w] if _w in _PINS else V[_w]
                snap_flag[_w] = False
                if _w in _PINS:
                    _pin_rev.add(_w)
                _rev.add(_w)
    _left = sum(1 for b, _nc, _g in blocks if not _block_faces_ok(b))
    print(f"reverted {len(_rev)} snaps, face-bad remaining: {_left}",
          file=sys.stderr)
    for _b, _nc, _g in blocks:
        if not _block_faces_ok(_b):
            _pts = snapped[np.array(_b)]
            _bc = _pts.mean(axis=0)
            _worst = min(float(np.dot(_newell(_pts[np.array(f)]),
                                      _pts[np.array(f)].mean(axis=0) - _bc))
                         for f in _OUT_FACES)
            _ijk = vijk.get(_b[0], "?")
            print(f"  TWISTED ijk={_ijk} worstFaceDot={_worst:.3g} "
                  f"centre=({float(_bc[0]):.3f},{float(_bc[1]):.3f},{float(_bc[2]):.3f})",
                  file=sys.stderr)
            for _vv in _b:
                _dsp = np.linalg.norm(snapped[_vv] - V[_vv])
                print(f"    v={_vv} base=({V[_vv][0]:.4f},{V[_vv][1]:.4f},{V[_vv][2]:.4f}) "
                      f"now=({snapped[_vv][0]:.4f},{snapped[_vv][1]:.4f},{snapped[_vv][2]:.4f}) "
                      f"disp={_dsp:.4f}", file=sys.stderr)
# Edges on still-twisted blocks must stay straight: a curved subdivided edge
# on a warped face is what folds cells even when block vertexes pass. Collect
# their edge keys; the auto-drop below removes them regardless of score.
_FACEBAD_EDGE_KEYS = set()
for _b, _nc, _g in blocks:
    if not _block_faces_ok(_b):
        _v = list(_b)
        for _a, _c in [(_v[0], _v[1]), (_v[1], _v[2]), (_v[2], _v[3]),
                       (_v[3], _v[0]), (_v[4], _v[5]), (_v[5], _v[6]),
                       (_v[6], _v[7]), (_v[7], _v[4]), (_v[0], _v[4]),
                       (_v[1], _v[5]), (_v[2], _v[6]), (_v[3], _v[7])]:
            _FACEBAD_EDGE_KEYS.add((min(_a, _c), max(_a, _c)))
if _FACEBAD_EDGE_KEYS:
    print(f"edges forced straight on twisted blocks: {len(_FACEBAD_EDGE_KEYS)}",
          file=sys.stderr)

# ------------------------------------------------- negative-volume guard
def hex_volume(pts):
    """signed volume of hex via centre-fan (faces wound outward for
    blockMesh ordering 0-3 bottom, 4-7 top). Positive if ordered."""
    c = pts.mean(axis=0)
    # outward-wound faces
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]
    vol = 0.0
    for f in faces:
        a, b, cc, d = pts[f[0]], pts[f[1]], pts[f[2]], pts[f[3]]
        # quad -> 2 tets with centre
        for tri in [(a, b, cc), (a, cc, d)]:
            m = tri[0] - c; n = tri[1] - c; o = tri[2] - c
            vol += np.dot(m, np.cross(n, o)) / 6.0
    return vol

# map vertex -> fluid blocks using it (for volume check only fluid blocks matter)
bad = 0
for b, _nc, _g in blocks:
    pts = snapped[np.array(b)]
    if hex_volume(pts) <= 0:
        bad += 1
print(f"blocks with non-positive volume after snap: {bad}/{len(blocks)}", file=sys.stderr)
if bad > 0:
    # revert snaps for vertices of bad blocks iteratively (largest displacement first)
    disp = np.linalg.norm(snapped - V, axis=1)
    # vertices sorted by displacement desc
    order = np.argsort(-disp)
    reverted = set()
    for v in order:
        if disp[v] < 1e-9 or v in reverted:
            continue
        # does v belong to a bad block?
        # recompute bad blocks containing v
        found = False
        for b, _nc, _g in blocks:
            if v in b:
                if hex_volume(snapped[np.array(b)]) <= 0:
                    found = True
                    break
        if found:
            for _w in (v, _mirror(v)):
                snapped[_w] = presnap[_w] if _w in _PINS else V[_w]
                snap_flag[_w] = False
                if _w in _PINS:
                    _pin_rev.add(_w)
                reverted.add(_w)
    # recount
    bad2 = sum(1 for b, _nc, _g in blocks if hex_volume(snapped[np.array(b)]) <= 0)
    print(f"reverted {len(reverted)} snaps, remaining bad: {bad2}", file=sys.stderr)
    if bad2 > 0 and not _SNAP_ALL:
        print("ERROR: still have negative volumes - aborting snap, using Cartesian", file=sys.stderr)
        snapped = V.copy()

# ------------------------------------------------- second chance for pins
# A pin reverted as collateral (largest displacement in a block that was bad
# for other reasons) gets one retry now the real culprits are reverted.
_pin_kept = 0
for _v in sorted(set(_pin_rev),
                 key=lambda w: float(np.linalg.norm(np.asarray(_PINS[w]) - V[w]))):
    if _v not in _PINS or not (0 <= _v < len(snapped)):
        continue
    _save = snapped[_v].copy()
    snapped[_v] = np.asarray(_PINS[_v], dtype=float)
    _ok = True
    for _b, _nc, _g in blocks:
        if _v in _b:
            _pts = snapped[np.array(_b)]
            if not _block_faces_ok(_b) or hex_volume(_pts) <= 0:
                _ok = False
                break
    if _ok:
        _pin_kept += 1
    else:
        snapped[_v] = _save
if _pin_rev:
    print(f"pins second chance: {_pin_kept}/{len(set(_pin_rev))} kept",
          file=sys.stderr)

# ------------------------------------------------- apply SDF-projected pins LAST
# Keep manual vertex edits, but place their body-surface anchors on the same
# SDF used by projected edges. Raw pins can sit several cm off the STL and
# leave a discontinuity where an SDF polyLine meets its endpoint.
_PIN_N = _PIN_PROJECTED = _PIN_REJECTED = 0
for _v, _pp in _PINS.items():
    _raw = np.asarray(_pp, dtype=float)
    _target = np.asarray(SDF.project(_raw), dtype=float)
    _save = snapped[_v].copy()
    _accepted = False
    for _alpha in (1.0, 0.75, 0.5, 0.25):
        snapped[_v] = _raw + _alpha * (_target - _raw)
        _valid = True
        for _b, _nc, _g in blocks:
            if _v in _b and (not _block_faces_ok(_b) or hex_volume(snapped[np.array(_b)]) <= 0):
                _valid = False
                break
        if _valid:
            _accepted = True
            _PIN_N += 1
            _PIN_PROJECTED += 1 if _alpha == 1.0 else 0
            break
    if not _accepted:
        snapped[_v] = _save
        _PIN_REJECTED += 1
    snap_flag[_v] = False
if _PIN_N:
    print(f"pinned {_PIN_N} hand-placed vertexes; {_PIN_PROJECTED} projected fully onto SDF, "
          f"{_PIN_REJECTED} reverted by block validity", file=sys.stderr)

# Ensure every body-face corner (including a corner held by an earlier cage
# guard) lands on the SDF before the projected polylines are generated.
_frozen_vertices = set()
try:
    _frozen_vertices = set(int(v) for v in _json.load(open(os.path.join(CASE, "system", "freeze.json"))).get("vertices", []))
except (OSError, ValueError, TypeError):
    pass
_wall_vertices = sorted({int(v) for _face in patches["body"] for v in _face})
_wall_moved = _wall_rejected = 0
for _v in _wall_vertices:
    if _v in _frozen_vertices:
        continue
    _start = snapped[_v].copy()
    _target = np.asarray(SDF.project(_start), dtype=float)
    if np.linalg.norm(_target - _start) <= 0.001:
        continue
    _accepted = False
    for _alpha in (1.0, 0.75, 0.5, 0.25):
        snapped[_v] = _start + _alpha * (_target - _start)
        _valid = True
        for _b, _nc, _g in blocks:
            if _v in _b and (not _block_faces_ok(_b) or hex_volume(snapped[np.array(_b)]) <= 0):
                _valid = False
                break
        if _valid:
            _accepted = True
            _wall_moved += 1
            break
    if not _accepted:
        snapped[_v] = _start
        _wall_rejected += 1
print(f"body vertices projected onto SDF: {_wall_moved}; kept for block validity: {_wall_rejected}", file=sys.stderr)

# ------------------------------------------------- outer boundary orthogonalization
def _orthogonalize_outer_boundaries():
    # 1. Top boundary (j = ny): match x, z to layer below (j = ny - 1), pin y to ys[-1]
    for i in range(nx + 1):
        for k in range(nz + 1):
            v_out = vid(i, ny, k)
            v_in = vid(i, ny - 1, k)
            snapped[v_out][0] = float(snapped[v_in][0])
            snapped[v_out][1] = float(ys[-1])
            snapped[v_out][2] = float(snapped[v_in][2])

    # 2. Side boundaries (k = 0 and k = nz): match x, y to adjacent layer, pin z
    for i in range(nx + 1):
        for j in range(ny + 1):
            if j == 0:
                continue  # ground kept as-is
            # -z side
            v_out_lo = vid(i, j, 0)
            v_in_lo = vid(i, j, 1)
            snapped[v_out_lo][0] = float(snapped[v_in_lo][0])
            snapped[v_out_lo][1] = float(snapped[v_in_lo][1])
            snapped[v_out_lo][2] = float(zs[0])

            # +z side
            v_out_hi = vid(i, j, nz)
            v_in_hi = vid(i, j, nz - 1)
            snapped[v_out_hi][0] = float(snapped[v_in_hi][0])
            snapped[v_out_hi][1] = float(snapped[v_in_hi][1])
            snapped[v_out_hi][2] = float(zs[-1])

    # 3. Inlet boundary (i = 0): match y, z to i = 1, pin x to xs[0]
    for j in range(ny + 1):
        if j == 0:
            continue  # ground kept as-is
        for k in range(nz + 1):
            v_out = vid(0, j, k)
            v_in = vid(1, j, k)
            snapped[v_out][0] = float(xs[0])
            snapped[v_out][1] = float(snapped[v_in][1])
            snapped[v_out][2] = float(snapped[v_in][2])

    # 4. Outlet boundary (i = nx): match y, z to i = nx - 1, pin x to xs[-1]
    for j in range(ny + 1):
        if j == 0:
            continue  # ground kept as-is
        for k in range(nz + 1):
            v_out = vid(nx, j, k)
            v_in = vid(nx - 1, j, k)
            snapped[v_out][0] = float(xs[-1])
            snapped[v_out][1] = float(snapped[v_in][1])
            snapped[v_out][2] = float(snapped[v_in][2])

    # 5. Fix 8 domain box corners
    for ci in (0, nx):
        for cj in (0, ny):
            for ck in (0, nz):
                snapped[vid(ci, cj, ck)] = [float(xs[ci]), float(ys[cj]), float(zs[ck])]

    # 6. Align box edges cleanly between intersecting outer boundary planes:
    for i in range(1, nx):
        snapped[vid(i, ny, 0)] = [float(snapped[vid(i, ny - 1, 1)][0]), float(ys[-1]), float(zs[0])]
        snapped[vid(i, ny, nz)] = [float(snapped[vid(i, ny - 1, nz - 1)][0]), float(ys[-1]), float(zs[-1])]
    for k in range(1, nz):
        snapped[vid(0, ny, k)] = [float(xs[0]), float(ys[-1]), float(snapped[vid(1, ny - 1, k)][2])]
        snapped[vid(nx, ny, k)] = [float(xs[-1]), float(ys[-1]), float(snapped[vid(nx - 1, ny - 1, k)][2])]

_orthogonalize_outer_boundaries()
print("outer boundaries orthogonalized (top, sides, inlet, outlet)", file=sys.stderr)

# ------------------------------------------------- outer block grading calculation
def _slab_cell_sizes(L, n, g):
    if n <= 1 or abs(g - 1.0) < 1e-9:
        d = L / max(n, 1)
        return d, d
    t = g ** (1.0 / (n - 1))
    if abs(t - 1.0) < 1e-9:
        d = L / n
        return d, d
    d0 = L * (t - 1.0) / (t ** n - 1.0)
    return d0, d0 * (t ** (n - 1))

def _solve_growth_ratio(L, n, d):
    if n <= 1 or d <= 0 or L <= 0:
        return 1.0
    avg = L / float(n)
    if abs(avg - d) < 1e-9:
        return 1.0
    target = L / d
    if target < n:
        lo, hi = 0.01, 1.0
    else:
        lo, hi = 1.0, 50.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if abs(mid - 1.0) < 1e-9:
            val = float(n)
        else:
            val = (mid ** n - 1.0) / (mid - 1.0)
        if val < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

# Precalculate adjacent inner slab cell sizes
_d_adj_inlet = None
if len(x_lines) > 2 and x_lines[2][1] > 0:
    _L = xs[2] - xs[1]
    _d_adj_inlet, _ = _slab_cell_sizes(_L, x_lines[2][1], x_lines[2][2])

_d_adj_outlet = None
if len(x_lines) > 2 and x_lines[-2][1] > 0:
    _L = xs[-2] - xs[-3]
    _, _d_adj_outlet = _slab_cell_sizes(_L, x_lines[-2][1], x_lines[-2][2])

_d_adj_top = None
if len(y_lines) > 2 and y_lines[-2][1] > 0:
    _L = ys[-2] - ys[-3]
    _, _d_adj_top = _slab_cell_sizes(_L, y_lines[-2][1], y_lines[-2][2])

_d_adj_side_lo = None
if len(z_lines) > 2 and z_lines[2][1] > 0:
    _L = zs[2] - zs[1]
    _d_adj_side_lo, _ = _slab_cell_sizes(_L, z_lines[2][1], z_lines[2][2])

_d_adj_side_hi = None
if len(z_lines) > 2 and z_lines[-2][1] > 0:
    _L = zs[-2] - zs[-3]
    _, _d_adj_side_hi = _slab_cell_sizes(_L, z_lines[-2][1], z_lines[-2][2])

# Slab-consistent outer block gradings (required by OpenFOAM multi-block face merging)
_gy_top_slab = None
if len(y_lines) > 2 and _d_adj_top is not None and y_lines[-1][1] >= 2:
    _Ly_slab = ys[-1] - ys[-2]
    _t = _solve_growth_ratio(_Ly_slab, y_lines[-1][1], _d_adj_top)
    _gy_top_slab = min(max(_t ** (y_lines[-1][1] - 1), 0.01), 100.0)

_gz_side_lo_slab = None
if len(z_lines) > 2 and _d_adj_side_lo is not None and z_lines[1][1] >= 2:
    _Lz_slab = zs[1] - zs[0]
    _t = _solve_growth_ratio(_Lz_slab, z_lines[1][1], _d_adj_side_lo)
    _gz_side_lo_slab = min(max(_t ** (-(z_lines[1][1] - 1)), 0.01), 100.0)

_gz_side_hi_slab = None
if len(z_lines) > 2 and _d_adj_side_hi is not None and z_lines[-1][1] >= 2:
    _Lz_slab = zs[-1] - zs[-2]
    _t = _solve_growth_ratio(_Lz_slab, z_lines[-1][1], _d_adj_side_hi)
    _gz_side_hi_slab = min(max(_t ** (z_lines[-1][1] - 1), 0.01), 100.0)

_gx_inlet_slab = None
if len(x_lines) > 2 and _d_adj_inlet is not None and x_lines[1][1] >= 2:
    _Lx_slab = xs[1] - xs[0]
    _t = _solve_growth_ratio(_Lx_slab, x_lines[1][1], _d_adj_inlet)
    _gx_inlet_slab = min(max(_t ** (-(x_lines[1][1] - 1)), 0.01), 100.0)

_gx_outlet_slab = None
if len(x_lines) > 2 and _d_adj_outlet is not None and x_lines[-1][1] >= 2:
    _Lx_slab = xs[-1] - xs[-2]
    _t = _solve_growth_ratio(_Lx_slab, x_lines[-1][1], _d_adj_outlet)
    _gx_outlet_slab = min(max(_t ** (x_lines[-1][1] - 1), 0.01), 100.0)

_new_blocks = []
for _b_idx, (_hv, _nc, _g) in enumerate(blocks):
    _i, _j, _k = block_ijk[_b_idx]
    _gx, _gy, _gz = _g

    if _j == ny - 1 and _gy_top_slab is not None:
        _gy = _gy_top_slab
    if _k == 0 and _gz_side_lo_slab is not None:
        _gz = _gz_side_lo_slab
    elif _k == nz - 1 and _gz_side_hi_slab is not None:
        _gz = _gz_side_hi_slab
    if _i == 0 and _gx_inlet_slab is not None:
        _gx = _gx_inlet_slab
    elif _i == nx - 1 and _gx_outlet_slab is not None:
        _gx = _gx_outlet_slab

    _new_blocks.append((_hv, _nc, (_gx, _gy, _gz)))

blocks = _new_blocks
print("outer blocks grading calculated based on orthogonalized slab lengths", file=sys.stderr)

# ------------------------------------------------- FULL edge projection
# Every wall edge gets a polyLine whose intermediates are projected onto the
# whole-car SDF. Monotonicity along the edge direction is enforced so
# the polyLine can't loop back; deviation is NOT filtered here -- smoothing
# above carries the curvature into the volume. If checkMesh still reports
# negatives, the auto-drop loop below removes the most deviant edges first.
wall_faces = patches["body"]
from collections import defaultdict
edge_faces = defaultdict(list)
for f in wall_faces:
    for a, b in [(f[0], f[1]), (f[1], f[2]), (f[2], f[3]), (f[3], f[0])]:
        key = (a, b) if a < b else (b, a)
        edge_faces[key].append(f)

def _project_poly(a, b, n_mid):
    """Sample an edge and project all controls through the whole-car SDF."""
    pa = np.asarray(snapped[a], dtype=float)
    pb = np.asarray(snapped[b], dtype=float)
    pts = []
    delta = pb - pa
    prev_t = 0.0
    prev_q = pa.copy()
    for s in range(1, n_mid + 1):
        t = s / (n_mid + 1)
        p_try = pa + t * delta
        # Continue from the preceding surface point so nearby branches at the
        # wheel arch cannot make independent Newton projections jump sides.
        seed = prev_q + (t - prev_t) * delta
        q = np.asarray(SDF.project(seed), dtype=float)
        pts.append(q)
        prev_q, prev_t = q, t
    return pts

# Wheel arches use denser sampling of the whole-car SDF than other edges.
_WHEEL_X_INTERVALS = {(3, 4), (9, 10)}
def _is_wheel_arch_edge(ia, ib, ja, jb, ka, kb):
    return (ja == jb == 1 and ka == kb and ka in (2, 5)
            and tuple(sorted((ia, ib))) in _WHEEL_X_INTERVALS)

edges_out = []
for (a, b), _fl in edge_faces.items():
    ia, ja, ka = vijk[a]; ib, jb, kb = vijk[b]
    if ja == jb and ka == kb and abs(ib - ia) == 1:  # x-directed body edge
        xm = 0.5 * (xs[min(ia, ib)] + xs[min(ia, ib) + 1])
        if not (-2.7 < xm < 2.7):
            continue
        _wheel_edge = _is_wheel_arch_edge(ia, ib, ja, jb, ka, kb)
        _tight_x = ja == jb == 1 and ka == kb and min(ia, ib) == 1 and ka in (2, 4)
        _pts = _project_poly(a, b, 40 if _wheel_edge else (24 if _tight_x else 12))
        edges_out.append((a, b, _pts))
    elif ia == ib and ka == kb and abs(jb - ja) == 1:  # y: 2 mids
        y0 = ys[min(ja, jb)]
        if not (-0.65 < y0 < 0.85):
            continue
        _tight_y = min(ja, jb) == 0 and ia in (3, 9) and ka in (1, nz-1)
        edges_out.append((a, b, _project_poly(a, b, 12 if _tight_y else 8)))
    elif ia == ib and ja == jb and abs(kb - ka) == 1:  # z: 3 mids
        xa = xs[ia]
        if not (-2.4 < xa < 2.4):
            continue
        _klo = min(ka, kb)
        _tight_z = ja == 3 and ia in (2, 3) and _klo in (1, 5)
        edges_out.append((a, b, _project_poly(a, b, 16 if _tight_z else 8)))
    else:
        continue

print(f"curved edges (pre-filter): {len(edges_out)}", file=sys.stderr)
# ------------------------------------------------- edge z-symmetry
# Mirror +z edges onto -z only while the paired dict endpoints are still
# symmetric. Hand-placed pins are applied after vertex symmetry and can move
# one endpoint independently; copying the opposite curve in that case makes
# its control points miss the actual negative-z endpoints, producing a kink.
EKEY = {}
for _a, _b, _p in edges_out:
    EKEY[(min(_a, _b), max(_a, _b))] = (_a, _b, _p)
def _mir2(v):
    i, j, k = vijk[v]
    return vid(i, j, nz - k)
def _mirror_point(p):
    return np.array([p[0], p[1], -p[2]], dtype=float)
_done = set()
_sym = []
for (a, b, pts) in edges_out:
    if (a, b) in _done or (b, a) in _done:
        continue
    ma, mb = _mir2(a), _mir2(b)
    key2 = (min(ma, mb), max(ma, mb))
    if key2 not in EKEY:
        _sym.append((a, b, pts))
        _done.add((a, b))
        continue
    a2, b2, pts2 = EKEY[key2]
    # z-directed pairs reverse endpoint order under reflection; the alternate
    # pairing also handles a self-mirrored edge crossing the centre plane.
    _err_direct = max(
        float(np.linalg.norm(snapped[a] - _mirror_point(snapped[a2]))),
        float(np.linalg.norm(snapped[b] - _mirror_point(snapped[b2])))
    )
    _err_reverse = max(
        float(np.linalg.norm(snapped[a] - _mirror_point(snapped[b2]))),
        float(np.linalg.norm(snapped[b] - _mirror_point(snapped[a2])))
    )
    if min(_err_direct, _err_reverse) > 1e-5:
        _sym.append((a, b, pts))
        if key2 != (min(a, b), max(a, b)):
            _sym.append((a2, b2, pts2))
            _done.add((a2, b2))
            _done.add((b2, a2))
        _done.add((a, b))
        _done.add((b, a))
        continue
    ia, ja, ka = vijk[a]; ib, jb, kb = vijk[b]
    if key2 == (min(a, b), max(a, b)):
        # self-mirrored centre-crossing z-edge: average across the centre
        n = len(pts)
        P = [np.asarray(p, dtype=float) for p in pts]
        Q = []
        for s in range(n):
            q1, q2 = P[s], P[n - 1 - s]
            x = 0.5 * (q1[0] + q2[0]); y = 0.5 * (q1[1] + q2[1])
            if s == n - 1 - s:
                Q.append(np.array([x, y, 0.0]))
            elif s < n - 1 - s:
                az = 0.5 * (abs(q1[2]) + abs(q2[2]))
                Q.append(np.array([x, y, -az]))
            else:
                az = 0.5 * (abs(P[n - 1 - s][2]) + abs(P[s][2]))
                Q.append(np.array([Q[n - 1 - s][0], Q[n - 1 - s][1],
                                   abs(Q[n - 1 - s][2])]))
        _sym.append((a, b, Q))
        _done.add((a, b))
    elif ka == kb:
        # x- or y-edge: same direction sense on both sides, copy +z -> -z
        if float(V[a][2] + V[b][2]) >= 0:
            keep_pts = [np.asarray(p, dtype=float) for p in pts]
            _sym.append((a, b, keep_pts))
            _sym.append((a2, b2, [_mirror_point(p) for p in keep_pts]))
        else:
            keep_pts = [np.asarray(p, dtype=float) for p in pts2]
            _sym.append((a2, b2, keep_pts))
            _sym.append((a, b, [_mirror_point(p) for p in keep_pts]))
        _done.add((a, b))
        _done.add((a2, b2))
        _done.add((b2, a2))
    else:
        # z-edge pair: mirror flips direction, copy reversed +z -> -z
        if float(V[a][2] + V[b][2]) >= 0:
            keep_pts = [np.asarray(p, dtype=float) for p in pts]
            _sym.append((a, b, keep_pts))
            _sym.append((a2, b2, [_mirror_point(p) for p in keep_pts[::-1]]))
        else:
            keep_pts = [np.asarray(p, dtype=float) for p in pts2]
            _sym.append((a2, b2, keep_pts))
            _sym.append((a, b, [_mirror_point(p) for p in keep_pts[::-1]]))
        _done.add((a, b))
        _done.add((a2, b2))
        _done.add((b2, a2))
edges_out = _sym
print(f"curved edges (symmetrised or independently projected): {len(edges_out)}", file=sys.stderr)
# AUTO-DROP: keep maximal smooth coverage with zero negatives. Score each edge
# by max midpoint deviation normalised by local cell length; drop the most
# deviant 8% (these fold subdivided cells). Terrain-following already cut
# negatives 3155 -> 320; this removes the remaining tail while keeping ~92%
# of wall edges ON the SDF (vs 8% with the old 0.15-cell filter).
def _edge_score(item):
    a, b, pts = item
    pa = np.asarray(snapped[a], dtype=float); pb = np.asarray(snapped[b], dtype=float)
    L = float(np.linalg.norm(pb - pa))
    if L < 1e-12 or not pts:
        return 0.0
    # local cells along edge from block counts: use min edge length/3 as proxy
    # (x:2-4, y:2-6, z:2-6 cells per block edge)
    cell = L / 3.0
    maxdev = 0.0
    for q in pts:
        q = np.asarray(q, dtype=float)
        t = float(np.dot(q - pa, pb - pa) / (L * L))
        t = min(max(t, 0.0), 1.0)
        dev = float(np.linalg.norm(q - (pa + t * (pb - pa))))
        maxdev = max(maxdev, dev)
    return maxdev / (cell + 1e-12)
def _is_roof_z_edge(item):
    a, b, _pts = item
    ia, ja, ka = vijk[a]; ib, jb, kb = vijk[b]
    pa, pb = snapped[a], snapped[b]
    z_edge = (ia == ib and ja == jb and abs(kb - ka) == 1
              and ys[ja] > 0.4
              and max(abs(pa[2]), abs(pb[2])) < 1.05
              and -2.4 < xs[ia] < 2.4)
    y_shoulder_edge = (ia == ib and ka == kb and abs(jb - ja) == 1
                       and ys[min(ja, jb)] > 0.1
                       and 0.78 < min(abs(pa[2]), abs(pb[2]))
                       and max(abs(pa[2]), abs(pb[2])) < 1.05
                       and max(pa[1], pb[1]) > -0.15
                       and -2.4 < xs[ia] < 2.4)
    return z_edge or y_shoulder_edge
scores = np.array([_edge_score(e) for e in edges_out])
if _SNAP_ALL:
    print(f"SNAP_ALL: keeping all {len(edges_out)} curved edges", file=sys.stderr)
    keep = list(edges_out)
elif len(scores):
    # Gentle majority only: steep subdivided curves are what skew/fold.
    _EDGE_PCT = float(os.environ.get("EDGE_PCT", "92"))
    thresh = float(np.percentile(scores, _EDGE_PCT)) if len(scores) else 0.0
    keep = []
    n_forced = 0
    for e, s in zip(edges_out, scores):
        _k = (min(e[0], e[1]), max(e[0], e[1]))
        if _k in _FACEBAD_EDGE_KEYS and not _is_roof_z_edge(e):
            n_forced += 1
            continue
        if s <= thresh or _is_roof_z_edge(e):
            keep.append(e)
    print(f"edge deviation thr(p{_EDGE_PCT:.0f})={thresh:.3f} p99={np.percentile(scores,99):.3f} max={scores.max():.3f} (norm. by cell); keeping {len(keep)}/{len(edges_out)} (+{n_forced} forced straight on twisted blocks)", file=sys.stderr)
else:
    keep = []
    print("edge deviation: no wall edges (no solid cut); keeping 0/0", file=sys.stderr)
edges_out = keep
print(f"curved edges: {len(edges_out)}", file=sys.stderr)
# SAFETY: straight-edge test flag -- set True to disable all curvature
# (isolates vertex-snap vs edge-curve folding). Keep False for final.
STRAIGHT_TEST = False
if STRAIGHT_TEST:
    print("STRAIGHT_TEST: dropping all curved edges", file=sys.stderr)
    edges_out = []
# SANITISE: drop degenerate polyLines (coincident consecutive points make
# polyLine::calcParam divide by zero -> SIGFPE in blockMesh). Compare AFTER
# %.6g rounding (the file precision): full-precision neighbours can round
# identical. Keep the edge only if a mid survives distinct from both ends.
def _r6(p):
    return tuple(float("%.6g" % c) for c in p)
_clean = []
for a, b, pts in edges_out:
    pa = _r6(snapped[a])
    pb = _r6(snapped[b])
    seq = [pa] + [_r6(q) for q in pts] + [pb]
    keep = [seq[0]]
    for q in seq[1:]:
        if q != keep[-1]:
            keep.append(q)
    if len(keep) >= 4:  # endpoints + >=1 distinct mid (degenerate ends trimmed)
        # trim endpoints, drop if nothing left between
        mids = [q for q in keep[1:-1] if q != pa and q != pb]
        if mids:
            _clean.append((a, b, [np.array(q) for q in mids]))
print(f"curved edges after degenerate filter: {len(_clean)}/{len(edges_out)}",
      file=sys.stderr)
edges_out = _clean

# Validate the actual piecewise-linear blockMesh edge geometry against the STL.
# Sample inside every segment, not only at polyLine controls (controls can be
# on the surface while the chords between them cut across it).
_AUDIT_FRAC = float(os.environ.get("POLY_AUDIT_FRAC", "0.25"))
def _edge_cell_scale(a, b):
    ia, ja, ka = vijk[a]; ib, jb, kb = vijk[b]
    axis = int(np.argmax(np.abs(np.asarray([ib-ia, jb-ja, kb-ka]))))
    length = float(np.linalg.norm(snapped[a] - snapped[b]))
    counts = [int(nc[axis]) for hv, nc, _g in blocks if a in hv and b in hv]
    return length / max(counts) if counts else length

def _audit_polyline(a, b, pts):
    chain = [np.asarray(snapped[a], float)] + [np.asarray(q, float) for q in pts] + [np.asarray(snapped[b], float)]
    worst = (0.0, None)
    for p0, p1 in zip(chain, chain[1:]):
        for t in (0.25, 0.5, 0.75):
            p = p0 + t * (p1 - p0)
            _q, d = closest_on_stl_for_audit(p, ncand=80)
            if d > worst[0]:
                worst = (float(d), p)
    tol = _AUDIT_FRAC * _edge_cell_scale(a, b)
    return worst[0], tol, worst[1]

_audit = [(a, b, *_audit_polyline(a, b, pts)) for a, b, pts in edges_out]
_audit_bad = [r for r in _audit if r[2] > r[3]]
_audit_worst = max(_audit, key=lambda r: r[2], default=None)
print(f"STL polyLine audit: {_AUDIT_FRAC:.2f} local-cell tolerance; {len(_audit_bad)}/{len(_audit)} edges over tolerance" +
      (f"; worst edge { _audit_worst[0]}-{_audit_worst[1]} gap={_audit_worst[2]:.5f} tol={_audit_worst[3]:.5f}" if _audit_worst else ""), file=sys.stderr)
for a, b, gap, tol, p in sorted(_audit_bad, key=lambda r: r[2], reverse=True)[:12]:
    print(f"  STL gap edge {a}-{b}: {gap:.5f} m > {tol:.5f} m at {np.asarray(p).round(5).tolist()}", file=sys.stderr)
_audit_path = os.path.join(os.path.dirname(OUT), "polyLineAudit.json")
with open(_audit_path, "w") as _af:
    _json.dump({"tolerance_fraction_of_local_cell": _AUDIT_FRAC,
               "edge_count": len(_audit),
               "failing_count": len(_audit_bad),
               "edges": [{"vertices": [a, b], "max_gap_m": gap,
                          "tolerance_m": tol,
                          "worst_sample": (np.asarray(p).tolist() if p is not None else None),
                          "pass": gap <= tol}
                         for a, b, gap, tol, p in _audit]}, _af, indent=2)
print(f"STL polyLine audit report: {_audit_path}", file=sys.stderr)
if _audit_bad and os.environ.get("POLY_AUDIT_STRICT", "0") != "0":
    raise SystemExit("STL polyLine audit failed; refusing to write blockMeshDict")

# ------------------------------------------------- write dict
ncells = sum(nc[0] * nc[1] * nc[2] for _hv, nc, _g in blocks)
print(f"approx cells: {ncells}", file=sys.stderr)
backup_dict()
with open(OUT, "w") as f:
    f.write("""/*--------------------------------*- C++ -*----------------------------------*\\
  Generated by make_car_blockMeshDict.py - body-fitted car, SDF-projected
  vertices/edges with negative-volume guards, no snappyHexMesh
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}

convertToMeters 1;
""")
    import json as _jsplit
    f.write("// SPLITS: %s\n" % _jsplit.dumps({"x": xs, "y": ys, "z": zs}))
    f.write("\nvertices\n(\n")
    for p in snapped:
        f.write("    (%.6g %.6g %.6g)\n" % tuple(p))
    f.write(");\n\nblocks\n(\n")
    for hv, nc, g in blocks:
        f.write("    hex (%s) (%d %d %d) simpleGrading (%.6g %.6g %.6g)\n"
                % (" ".join(map(str, hv)), nc[0], nc[1], nc[2], g[0], g[1], g[2]))
    f.write(");\n\nedges\n(\n")
    for a, b, pts in edges_out:
        f.write("    polyLine %d %d (" % (a, b))
        for q in pts:
            f.write("(%.6g %.6g %.6g)" % tuple(q))
        f.write(")\n")
    f.write(");\n\nboundary\n(\n")
    ptype = {"inlet": "patch", "outlet": "patch", "sides": "patch", "top": "patch",
             "ground": "wall", "body": "wall"}
    for name in ["inlet", "outlet", "sides", "top", "ground", "body"]:
        f.write("    %s\n    {\n        type %s;\n        faces\n        (\n" % (name, ptype[name]))
        for fa in patches[name]:
            f.write("            (%d %d %d %d)\n" % fa)
        f.write("        );\n    }\n")
    f.write(");\n\nmergePatchPairs\n(\n);\n")
print(f"wrote {OUT}", file=sys.stderr)
