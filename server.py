#!/usr/bin/env python3
"""Block-split GUI backend (Python standard library only).

Serves the single-page Three.js GUI and the JSON API it needs:
  GET  /                          -> gui/index.html
  GET  /vendor/<file>             -> vendored three.js (offline)
  GET  /api/stl?stride=N          -> decimated car surface {positions,index,bounds,count}
  GET  /api/tables                -> roof/half-width tables + tuned splits + domain
  POST /api/project               -> signed-distance-field projection of a split face
  POST /api/export                -> write splits.json, run exporter, opt blockMesh/checkMesh
  GET  /api/dict                  -> download system/blockMeshDict

Run from the case directory:
    python3 gui/server.py [port]      (default 8765)
"""
import json
import os
import re
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GUI = os.path.join(CASE, "gui")
SYS = os.path.join(CASE, "system")
STL = os.path.join(CASE, "car_simplified.stl")
GENERATOR = os.path.join(SYS, "make_car_blockMeshDict.py")
sys.path.insert(0, SYS)
from sdf import load_field, reload_field
SDF_PATH = os.path.join(SYS, "car_sdf.npy")
SDF_BUILDER = os.path.join(SYS, "build_car_sdf.py")
SDF_EXTRACTOR = os.path.join(SYS, "extract_car_sdf_surface.py")
SDF = load_field(SDF_PATH)

ROOF = [(-2.267, -0.118), (-2.10, 0.008), (-1.90, 0.090), (-1.70, 0.129),
        (-1.50, 0.143), (-1.30, 0.154), (-1.10, 0.164), (-0.90, 0.218),
        (-0.70, 0.334), (-0.50, 0.465), (-0.30, 0.558), (-0.10, 0.600),
        (0.10, 0.610), (0.30, 0.610), (0.50, 0.588), (0.70, 0.560),
        (0.90, 0.511), (1.10, 0.451), (1.30, 0.380), (1.50, 0.303),
        (1.70, 0.230), (1.90, 0.158), (2.10, 0.080), (2.267, 0.003)]
HALFW = [(-2.267, 0.45), (-2.10, 0.90), (-1.90, 0.986), (-1.70, 1.012),
         (-1.50, 1.012), (-1.30, 1.012), (-1.10, 1.008), (-0.90, 0.996),
         (-0.70, 1.000), (-0.50, 1.000), (-0.30, 1.000), (-0.10, 1.000),
         (0.10, 1.000), (0.30, 0.999), (0.50, 0.990), (0.70, 0.988),
         (0.90, 1.012), (1.10, 1.012), (1.30, 1.012), (1.50, 0.996),
         (1.70, 0.972), (1.90, 0.908), (2.10, 0.732), (2.267, 0.44)]
UNDER, GROUND, BELT = -0.41, -0.59, 0.17
FX0, FX1, RX0, RX1 = -1.653, -1.171, 0.879, 1.375
WZIN, WZOUT, WTOP = 0.74, 1.012, -0.10
# wheel tyre circles (X-Y at axle): front/rear centre x, radius -> axle y
WHEEL_R = 0.205
WHEEL_CX = {"f": 0.5 * (FX0 + FX1), "r": 0.5 * (RX0 + RX1)}
WHEEL_CY = GROUND + WHEEL_R
WHEEL_Z0, WHEEL_Z1 = WZIN, WZOUT


def _wheel_key(cx, cy, cz):
    # return "f"/"r" if (cx,cy,cz) sits in a lower-half tyre box, else None
    if abs(cz) < WHEEL_Z0 or abs(cz) > WHEEL_Z1:
        return None
    if cy >= WHEEL_CY:
        return None
    if FX0 <= cx <= FX1:
        return "f"
    if RX0 <= cx <= RX1:
        return "r"
    return None
TUNED = {
    "x": [-8.0, -2.267, -2.1, -1.9, -1.65, -1.41, -1.17, -0.9, -0.7, -0.5,
          -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.88, 1.125, 1.37, 1.5, 1.7,
          1.9, 2.1, 2.267, 12.0],
    "y": [-0.59, -0.41, -0.15, 0.05, 0.17, 0.30, 0.45, 0.55, 0.61, 4.0],
    "z": [-4.0, -1.012, -0.7, 0.7, 1.012, 4.0],
}
DOMAIN = {"xmin": -8.0, "xmax": 12.0, "ymin": -0.59, "ymax": 4.0,
          "zmin": -4.0, "zmax": 4.0}
RREF, WREF = 0.61, 1.012
_SH_YN = [0.0, 0.386, 0.848, 0.973, 1.0]
_SH_AZ = [1.0, 0.8, 0.6, 0.4, 0.2]


def _build_smooth_W():
    import numpy as np
    _xs = np.arange(-3.0, 3.001, 0.05)

    def _rawW(_x):
        return _half_w(_x) if -2.267 <= _x <= 2.267 else WREF

    _raw = np.array([_rawW(_x) for _x in _xs])
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


def _smooth_W(x):
    import numpy as np
    if "t" not in _SW:
        _SW["xs"], _SW["w"] = _build_smooth_W()
    _SW_XS, _SW_W = _SW["xs"], _SW["w"]
    if x <= _SW_XS[0]:
        return float(_SW_W[0])
    if x >= _SW_XS[-1]:
        return float(_SW_W[-1])
    _i = int(np.searchsorted(_SW_XS, x)) - 1
    _x0, _y0 = _SW_XS[_i], _SW_W[_i]
    _x1, _y1 = _SW_XS[_i + 1], _SW_W[_i + 1]
    return float(_y0 + (_y1 - _y0) * (x - _x0) / (_x1 - _x0))


_SW = {}


def _interp_t(table, x):
    import numpy as np
    table = np.asarray(table, float)
    if x <= table[0, 0]:
        return float(table[0, 1])
    if x >= table[-1, 0]:
        return float(table[-1, 1])
    i = int(np.searchsorted(table[:, 0], x)) - 1
    x0, y0 = table[i]
    x1, y1 = table[i + 1]
    return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))


def _roof_c(x):
    return _interp_t(ROOF, x)


def _half_w(x):
    return _interp_t(HALFW, x)


def _roof_yz(x, z):
    import numpy as np
    R = _roof_c(x)
    W = _half_w(x)
    if W <= 1e-9:
        return R
    az = min(abs(z) / W, 1.0)
    return R - max(0.0, R - BELT) * (az ** 2.5)


def _is_solid(cx, cy, cz):
    az = abs(cz)
    if ((FX0 < cx < FX1) or (RX0 < cx < RX1)) and (WZIN < az < WZOUT) \
            and (GROUND < cy < WTOP):
        return True
    if cx < -2.267 or cx > 2.267:
        return False
    if cy < UNDER or cy > 0.65:
        return False
    W = _half_w(cx)
    if az > W:
        return False
    if cy > _roof_yz(cx, cz):
        return False
    return True


def _shoulder_az(yn):
    import numpy as np
    yn = min(max(yn, 0.0), 1.0)
    return float(np.interp(yn, _SH_YN, _SH_AZ))


def _deform(x, y, z, flat=False):
    """Terrain-following deform. Mirrors make_car_blockMeshDict exactly."""
    import numpy as np
    if flat:
        return y, z
    in_car = -2.267 <= x <= 2.267
    Rx = _roof_c(x) if in_car else RREF
    Wx = _smooth_W(x)
    Rz = _roof_yz(x, z) if in_car else RREF
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
        z2 = z + s * (Wx - WREF) * (az / WREF)
    else:
        z2 = z + s * (Wx - WREF) * float(np.exp(-(az - WREF) / 1.5))
    if in_car and BELT < y2 < Rx and Rx > BELT + 0.05 and abs(z2) <= Wx:
        yn = (y2 - BELT) / (Rx - BELT)
        Ws = _shoulder_az(yn) * Wx
        if Wx > 1e-9:
            zs = s * abs(z2) * (Ws / Wx)
            z2 = z2 + 0.6 * (zs - z2)
    return y2, z2


def _clean_triples(v):
    out = []
    if isinstance(v, list):
        for t in v:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                try:
                    out.append([int(t[0]), int(t[1]), int(t[2])])
                except (TypeError, ValueError):
                    pass
    return out


_tri_cache = {}


def _tri_bboxes():
    if "b" not in _tri_cache:
        import numpy as np
        T0, T1, T2 = load_stl()
        lo = np.minimum(np.minimum(T0, T1), T2)
        hi = np.maximum(np.maximum(T0, T1), T2)
        _tri_cache["b"] = (T0, T1, T2, lo, hi)
    return _tri_cache["b"]


def _ray_hits_inside(pts, direction):
    """Even/odd ray-cast per point (vectorized prefilter + exact MT).

    pts: (M,3) query points (deformed block centres). Returns bool array:
    True = odd crossings = inside the closed STL.
    """
    import numpy as np
    T0, T1, T2, lo, hi = _tri_bboxes()
    d = int(np.argmax(np.abs(direction)))
    sgn = 1.0 if direction[d] >= 0 else -1.0
    t1 = (d + 1) % 3
    t2 = (d + 2) % 3
    # jitter transverse coords to dodge exact vertex/edge hits (deterministic)
    P = np.asarray(pts, float) + np.array([1e-7, 2e-7, 3e-7])[:3]
    inside = np.zeros(len(P), dtype=bool)
    E1 = T1 - T0
    E2 = T2 - T0
    for n in range(len(P)):
        o = P[n]
        if sgn > 0:
            m = (hi[:, d] > o[d]) & (lo[:, t1] <= o[t1] + 1e-9) \
                & (hi[:, t1] >= o[t1] - 1e-9) & (lo[:, t2] <= o[t2] + 1e-9) \
                & (hi[:, t2] >= o[t2] - 1e-9)
        else:
            m = (lo[:, d] < o[d]) & (lo[:, t1] <= o[t1] + 1e-9) \
                & (hi[:, t1] >= o[t1] - 1e-9) & (lo[:, t2] <= o[t2] + 1e-9) \
                & (hi[:, t2] >= o[t2] - 1e-9)
        idx = np.nonzero(m)[0]
        if len(idx) == 0:
            continue
        a = T0[idx]
        e1 = E1[idx]
        e2 = E2[idx]
        if sgn > 0:
            dv = np.array([1.0 if d == 0 else 0.0, 1.0 if d == 1 else 0.0,
                           1.0 if d == 2 else 0.0])
        else:
            dv = -np.array([1.0 if d == 0 else 0.0, 1.0 if d == 1 else 0.0,
                            1.0 if d == 2 else 0.0])
        pvec = np.cross(dv, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        ok = np.abs(det) > 1e-12
        inv = np.zeros_like(det)
        inv[ok] = 1.0 / det[ok]
        tvec = o - a
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        ok &= (u >= -1e-9) & (u <= 1.0 + 1e-9)
        qvec = np.cross(tvec, e1)
        v = np.einsum("ij,ij->i", qvec, np.broadcast_to(dv, qvec.shape)) * inv
        ok &= (v >= -1e-9) & (u + v <= 1.0 + 1e-9)
        t = np.einsum("ij,ij->i", e2, qvec) * inv
        ok &= t > 1e-7
        inside[n] = (int(np.count_nonzero(ok)) % 2) == 1
    return inside


_class_cache = {}


def classify_raycast(centres):
    """Majority vote of +x/-x/+y rays (third only to break 1-1 ties)."""
    import numpy as np
    C = np.asarray(centres, float)
    v0 = _ray_hits_inside(C, (1.0, 0.0, 0.0))
    v1 = _ray_hits_inside(C, (-1.0, 0.0, 0.0))
    solid = v0 & v1
    tied = v0 ^ v1
    if bool(tied.any()):
        v2 = _ray_hits_inside(C[tied], (0.0, 1.0, 0.0))
        s = solid.copy()
        s[tied] = (v0[tied].astype(int) + v1[tied].astype(int)
                   + v2.astype(int)) >= 2
        return s, {"unanimous": int((~tied).sum()), "voted": int(tied.sum())}
    return solid, {"unanimous": int(len(C)), "voted": 0}


def _auto_lines(coords, axis):
    """Mirror of the generator rule: per-interval (n, grading)."""
    coords = sorted(set(float(c) for c in coords))
    out = []
    for prev, cur in zip(coords[:-1], coords[1:]):
        L = cur - prev
        if L <= 1e-9:
            out.append((2, 1))
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
        out.append((n, g))
    return out


def _resolve_lines(vals, axis):
    """Full [[c,n,g],...] lines: stored splits.json *_lines if coords match,
    else coords + auto rule (same as the generator)."""
    vals = sorted(set(float(v) for v in vals))
    try:
        with open(os.path.join(SYS, "splits.json")) as f:
            d = json.load(f)
        stored = d.get(axis + "_lines")
        if stored:
            lc = [float(c) for c, n, g in stored]
            if len(lc) == len(vals) and all(abs(a - b) < 1e-9 for a, b in zip(lc, vals)):
                return [[float(c), int(n), float(g)] for c, n, g in stored]
    except (OSError, ValueError, TypeError, KeyError):
        pass
    auto = _auto_lines(vals, axis)
    return [[vals[0], 0, 1]] + [[c, n, g] for c, (n, g) in zip(vals[1:], auto)]


def build_interface(xs, ys, zs, solid_ov, fluid_ov, method="raycast", nlines=None,
                    flat=False):
    """Deformed block grid + solid test + overrides -> interface tris.

    method="raycast" (default): even/odd closest... parity of STL ray
    crossings at each deformed block centre — follows the true surface.
    method="analytic": legacy lookup-table test (for comparison).
    """
    import numpy as np
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    zs = [float(v) for v in zs]
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1

    def vid(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    V = np.empty(((nx + 1) * (ny + 1) * (nz + 1), 3))
    for k in range(nz + 1):
        for j in range(ny + 1):
            for i in range(nx + 1):
                y2, z2 = _deform(xs[i], ys[j], zs[k], flat)
                V[vid(i, j, k)] = (xs[i], y2, z2)
    solid = np.zeros((nx, ny, nz), dtype=bool)
    centres = {}
    clist = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                cx = 0.5 * (xs[i] + xs[i + 1])
                cy = 0.5 * (ys[j] + ys[j + 1])
                cz = 0.5 * (zs[k] + zs[k + 1])
                cy2, cz2 = _deform(cx, cy, cz, flat)
                centres[(i, j, k)] = [cx, cy2, cz2]
                clist.append([cx, cy2, cz2])
    vote_info = {"unanimous": 0, "voted": 0}
    if method == "analytic":
        for (i, j, k), c in zip(centres.keys(), clist):
            solid[i, j, k] = _is_solid(c[0], c[1], c[2])
    else:
        key = (tuple(round(v, 6) for v in xs),
               tuple(round(v, 6) for v in ys),
               tuple(round(v, 6) for v in zs),
               bool(flat))
        hit = _class_cache.get(key)
        if hit is None:
            arr, vote_info = classify_raycast(clist)
            if len(_class_cache) >= 4:
                _class_cache.pop(next(iter(_class_cache)))
            _class_cache[key] = (arr.copy(), vote_info)
            hit = (arr, vote_info)
        arr, vote_info = hit
        for (n, (i, j, k)) in enumerate(centres.keys()):
            solid[i, j, k] = bool(arr[n])
    n_ov = 0
    for (a, b, c) in solid_ov:
        if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz and not solid[a, b, c]:
            solid[a, b, c] = True
            n_ov += 1
    for (a, b, c) in fluid_ov:
        if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz and solid[a, b, c]:
            solid[a, b, c] = False
            n_ov += 1

    def S(i, j, k):
        if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz:
            return bool(solid[i, j, k])
        return None

    # (neighbour offset, 4 vertex (i,j,k) offsets)
    F = [((-1, 0, 0), [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)]),
         ((1, 0, 0), [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]),
         ((0, -1, 0), [(0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0)]),
         ((0, 1, 0), [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)]),
         ((0, 0, -1), [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]),
         ((0, 0, 1), [(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1)])]
    tris = []
    wallq = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                s0 = bool(solid[i, j, k])
                for _di, (d, cn) in enumerate(F):
                    nb = S(i + d[0], j + d[1], k + d[2])
                    if nb is None or nb == s0:
                        continue
                    if not s0:
                        continue  # one emission per interface, solid side only
                    q = [V[vid(i + a, j + b, k + c)] for (a, b, c) in cn]
                    tris += [q[0], q[1], q[2], q[0], q[2], q[3]]
                    wallq.append({"b": [i, j, k], "d": _di,
                                  "q": [[float(v) for v in p] for p in q],
                                  "e": [[i + a, j + b, k + c] for (a, b, c) in cn]})
    return {"wall": [[float(v) for v in p] for p in tris],
            "wallq": wallq,
            "blocks": [{"k": [i, j, k], "c": centres[(i, j, k)],
                        "s": int(solid[i, j, k]),
                        "w": 1 if (solid[i, j, k] and _wheel_key(*centres[(i, j, k)])) else 0,
                        "box": [[float(v) for v in V[vid(i + a, j + b, k + c)]]
                                for (a, b, c) in
                                [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                 (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]],
                        "nc": [int(nlines["x"][i]) if nlines else 0,
                               int(nlines["y"][j]) if nlines else 0,
                               int(nlines["z"][k]) if nlines else 0]} for k in range(nz)
                       for j in range(ny) for i in range(nx)],
            "counts": {"solid": int(solid.sum()),
                       "fluid": int((~solid).sum()),
                       "wheel": sum(1 for (i, j, k) in centres
                                    if solid[i, j, k]
                                    and _wheel_key(*centres[(i, j, k)])),
                       "faces": len(tris) // 6,
                       "overrides": n_ov,
                       "method": method,
                       "unanimous": int(vote_info.get("unanimous", 0)),
                       "voted": int(vote_info.get("voted", 0))}}

_stl_cache = {}
_sdf_surface_cache = {}
_lock = threading.Lock()


def load_stl():
    if "t" not in _stl_cache:
        import numpy as np
        with open(STL, "rb") as f:
            f.read(80)
            n = struct.unpack("<I", f.read(4))[0]
            raw = f.read()
        T0 = np.empty((n, 3))
        T1 = np.empty((n, 3))
        T2 = np.empty((n, 3))
        for i in range(n):
            v = struct.unpack_from("<12f", raw, i * 50)
            T0[i] = v[3:6]
            T1[i] = v[6:9]
            T2[i] = v[9:12]
        _stl_cache["t"] = (T0, T1, T2)
    return _stl_cache["t"]


def run(cmd, timeout=600):
    # Serialize generator/build commands so no generator reads the SDF while
    # the Distance Field button is replacing it.
    with _lock:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + "\n" + (r.stderr or "")


def _dict_grid():
    """Grid (x/y/z split lists) the CURRENT blockMeshDict was built from (the
    // SPLITS: stamp). Vertex ids in the dict follow THIS grid, never a newer
    splits.json. Falls back to splits.json. Returns {"x","y","z",...} or None."""
    try:
        with open(os.path.join(SYS, "blockMeshDict")) as f:
            for line in f:
                if line.startswith("// SPLITS:"):
                    sp = json.loads(line[len("// SPLITS:"):])
                    g = {a: [float(v) for v in sp.get(a, [])] for a in "xyz"}
                    if all(g[a] for a in "xyz"):
                        return g
                    break
    except (OSError, ValueError, TypeError):
        pass
    try:
        with open(os.path.join(SYS, "splits.json")) as f:
            d = json.load(f)
        return {a: [float(v) for v in d[a]] for a in "xyz"}
    except (OSError, ValueError, TypeError, KeyError):
        return None


def _closest_on_tris(p, A, B, C):
    """Closest point to p on each triangle (A,B,C: (n,3)) -> (n,3). Ericson."""
    import numpy as np
    ein = lambda a, b: np.einsum("ij,ij->i", a, b)
    tiny = 1e-30
    ab, ac = B - A, C - A
    ap, bp, cp = p - A, p - B, p - C
    d1, d2 = ein(ab, ap), ein(ac, ap)
    d3, d4 = ein(ab, bp), ein(ac, bp)
    d5, d6 = ein(ab, cp), ein(ac, cp)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    den = va + vb + vc
    den = np.where(np.abs(den) < tiny, tiny, den)
    out = A + ab * (vb / den)[:, None] + ac * (vc / den)[:, None]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)            # edge AB
    t = d1 / np.where(np.abs(d1 - d3) < tiny, tiny, d1 - d3)
    out[m] = (A + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)            # edge AC
    t = d2 / np.where(np.abs(d2 - d6) < tiny, tiny, d2 - d6)
    out[m] = (A + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)  # edge BC
    w = (d4 - d3) + (d5 - d6)
    t = (d4 - d3) / np.where(np.abs(w) < tiny, tiny, w)
    out[m] = (B + (C - B) * t[:, None])[m]
    m = (d1 <= 0) & (d2 <= 0)                        # vertex A
    out[m] = A[m]
    m = (d3 >= 0) & (d4 <= d3)                       # vertex B
    out[m] = B[m]
    m = (d6 >= 0) & (d5 <= d6)                       # vertex C
    out[m] = C[m]
    return out


_snap_cache = {}


def _snap_tris(target):
    """Triangle soup (A,B,C) of the snap target: "stl" = car STL,
    "smooth" = smooth distance-field mesh (system/car_sdf_surface.npz)."""
    import numpy as np
    if target == "smooth":
        if "smooth" not in _snap_cache:
            with np.load(os.path.join(SYS, "car_sdf_surface.npz")) as m:
                P = np.asarray(m["positions"], float).reshape(-1, 3)
                F = np.asarray(m["faces"]).reshape(-1, 3).astype(np.int64)
            _snap_cache["smooth"] = (P[F[:, 0]], P[F[:, 1]], P[F[:, 2]])
        return _snap_cache["smooth"]
    return load_stl()


def snap_points(P, target):
    """Nearest point on the target surface for each row of P."""
    import numpy as np
    A, B, C = _snap_tris(target)
    out = np.empty_like(P)
    dist = np.empty(len(P))
    for n in range(len(P)):
        q = _closest_on_tris(P[n], A, B, C)
        d2 = ((q - P[n]) ** 2).sum(axis=1)
        k = int(np.argmin(d2))
        out[n] = q[k]
        dist[n] = float(np.sqrt(d2[k]))
    return out, dist


def _key_to_vid(key, d):
    try:
        p = str(key).split("|")
        if len(p) == 3:
            x, y, z = (float(c) for c in p)
            xs, ys, zs = d["x"], d["y"], d["z"]
            i = next(i for i, v in enumerate(xs) if abs(v - x) < 1e-6)
            j = next(j for j, v in enumerate(ys) if abs(v - y) < 1e-6)
            k = next(k for k, v in enumerate(zs) if abs(v - z) < 1e-6)
            nx, ny = len(xs) - 1, len(ys) - 1
            return i + (nx + 1) * (j + (ny + 1) * k), (x, y, z)
        return int(key), None
    except (ValueError, StopIteration, KeyError, TypeError):
        return None, None


def _foam_load(path):
    """-> (raw, body_start, binary, label_bits, scalar_bits, class)"""
    raw = open(path, "rb").read()
    m = re.search(rb"FoamFile\s*\{(.*?)\}", raw, re.S)
    head = m.group(1).decode("latin-1") if m else ""
    am = re.search(r"label=(\d+)", head)
    sm = re.search(r"scalar=(\d+)", head)
    cm = re.search(r"class\s+(\w+)", head)
    return (raw, m.end() if m else 0, bool(re.search(r"format\s+binary", head)),
            int(am.group(1)) if am else 32, int(sm.group(1)) if sm else 64,
            cm.group(1) if cm else "")


def _foam_skip(raw, pos):
    """Skip whitespace and // and /* */ comments."""
    n = len(raw)
    while pos < n:
        c = raw[pos:pos + 1]
        if c in b" \t\r\n":
            pos += 1
        elif raw[pos:pos + 2] == b"//":
            e = raw.find(b"\n", pos)
            pos = n if e < 0 else e + 1
        elif raw[pos:pos + 2] == b"/*":
            e = raw.find(b"*/", pos)
            pos = n if e < 0 else e + 2
        else:
            break
    return pos


def _foam_list(raw, pos, binary, dtype, ncomp=1):
    """Parse 'N(...)' at/after pos -> (numpy array shape (N,ncomp) or (N,), end).
    The binary payload starts right after '(' (no whitespace skipping: a first
    label equal to 10/32/... must not be mistaken for whitespace)."""
    import numpy as np
    pos = _foam_skip(raw, pos)
    m = re.compile(rb"(\d+)\s*\(").match(raw, pos)
    if not m:
        raise ValueError("list header not found")
    n = int(m.group(1))
    off = m.end()
    if binary:
        dt = np.dtype(dtype)
        a = np.frombuffer(raw, dtype=dt, count=n * ncomp, offset=off)
        end = off + n * ncomp * dt.itemsize
        if raw[end:end + 1] == b")":
            end += 1
    elif ncomp == 1:
        e = raw.index(b")", off)
        a = np.array(raw[off:e].split(), dtype=dtype)
        end = e + 1
    else:
        vals = []
        end = off
        for k, mm in enumerate(re.finditer(rb"\(\s*([^()]+?)\s*\)", raw[off:])):
            vals.extend(float(t) for t in mm.group(1).split())
            end = off + mm.end()
            if k + 1 >= n:
                break
        a = np.array(vals, dtype=dtype)
    if ncomp > 1:
        a = a.reshape(-1, ncomp)
    return a, end


def _read_label_list(path):
    """OpenFOAM labelList / cellSet / faceSet file -> list[int] ([] on failure)."""
    try:
        raw, pos, binary, lb, sb, _c = _foam_load(path)
        a, _e = _foam_list(raw, pos, binary, "<i4" if lb == 32 else "<i8")
        return [int(v) for v in a]
    except (OSError, ValueError, IndexError):
        return []


def _read_cell_set(path):
    """Parse an OpenFOAM cellSet/faceSet -> list[int] (ascii or binary)."""
    return _read_label_list(path)


def _read_points(path):
    raw, pos, binary, lb, sb, _c = _foam_load(path)
    a, _e = _foam_list(raw, pos, binary, "<f8" if sb == 64 else "<f4", 3)
    return a.astype(float)


def _read_faces(path):
    """polyMesh/faces -> (offsets, flat) numpy int arrays (faceCompactList, or
    an ascii faceList)."""
    import numpy as np
    raw, pos, binary, lb, sb, cls = _foam_load(path)
    dt = "<i4" if lb == 32 else "<i8"
    if cls == "faceCompactList":
        offs, e = _foam_list(raw, pos, binary, dt)
        flat, _e = _foam_list(raw, e, binary, dt)
        return offs.astype(np.int64), flat.astype(np.int64)
    if binary:
        raise ValueError("binary non-compact faceList is not supported")
    p = _foam_skip(raw, pos)
    m = re.compile(rb"(\d+)\s*\(").match(raw, p)
    offs, flat = [0], []
    for mm in re.finditer(rb"\d+\s*\(([^()]*)\)", raw[m.end():]):
        v = [int(t) for t in mm.group(1).split()]
        flat.extend(v)
        offs.append(len(flat))
    return np.array(offs, dtype=np.int64), np.array(flat, dtype=np.int64)


def _problem_tris(cells, nof, skf, pyf, cap=20000):
    """Triangles (flat xyz lists) of the whole bad cells / faces, read straight
    from constant/polyMesh. cells: cell ids; others: face ids."""
    import numpy as np
    pd = os.path.join(CASE, "constant", "polyMesh")
    P = _read_points(os.path.join(pd, "points"))
    offs, flat = _read_faces(os.path.join(pd, "faces"))
    nf = len(offs) - 1

    def tris(fids):
        out = []
        for f in list(fids)[:cap]:
            if not 0 <= f < nf:
                continue
            v = flat[offs[f]:offs[f + 1]]
            for i in range(1, len(v) - 1):
                out.extend(P[v[0]].tolist() + P[v[i]].tolist() + P[v[i + 1]].tolist())
        return [round(x, 5) for x in out]
    res = {"nonortho": tris(nof), "skew": tris(skf), "pyramids": tris(pyf)}
    if cells:
        owner = np.array(_read_label_list(os.path.join(pd, "owner")), dtype=np.int64)
        nei = np.array(_read_label_list(os.path.join(pd, "neighbour")), dtype=np.int64)
        cs = np.array(sorted(set(cells)), dtype=np.int64)
        mask = np.isin(owner, cs)
        if len(nei):
            mask[:len(nei)] |= np.isin(nei, cs)
        res["cells"] = tris(np.nonzero(mask)[0].tolist())
    else:
        res["cells"] = []
    return res


def _read_cell_centres(path):
    """Parse an OpenFOAM volVectorField internalField (ascii or binary)
    -> list of [x, y, z] cell centres (empty on failure)."""
    import re
    import numpy as np
    try:
        raw = open(path, "rb").read()
    except OSError:
        return []
    m = re.search(rb"internalField\s+nonuniform\s+List<vector>\s*(\d+)\s*\(", raw)
    if not m:
        return []
    n = int(m.group(1))
    off = m.end()
    head = raw[:m.start()]
    if b"format" in head and b"binary" in head:
        if off + 24 * n > len(raw):
            return []
        a = np.frombuffer(raw, dtype="<f8", count=3 * n, offset=off).reshape(n, 3)
        return [[float(x), float(y), float(z)] for x, y, z in a]
    body = raw[off:].decode("latin-1")
    try:
        body = body[:body.index(")")]
    except ValueError:
        pass
    v = [float(t) for t in body.split()]
    return [[v[3 * i], v[3 * i + 1], v[3 * i + 2]] for i in range(min(n, len(v) // 3))]


def _on_boundary(vi, d):
    """True if vertex id sits on any outer-domain plane (i/j/k at 0 or max).
    The blockMesh outer walls must stay flat, so these are never projected."""
    try:
        xs, ys, zs = d["x"], d["y"], d["z"]
        nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1
        i = vi % (nx + 1)
        j = (vi // (nx + 1)) % (ny + 1)
        k = vi // ((nx + 1) * (ny + 1))
        return i in (0, nx) or j in (0, ny) or k in (0, nz)
    except (KeyError, TypeError, ValueError):
        return False


def _dead_verts():
    """Vertex ids on any dead (solid) block, reconstructed from the current
    dict's live blocks plus the full grid. Returns None if unreadable."""
    try:
        with open(os.path.join(SYS, "blockMeshDict")) as _f:
            txt = _f.read()
        d = _dict_grid()
        xs, ys, zs = d["x"], d["y"], d["z"]
    except (OSError, KeyError, ValueError, TypeError):
        return None
    m = re.search(r"blocks\s*\(\s*(.*?)\n\)\s*;", txt, re.S)
    if not m:
        return None
    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(zs) - 1

    def vid(i, j, k):
        return i + (nx + 1) * (j + (ny + 1) * k)

    live = set()
    for h in re.findall(r"hex\s*\(([^()]+)\)", m.group(1)):
        try:
            v0 = int(h.split()[0])
        except (ValueError, IndexError):
            continue
        live.add((v0 % (nx + 1), (v0 // (nx + 1)) % (ny + 1),
                  v0 // ((nx + 1) * (ny + 1))))
    out = set()
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if (i, j, k) in live:
                    continue
                out.update([vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j + 1, k),
                            vid(i, j + 1, k), vid(i, j, k + 1), vid(i + 1, j, k + 1),
                            vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1)])
    return out


def _pin_vetoed(vi, d, dead):
    """A vertex projection is vetoed only if it sits on an outer-domain plane
    AND on no dead block. Outer walls over live (fluid) blocks must stay flat,
    but the car touches the ground boundary (wheel base) and those boundary
    vertices are shared with live blocks, so a boundary vertex on any dead
    block is allowed."""
    if vi is None or not _on_boundary(vi, d):
        return False
    if dead is None:
        return True
    return vi not in dead


def _veto_boundary_pins(pins, d, dead=None):
    """Drop {key:[x,y,z]} projections that land on an outer-domain plane with
    no dead block under them (see _pin_vetoed)."""
    if dead is None:
        dead = _dead_verts()
    out = {}
    for _k, _v in (pins or {}).items():
        try:
            _p = [float(c) for c in _v]
        except (TypeError, ValueError):
            continue
        if len(_p) != 3:
            continue
        if _pin_vetoed(_key_to_vid(_k, d)[0], d, dead):
            continue
        out[_k] = _p
    return out


def _backup_dict():
    """Keep the previous blockMeshDict as blockMeshDict.bak before any write."""
    try:
        dp = os.path.join(SYS, "blockMeshDict")
        if not os.path.exists(dp):
            return
        txt = open(dp).read()
        if "vertices" in txt and "blocks" in txt and txt.count("hex (") > 1:
            with open(dp + ".bak", "w") as f:
                f.write(txt)
    except OSError:
        pass


def _checkmesh_run():
    """Run checkMesh -writeSets on the current constant/polyMesh and return the
    result dict served by /api/checkmesh (counts, issues, problem locations)."""
    import re as _re
    code, log = run(["bash", "-lc",
                     "source /opt/openfoam12/etc/bashrc; "
                     f"cd {CASE}; rm -rf constant/polyMesh/sets; "
                     "checkMesh -writeSets 2>&1"], timeout=900)
    def _num(pat):
        m = _re.search(pat, log)
        return int(m.group(1)) if m else 0

    def _flt(pat):
        m = _re.search(pat, log)
        try:
            return float(m.group(1)) if m else None
        except ValueError:
            return None
    mesh_ok = (("Mesh OK" in log) and ("Failed" not in log)
               and "Error in face pyramids" not in log)
    setdir = os.path.join(CASE, "constant", "polyMesh", "sets")

    def _set(name):
        _q = os.path.join(setdir, name)
        return _read_cell_set(_q) if os.path.exists(_q) else []
    idx = _set("zeroVolumeCells")      # cells (negative / zero volume)
    nof = _set("nonOrthoFaces")        # faces (severely non-orthogonal)
    skf = _set("skewFaces")            # faces (highly skew)
    pyf = _set("wrongOrientedFaces")   # faces (error in face pyramids)
    cells, nonortho, skew, pyramids = [], [], [], []
    if idx or nof or skf or pyf:
        run(["bash", "-lc",
             "source /opt/openfoam12/etc/bashrc; "
             f"cd {CASE}; postProcess -func writeCellCentres 2>&1 "
             "| tail -n 2"], timeout=900)
        C = _read_cell_centres(os.path.join(CASE, "constant", "C"))
        cells = [{"i": int(i), "c": C[i]}
                 for i in idx if 0 <= i < len(C)]
        if nof or skf or pyf:
            owner = _read_label_list(os.path.join(
                CASE, "constant", "polyMesh", "owner"))

            def _faces(fl):
                # a face is shown at the centre of its owner cell
                return [{"i": int(f), "c": C[owner[f]]}
                        for f in fl
                        if 0 <= f < len(owner) and 0 <= owner[f] < len(C)]
            nonortho = _faces(nof)
            skew = _faces(skf)
            pyramids = _faces(pyf)
        for _f in ("C", "Ccx", "Ccy", "Ccz"):
            try:
                os.remove(os.path.join(CASE, "constant", _f))
            except OSError:
                pass
    tris, geom_error = {}, None
    if idx or nof or skf or pyf:
        try:
            tris = _problem_tris(idx, nof, skf, pyf)
        except Exception as e:
            geom_error = repr(e)
    n_cells = (_num(r"\b(\d+)\s+cells\b")
               or _num(r"(?m)^\s*cells:\s*(\d+)\s*$"))
    # Counts: trust checkMesh's own log line, fall back to the set size.
    n_bad = _num(r"negative volume cells:\s*(\d+)") or len(idx)
    n_nonortho = (_num(r"severely non-orthogonal[^\n]*:\s*(\d+)")
                  or len(nof))
    n_skew = _num(r"(\d+)\s+highly skew faces detected") or len(skf)
    n_pyr = (_num(r"(?i)error in face pyramids:\s*(\d+)") or len(pyf))
    issues = [
        {"key": "negvol", "label": "negative volume cells",
         "n": n_bad, "max": _flt(r"Minimum negative volume:\s*([-+.\deE]+)")},
        {"key": "nonortho", "label": "non-orthogonal faces (severe)",
         "n": n_nonortho,
         "max": _flt(r"Mesh non-orthogonality Max:\s*([-+.\deE]+)")},
        {"key": "skew", "label": "highly skew faces",
         "n": n_skew, "max": _flt(r"Max skewness\s*=\s*([-+.\deE]+)")},
        {"key": "pyramid", "label": "face pyramid errors",
         "n": n_pyr, "max": None},
    ]
    return ({"ok": True,
                       "log": "\n".join(log.strip().splitlines()[-30:]),
                       "mesh_ok": mesh_ok,
                       "n_cells": n_cells,
                       "n_bad": n_bad,
                       "n_severe_nonortho": n_nonortho,
                       "n_nonortho": n_nonortho,
                       "n_skew": n_skew,
                       "n_pyramid": n_pyr,
                       "issues": issues,
                       "cells": cells,
                       "nonortho": nonortho,
                       "skew": skew,
                       "pyramids": pyramids,
                       "tris": tris,
                       "geom_error": geom_error})


# ---------------------------------------------------------------------------
# Mesh optimiser: moves SURFACE vertices near checkMesh problem cells, keeping
# them on the surface, and keeps a move only if checkMesh actually improves.
# ---------------------------------------------------------------------------
_opt_mu = threading.Lock()
_opt = {"running": False, "cancel": False, "done": False, "log": [], "iter": 0,
        "max_iter": 0, "score": None, "base": None, "targets": [], "trial": "",
        "message": "", "result": None, "error": None, "cancelled": False}
_OPT_BLOCKED = ("/api/pinpatch", "/api/unpin", "/api/export", "/api/blockMesh",
                "/api/checkmesh", "/api/modify", "/api/saveover", "/api/savepins",
                "/api/interface", "/api/sdf/build")


def _opt_running():
    with _opt_mu:
        return _opt["running"]


def _opt_log(msg):
    with _opt_mu:
        _opt["log"].append(str(msg))


def _opt_set(**kw):
    with _opt_mu:
        _opt.update(kw)


def _fmt_num(v):
    t = repr(float(v))
    return t[:-2] if t.endswith(".0") else t


def _vid_key(vid, d):
    xs, ys, zs = d["x"], d["y"], d["z"]
    nx1, ny1 = len(xs), len(ys)
    return "%s|%s|%s" % (_fmt_num(xs[vid % nx1]), _fmt_num(ys[(vid // nx1) % ny1]),
                         _fmt_num(zs[vid // (nx1 * ny1)]))


def _read_dict_surface():
    """(verts Nx3, quads, neighbours{vid:set}) of the body+wheels wall."""
    import numpy as np
    txt = open(os.path.join(SYS, "blockMeshDict")).read()
    m = re.search(r"vertices\s*\(\s*(.*?)\)\s*;\s*blocks", txt, re.S)
    verts = np.array([[float(x) for x in ln.split()]
                      for ln in re.findall(r"\(([^()]+)\)", m.group(1))])
    quads = []
    for pn in ("body", "wheels"):
        m2 = re.search(pn + r"\s*\{\s*type\s+\w+;\s*faces\s*\(\s*(.*?)\)\s*;\s*\}",
                       txt, re.S)
        if m2:
            for ln in re.findall(r"\(([^()]+)\)", m2.group(1)):
                quads.append([int(x) for x in ln.split()])
    nbr = {}
    for q in quads:
        for a, b in zip(q, q[1:] + q[:1]):
            if a != b:
                nbr.setdefault(a, set()).add(b)
                nbr.setdefault(b, set()).add(a)
    return verts, quads, nbr


def _snapshot():
    sp = os.path.join(SYS, "splits.json")
    return (open(os.path.join(SYS, "blockMeshDict")).read(),
            open(sp).read() if os.path.exists(sp) else None)


def _restore(snap):
    with open(os.path.join(SYS, "blockMeshDict"), "w") as f:
        f.write(snap[0])
    if snap[1] is not None:
        with open(os.path.join(SYS, "splits.json"), "w") as f:
            f.write(snap[1])


def _write_vertices(moves):
    """Write {vid:[x,y,z]} into blockMeshDict and splits.json pins."""
    gd = _dict_grid()
    dp = os.path.join(SYS, "blockMeshDict")
    txt = open(dp).read()
    m = re.search(r"vertices\s*\(\s*(.*?)\)\s*;\s*blocks", txt, re.S)
    vlines = re.findall(r"\(([^()]+)\)", m.group(1))
    for vid, p in moves.items():
        if 0 <= vid < len(vlines):
            vlines[vid] = "%.6g %.6g %.6g" % tuple(float(c) for c in p)
    txt = (txt[:m.start(1)] + "\n" + "\n".join("    (%s)" % ln for ln in vlines)
           + "\n" + txt[m.end(1):])
    with open(dp, "w") as f:
        f.write(txt)
    sp = os.path.join(SYS, "splits.json")
    try:
        cur = json.load(open(sp))
    except (OSError, ValueError):
        cur = {}
    pins = cur.get("pins") if isinstance(cur.get("pins"), dict) else {}
    have = {}
    for k in pins:
        v = _key_to_vid(k, gd)[0]
        if v is not None:
            have[v] = k
    for vid, p in moves.items():
        pins.pop(have.get(vid, ""), None)
        pins[_vid_key(vid, gd)] = [float(c) for c in p]
    cur["pins"] = pins
    with open(sp, "w") as f:
        json.dump(cur, f)


def _run_blockmesh():
    code, log = run(["bash", "-lc", "source /opt/openfoam12/etc/bashrc; "
                     f"cd {CASE}; rm -rf constant/polyMesh; blockMesh 2>&1"],
                    timeout=900)
    ok = (code == 0) and ("FOAM FATAL" not in log) and ("FOAM exiting" not in log)
    return ok, log


def _mesh_and_check():
    ok, blog = _run_blockmesh()
    if not ok:
        return None, blog
    return _checkmesh_run(), blog


def _issue(r, key, field="n"):
    for it in r.get("issues", []):
        if it.get("key") == key:
            return it.get(field) or 0
    return 0


def _n_problems(r):
    return sum(_issue(r, k) for k in ("negvol", "pyramid", "skew", "nonortho"))


def _score(r):
    if r is None:
        return float("inf")
    return (1000 * _issue(r, "negvol") + 1000 * _issue(r, "pyramid")
            + 50 * _issue(r, "skew") + 10 * _issue(r, "nonortho")
            + 0.1 * _issue(r, "nonortho", "max") + 1.0 * _issue(r, "skew", "max"))


def _summ(r):
    if r is None:
        return "mesh generation failed"
    return ("%d negative-volume, %d face-pyramid, %d highly-skew, %d non-orthogonal"
            % (_issue(r, "negvol"), _issue(r, "pyramid"), _issue(r, "skew"),
               _issue(r, "nonortho")))


def _problem_points(r):
    out = []
    for key, w, kind in (("cells", 10, "negative-volume"), ("pyramids", 10, "pyramid-error"),
                         ("skew", 3, "skew"), ("nonortho", 1, "non-orthogonal")):
        for e in r.get(key) or []:
            c = e.get("c") if isinstance(e, dict) else None
            if c and len(c) == 3:
                out.append((c, w, kind))
    return out


def _propose(verts, ids, nbr, relax, target):
    """Laplacian step toward the neighbour mean, then back onto the surface."""
    import numpy as np
    P = verts[ids]
    new = np.empty_like(P)
    lv = np.empty(len(ids))
    for n, v in enumerate(ids):
        nb = verts[list(nbr[v])]
        lv[n] = np.linalg.norm(nb - P[n], axis=1).mean()
        new[n] = P[n] + relax * (nb.mean(axis=0) - P[n])
    snapped, _ = snap_points(new, target)
    step = snapped - P
    d = np.linalg.norm(step, axis=1)
    cap = 0.5 * lv                              # never move more than half an edge
    over = d > cap
    if over.any():
        clipped = P[over] + step[over] * (cap[over] / np.maximum(d[over], 1e-12))[:, None]
        snapped[over], _ = snap_points(clipped, target)
    return snapped, np.linalg.norm(snapped - P, axis=1)


def _opt_job(params):
    import numpy as np
    max_iter = max(1, min(30, int(params.get("max_iter", 6))))
    max_verts = max(5, min(2000, int(params.get("max_verts", 300))))
    target = "stl" if str(params.get("target", "smooth")) == "stl" else "smooth"
    if target == "smooth" and not os.path.exists(os.path.join(SYS, "car_sdf_surface.npz")):
        target = "stl"
        _opt_log("smooth surface mesh not built yet - snapping to the STL instead")
    _opt_set(max_iter=max_iter)
    gd = _dict_grid()
    if gd is None:
        raise RuntimeError("no blockMeshDict / grid - write the blockMeshDict first")
    dead = _dead_verts()
    verts, quads, nbr = _read_dict_surface()
    if not nbr:
        raise RuntimeError("no body/wheels surface faces in blockMeshDict")
    surf = np.array(sorted(nbr))
    fixed = [v for v in surf if _pin_vetoed(int(v), gd, dead)]
    fixed_set = set(int(v) for v in fixed)
    edges = {(min(a, b), max(a, b)) for a in nbr for b in nbr[a]}
    # Vertices on the rim of the wall patch only slide ALONG the rim (so the
    # car outline does not shrink); interior vertices use all their neighbours.
    ecount = {}
    for q in quads:
        for a, b in zip(q, q[1:] + q[:1]):
            e = (min(a, b), max(a, b))
            ecount[e] = ecount.get(e, 0) + 1
    bnbr = {}
    for (a, b), c in ecount.items():
        if c == 1:
            bnbr.setdefault(a, set()).add(b)
            bnbr.setdefault(b, set()).add(a)
    nbr_use = {v: (bnbr[v] if v in bnbr else nbr[v]) for v in nbr}
    L = float(np.mean([np.linalg.norm(verts[a] - verts[b]) for a, b in edges]))
    # keep a safe copy of the starting point
    try:
        import shutil
        shutil.copy(os.path.join(SYS, "blockMeshDict"),
                    os.path.join(SYS, "blockMeshDict.before_optimise"))
        sp = os.path.join(SYS, "splits.json")
        if os.path.exists(sp):
            shutil.copy(sp, sp + ".before_optimise")
    except OSError:
        pass
    _opt_log("Optimiser: %d surface vertices (%d on outer boundary, kept fixed), "
             "mean surface edge %.1f mm, snapping to the %s surface"
             % (len(surf), len(fixed_set), L * 1000, "smooth" if target == "smooth" else "STL"))
    _opt_log("Baseline: blockMesh + checkMesh ...")
    _opt_set(message="baseline blockMesh + checkMesh", trial="baseline")
    best_r, blog = _mesh_and_check()
    if best_r is None:
        raise RuntimeError("blockMesh failed: " + " | ".join(blog.strip().splitlines()[-3:]))
    best_s = _score(best_r)
    _opt_set(score=best_s, base=best_s, result=best_r)
    _opt_log("Baseline: " + _summ(best_r))
    snap_best = _snapshot()
    mesh_is_best = True
    relax = 0.5
    R = 2.5 * L

    for it in range(1, max_iter + 1):
        with _opt_mu:
            if _opt["cancel"]:
                break
        if _n_problems(best_r) == 0:
            _opt_log("No checkMesh problems left - done.")
            break
        _opt_set(iter=it)
        probs = sorted(_problem_points(best_r), key=lambda t: -t[1])[:3000]
        PS = verts[surf]
        wt = np.zeros(len(surf))
        kinds = [dict() for _ in surf]
        far = 0
        for c, w, kind in probs:
            d = np.linalg.norm(PS - np.asarray(c), axis=1)
            k = min(4, len(d) - 1)
            for n in np.argpartition(d, k)[:4] if len(d) > 4 else range(len(d)):
                if d[n] <= R:
                    wt[n] += w / (1.0 + d[n] / L)
                    kinds[n][kind] = kinds[n].get(kind, 0) + 1
            if d.min() > R:
                far += 1
        order = [int(n) for n in np.argsort(-wt)
                 if wt[n] > 0 and int(surf[n]) not in fixed_set][:max_verts]
        if far:
            _opt_log("  %d problem location(s) are too far from the surface to be "
                     "fixed by surface vertices" % far)
        if not order:
            _opt_log("Iteration %d: no movable surface vertices near the problems - stopping." % it)
            break
        ids = [int(surf[n]) for n in order]
        _opt_log("Iteration %d/%d: %d surface vertices sit next to problem cells"
                 % (it, max_iter, len(ids)))
        improved = False
        for a in (relax, relax / 2.0, relax / 4.0):
            with _opt_mu:
                if _opt["cancel"]:
                    break
            new, moved = _propose(verts, ids, nbr_use, a, target)
            tgt = [{"vid": v, "from": verts[v].tolist(), "to": new[n].tolist()}
                   for n, v in enumerate(ids)]
            _opt_set(targets=tgt, trial="it%d-a%.3f" % (it, a),
                     message="iteration %d/%d, relax %.2f: %d vertices" % (it, max_iter, a, len(ids)))
            for n, v in list(enumerate(ids))[:25]:
                near = ", ".join("%d %s" % (c, k) for k, c in sorted(kinds[order[n]].items()))
                _opt_log("  fixing vertex %d (grid %s): near %s -> move %.2f mm"
                         % (v, _vid_key(v, gd), near, moved[n] * 1000))
            if len(ids) > 25:
                _opt_log("  ... and %d more vertices" % (len(ids) - 25))
            _write_vertices({v: new[n] for n, v in enumerate(ids)})
            _opt_log("  trial relax %.2f: blockMesh + checkMesh ..." % a)
            r, blog = _mesh_and_check()
            s = _score(r)
            if r is not None and s < best_s - 1e-9:
                _opt_log("  trial relax %.2f: %s  (score %.1f -> %.1f) ACCEPTED"
                         % (a, _summ(r), best_s, s))
                for n, v in enumerate(ids):
                    verts[v] = new[n]
                best_r, best_s = r, s
                snap_best = _snapshot()
                mesh_is_best = True
                relax = a
                improved = True
                _opt_set(score=best_s, result=best_r)
                break
            _opt_log("  trial relax %.2f: %s  (score %s) rejected, reverted"
                     % (a, _summ(r), "n/a" if r is None else "%.1f" % s))
            _restore(snap_best)
            mesh_is_best = False
        with _opt_mu:
            cancelled = _opt["cancel"]
        if cancelled:
            break
        if not improved:
            _opt_log("No further improvement from moving these vertices - stopping.")
            break
    with _opt_mu:
        was_cancelled = _opt["cancel"]
    if not mesh_is_best:
        _opt_log("Restoring best mesh (blockMesh) ...")
        _opt_set(message="restoring best mesh", targets=[])
        _restore(snap_best)
        _run_blockmesh()
    base = _opt["base"]
    _opt_log(("Stopped by user. " if was_cancelled else "Finished. ")
             + "Result: " + _summ(best_r) + "  (score %.1f -> %.1f)" % (base, best_s))
    _opt_set(result=best_r, cancelled=was_cancelled)


def _opt_thread(params):
    import traceback as _tb
    try:
        _opt_job(params)
    except Exception as e:
        _opt_set(error=str(e))
        _opt_log("ERROR: %s" % e)
        try:
            with open("/tmp/optimise_err.log", "a") as f:
                f.write(_tb.format_exc() + "\n")
        except OSError:
            pass
    finally:
        _opt_set(running=False, done=True, targets=[], trial="")


def _opt_start(params):
    with _opt_mu:
        if _opt["running"]:
            return False
        _opt.update({"running": True, "cancel": False, "done": False, "log": [],
                     "iter": 0, "score": None, "base": None, "targets": [],
                     "trial": "", "message": "starting", "result": None,
                     "error": None, "cancelled": False})
    threading.Thread(target=_opt_thread, args=(params,), daemon=True).start()
    return True



def _same(a, b):
    try:
        a = [float(v) for v in (a or [])]
        b = [float(v) for v in (b or [])]
    except (TypeError, ValueError):
        return False
    return len(a) == len(b) and all(abs(x - y) < 1e-9 for x, y in zip(a, b))


def _infer_grid(nverts):
    """(nx+1)(ny+1)(nz+1) == nverts -> axis plane counts, closest to cubic."""
    import math
    n = int(round(nverts ** (1.0 / 3.0)))
    for cand in range(max(2, n - 6), n + 7):
        if nverts % cand:
            continue
        rest = nverts // cand
        m = int(round(rest ** 0.5))
        for m2 in range(max(2, m - 6), m + 7):
            if rest % m2 == 0:
                a = [cand, m2, rest // m2]
                a.sort()
                return ([[0.0] * a[0], [0.0] * a[1], [0.0] * a[2]])
    return [None, None, None]


class H(BaseHTTPRequestHandler):
    server_version = "blockMesh GUI/1.0"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", name=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if name:
            self.send_header("Content-Disposition",
                             f"attachment; filename={name}")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Browsers can close a request while the response is being sent
            # (for example during navigation or a reload). Treat that as a
            # disconnected client rather than an unhandled server exception.
            pass

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
        except (TypeError, ValueError):
            n = 0
        raw = self.rfile.read(n) if n > 0 else b""
        try:
            return json.loads(raw.decode() or "{}")
        except (ValueError, UnicodeDecodeError):
            return None

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            with open(os.path.join(GUI, "index.html"), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")
        elif u.path.startswith("/vendor/"):
            name = os.path.basename(u.path)
            if name not in ("three.min.js", "OrbitControls.js"):
                return self._json({"error": "not found"}, 404)
            with open(os.path.join(GUI, "vendor", name), "rb") as f:
                self._send(200, f.read(), "application/javascript")
        elif u.path == "/api/sdfsurface":
            import numpy as np
            _path = os.path.join(SYS, "car_sdf_surface.npz")
            try:
                with _lock:
                    if "mesh" not in _sdf_surface_cache:
                        with np.load(_path) as _m:
                            _sdf_surface_cache["mesh"] = {
                                "positions": _m["positions"].reshape(-1).tolist(),
                                "index": _m["faces"].reshape(-1).tolist()}
                    mesh = _sdf_surface_cache["mesh"]
                self._json(mesh)
            except (OSError, ValueError, KeyError) as e:
                self._json({"error": f"distance field surface unavailable: {e}"}, 404)
        elif u.path == "/api/stl":
            import numpy as np
            q = parse_qs(u.query)
            try:
                stride = max(1, min(24, int(q.get("stride", ["6"])[0])))
            except ValueError:
                stride = 6
            T0, T1, T2 = load_stl()
            sel = np.arange(0, len(T0), stride)
            pos = np.concatenate([T0[sel], T1[sel], T2[sel]], axis=0)
            m = len(sel)
            idx = np.concatenate(
                [np.arange(m)[:, None], np.arange(m)[:, None] + m,
                 np.arange(m)[:, None] + 2 * m], axis=1)
            vmin = [float(v) for v in pos.min(axis=0)]
            vmax = [float(v) for v in pos.max(axis=0)]
            self._json({"positions": pos.ravel().tolist(),
                        "index": idx.ravel().tolist(),
                        "tris": int(len(T0)), "shown": int(m),
                        "bounds": [vmin, vmax]})
        elif u.path == "/api/tables":
            self._json({"roof": ROOF, "halfw": HALFW,
                        "under": UNDER, "ground": GROUND, "belt": BELT,
                        "wheels": {"fx": [FX0, FX1], "rx": [RX0, RX1],
                                   "zin": WZIN, "zout": WZOUT, "top": WTOP},
                        "tuned": TUNED, "domain": DOMAIN})
        elif u.path == "/api/dict":
            with open(os.path.join(SYS, "blockMeshDict"), "rb") as f:
                self._send(200, f.read(), "text/plain", name="blockMeshDict")
        elif u.path == "/api/dictwall":
            # The ACTUAL wall as written in system/blockMeshDict: snapped
            # vertices + body/wheels faces + curved polyLine edges. This is
            # what the mesh is built from — not an ideal projection.
            try:
                txt = open(os.path.join(SYS, "blockMeshDict")).read()
            except OSError as e:
                return self._json({"error": f"no blockMeshDict: {e}"}, 404)
            try:
                import re as _re
                m = _re.search(r"vertices\s*\(\s*(.*?)\)\s*;\s*blocks", txt, _re.S)
                verts = [[float(x) for x in ln.split()]
                         for ln in _re.findall(r"\(([^()]+)\)", m.group(1))]
                quads = []
                fidx = []
                for _pn in ("body", "wheels"):
                    m2 = _re.search(_pn + r"\s*\{\s*type\s+\w+;\s*faces\s*\(\s*(.*?)\)\s*;\s*\}",
                                    txt, _re.S)
                    if m2:
                        for ln in _re.findall(r"\(([^()]+)\)", m2.group(1)):
                            idx = [int(x) for x in ln.split()]
                            quads.append([verts[q] for q in idx])
                            fidx.append(idx)
                curves = []
                curve_ids = []
                m3 = _re.search(r"edges\s*\(\s*(.*?)\)\s*;\s*boundary", txt, _re.S)
                if m3:
                    for em in _re.finditer(r"polyLine\s+(\d+)\s+(\d+)\s*\((.*?)\)(?=\s*polyLine|\s*$)",
                                           m3.group(1), _re.S):
                        a, b = int(em.group(1)), int(em.group(2))
                        mid = [[float(x) for x in ln.split()]
                               for ln in _re.findall(r"\(([^()]+)\)", em.group(3))]
                        curves.append([verts[a]] + mid + [verts[b]])
                        curve_ids.append([a, b])
            except (ValueError, AttributeError) as e:
                return self._json({"error": f"dict parse failed: {e}"}, 500)

            # The grid the dict was ACTUALLY built from (stamped // SPLITS:).
            # Clients must decode vertex ids with THESE, never with splits.json
            # — a stale dict vs. splits.json stride mismatch silently remapped
            # every projection onto the wrong vertex.
            gxs, gys, gzs = None, None, None
            try:
                with open(os.path.join(SYS, "blockMeshDict")) as _f:
                    for _line in _f:
                        if _line.startswith("// SPLITS:"):
                            _sp = json.loads(_line[len("// SPLITS:"):])
                            gxs = [float(v) for v in _sp.get("x", [])]
                            gys = [float(v) for v in _sp.get("y", [])]
                            gzs = [float(v) for v in _sp.get("z", [])]
                            break
            except (OSError, ValueError, TypeError):
                pass
            if not (gxs and gys and gzs):
                # no stamp: infer from vertex count
                gxs, gys, gzs = _infer_grid(len(verts))

            drift = False
            try:
                _sd = json.load(open(os.path.join(SYS, "splits.json")))
                drift = not (_same(_sd.get("x"), gxs) and _same(_sd.get("y"), gys)
                             and _same(_sd.get("z"), gzs))
            except (OSError, ValueError, TypeError):
                pass

            out = {"quads": quads, "curves": curves, "curveIds": curve_ids, "verts": verts,
                   "fidx": fidx, "x": gxs, "y": gys, "z": gzs,
                   "nx": len(gxs) - 1, "ny": len(gys) - 1, "nz": len(gzs) - 1,
                   "drift": drift}
            if drift:
                out["drift_msg"] = ("dict grid %dx%dx%d != splits.json — "
                                    "regenerate (Export) to resync"
                                    % (len(gxs) - 1, len(gys) - 1, len(gzs) - 1))
            return self._json(out)
        elif u.path == "/api/optimise/status":
            try:
                since = int(parse_qs(u.query).get("since", ["0"])[0])
            except (TypeError, ValueError):
                since = 0
            with _opt_mu:
                snap = dict(_opt)
                lines = list(_opt["log"][max(0, since):])
                n = len(_opt["log"])
            self._json({"running": snap["running"], "done": snap["done"],
                        "log": lines, "n": n, "iter": snap["iter"],
                        "max_iter": snap["max_iter"], "score": snap["score"],
                        "base": snap["base"], "targets": snap["targets"],
                        "trial": snap["trial"], "message": snap["message"],
                        "error": snap["error"], "cancelled": snap["cancelled"],
                        "result": snap["result"] if not snap["running"] else None})
        elif u.path == "/api/read_dict":
            # Splits that produced system/blockMeshDict: preferred source is
            # the SPLITS comment stamped by the generator; fallback is
            # system/splits.json (written by the GUI on every export).
            try:
                found = None
                with open(os.path.join(SYS, "blockMeshDict")) as f:
                    for line in f:
                        if line.startswith("// SPLITS:"):
                            found = json.loads(line[len("// SPLITS:"):])
                            break
                if found and all(k in found for k in ("x", "y", "z")):
                    try:
                        _sd = json.load(open(os.path.join(SYS, "splits.json")))
                        _pins = _sd.get("pins", {})
                        _solid = _sd.get("solid", [])
                        _fluid = _sd.get("fluid", [])
                    except (OSError, ValueError):
                        _pins, _solid, _fluid = {}, [], []
                    return self._json({"x": [float(v) for v in found["x"]],
                                       "y": [float(v) for v in found["y"]],
                                       "z": [float(v) for v in found["z"]],
                                       "pins": _pins if isinstance(_pins, dict) else {},
                                       "solid": _solid if isinstance(_solid, list) else [],
                                       "fluid": _fluid if isinstance(_fluid, list) else [],
                                       "source": "blockMeshDict"})
                with open(os.path.join(SYS, "splits.json")) as f:
                    d = json.load(f)
                return self._json({"x": [float(v) for v in d["x"]],
                                   "y": [float(v) for v in d["y"]],
                                   "z": [float(v) for v in d["z"]],
                                   "pins": d.get("pins", {}) if isinstance(d.get("pins"), dict) else {},
                                   "solid": d.get("solid", []) if isinstance(d.get("solid"), list) else [],
                                   "fluid": d.get("fluid", []) if isinstance(d.get("fluid"), list) else [],
                                   "source": "splits.json"})
            except (OSError, ValueError, KeyError, TypeError) as e:
                return self._json({"error": f"no readable splits: {e}"}, 404)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        u = urlparse(self.path)
        data = self._body()
        if data is None:
            return self._json({"error": "bad JSON"}, 400)
        if u.path == "/api/optimise/start":
            if not _opt_start(data if isinstance(data, dict) else {}):
                return self._json({"ok": False, "error": "optimiser already running"}, 409)
            return self._json({"ok": True})
        if u.path == "/api/optimise/stop":
            _opt_set(cancel=True)
            return self._json({"ok": True})
        if u.path in _OPT_BLOCKED and _opt_running():
            return self._json({"error": "the mesh optimiser is running - stop it first"}, 409)
        if u.path == "/api/sdf/build":
            global SDF
            try:
                spacing_cm = float(data["spacing_cm"])
                if not 1.0 <= spacing_cm <= 10.0:
                    raise ValueError("mesh size must be between 1 and 10 cm")
            except (KeyError, TypeError, ValueError) as e:
                return self._json({"error": f"invalid distance-field mesh size: {e}"}, 400)
            spacing_m = spacing_cm / 100.0
            try:
                with _lock:
                    _build = subprocess.run([sys.executable, SDF_BUILDER, f"{spacing_m:.6f}"],
                                            cwd=CASE, capture_output=True, text=True, timeout=300)
                    if _build.returncode != 0:
                        raise RuntimeError(_build.stderr[-3000:] or _build.stdout[-3000:] or "distance-field build failed")
                    _surface = subprocess.run([sys.executable, SDF_EXTRACTOR],
                                              cwd=CASE, capture_output=True, text=True, timeout=300)
                    if _surface.returncode != 0:
                        raise RuntimeError(_surface.stderr[-3000:] or _surface.stdout[-3000:] or "zero-surface extraction failed")
                    SDF = reload_field(SDF_PATH)
                    _sdf_surface_cache.clear()
                    _snap_cache.pop("smooth", None)
                with open(os.path.join(SYS, "car_sdf.json"), encoding="utf-8") as _f:
                    _meta = json.load(_f)
                return self._json({"ok": True, "spacing_cm": spacing_cm,
                                   "shape_xyz": _meta.get("shape_xyz"),
                                   "build_log": _build.stdout.strip()})
            except (OSError, subprocess.TimeoutExpired, RuntimeError, ValueError) as e:
                return self._json({"error": f"distance-field build failed: {e}"}, 500)
        if u.path == "/api/project":
            try:
                import numpy as np
                P = np.asarray(data["points"], dtype=float)
                if P.ndim != 2 or P.shape[1] != 3:
                    raise ValueError("points must be Nx3")
                with _lock:
                    out = SDF.project(P)
                D = np.linalg.norm(out - P, axis=1)
            except (KeyError, TypeError, ValueError) as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            return self._json({"points": out.tolist(),
                               "stats": {"n": int(len(D)),
                                         "mean_mm": float(D.mean()*1000) if len(D) else 0.0,
                                         "p90_mm": float(np.percentile(D,90)*1000) if len(D) else 0.0,
                                         "max_mm": float(D.max()*1000) if len(D) else 0.0}})
        if u.path == "/api/projectmesh":
            try:
                import numpy as np
                P = np.asarray(data["points"], dtype=float)
                if P.ndim != 2 or P.shape[1] != 3:
                    raise ValueError("points must be Nx3")
                with _lock:
                    out = SDF.project(P)
                D = np.linalg.norm(out - P, axis=1)
            except (KeyError, TypeError, ValueError) as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            return self._json({"points": out.tolist(),
                               "stats": {"n": int(len(D)),
                                         "mean_mm": float(D.mean()*1000) if len(D) else 0.0,
                                         "max_mm": float(D.max()*1000) if len(D) else 0.0}})
        if u.path == "/api/export":
            try:
                xs = sorted(set(float(v) for v in data["x"]))
                ys = sorted(set(float(v) for v in data["y"]))
                zs = sorted(set(float(v) for v in data["z"]))
                solid_ov = _clean_triples(data.get("solid", []))
                fluid_ov = _clean_triples(data.get("fluid", []))
            except (KeyError, TypeError, ValueError):
                return self._json({"error": "x/y/z float lists required"}, 400)
            for vals, lo, hi, nm in ((xs, -50, 50, "x"), (ys, -50, 50, "y"),
                                     (zs, -50, 50, "z")):
                if len(vals) < 2:
                    return self._json({"error": f"{nm}: need >= 2 splits"}, 400)
            try:
                _old = json.load(open(os.path.join(SYS, "splits.json")))
                if not isinstance(_old, dict):
                    _old = {}
            except (OSError, ValueError):
                _old = {}
            payload = {"x": xs, "y": ys, "z": zs,
                       "solid": solid_ov, "fluid": fluid_ov}
            for _k, _vals in (("x_lines", xs), ("y_lines", ys), ("z_lines", zs)):
                _ax = _k[0]
                try:
                    _stored = _old.get(_k)
                    if _stored:
                        _lc = [float(c) for c, n, g in _stored]
                        if len(_lc) == len(_vals) and all(abs(a - b) < 1e-9 for a, b in zip(_lc, _vals)):
                            payload[_k] = _stored
                except (TypeError, ValueError, KeyError):
                    pass
            if "snap_all" in _old:
                payload["snap_all"] = _old["snap_all"]
            if "snap_all" in data:
                payload["snap_all"] = data["snap_all"]
            if "flat" in _old:
                payload["flat"] = _old["flat"]
            if "flat" in data:
                payload["flat"] = data["flat"]
            for _k in ("pins",):
                if _k in data and isinstance(data[_k], dict):
                    payload[_k] = _veto_boundary_pins(data[_k], _old)
                elif _k in _old:
                    payload[_k] = _veto_boundary_pins(_old[_k], _old)
            with open(os.path.join(SYS, "splits.json"), "w") as f:
                json.dump(payload, f)
            code, log = run([sys.executable, GENERATOR], timeout=600)
            tail = "\n".join(log.strip().splitlines()[-14:])
            if code != 0:
                return self._json({"ok": False, "log": tail})
            out = {"ok": True, "log": tail}
            if data.get("runMesh"):
                _, mlog = run(["bash", "-lc",
                               "source /opt/openfoam12/etc/bashrc; "
                               f"cd {CASE}; rm -rf constant/polyMesh; "
                               "blockMesh 2>&1 | tail -n 18"], timeout=600)
                out["mesh_log"] = mlog.strip()
            if data.get("runCheck"):
                _, clog = run(["bash", "-lc",
                               "source /opt/openfoam12/etc/bashrc; "
                               f"cd {CASE}; checkMesh 2>&1 | tail -n 30"],
                              timeout=600)
                out["check_log"] = clog.strip()
            return self._json(out)
        if u.path == "/api/blockMesh":
            # Run blockMesh only (same invocation as checkMesh / export runMesh).
            import traceback as _tb
            try:
                code, log = run(["bash", "-lc",
                                 "source /opt/openfoam12/etc/bashrc; "
                                 f"cd {CASE}; rm -rf constant/polyMesh; "
                                 "blockMesh 2>&1"], timeout=900)
                ok = (code == 0) and ("FOAM FATAL" not in log) \
                    and ("FOAM exiting" not in log)
                return self._json({"ok": ok,
                                   "log": "\n".join(log.strip().splitlines()[-30:])})
            except Exception as e:
                return self._json({"ok": False, "error": repr(e),
                                   "trace": _tb.format_exc()}, 500)
        if u.path == "/api/checkmesh":
            # Run checkMesh, collect the problem sets it writes (-writeSets) and
            # return their locations so the GUI can show them.
            import traceback as _tb
            try:
                return self._json(_checkmesh_run())
            except Exception as e:
                try:
                    with open("/tmp/checkmesh_err.log", "a") as _f:
                        _f.write(_tb.format_exc() + "\n")
                except OSError:
                    pass
                return self._json({"ok": False, "error": repr(e),
                                   "trace": _tb.format_exc()}, 500)
        if u.path == "/api/pinpatch":
            # Fast path: only vertex projections changed. Move those dict
            # vertices in place (no full regenerate) and persist to splits.json.
            try:
                _pins = data.get("pins", {})
                if not isinstance(_pins, dict):
                    raise ValueError
                _dead = _dead_verts()
                _gd = _dict_grid()
                if _gd is None:
                    raise ValueError("no grid")
                _clean = {}
                _vetoed = []
                for _k, _v in _pins.items():
                    _p = [float(c) for c in _v]
                    if len(_p) != 3:
                        continue
                    _vi, _ = _key_to_vid(_k, _gd)
                    if _vi is None:
                        continue
                    if _pin_vetoed(_vi, _gd, _dead):
                        _vetoed.append(str(_k))
                        continue
                    _clean[str(_vi)] = _p
            except (TypeError, ValueError, KeyError, OSError):
                return self._json({"error": "vertex projections {key:[x,y,z]} required"}, 400)
            try:
                _dp = os.path.join(SYS, "blockMeshDict")
                txt = open(_dp).read()
                import re as _re
                m = _re.search(r"vertices\s*\(\s*(.*?)\)\s*;\s*blocks", txt, _re.S)
                vlines = _re.findall(r"\(([^()]+)\)", m.group(1))
            except (OSError, AttributeError) as e:
                return self._json({"error": f"dict read failed: {e}"}, 500)
            _n = 0
            for _k, _p in _clean.items():
                _vi = int(_k)
                if 0 <= _vi < len(vlines) and len(_p) == 3 \
                        and all(abs(c) < 1e6 for c in _p):
                    vlines[_vi] = "%.6g %.6g %.6g" % tuple(_p)
                    _n += 1
            try:
                _sp = os.path.join(SYS, "splits.json")
                cur = json.load(open(_sp))
                cur["pins"] = _veto_boundary_pins(_pins, _gd, _dead)
                with open(_sp, "w") as f:
                    json.dump(cur, f)
                _backup_dict()
                txt = txt[:m.start(1)] + "\n" + "\n".join(
                    "    (%s)" % ln for ln in vlines) + "\n" \
                    + txt[m.end(1):]
                with open(_dp, "w") as f:
                    f.write(txt)
            except (OSError, ValueError) as e:
                return self._json({"error": f"write failed: {e}"}, 500)
            return self._json({"ok": True, "patched": _n, "vetoed": _vetoed})
        if u.path == "/api/saveover":
            # Persist staged DEAD/LIVE block overrides into splits.json without
            # regenerating blockMeshDict, so a reload restores them.
            try:
                _solid = _clean_triples(data.get("solid", []))
                _fluid = _clean_triples(data.get("fluid", []))
            except (TypeError, ValueError):
                return self._json({"error": "solid/fluid must be [i,j,k] lists"}, 400)
            try:
                _p = os.path.join(SYS, "splits.json")
                cur = json.load(open(_p))
            except (OSError, ValueError) as e:
                return self._json({"error": f"no splits.json: {e}"}, 400)
            if _solid:
                cur["solid"] = _solid
            else:
                cur.pop("solid", None)
            if _fluid:
                cur["fluid"] = _fluid
            else:
                cur.pop("fluid", None)
            with open(_p, "w") as f:
                json.dump(cur, f)
            return self._json({"ok": True, "n_solid": len(_solid), "n_fluid": len(_fluid)})
        if u.path == "/api/savepins":
            try:
                _pins = data.get("pins", {})
                if not isinstance(_pins, dict):
                    raise ValueError
            except (TypeError, ValueError, AttributeError):
                return self._json({"error": "vertex projections object required"}, 400)
            try:
                _p = os.path.join(SYS, "splits.json")
                cur = json.load(open(_p))
            except (OSError, ValueError) as e:
                return self._json({"error": f"no splits.json: {e}"}, 400)
            _dead = _dead_verts()
            cur["pins"] = {str(_k): [float(c) for c in _v]
                           for _k, _v in _pins.items()
                           if isinstance(_v, (list, tuple)) and len(_v) == 3
                           and not _pin_vetoed(_key_to_vid(_k, cur)[0], cur, _dead)}
            with open(_p, "w") as f:
                json.dump(cur, f)
            return self._json({"ok": True, "n": len(cur["pins"])})
        if u.path == "/api/unpin":
            _key = str(data.get("key", "") or "")
            try:
                _p = os.path.join(SYS, "splits.json")
                cur = json.load(open(_p))
            except (OSError, ValueError) as e:
                return self._json({"error": f"no splits.json: {e}"}, 400)
            _gd = _dict_grid()
            if _gd is None:
                return self._json({"error": "no grid (blockMeshDict / splits.json)"}, 400)
            _vi = None
            try:
                if data.get("vid") is not None:
                    _vi = int(data["vid"])
            except (TypeError, ValueError):
                _vi = None
            if _vi is None and _key:
                _vi, _ = _key_to_vid(_key, _gd)
            if _vi is None:
                return self._json({"error": "unknown vertex"}, 400)
            # drop every pin that resolves to this grid vertex
            _pins = cur.get("pins")
            if isinstance(_pins, dict):
                for _k in list(_pins):
                    if _k == _key or _key_to_vid(_k, _gd)[0] == _vi:
                        del _pins[_k]
                if not _pins:
                    del cur["pins"]
                with open(_p, "w") as f:
                    json.dump(cur, f)
            # Put the dict vertex back on its un-projected corner: the raw grid
            # point run through the terrain deform (no pin, no projection).
            try:
                _xs, _ys, _zs = _gd["x"], _gd["y"], _gd["z"]
                _nx1, _ny1 = len(_xs), len(_ys)
                if not 0 <= _vi < _nx1 * _ny1 * len(_zs):
                    raise IndexError(_vi)
                _i = _vi % _nx1
                _j = (_vi // _nx1) % _ny1
                _k3 = _vi // (_nx1 * _ny1)
                _y2, _z2 = _deform(float(_xs[_i]), float(_ys[_j]), float(_zs[_k3]),
                                   bool(cur.get("flat", False)))
                _base = [float(_xs[_i]), float(_y2), float(_z2)]
            except (KeyError, IndexError, TypeError, ValueError) as e:
                return self._json({"error": f"corner lookup failed: {e!r}"}, 500)
            try:
                _dp = os.path.join(SYS, "blockMeshDict")
                txt = open(_dp).read()
                m = re.search(r"vertices\s*\(\s*(.*?)\)\s*;\s*blocks", txt, re.S)
                vlines = re.findall(r"\(([^()]+)\)", m.group(1))
                if not 0 <= _vi < len(vlines):
                    raise IndexError(_vi)
                vlines[_vi] = "%.6g %.6g %.6g" % tuple(_base)
                _backup_dict()
                txt = txt[:m.start(1)] + "\n" + "\n".join(
                    "    (%s)" % ln for ln in vlines) + "\n" \
                    + txt[m.end(1):]
                with open(_dp, "w") as f:
                    f.write(txt)
            except (OSError, AttributeError, IndexError, ValueError) as e:
                return self._json({"error": f"blockMeshDict write failed: {e!r}"}, 500)
            return self._json({"ok": True, "vid": _vi, "point": _base})
        if u.path == "/api/snap":
            # Nearest point on a surface: target "stl" (car STL) or "smooth"
            # (smooth distance-field mesh). {points:[[x,y,z],...], target}
            try:
                import numpy as np
                P = np.asarray(data["points"], dtype=float)
                if P.ndim != 2 or P.shape[1] != 3 or not len(P) or len(P) > 5000:
                    raise ValueError("points must be Nx3 (1..5000)")
                if not np.isfinite(P).all():
                    raise ValueError("points must be finite")
                target = str(data.get("target", "stl"))
                if target not in ("stl", "smooth"):
                    raise ValueError("target must be 'stl' or 'smooth'")
            except (KeyError, TypeError, ValueError) as e:
                return self._json({"error": f"bad request: {e}"}, 400)
            try:
                out, dist = snap_points(P, target)
            except (OSError, KeyError, ValueError) as e:
                return self._json({"error": f"{target} surface unavailable: {e}"}, 404)
            except Exception as e:
                return self._json({"error": f"snap failed: {e!r}"}, 500)
            return self._json({"points": out.tolist(),
                               "dist_mm": [float(d) * 1000.0 for d in dist],
                               "target": target})
        if u.path == "/api/modify":            # Per-block cell counts: {block:[i,j,k], nc:[nx|0,ny|0,nz|0]}
            # (0/null = keep). Persists full *_lines into splits.json, regenerates.
            try:
                _bi = [int(v) for v in data["block"]]
                _nc = [(0 if v in (None, "") else int(v)) for v in data["nc"]]
                if len(_bi) != 3 or len(_nc) != 3:
                    raise ValueError
                for _n in _nc:
                    if _n < 0 or _n > 200:
                        raise ValueError
            except (KeyError, TypeError, ValueError):
                return self._json({"error": "block:[i,j,k] nc:[nx,ny,nz] (0=keep, 1..200)"}, 400)
            try:
                _p = os.path.join(SYS, "splits.json")
                cur = json.load(open(_p))
                _bx, _by, _bz = cur["x"], cur["y"], cur["z"]
            except (OSError, ValueError, KeyError) as e:
                return self._json({"error": f"no splits.json: {e}"}, 400)
            _lx, _ly, _lz = (_resolve_lines(_bx, "x"), _resolve_lines(_by, "y"),
                             _resolve_lines(_bz, "z"))
            _nx, _ny, _nz = len(_lx) - 1, len(_ly) - 1, len(_lz) - 1
            _i, _j, _k = _bi
            if not (0 <= _i < _nx and 0 <= _j < _ny and 0 <= _k < _nz):
                return self._json({"error": f"block range 0..{_nx - 1},0..{_ny - 1},0..{_nz - 1}"}, 400)
            if _nc[0]:
                _lx[_i + 1][1] = _nc[0]
            if _nc[1]:
                _ly[_j + 1][1] = _nc[1]
            if _nc[2]:
                _lz[_k + 1][1] = _nc[2]
            cur["x_lines"], cur["y_lines"], cur["z_lines"] = _lx, _ly, _lz
            with open(_p, "w") as f:
                json.dump(cur, f)
            code, log = run([sys.executable, GENERATOR], timeout=600)
            tail = "\n".join(log.strip().splitlines()[-14:])
            if code != 0:
                return self._json({"ok": False, "log": tail})
            return self._json({"ok": True, "log": tail,
                               "nc": [_lx[_i + 1][1], _ly[_j + 1][1], _lz[_k + 1][1]]})
        if u.path == "/api/interface":
            try:
                xs = sorted(set(float(v) for v in data["x"]))
                ys = sorted(set(float(v) for v in data["y"]))
                zs = sorted(set(float(v) for v in data["z"]))
                solid_ov = _clean_triples(data.get("solid", []))
                fluid_ov = _clean_triples(data.get("fluid", []))
            except (KeyError, TypeError, ValueError):
                return self._json({"error": "x/y/z float lists required"}, 400)
            if min(len(xs), len(ys), len(zs)) < 2:
                return self._json({"error": "need >= 2 splits per axis"}, 400)
            if (len(xs) - 1) * (len(ys) - 1) * (len(zs) - 1) > 20000:
                return self._json({"error": "grid too big for preview"}, 400)
            try:
                method = str(data.get("method", "raycast"))
                if method not in ("raycast", "analytic"):
                    method = "raycast"
                _nl = {"x": [e[1] for e in _resolve_lines(xs, "x")[1:]],
                       "y": [e[1] for e in _resolve_lines(ys, "y")[1:]],
                       "z": [e[1] for e in _resolve_lines(zs, "z")[1:]]}
                try:
                    _flat = bool(json.load(open(os.path.join(SYS, "splits.json"))).get("flat", False))
                except (OSError, ValueError):
                    _flat = False
                return self._json(build_interface(xs, ys, zs, solid_ov,
                                                  fluid_ov, method, _nl, _flat))
            except Exception as e:
                return self._json({"error": f"interface failed: {e}"}, 500)
        return self._json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print(f"blockMesh GUI on http://0.0.0.0:{port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
