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
# Geometry (STL) location: chosen in the GUI "Geometry" panel and persisted in
# system/geometry.json. Relative paths are resolved against the case directory.
GEOMETRY_CFG = os.path.join(SYS, "geometry.json")
DEFAULT_STL = os.path.join("constant", "geometry", "car_simplified.stl")
LEGACY_STL = "car_simplified.stl"
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
    # Outer boundary orthogonalization (Top, Sides, Inlet, Outlet)
    for i in range(nx + 1):
        for k in range(nz + 1):
            V[vid(i, ny, k)] = (V[vid(i, ny - 1, k)][0], ys[-1], V[vid(i, ny - 1, k)][2])

    for i in range(nx + 1):
        for j in range(ny + 1):
            if j == 0:
                continue
            V[vid(i, j, 0)] = (V[vid(i, j, 1)][0], V[vid(i, j, 1)][1], zs[0])
            V[vid(i, j, nz)] = (V[vid(i, j, nz - 1)][0], V[vid(i, j, nz - 1)][1], zs[-1])

    for j in range(ny + 1):
        if j == 0:
            continue
        for k in range(nz + 1):
            V[vid(0, j, k)] = (xs[0], V[vid(1, j, k)][1], V[vid(1, j, k)][2])
            V[vid(nx, j, k)] = (xs[-1], V[vid(nx - 1, j, k)][1], V[vid(nx - 1, j, k)][2])

    for ci in (0, nx):
        for cj in (0, ny):
            for ck in (0, nz):
                V[vid(ci, cj, ck)] = (xs[ci], ys[cj], zs[ck])

    for i in range(1, nx):
        V[vid(i, ny, 0)] = (V[vid(i, ny - 1, 1)][0], ys[-1], zs[0])
        V[vid(i, ny, nz)] = (V[vid(i, ny - 1, nz - 1)][0], ys[-1], zs[-1])
    for k in range(1, nz):
        V[vid(0, ny, k)] = (xs[0], ys[-1], V[vid(1, ny - 1, k)][2])
        V[vid(nx, ny, k)] = (xs[-1], ys[-1], V[vid(nx - 1, ny - 1, k)][2])

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

# ---------------------------------------------------------------------------
# Grading of the OUTER (live) slabs so their cells match the INNER (dead/solid)
# blocks. Grading lives in splits.json as per-interval lines [coord, n, g]
# (g = simpleGrading expansion ratio = last cell / first cell, increasing coord).
# ---------------------------------------------------------------------------
def _slab_cells(L, n, g):
    """(first, last) cell size of a slab of length L, n cells, ratio g=last/first."""
    n = max(int(n), 1)
    if n == 1 or abs(g - 1.0) < 1e-12:
        return L / n, L / n
    q = g ** (1.0 / (n - 1))
    first = L * (q - 1.0) / (q ** n - 1.0)
    return first, first * g


def _solve_growth(L, n, d):
    """Per-cell growth t so that d*(1+t+...+t^(n-1)) = L (None if impossible)."""
    if n < 2 or d <= 0 or L / d < 1.0 + 1e-12:
        return None
    target = L / d

    def f(t):
        return n if abs(t - 1.0) < 1e-12 else (t ** n - 1.0) / (t - 1.0)
    lo, hi = 1e-3, min(1e3, 10 ** (250.0 / n))
    if f(lo) > target or f(hi) < target:
        return None
    for _ in range(200):
        mid = (lo * hi) ** 0.5
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo * hi) ** 0.5


def _grade_outer_lines(xs, ys, zs, solid_ov, fluid_ov, flat=False, max_ratio=100.0, lines0=None):
    """New per-axis lines with the outer slabs graded to match the inner cells.
    Returns ({"x":lines,"y":lines,"z":lines}, plan rows, summary dict)."""
    import math
    info = build_interface(xs, ys, zs, solid_ov, fluid_ov, "raycast", None, flat)
    sol = [b["k"] for b in info["blocks"] if b["s"]]
    if not sol:
        raise ValueError("no dead (solid) blocks found - nothing to match the outer blocks to")
    coords = {"x": sorted(xs), "y": sorted(ys), "z": sorted(zs)}
    out, plan = {}, []
    for a, ax in enumerate("xyz"):
        src = lines0[ax] if lines0 else _resolve_lines(coords[ax], ax)
        lines = [[float(c), int(n), float(g)] for c, n, g in src]
        idx = sorted({k[a] for k in sol})
        lo, hi = idx[0], idx[-1]
        ns = len(lines) - 1                                   # slabs 0..ns-1 = lines[s+1]

        def slab(s):
            return lines[s + 1][0] - lines[s][0], lines[s + 1][1], lines[s + 1][2]
        for side, order in (("low", range(lo - 1, -1, -1)), ("high", range(hi + 1, ns))):
            for s in order:
                L, n, g_old = slab(s)
                nb = s + 1 if side == "low" else s - 1        # slab next to it, towards the car
                Lb, nbn, gb = slab(nb)
                fb, lb = _slab_cells(Lb, nbn, gb)
                d_adj = fb if side == "low" else lb           # cell size at the shared plane
                row = {"axis": ax, "slab": s, "side": side, "c0": lines[s][0],
                       "c1": lines[s + 1][0], "n": n, "old_g": g_old,
                       "match_mm": d_adj * 1000.0, "note": ""}
                t = _solve_growth(L, n, d_adj) if n >= 2 else None
                if t is None:
                    g_new = 1.0
                    row["note"] = ("single cell" if n < 2 else
                                   "slab shorter than the neighbouring cell" if L / max(d_adj, 1e-12) < 1.0
                                   else "growth out of range - increase/decrease n")
                elif t < 1.0 - 1e-9:
                    g_new = 1.0                                # finer than the inner cells: keep uniform
                    row["note"] = "cells already finer than the inner ones - left uniform"
                else:
                    g_new = t ** (n - 1) if side == "high" else t ** (-(n - 1))
                    gmax = float(max_ratio)
                    if g_new > gmax or g_new < 1.0 / gmax:
                        g_new = min(max(g_new, 1.0 / gmax), gmax)
                        row["note"] = "grading clamped to %g" % gmax
                f_new, l_new = _slab_cells(L, n, g_new)
                row.update({"new_g": g_new, "growth": (t if t else 1.0),
                            "outer_mm": (f_new if side == "low" else l_new) * 1000.0,
                            "inner_mm_after": (l_new if side == "low" else f_new) * 1000.0})
                lines[s + 1][2] = g_new
                plan.append(row)
        out[ax] = lines
    changed = sum(1 for r in plan if abs(r["new_g"] - r["old_g"]) > 1e-9)
    worst = max([r["growth"] for r in plan] or [1.0])
    return out, plan, {"slabs": len(plan), "changed": changed, "max_growth": worst,
                       "dead_blocks": len(sol)}


_stl_cache = {}
_sdf_surface_cache = {}
_lock = threading.Lock()


def _stl_rel():
    """STL path as configured (relative to the case if possible), or the default."""
    try:
        with open(GEOMETRY_CFG) as f:
            rel = json.load(f).get("stl")
        if isinstance(rel, str) and rel.strip():
            return rel.strip().replace("\\", "/")
    except (OSError, ValueError, AttributeError):
        pass
    if os.path.isfile(os.path.join(CASE, DEFAULT_STL)):
        return DEFAULT_STL.replace(os.sep, "/")
    return LEGACY_STL


def _stl_path():
    """Absolute path of the geometry STL currently selected in the GUI."""
    rel = _stl_rel()
    return rel if os.path.isabs(rel) else os.path.join(CASE, rel)


def _stl_tris(path=None):
    """Triangle count of a binary STL, or None if the file is not one."""
    p = path or _stl_path()
    try:
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            f.read(80)
            n = struct.unpack("<I", f.read(4))[0]
    except (OSError, struct.error):
        return None
    return n if size == 84 + 50 * n else None


_stl_sha_cache = {}


def _stl_sha(path=None):
    """SHA-256 of the STL (cached on path/size/mtime), or None if unreadable."""
    import hashlib
    p = os.path.abspath(path or _stl_path())
    try:
        st = os.stat(p)
    except OSError:
        return None
    key = (p, st.st_size, st.st_mtime_ns)
    if _stl_sha_cache.get("key") != key:
        h = hashlib.sha256()
        try:
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        except OSError:
            return None
        _stl_sha_cache.clear()
        _stl_sha_cache.update({"key": key, "sha": h.hexdigest()})
    return _stl_sha_cache["sha"]


def _sdf_stale(stl):
    """True when the distance field was not built from this exact STL."""
    try:
        with open(os.path.join(SYS, "car_sdf.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return True
    ref = meta.get("source_sha256")
    if not ref:
        src = meta.get("source")
        if not src:
            return True
        sp = src if os.path.isabs(src) else os.path.join(CASE, src)
        # old metadata: compare the recorded source's bytes instead
        ref = _stl_sha(sp)
    if not ref:
        return True
    cur = _stl_sha(stl)
    return cur is None or cur != ref


def _geometry_info():
    """What the GUI shows in the Geometry panel. Never raises."""
    rel = _stl_rel()
    p = _stl_path()
    info = {"stl": rel, "exists": os.path.isfile(p), "tris": None,
            "sdf_stale": True}
    if info["exists"]:
        info["tris"] = _stl_tris(p)
    info["sdf_stale"] = _sdf_stale(p)
    return info


def load_stl(path=None):
    p = path or _stl_path()
    if _stl_cache.get("path") != p or "t" not in _stl_cache:
        import numpy as np
        with open(p, "rb") as f:
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
        _stl_cache["path"] = p
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
    """Nearest point on the target surface for each row of P (exact; triangles
    are pruned with bounding spheres so large meshes stay fast)."""
    import numpy as np
    A, B, C = _snap_tris(target)
    key = "cr_" + target
    if key not in _snap_cache:
        cen = (A + B + C) / 3.0
        r = np.maximum.reduce([np.linalg.norm(A - cen, axis=1),
                               np.linalg.norm(B - cen, axis=1),
                               np.linalg.norm(C - cen, axis=1)])
        _snap_cache[key] = (cen, r)
    cen, r = _snap_cache[key]
    out = np.empty_like(P)
    dist = np.empty(len(P))
    for n in range(len(P)):
        d = np.linalg.norm(cen - P[n], axis=1)
        cand = np.nonzero(d - r <= (d + r).min())[0]
        q = _closest_on_tris(P[n], A[cand], B[cand], C[cand])
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


def _mesh_quality_arrays(P, offs, flat, owner, nei, chunk=400000):
    """OpenFOAM-style mesh quality straight from polyMesh geometry.
    Returns dict: angle (non-orthogonality in degrees, internal faces),
    skew (all faces), pyr_bad (bool, all faces), vol (cell volumes),
    fC (face centres), cC (cell centres)."""
    import numpy as np
    nf = len(offs) - 1
    ni = len(nei)
    sizes = np.diff(offs)
    fC = np.zeros((nf, 3))
    fA = np.zeros((nf, 3))
    groups = []
    for n in np.unique(sizes):
        ids_all = np.nonzero(sizes == n)[0]
        for s in range(0, len(ids_all), chunk):
            groups.append((int(n), ids_all[s:s + chunk]))
    for n, ids in groups:
        V = P[flat[offs[ids][:, None] + np.arange(n)[None, :]]]          # (m,n,3)
        if n == 3:
            fC[ids] = V.mean(axis=1)
            fA[ids] = 0.5 * np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
            continue
        est = V.mean(axis=1)
        Vn = np.roll(V, -1, axis=1)
        nrm = np.cross(Vn - V, est[:, None, :] - V)                     # (m,n,3)
        a = np.linalg.norm(nrm, axis=2)
        sumA = a.sum(axis=1)
        sumAc = (a[:, :, None] * (V + Vn + est[:, None, :])).sum(axis=1)
        good = sumA > 1e-150
        fC[ids] = est
        fC[ids[good]] = sumAc[good] / (3.0 * sumA[good])[:, None]
        fA[ids] = 0.5 * nrm.sum(axis=1)
    nc = int(max(owner.max(initial=-1), nei.max(initial=-1))) + 1
    cEst = np.zeros((nc, 3))
    cnt = np.zeros(nc)
    np.add.at(cEst, owner, fC)
    np.add.at(cnt, owner, 1.0)
    if ni:
        np.add.at(cEst, nei, fC[:ni])
        np.add.at(cnt, nei, 1.0)
    cEst /= np.maximum(cnt, 1.0)[:, None]
    pyr_o = np.einsum("ij,ij->i", fA, fC - cEst[owner])
    vol3 = np.zeros(nc)
    cs = np.zeros((nc, 3))
    np.add.at(vol3, owner, pyr_o)
    np.add.at(cs, owner, pyr_o[:, None] * (0.75 * fC + 0.25 * cEst[owner]))
    pyr_n = np.zeros(0)
    if ni:
        pyr_n = np.einsum("ij,ij->i", fA[:ni], cEst[nei] - fC[:ni])
        np.add.at(vol3, nei, pyr_n)
        np.add.at(cs, nei, pyr_n[:, None] * (0.75 * fC[:ni] + 0.25 * cEst[nei]))
    ok = np.abs(vol3) > 1e-300
    cC = cEst.copy()
    cC[ok] = cs[ok] / vol3[ok][:, None]
    vol = vol3 / 3.0
    # non-orthogonality (internal faces)
    d = cC[nei] - cC[owner[:ni]] if ni else np.zeros((0, 3))
    dn = np.linalg.norm(d, axis=1) * np.linalg.norm(fA[:ni], axis=1)
    cosv = np.einsum("ij,ij->i", d, fA[:ni]) / np.maximum(dn, 1e-300)
    angle = np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))
    # skewness (internal + boundary faces)
    Cpf = fC - cC[owner]
    dd = np.zeros((nf, 3))
    dd[:ni] = d
    if nf > ni:
        nb = fA[ni:] / np.maximum(np.linalg.norm(fA[ni:], axis=1), 1e-300)[:, None]
        dd[ni:] = nb * np.einsum("ij,ij->i", nb, Cpf[ni:])[:, None]
    rvs = 1e-150
    sv = Cpf - (np.einsum("ij,ij->i", fA, Cpf)
                / (np.einsum("ij,ij->i", fA, dd) + rvs))[:, None] * dd
    svn = np.linalg.norm(sv, axis=1)
    svHat = sv / (svn + rvs)[:, None]
    fd = np.where(np.arange(nf) < ni, 0.2, 0.4) * np.linalg.norm(dd, axis=1) + 1e-300
    for n, ids in groups:
        V = P[flat[offs[ids][:, None] + np.arange(n)[None, :]]]
        proj = np.abs(np.einsum("mnk,mk->mn", V - fC[ids][:, None, :], svHat[ids]))
        fd[ids] = np.maximum(fd[ids], proj.max(axis=1))
    skew = svn / fd
    # face pyramids: apex (cell centre) must lie on the inner side of every face
    pyr_bad = np.zeros(nf, dtype=bool)
    pyr_bad |= (pyr_o / 3.0) <= 1e-15
    if ni:
        pyr_bad[:ni] |= (pyr_n / 3.0) <= 1e-15
    return {"angle": angle, "skew": skew, "pyr_bad": pyr_bad, "vol": vol,
            "fC": fC, "cC": cC}


def _mesh_quality():
    """Problem elements computed from constant/polyMesh itself:
    {"neg": cells, "nonortho": faces (>70 deg), "skew": faces (>4), "pyr": faces}."""
    import numpy as np
    pd = os.path.join(CASE, "constant", "polyMesh")
    P = _read_points(os.path.join(pd, "points"))
    offs, flat = _read_faces(os.path.join(pd, "faces"))
    owner = np.array(_read_label_list(os.path.join(pd, "owner")), dtype=np.int64)
    nei = np.array(_read_label_list(os.path.join(pd, "neighbour")), dtype=np.int64)
    q = _mesh_quality_arrays(P, offs, flat, owner, nei)
    return {"neg": [int(i) for i in np.nonzero(q["vol"] < 1e-300)[0]],
            "nonortho": [int(i) for i in np.nonzero(q["angle"] > 70.0)[0]],
            "skew": [int(i) for i in np.nonzero(q["skew"] > 4.0)[0]],
            "pyr": [int(i) for i in np.nonzero(q["pyr_bad"])[0]]}


def _mesh_geometry():
    """(face_centroids, cell_centres, owner) of constant/polyMesh, read directly
    (no postProcess needed)."""
    import numpy as np
    pd = os.path.join(CASE, "constant", "polyMesh")
    P = _read_points(os.path.join(pd, "points"))
    offs, flat = _read_faces(os.path.join(pd, "faces"))
    owner = np.array(_read_label_list(os.path.join(pd, "owner")), dtype=np.int64)
    nei = np.array(_read_label_list(os.path.join(pd, "neighbour")), dtype=np.int64)
    cnt = np.diff(offs)
    fc = np.add.reduceat(P[flat], offs[:-1], axis=0) / np.maximum(cnt, 1)[:, None]
    nc = int(max(owner.max(initial=-1), nei.max(initial=-1))) + 1
    sums = np.zeros((nc, 3))
    n = np.zeros(nc)
    np.add.at(sums, owner, fc[:len(owner)])
    np.add.at(n, owner, 1.0)
    if len(nei):
        np.add.at(sums, nei, fc[:len(nei)])
        np.add.at(n, nei, 1.0)
    return fc, sums / np.maximum(n, 1.0)[:, None], owner


# ---------------------------------------------------------------------------
# Bad-element display via foamToVTK: checkMesh writes face/cell sets, foamToVTK
# turns each one into a VTK file under <case>/VTK, and we read that geometry.
# ---------------------------------------------------------------------------
_VTK_FACES = {                       # VTK cell type -> faces (vertex indices)
    12: [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],   # hexahedron
    10: [(0, 2, 1), (0, 1, 3), (1, 2, 3), (0, 3, 2)],                                           # tetrahedron
    14: [(0, 3, 2, 1), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],                             # pyramid
    13: [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],                       # wedge
}


def _vtk_ints(raw, pos, count, binary):
    import numpy as np
    if binary:
        a = np.frombuffer(raw, dtype=">i4", count=count, offset=pos)
        return a.astype(np.int64), pos + 4 * count
    parts = raw[pos:].split(None, count)
    vals = np.array(parts[:count], dtype=np.int64)
    rest = parts[count] if len(parts) > count else b""
    return vals, len(raw) - len(rest)


def _vtk_read(path, cap=20000):
    """Read a legacy VTK file (ASCII or BINARY; POLYDATA or UNSTRUCTURED_GRID) as
    written by foamToVTK. Returns {"tris": flat xyz list, "centres": [[x,y,z]..],
    "count": number of faces/cells in the file}."""
    import numpy as np
    raw = open(path, "rb").read()
    if not raw.startswith(b"# vtk"):
        raise ValueError("not a legacy VTK file (XML .vtp/.vtu are not supported)")
    binary = b"BINARY" in raw[:300].upper().split(b"DATASET")[0]
    pos = raw.index(b"DATASET")
    kw = re.compile(rb"(POINTS|POLYGONS|CELLS|CELL_TYPES|POINT_DATA|CELL_DATA)[ \t]*([^\n]*)\n")
    P = polys = cells = ctypes = None
    while True:
        m = kw.search(raw, pos)
        if not m:
            break
        k, args, pos = m.group(1), m.group(2).split(), m.end()
        if k in (b"POINT_DATA", b"CELL_DATA"):
            break
        if k == b"POINTS":
            n = int(args[0])
            dt = np.dtype(">f8" if len(args) > 1 and args[1] == b"double" else ">f4")
            if binary:
                a = np.frombuffer(raw, dtype=dt, count=3 * n, offset=pos)
                pos += 3 * n * dt.itemsize
            else:
                parts = raw[pos:].split(None, 3 * n)
                a = np.array(parts[:3 * n], dtype=float)
                pos = len(raw) - len(parts[3 * n] if len(parts) > 3 * n else b"")
            P = a.astype(float).reshape(-1, 3)
        elif k in (b"POLYGONS", b"CELLS"):
            n, size = int(args[0]), int(args[1])
            a, pos = _vtk_ints(raw, pos, size, binary)
            if k == b"POLYGONS":
                polys = (n, a)
            else:
                cells = (n, a)
        elif k == b"CELL_TYPES":
            n = int(args[0])
            ctypes, pos = _vtk_ints(raw, pos, n, binary)
    if P is None:
        raise ValueError("no POINTS in VTK file")
    tris, centres, count = [], [], 0

    def add_face(ids):
        nonlocal tris
        for i in range(1, len(ids) - 1):
            tris.extend(P[ids[0]].tolist() + P[ids[i]].tolist() + P[ids[i + 1]].tolist())

    def entries(n, a):
        p = 0
        for _ in range(n):
            c = int(a[p])
            yield a[p + 1:p + 1 + c]
            p += 1 + c
    if polys is not None:                                   # POLYDATA: each polygon = one face
        for ids in entries(*polys):
            count += 1
            if len(centres) < cap:
                add_face(ids)
                centres.append(P[ids].mean(axis=0).tolist())
    elif cells is not None:                                 # UNSTRUCTURED_GRID
        for ci, ids in enumerate(entries(*cells)):
            count += 1
            if len(centres) >= cap:
                continue
            t = int(ctypes[ci]) if ctypes is not None and ci < len(ctypes) else 12
            if t in (5, 7, 9):                              # a single face
                add_face(ids)
            elif t == 42:                                   # polyhedron: [nFaces, n0, ids.., n1, ids..]
                nf, q = int(ids[0]), 1
                for _f in range(nf):
                    c = int(ids[q]); add_face(ids[q + 1:q + 1 + c]); q += 1 + c
            elif t in _VTK_FACES:
                for f in _VTK_FACES[t]:
                    add_face([ids[j] for j in f])
            else:
                continue
            centres.append(P[ids].mean(axis=0).tolist() if t != 42 else P[np.unique(ids[1:])].mean(axis=0).tolist())
    else:
        raise ValueError("VTK file has no POLYGONS or CELLS")
    return {"tris": [round(x, 5) for x in tris], "centres": centres, "count": count}


def _set_is_cell(path):
    """True for a cellSet file, False for a faceSet (from its FoamFile header)."""
    try:
        head = open(path, "rb").read(600).decode("latin-1")
        return bool(re.search(r"class\s+cellSet", head))
    except OSError:
        return False


def _foam_to_vtk(name, is_cell):
    """foamToVTK -faceSet/-cellSet <name>; returns the newest VTK file made for it
    (None if nothing was produced) plus the command output."""
    import time as _time
    opt = "-cellSet" if is_cell else "-faceSet"
    t0 = _time.time() - 1.0
    code, out = run(["bash", "-lc",
                     "source /opt/openfoam12/etc/bashrc; "
                     f"cd {CASE}; "
                     f"o=$(foamToVTK -ascii -noInternal -noBoundary -noFaceZones -noLagrangian "
                     f"-noPointValues {opt} {name} 2>&1) "
                     f"|| o=$(foamToVTK -ascii {opt} {name} 2>&1) "
                     f"|| o=$(foamToVTK {opt} {name} 2>&1); echo \"$o\""], timeout=900)
    hits = []
    for root, _d, files in os.walk(os.path.join(CASE, "VTK")):
        for f in files:
            p = os.path.join(root, f)
            if name in p and f.endswith(".vtk"):
                try:
                    mt = os.path.getmtime(p)
                except OSError:
                    continue
                if mt >= t0:
                    hits.append((mt, p))
    hits.sort()
    return (hits[-1][1] if hits else None), out


_SET_KINDS = (          # key, slot in the response, candidate set names, label
    ("negvol",   "cells",    ["negativeVolumeCells", "zeroVolumeCells"], "negative volume cells"),
    ("nonortho", "nonortho", ["nonOrthoFaces"],                          "non-orthogonal faces (severe)"),
    ("skew",     "skew",     ["skewFaces"],                              "highly skew faces"),
    ("pyramid",  "pyramids", ["wrongOrientedFaces"],                     "face pyramid errors"),
    ("aspect",   "aspect",   ["highAspectRatioCells"],                   "high aspect ratio cells"),
)


def _checkmesh_run():
    """Run checkMesh, turn each bad-element set it wrote into VTK with foamToVTK,
    read that geometry, and return the dict served by /api/checkmesh."""
    import re as _re
    # -allGeometry also checks aspect ratio. OpenFOAM 12 wants "-writeSets <format>";
    # other builds take a bare flag or none: try them in turn (sets are written anyway).
    flags = "-allGeometry -allTopology"
    code, log = run(["bash", "-lc",
                     "source /opt/openfoam12/etc/bashrc; "
                     f"cd {CASE}; rm -rf constant/polyMesh/sets; "
                     f"out=$(checkMesh {flags} -writeSets vtk 2>&1) "
                     f"|| out=$(checkMesh {flags} -writeSets 2>&1) "
                     f"|| out=$(checkMesh {flags} 2>&1) "
                     "|| out=$(checkMesh 2>&1); echo \"$out\""], timeout=900)

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
    notes, srcs = [], {}

    # ---- which set file belongs to which check ---------------------------------
    logged = {}
    for _n, desc, nm in _re.findall(r"Writing\s+(\d+)\s+([^\n]*?)\s+to set\s+(\w+)", log):
        d = desc.lower()
        if "aspect" in d:
            logged["aspect"] = nm
        elif "zero" in d or "negative" in d or "volume" in d:
            logged["negvol"] = nm
        elif "non-orth" in d or "nonorth" in d:
            logged["nonortho"] = nm
        elif "skew" in d:
            logged["skew"] = nm
        elif "orient" in d or "pyramid" in d:
            logged["pyramid"] = nm
    setname, setpath = {}, {}
    for key, slot, cands, label in _SET_KINDS:
        for nm in ([logged[key]] if key in logged else []) + cands:
            p = os.path.join(setdir, nm)
            if os.path.exists(p):
                setname[key], setpath[key] = nm, p
                break

    # ---- geometry: foamToVTK -> VTK file -> read ---------------------------------
    geo = {}                                   # key -> {"tris","centres","count"}
    for key, slot, cands, label in _SET_KINDS:
        if key not in setpath:
            continue
        try:
            vtk, out = _foam_to_vtk(setname[key], _set_is_cell(setpath[key]))
            if not vtk:
                notes.append("foamToVTK made no VTK file for set '%s': %s" % (
                    setname[key], " | ".join(out.strip().splitlines()[-2:]) or "no output"))
                continue
            geo[key] = _vtk_read(vtk)
            srcs[key] = "vtk"
        except Exception as e:
            notes.append("could not read VTK for '%s': %r" % (setname[key], e))

    # ---- fall back to the polyMesh files for anything VTK did not give ---------------
    ids = {k: (_read_cell_set(setpath[k]) if k in setpath else []) for k, *_ in _SET_KINDS}
    for k in list(ids):
        if k in geo:
            ids[k] = []                        # already have the geometry
    nonortho_max = _flt(r"Mesh non-orthogonality Max:\s*([-+.\deE]+)")
    skew_max = _flt(r"Max skewness\s*=\s*([-+.\deE]+)")
    want = {
        "negvol": bool(_num(r"negative volume cells:\s*(\d+)")) and "negvol" not in geo and not ids["negvol"],
        "nonortho": bool(_num(r"severely non-orthogonal[^\n]*:\s*(\d+)")
                         or (nonortho_max and nonortho_max > 70.0))
                    and "nonortho" not in geo and not ids["nonortho"],
        "skew": bool(_num(r"(\d+)\s+highly skew faces detected")
                     or (skew_max and skew_max > 4.0)) and "skew" not in geo and not ids["skew"],
        "pyramid": bool(_num(r"(?i)error in face pyramids:\s*(\d+)"))
                   and "pyramid" not in geo and not ids["pyramid"],
    }
    if any(want.values()):
        # checkMesh writes no set for some warnings (e.g. severe non-orthogonality):
        # work the faces out from the mesh itself.
        try:
            qm = _mesh_quality()
            for key, qk in (("negvol", "neg"), ("nonortho", "nonortho"),
                            ("skew", "skew"), ("pyramid", "pyr")):
                if want[key] and qm[qk]:
                    ids[key] = qm[qk]
                    srcs[key] = "computed"
        except Exception as e:
            notes.append("could not work out the problem faces from the mesh: " + repr(e))
    pm = {}                                    # polyMesh-derived centres + triangles
    if any(ids.values()):
        try:
            fc, cc, owner = _mesh_geometry()
            pm["fc"], pm["cc"] = fc, cc
            pm["tris"] = _problem_tris(ids["negvol"], ids["nonortho"], ids["skew"], ids["pyramid"])
            pm["tris"]["aspect"] = (_problem_tris(ids["aspect"], [], [], [])["cells"]
                                    if ids["aspect"] else [])
        except Exception as e:
            notes.append("could not read the mesh geometry: " + repr(e))

    cells, nonortho, skew, pyramids, aspect = [], [], [], [], []
    lists = {"negvol": cells, "nonortho": nonortho, "skew": skew, "pyramid": pyramids,
             "aspect": aspect}
    tris = {"cells": [], "nonortho": [], "skew": [], "pyramids": [], "aspect": []}
    tri_key = {"negvol": "cells", "nonortho": "nonortho", "skew": "skew",
               "pyramid": "pyramids", "aspect": "aspect"}
    for key, slot, cands, label in _SET_KINDS:
        if key in geo:
            g = geo[key]
            tris[slot] = g["tris"]
            lists[key].extend({"i": n, "c": c} for n, c in enumerate(g["centres"]))
        elif ids.get(key) and "tris" in pm:
            tris[slot] = pm["tris"].get(tri_key.get(key, ""), [])
            ctr = pm["cc"] if key in ("negvol", "aspect") else pm["fc"]
            lists[key].extend({"i": int(i), "c": [float(v) for v in ctr[i]]}
                              for i in ids[key] if 0 <= i < len(ctr))

    # ---- counts: trust checkMesh's own log line, else what we found ----------------------
    n_cells = (_num(r"(?m)^\s*cells:\s*(\d+)\s*$") or _num(r"(?m)^\s*cells\s*[:=]\s*(\d+)"))

    def found(key):
        return geo[key]["count"] if key in geo else len(ids.get(key, []))
    n = {
        "negvol": _num(r"negative volume cells:\s*(\d+)") or found("negvol"),
        "nonortho": _num(r"severely non-orthogonal[^\n]*:\s*(\d+)") or found("nonortho"),
        "skew": _num(r"(\d+)\s+highly skew faces detected") or found("skew"),
        "pyramid": _num(r"(?i)error in face pyramids:\s*(\d+)") or found("pyramid"),
        "aspect": (_num(r"(?i)High aspect ratio cells found[^\n]*?number of cells\s+(\d+)")
                   or found("aspect")),
    }
    maxv = {
        "negvol": _flt(r"Minimum negative volume:\s*([-+.\deE]+)"),
        "nonortho": nonortho_max,
        "skew": skew_max,
        "pyramid": None,
        "aspect": _flt(r"(?i)Max aspect ratio\s*[:=]\s*([-+.\deE]+)"),
    }
    for key, slot, cands, label in _SET_KINDS:
        if n[key] and not (lists[key] or tris[slot]):
            notes.append("checkMesh reports %d %s but no geometry could be read for them "
                         "(set '%s' %s)" % (n[key], label, setname.get(key, cands[0]),
                                            "found" if key in setpath else "not written"))
    if not _re.search(r"Mesh OK|Failed \d+ mesh checks|\bcells:", log):
        notes.append("checkMesh output not recognised - see the log: " + " | ".join(
            log.strip().splitlines()[-2:]))
    issues = [{"key": key, "label": label, "n": n[key], "max": maxv[key],
               "src": srcs.get(key, "set")} for key, slot, cands, label in _SET_KINDS]
    return ({"ok": True,
             "log": "\n".join(log.strip().splitlines()[-30:]),
             "mesh_ok": mesh_ok,
             "n_cells": n_cells,
             "n_bad": n["negvol"], "n_severe_nonortho": n["nonortho"],
             "n_nonortho": n["nonortho"], "n_skew": n["skew"],
             "n_pyramid": n["pyramid"], "n_aspect": n["aspect"],
             "issues": issues,
             "cells": cells, "nonortho": nonortho, "skew": skew,
             "pyramids": pyramids, "aspect": aspect,
             "tris": tris,
             "geom_error": None,
             "notes": notes})


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
                "/api/interface", "/api/interface_edges", "/api/setvertices",
                "/api/sdf/build", "/api/geometry")


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


def _write_vertices(moves, unpin=()):
    """Write {vid:[x,y,z]} into blockMeshDict and splits.json pins (vertices in
    `unpin` get the position but no pin)."""
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
    _refresh_interface_edges(set(moves))       # edges follow the moved vertices
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
        if vid not in unpin:
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


def _surface_normals(P, target):
    """Smoothed unit surface normal at the closest surface point of each row of P
    (average of nearby triangle normals, sign-aligned). Orientation is arbitrary."""
    import numpy as np
    A, B, C = _snap_tris(target)
    key = "cent_" + target
    if key not in _snap_cache:
        cen = (A + B + C) / 3.0
        n3 = np.cross(B - A, C - A)
        n3 = n3 / np.maximum(np.linalg.norm(n3, axis=1), 1e-30)[:, None]
        e = np.linalg.norm(B[:2000] - A[:2000], axis=1)
        _snap_cache[key] = (cen, n3, float(np.median(e)) if len(e) else 1.0)
    cen, n3, med = _snap_cache[key]
    out = np.zeros_like(P)
    for n in range(len(P)):
        q = _closest_on_tris(P[n], A, B, C)
        k = int(np.argmin(((q - P[n]) ** 2).sum(axis=1)))
        near = np.nonzero(np.linalg.norm(cen - q[k], axis=1) < 3.0 * med)[0]
        if k not in near:
            near = np.append(near, k)
        sgn = np.sign(n3[near] @ n3[k])
        sgn[sgn == 0] = 1.0
        v = (n3[near] * sgn[:, None]).sum(axis=0)
        ln = np.linalg.norm(v)
        out[n] = v / ln if ln > 1e-12 else n3[k]
    return out


def _hex_adjacency():
    """{vertex: set(vertices joined to it by a block edge)} over the live blocks."""
    txt = open(os.path.join(SYS, "blockMeshDict")).read()
    m = re.search(r"blocks\s*\(\s*(.*?)\n\)\s*;", txt, re.S)
    adj = {}
    he = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
          (0, 4), (1, 5), (2, 6), (3, 7)]
    for h in re.findall(r"hex\s*\(([^()]+)\)", m.group(1) if m else ""):
        v = [int(x) for x in h.split()[:8]]
        if len(v) < 8:
            continue
        for a, b in he:
            adj.setdefault(v[a], set()).add(v[b])
            adj.setdefault(v[b], set()).add(v[a])
    return adj


def _edge_section(txt):
    """(match of the edges section, existing curve pairs, polyLine spans)."""
    m = re.search(r"edges\s*\(\s*(.*?)\)\s*;\s*boundary", txt, re.S)
    if not m:
        return None, set(), []
    body = m.group(1)
    pairs = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in re.findall(
        r"(?:polyLine|arc|spline|BSpline|bspline)\s+(\d+)\s+(\d+)", body)}
    spans = []
    for pm in re.finditer(r"polyLine\s+(\d+)\s+(\d+)\s*\(", body):
        i, depth = pm.end(), 1
        while i < len(body) and depth:
            depth += (body[i] == "(") - (body[i] == ")")
            i += 1
        a, b = int(pm.group(1)), int(pm.group(2))
        spans.append(((min(a, b), max(a, b)), pm.start(), i))
    return m, pairs, spans


def _normal_edge_set():
    try:
        cur = json.load(open(os.path.join(SYS, "splits.json")))
        return {(min(int(a), int(b)), max(int(a), int(b)))
                for a, b in cur.get("normal_edges", [])}
    except (OSError, ValueError, TypeError):
        return set()


def _write_edge_curves(curves):
    """Add {(a,b): [[x,y,z],...]} as polyLine edges to blockMeshDict; curves the
    optimiser added earlier for the same edge are replaced. Remembered in
    splits.json ("normal_edges") so they are not mistaken for surface edges."""
    dp = os.path.join(SYS, "blockMeshDict")
    txt = open(dp).read()
    m, _pairs, spans = _edge_section(txt)
    if m is None:
        raise RuntimeError("no edges section in blockMeshDict")
    mine = _normal_edge_set()
    want = {(min(a, b), max(a, b)) for a, b in curves}
    body = m.group(1)
    for key, s0, e0 in sorted(spans, key=lambda t: -t[1]):
        if key in want and key in mine:
            body = body[:s0] + body[e0:]
    add = "".join("\n    polyLine %d %d (%s)" % (
        a, b, " ".join("(%.6g %.6g %.6g)" % tuple(p) for p in pts))
        for (a, b), pts in curves.items())
    body = body.rstrip() + add + "\n"
    txt = txt[:m.start(1)] + body + txt[m.end(1):]
    with open(dp, "w") as f:
        f.write(txt)
    sp = os.path.join(SYS, "splits.json")
    try:
        cur = json.load(open(sp))
    except (OSError, ValueError):
        cur = {}
    cur["normal_edges"] = sorted([list(k) for k in (mine | want)])
    with open(sp, "w") as f:
        json.dump(cur, f)


def _normal_curve(p0, p1, n, k=6):
    """Cubic Bezier from surface vertex p0 to p1 that leaves p0 along the surface
    normal n and arrives along the straight direction. None if the edge runs
    almost along the surface (nothing sensible to do)."""
    import numpy as np
    d = p1 - p0
    le = float(np.linalg.norm(d))
    if le < 1e-9:
        return None
    u = d / le
    n = np.asarray(n, float)
    if float(n @ u) < 0:
        n = -n
    if float(n @ u) < 0.2:
        return None
    h = le / 3.0
    c1, c2 = p0 + n * h, p1 - u * h
    pts = []
    for i in range(1, k + 1):
        t = i / (k + 1.0)
        b = ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * c1 \
            + 3 * (1 - t) * t * t * c2 + (t ** 3) * p1
        pts.append(b.tolist())
    return pts


def _project_for_edges(P):
    """Project points onto the car surface for edge curves: the distance field
    (same projection as "Update projection") when loaded, else the smooth mesh,
    else the STL."""
    import numpy as np
    if SDF is not None:
        try:
            with _lock:
                out = np.asarray(SDF.project(P), dtype=float)
            if out.shape == P.shape and np.isfinite(out).all():
                return out
        except Exception:
            pass
    target = "smooth" if os.path.exists(os.path.join(SYS, "car_sdf_surface.npz")) else "stl"
    return snap_points(P, target)[0]


def _refresh_interface_edges(changed=None, missing_only=False):
    """Make EVERY interface edge (edge of a body/wheels wall quad) a polyLine whose
    points lie on the car surface, so no interface edge is left straight.

    changed=None      -> all interface edges
    changed={vids}    -> only edges touching those vertices (they just moved)
    missing_only=True -> only edges that have no polyLine yet (keep existing curves)
    Returns a small summary dict; never raises."""
    import numpy as np
    try:
        dp = os.path.join(SYS, "blockMeshDict")
        txt = open(dp).read()
        verts, quads, _nbr = _read_dict_surface()
        if not quads:
            return {"ok": True, "edges": 0, "written": 0, "added": 0}
        wall = set()
        for q in quads:
            for a, b in zip(q, q[1:] + q[:1]):
                if a != b:
                    wall.add((min(a, b), max(a, b)))
        m, pairs, spans = _edge_section(txt)
        if m is None:
            return {"ok": False, "error": "no edges section in blockMeshDict"}
        span_of = {k: (s0, e0) for k, s0, e0 in spans}
        other = pairs - set(span_of)              # arc/spline etc.: leave alone
        mine = _normal_edge_set()                 # optimiser's off-surface curves
        chg = None if changed is None else set(int(v) for v in changed)
        todo = []
        for e in sorted(wall):
            if e in other or e in mine:
                continue
            if chg is not None and e[0] not in chg and e[1] not in chg:
                continue
            if missing_only and e in span_of:
                continue
            todo.append(e)
        n_missing = sum(1 for e in todo if e not in span_of)
        if not todo:
            return {"ok": True, "edges": len(wall), "written": 0, "added": 0,
                    "straight": sum(1 for e in wall if e not in span_of and e not in other)}
        L = float(np.mean([np.linalg.norm(verts[a] - verts[b]) for a, b in wall]))
        gd = _dict_grid()
        dead = _dead_verts()
        fixedv = set()
        if gd is not None:
            fixedv = {v for e in todo for v in e if _pin_vetoed(v, gd, dead)}
        samples, layout = [], []
        for a, b in todo:
            pa, pb = verts[a], verts[b]
            le = float(np.linalg.norm(pb - pa))
            k = max(4, min(24, int(round(8 * le / max(L, 1e-9)))))
            t = (np.arange(1, k + 1) / (k + 1.0))[:, None]
            samples.append(pa + (pb - pa) * t)
            layout.append((a, b, k))
        Q = _project_for_edges(np.vstack(samples))
        add, pos = [], 0
        for a, b, k in layout:
            pts = Q[pos:pos + k].copy()
            pos += k
            if a in fixedv and b in fixedv:        # edge on the outer boundary:
                for ax in range(3):                # stay in that boundary plane
                    if abs(verts[a][ax] - verts[b][ax]) < 1e-9:
                        pts[:, ax] = verts[a][ax]
            add.append("\n    polyLine %d %d (%s)" % (
                a, b, " ".join("(%.6g %.6g %.6g)" % tuple(p) for p in pts)))
        body = m.group(1)
        for key, (s0, e0) in sorted(((e, span_of[e]) for e in todo if e in span_of),
                                    key=lambda t: -t[1][0]):
            body = body[:s0] + body[e0:]
        body = body.rstrip() + "".join(add) + "\n"
        txt = txt[:m.start(1)] + body + txt[m.end(1):]
        with open(dp, "w") as f:
            f.write(txt)
        return {"ok": True, "edges": len(wall), "written": len(todo), "added": n_missing,
                "straight": 0}
    except Exception as e:
        return {"ok": False, "error": repr(e)}


def _opt_job(params):
    import numpy as np
    max_iter = max(1, min(30, int(params.get("max_iter", 6))))
    max_verts = max(5, min(2000, int(params.get("max_verts", 300))))
    target = "stl" if str(params.get("target", "smooth")) == "stl" else "smooth"
    use_normal = bool(params.get("normal_edges", False))
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
    surf_set = set(int(v) for v in surf)
    fixed_set = set(int(v) for v in surf if _pin_vetoed(int(v), gd, dead))
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
    adj = _hex_adjacency() if use_normal else {}
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
             "mean surface edge %.1f mm, snapping to the %s surface%s"
             % (len(surf), len(fixed_set), L * 1000, "smooth" if target == "smooth" else "STL",
                ", normal edges ON" if use_normal else ""))
    _opt_set(message="baseline: building mesh + checkMesh", trial="baseline")
    best_r, blog = _mesh_and_check()
    if best_r is None:
        raise RuntimeError("blockMesh failed: " + " | ".join(blog.strip().splitlines()[-3:]))
    best_s = _score(best_r)
    _opt_set(score=best_s, base=best_s, result=best_r)
    _opt_log("Baseline: " + _summ(best_r))
    _opt_set(message="baseline: " + _summ(best_r))
    snap_best = _snapshot()
    mesh_is_best = True
    relax = 0.5
    R = 2.5 * L

    def select(r):
        """Surface vertices (movable) next to the current problems, worst first."""
        probs = sorted(_problem_points(r), key=lambda t: -t[1])[:3000]
        PS = verts[surf]
        wt = np.zeros(len(surf))
        kinds = [dict() for _ in surf]
        far = 0
        for c, w, kind in probs:
            d = np.linalg.norm(PS - np.asarray(c), axis=1)
            for n in (np.argpartition(d, 4)[:4] if len(d) > 4 else range(len(d))):
                if d[n] <= R:
                    wt[n] += w / (1.0 + d[n] / L)
                    kinds[n][kind] = kinds[n].get(kind, 0) + 1
            if d.min() > R:
                far += 1
        order = [int(n) for n in np.argsort(-wt)
                 if wt[n] > 0 and int(surf[n]) not in fixed_set][:max_verts]
        return order, kinds, far

    def cancelled():
        with _opt_mu:
            return _opt["cancel"]

    for it in range(1, max_iter + 1):
        if cancelled():
            break
        if _n_problems(best_r) == 0:
            _opt_log("No checkMesh problems left - done.")
            break
        _opt_set(iter=it)
        tag = "iter %d/%d" % (it, max_iter)
        improved_a = improved_b = False

        # ---- stage A: slide surface vertices (Laplacian step, re-snapped) ----
        order, kinds, far = select(best_r)
        if far:
            _opt_log("  %d problem location(s) are too far from the surface to be "
                     "fixed by surface vertices" % far)
        if order:
            ids = [int(surf[n]) for n in order]
            _opt_log("Iteration %d/%d: %d surface vertices sit next to problem cells"
                     % (it, max_iter, len(ids)))
            for a in (relax, relax / 2.0, relax / 4.0):
                if cancelled():
                    break
                new, moved = _propose(verts, ids, nbr_use, a, target)
                tgt = [{"vid": v, "from": verts[v].tolist(), "to": new[n].tolist()}
                       for n, v in enumerate(ids)]
                _opt_set(targets=tgt, trial="it%d-a%.3f" % (it, a),
                         message="%s: moving %d vertices (step %.2f)" % (tag, len(ids), a))
                for n, v in list(enumerate(ids))[:25]:
                    near = ", ".join("%d %s" % (c, k) for k, c in sorted(kinds[order[n]].items()))
                    _opt_log("  fixing vertex %d (grid %s): near %s -> move %.2f mm"
                             % (v, _vid_key(v, gd), near, moved[n] * 1000))
                if len(ids) > 25:
                    _opt_log("  ... and %d more vertices" % (len(ids) - 25))
                _write_vertices({v: new[n] for n, v in enumerate(ids)})
                _opt_log("  trial step %.2f: blockMesh + checkMesh ..." % a)
                r, blog = _mesh_and_check()
                s = _score(r)
                if r is not None and s < best_s - 1e-9:
                    _opt_log("  trial step %.2f: %s  (score %.1f -> %.1f) ACCEPTED"
                             % (a, _summ(r), best_s, s))
                    for n, v in enumerate(ids):
                        verts[v] = new[n]
                    best_r, best_s = r, s
                    snap_best = _snapshot()
                    mesh_is_best = True
                    relax = a
                    improved_a = True
                    _opt_set(score=best_s, result=best_r,
                             message="%s: better - %s" % (tag, _summ(r)))
                    break
                _opt_log("  trial step %.2f: %s  (score %s) rejected, reverted"
                         % (a, _summ(r), "n/a" if r is None else "%.1f" % s))
                _opt_set(message="%s: step %.2f rejected" % (tag, a))
                _restore(snap_best)
                mesh_is_best = False
        else:
            _opt_log("Iteration %d: no movable surface vertices near the problems" % it)

        if cancelled():
            break

        # ---- stage B: edges leaving the surface approach it along the normal ---
        if use_normal and _n_problems(best_r) > 0:
            order, kinds, far = select(best_r)
            ids = [int(surf[n]) for n in order]
            if ids:
                nrm = _surface_normals(verts[ids], target)
                have = _edge_section(open(os.path.join(SYS, "blockMeshDict")).read())[1]
                mine = _normal_edge_set()
                curves, ends, tangential = {}, [], 0
                for n, v in enumerate(ids):
                    for w in sorted(adj.get(v, ())):
                        if w in surf_set:
                            continue              # lies on the surface already
                        key = (min(v, w), max(v, w))
                        if key in have and key not in mine:
                            continue              # already a projected/curved edge
                        pts = _normal_curve(verts[v], verts[w], nrm[n])
                        if pts is None:
                            tangential += 1
                            continue
                        curves[(v, w)] = pts
                        ends.append({"vid": v, "from": verts[v].tolist(),
                                     "to": verts[w].tolist()})
                if curves:
                    _opt_log("Iteration %d/%d: curving %d edge(s) at %d vertices so they "
                             "leave the surface along its normal%s"
                             % (it, max_iter, len(curves), len({v for v, _ in curves}),
                                " (%d skipped: nearly tangential)" % tangential if tangential else ""))
                    _opt_set(targets=ends, trial="it%d-normal" % it,
                             message="%s: curving %d edges" % (tag, len(curves)))
                    _write_edge_curves(curves)
                    r, blog = _mesh_and_check()
                    s = _score(r)
                    if r is not None and s < best_s - 1e-9:
                        _opt_log("  normal edges: %s  (score %.1f -> %.1f) ACCEPTED"
                                 % (_summ(r), best_s, s))
                        best_r, best_s = r, s
                        snap_best = _snapshot()
                        mesh_is_best = True
                        improved_b = True
                        _opt_set(score=best_s, result=best_r,
                                 message="%s: normal edges better - %s" % (tag, _summ(r)))
                    else:
                        _opt_log("  normal edges: %s  (score %s) rejected, reverted"
                                 % (_summ(r), "n/a" if r is None else "%.1f" % s))
                        _opt_set(message="%s: normal edges rejected" % tag)
                        _restore(snap_best)
                        mesh_is_best = False
                else:
                    _opt_log("Iteration %d: no off-surface edges to curve here" % it)
        if cancelled():
            break
        if not (improved_a or improved_b):
            _opt_log("No further improvement - stopping.")
            break
    was_cancelled = cancelled()
    if not mesh_is_best:
        _opt_log("Restoring best mesh (blockMesh) ...")
        _opt_set(message="restoring best mesh", targets=[])
        _restore(snap_best)
        _run_blockmesh()
    base = _opt["base"]
    _opt_log(("Stopped by user. " if was_cancelled else "Finished. ")
             + "Result: " + _summ(best_r) + "  (score %.1f -> %.1f)" % (base, best_s))
    _opt_set(result=best_r, cancelled=was_cancelled,
             message=("stopped: " if was_cancelled else "finished: ") + _summ(best_r))


def _opt_thread(params):
    import traceback as _tb
    try:
        _opt_job(params)
    except Exception as e:
        _opt_set(error=str(e), message="ERROR: %s" % e)
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
                        "tuned": TUNED, "domain": DOMAIN,
                        "geometry": _geometry_info()})
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
                    _nes = _normal_edge_set()
                    for em in _re.finditer(r"polyLine\s+(\d+)\s+(\d+)\s*\((.*?)\)(?=\s*polyLine|\s*$)",
                                           m3.group(1), _re.S):
                        a, b = int(em.group(1)), int(em.group(2))
                        if (min(a, b), max(a, b)) in _nes:
                            continue
                        mid = [[float(x) for x in ln.split()]
                               for ln in _re.findall(r"\(([^()]+)\)", em.group(3))]
                        curves.append([verts[a]] + mid + [verts[b]])
                        curve_ids.append([a, b])
            except (ValueError, AttributeError) as e:
                return self._json({"error": f"dict parse failed: {e}"}, 500)
            _wall_e = set()
            for _q in fidx:
                for _a, _b in zip(_q, _q[1:] + _q[:1]):
                    if _a != _b:
                        _wall_e.add((min(_a, _b), max(_a, _b)))
            _have_e = {(min(a_, b_), max(a_, b_)) for a_, b_ in curve_ids}
            n_straight = len(_wall_e - _have_e)

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
                   "drift": drift, "n_straight": n_straight}
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
        if u.path == "/api/geometry":
            # Geometry (STL) location, set from the GUI panel above the domain.
            # Persisted in system/geometry.json so the exporter and the SDF
            # builder read the very same file.
            try:
                raw = data.get("stl", "")
                if not isinstance(raw, str) or not raw.strip():
                    return self._json({"error": "STL path required"}, 400)
                p = raw.strip().strip('"').strip("'")
                full = os.path.normpath(p if os.path.isabs(p)
                                        else os.path.join(CASE, p))
                if not os.path.isfile(full):
                    return self._json({"error": "no such file: %s" % full}, 400)
                tris = _stl_tris(full)
                if tris is None:
                    return self._json(
                        {"error": "%s is not a binary STL (expected 84 + 50*n bytes); "
                                  "export the geometry as a binary STL" % full}, 400)
                rel = os.path.relpath(full, CASE)
                rel = full if rel.startswith("..") else rel
                rel = rel.replace("\\", "/")
                old = _stl_rel()
                with open(GEOMETRY_CFG, "w") as f:
                    json.dump({"stl": rel}, f, indent=2)
                _stl_cache.clear()
                _sdf_surface_cache.clear()
                for _k in ("smooth", "cent_smooth", "cr_smooth"):
                    _snap_cache.pop(_k, None)
                info = _geometry_info()
                info.update({"ok": True, "changed": rel != old, "tris": tris})
                return self._json(info)
            except OSError as e:
                return self._json({"error": "cannot use that geometry: %s" % e}, 400)
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
                    _snap_cache.pop("cent_smooth", None)
                    _snap_cache.pop("cr_smooth", None)
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
            # Automatic block grading: each outer slab is graded in its own direction so the
            # cell next to the inner (dead) blocks has the inner cell size - no jump in size.
            grading = None
            try:
                _gl, _gp, grading = _grade_outer_lines(
                    xs, ys, zs, solid_ov, fluid_ov, bool(payload.get("flat", False)))
                for _ax in "xyz":
                    payload[_ax + "_lines"] = _gl[_ax]
            except Exception as e:
                grading = {"error": str(e)}
            with open(os.path.join(SYS, "splits.json"), "w") as f:
                json.dump(payload, f)
            code, log = run([sys.executable, GENERATOR], timeout=600)
            tail = "\n".join(log.strip().splitlines()[-14:])
            if code != 0:
                return self._json({"ok": False, "log": tail})
            out = {"ok": True, "log": tail, "grading": grading}
            # whatever the generator left straight on the interface is curved now
            out["interface_edges"] = _refresh_interface_edges(None, missing_only=True)
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
            _old_v = list(vlines)
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
            _chg = {i for i, (a_, b_) in enumerate(zip(_old_v, vlines)) if a_ != b_}
            _ie = _refresh_interface_edges(_chg) if _chg else {"ok": True, "written": 0}
            return self._json({"ok": True, "patched": _n, "vetoed": _vetoed,
                               "interface_edges": _ie})
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
            _ie = _refresh_interface_edges({_vi})
            return self._json({"ok": True, "vid": _vi, "point": _base,
                               "interface_edges": _ie})
        if u.path == "/api/setvertices":
            # Undo/Redo of vertex moves: {states: [{vid, pos:[x,y,z], pinned: bool}]}
            try:
                moves, unp = {}, set()
                for st in data["states"]:
                    vid = int(st["vid"])
                    pos = [float(c) for c in st["pos"]]
                    if len(pos) != 3 or not all(c == c and abs(c) < 1e12 for c in pos):
                        raise ValueError("bad position")
                    moves[vid] = pos
                    if not st.get("pinned"):
                        unp.add(vid)
            except (KeyError, TypeError, ValueError) as e:
                return self._json({"ok": False, "error": f"bad request: {e}"}, 400)
            if not moves:
                return self._json({"ok": True, "n": 0})
            try:
                _backup_dict()
                _write_vertices(moves, unp)
            except Exception as e:
                return self._json({"ok": False, "error": repr(e)}, 500)
            return self._json({"ok": True, "n": len(moves)})
        if u.path == "/api/interface_edges":
            # {missing_only: bool=true}: make every interface edge a projected polyLine
            _backup_dict()
            return self._json(_refresh_interface_edges(
                None, missing_only=bool(data.get("missing_only", True))))
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
            _gs = None
            try:
                _gl, _gp, _gs = _grade_outer_lines(
                    _bx, _by, _bz, _clean_triples(cur.get("solid", [])),
                    _clean_triples(cur.get("fluid", [])), bool(cur.get("flat", False)),
                    lines0={"x": _lx, "y": _ly, "z": _lz})
                cur["x_lines"], cur["y_lines"], cur["z_lines"] = _gl["x"], _gl["y"], _gl["z"]
            except Exception as e:
                _gs = {"error": str(e)}
            cur.pop("auto_grade", None)                 # settings of the old on/off switch
            cur.pop("grading_mode", None)
            with open(_p, "w") as f:
                json.dump(cur, f)
            code, log = run([sys.executable, GENERATOR], timeout=600)
            tail = "\n".join(log.strip().splitlines()[-14:])
            if code != 0:
                return self._json({"ok": False, "log": tail})
            _ie = _refresh_interface_edges(None, missing_only=True)
            return self._json({"ok": True, "log": tail, "grading": _gs, "interface_edges": _ie,
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
