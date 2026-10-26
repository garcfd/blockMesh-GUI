#!/usr/bin/env python3
"""Build a signed-distance field for the car.

Only a narrow band around the surface is sampled with the real STL
distance function; the rest of the bounding box is never evaluated.

Passes
  1. coarse   exact distances on every `stride`-th node (cheap, small)
  2. band     trilinear upsample of (1); keep |d| <= band + margin
  3. fine     exact distances for the band nodes only (multicore)
  4. fill     everything else gets a unit-gradient ramp carried out from
              the band edge (EDT), so SDFField.project()'s Newton solve
              still finds a value + gradient anywhere in the box

The zero level set — all that matters for projection and for
extract_car_sdf_surface.py — is exact, because it lies inside the band.
The ramp never changes sign, so it cannot invent a second surface.

usage: build_car_sdf.py [STEP_m] [--band-cm N] [--stride N] [--threads N]
"""
import json, math, multiprocessing, os, sys, time

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

CASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STL = os.path.join(CASE, "car_simplified.stl")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "car_sdf.npy")
# Close bounds with a little margin so Newton projection never lands on a cap.
BOUNDS = (-2.40, 2.40, -0.72, 0.82, -1.14, 1.14)

_IMPL = None


def _get_impl():
    """Per-process STL distance function (safe under fork or spawn)."""
    global _IMPL
    if _IMPL is None:
        r = vtk.vtkSTLReader(); r.SetFileName(STL); r.Update()
        im = vtk.vtkImplicitPolyDataDistance(); im.SetInput(r.GetOutput())
        _IMPL = im
    return _IMPL


def _eval_chunk(job):
    start, xyz = job
    ev = _get_impl().EvaluateFunction
    out = np.empty(len(xyz), dtype=np.float32)
    for n in range(len(xyz)):
        p = xyz[n]
        out[n] = ev(float(p[0]), float(p[1]), float(p[2]))
    return start, out


def _parse(argv):
    step, band_cm, stride, threads = 0.006, 8.0, 3, None
    pos = []
    i = 1
    while i < len(argv):
        a = argv[i]
        if a.startswith("--"):
            k, _, v = a.partition("=")
            if i + 1 < len(argv) and not v:
                v = argv[i + 1]; i += 1
            if k == "--band-cm":
                band_cm = float(v)
            elif k == "--stride":
                stride = max(1, int(v))
            elif k == "--threads":
                threads = max(1, int(v))
            else:
                raise SystemExit(f"unknown option {k}")
        else:
            pos.append(float(a))
        i += 1
    if pos:
        step = pos[0]
    if not (0.0005 <= step <= 0.1):
        raise SystemExit(f"step {step} outside 0.5..100 mm")
    if not (1.0 <= band_cm <= 30.0):
        raise SystemExit("band must be between 1 and 30 cm")
    if threads is None:
        threads = min(8, os.cpu_count() or 1)
    return step, band_cm, stride, threads


def main():
    t0 = time.time()
    step, band_cm, stride, threads = _parse(sys.argv)
    band = band_cm / 100.0

    reader = vtk.vtkSTLReader(); reader.SetFileName(STL); reader.Update()
    poly = reader.GetOutput()
    if poly.GetNumberOfPolys() == 0:
        raise SystemExit("STL contains no triangles")

    # Grid, padded so the coarse pass lands exactly on stride boundaries.
    span = [BOUNDS[2 * a + 1] - BOUNDS[2 * a] for a in range(3)]
    dims = []
    for s in span:
        n = int(math.ceil(s / step))
        n = ((n + stride - 1) // stride) * stride   # n % stride == 0
        dims.append(n + 1)
    origin = list(BOUNDS[::2])
    spacing = [span[a] / (dims[a] - 1) for a in range(3)]

    # ---- pass 1: coarse exact -------------------------------------------
    cdim = [dims[a] // stride + 1 for a in range(3)]
    cbounds = []
    for a in range(3):
        cbounds += [origin[a], origin[a] + (cdim[a] - 1) * stride * spacing[a]]
    sample = vtk.vtkSampleFunction()
    implicit = _get_impl()
    sample.SetImplicitFunction(implicit)
    sample.SetModelBounds(*cbounds)
    sample.SetSampleDimensions(*cdim)
    sample.ComputeNormalsOff()
    sample.SetOutputScalarTypeToFloat()
    sample.Update()
    cfield = vtk_to_numpy(sample.GetOutput().GetPointData().GetScalars()).copy()
    cfield = cfield.reshape((cdim[2], cdim[1], cdim[0]))
    t1 = time.time()

    # ---- pass 2: band selection -----------------------------------------
    # Trilinear upsample of the coarse field. Near the surface the distance
    # function is locally planar, so the error there is a fraction of a cell
    # — the margin covers the crease case where it is not.
    from scipy.ndimage import map_coordinates
    zz, yy, xx = np.meshgrid(
        np.arange(dims[2], dtype=np.float64) / stride,
        np.arange(dims[1], dtype=np.float64) / stride,
        np.arange(dims[0], dtype=np.float64) / stride,
        indexing="ij")
    est = map_coordinates(cfield, np.stack([zz, yy, xx]), order=1,
                          mode="nearest").astype(np.float32)
    del zz, yy, xx
    margin = 0.5 * stride * min(spacing)
    mask = np.abs(est) <= (band + margin)
    n_band = int(mask.sum())
    n_total = int(est.size)
    if n_band == 0:
        raise SystemExit("band selection came up empty — STL/bounds mismatch?")
    del cfield

    # ---- pass 3: fine exact inside the band ------------------------------
    idx = np.flatnonzero(mask)
    px = origin[0] + (idx % dims[0]) * spacing[0]
    py = origin[1] + ((idx // dims[0]) % dims[1]) * spacing[1]
    pz = origin[2] + (idx // (dims[0] * dims[1])) * spacing[2]
    pts = np.stack([px, py, pz], axis=1)
    del px, py, pz
    field = np.zeros(n_total, dtype=np.float32)
    idx = np.asarray(idx)
    n_chunks = max(1, (n_band + 19999) // 20000)
    chunk = int(math.ceil(n_band / n_chunks))
    jobs = []
    for c in range(0, n_band, chunk):
        jobs.append((c, pts[c:c + chunk]))
    done = 0
    if threads > 1 and len(jobs) > 1:
        with multiprocessing.Pool(threads) as pool:
            for start, vals in pool.imap_unordered(_eval_chunk, jobs, chunksize=1):
                field[idx[start:start + len(vals)]] = vals
                done += len(vals)
                print(f"band {100 * done // n_band}%", flush=True)
    else:
        for start, vals in map(_eval_chunk, jobs):
            field[idx[start:start + len(vals)]] = vals
            done += len(vals)
            print(f"band {100 * done // n_band}%", flush=True)
    del pts
    t2 = time.time()

    # ---- pass 4: ramp the rest ------------------------------------------
    from scipy.ndimage import distance_transform_edt
    flat = field.reshape(dims[2], dims[1], dims[0])
    m = mask.reshape(dims[2], dims[1], dims[0])
    filled = True
    if not m.all():
        dist, (iy, ix, iz) = distance_transform_edt(
            ~m, return_indices=True, sampling=tuple(spacing))
        near = flat[iy, ix, iz]
        sgn = np.where(np.abs(near) > 1e-9, np.sign(near), np.sign(est.reshape(
            dims[2], dims[1], dims[0])[iy, ix, iz]))
        sgn = np.where(sgn == 0, 1.0, sgn)
        h = float(np.mean(spacing))
        ramp = sgn * (np.abs(near) + np.maximum(dist - h, 0.0))
        flat = np.where(m, flat, ramp.astype(np.float32))
    else:
        filled = False
    field = np.ascontiguousarray(flat)
    t3 = time.time()

    np.save(OUT, field, allow_pickle=False)
    with open(os.path.splitext(OUT)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"origin": origin, "spacing": spacing, "shape_xyz": dims,
                   "source": "car_simplified.stl",
                   "method": "exact signed distance to STL triangles in a "
                             "%g cm band, ramped beyond" % band_cm,
                   "band_cm": band_cm, "stride": stride,
                   "band_nodes": n_band, "total_nodes": n_total,
                   "ramped": filled, "dtype": "float32"}, f, indent=2)
    print("wrote %s; dimensions xyz=%s, spacing=%s, memory=%.1f MB"
          % (OUT, list(field.shape[::-1]), spacing, field.nbytes / 1e6))
    print("band %d/%d nodes (%.1f%%), threads=%d, ramped=%s"
          % (n_band, n_total, 100.0 * n_band / n_total, threads, filled))
    print("timing coarse %.1fs  band %.1fs  fill %.1fs  total %.1fs"
          % (t1 - t0, t2 - t1, t3 - t2, time.time() - t0))


if __name__ == "__main__":
    main()
