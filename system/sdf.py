"""Trilinear queries and projection against a sampled signed-distance field."""
import json
import os
import numpy as np

class SDFField:
    def __init__(self, path):
        self.path = path
        self.data = np.load(path, mmap_mode="r")
        with open(os.path.splitext(path)[0] + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        self.origin = np.asarray(meta["origin"], dtype=float)
        self.spacing = np.asarray(meta["spacing"], dtype=float)
        self.shape_xyz = np.asarray(meta["shape_xyz"], dtype=int)

    def sample_gradient(self, points):
        p = np.atleast_2d(np.asarray(points, dtype=float))
        u = (p - self.origin) / self.spacing
        hi = self.shape_xyz - 2
        if np.any(u < 0) or np.any(u > hi):
            raise ValueError("point lies outside whole-car SDF bounds")
        i = np.floor(u).astype(int)
        t = u - i
        x, y, z = i[:, 0], i[:, 1], i[:, 2]
        tx, ty, tz = t[:, 0], t[:, 1], t[:, 2]
        # Stored in z,y,x order; convert the eight field corners to float32.
        c000 = np.asarray(self.data[z, y, x], dtype=np.float32)
        c100 = np.asarray(self.data[z, y, x+1], dtype=np.float32)
        c010 = np.asarray(self.data[z, y+1, x], dtype=np.float32)
        c110 = np.asarray(self.data[z, y+1, x+1], dtype=np.float32)
        c001 = np.asarray(self.data[z+1, y, x], dtype=np.float32)
        c101 = np.asarray(self.data[z+1, y, x+1], dtype=np.float32)
        c011 = np.asarray(self.data[z+1, y+1, x], dtype=np.float32)
        c111 = np.asarray(self.data[z+1, y+1, x+1], dtype=np.float32)
        def mix(a,b,t): return a + (b-a)*t
        c00=mix(c000,c100,tx); c10=mix(c010,c110,tx)
        c01=mix(c001,c101,tx); c11=mix(c011,c111,tx)
        c0=mix(c00,c10,ty); c1=mix(c01,c11,ty)
        val=mix(c0,c1,tz)
        gx=(mix(mix(c100-c000,c110-c010,ty),
                mix(c101-c001,c111-c011,ty),tz)/self.spacing[0])
        gy=(mix(mix(c010-c000,c110-c100,tx),
                mix(c011-c001,c111-c101,tx),tz)/self.spacing[1])
        gz=(mix(mix(c001-c000,c101-c100,tx),
                mix(c011-c010,c111-c110,tx),ty)/self.spacing[2])
        return val, np.column_stack((gx,gy,gz))

    def project(self, points, iterations=12, tolerance=2e-5):
        one = np.asarray(points, dtype=float).ndim == 1
        p = np.atleast_2d(np.asarray(points, dtype=float)).copy()
        # Points outside the sampled box cannot be interpolated. Clamp them into
        # the box (the car touches the domain walls) so a single out-of-field
        # vertex does not abort the whole projection batch.
        u = (p - self.origin) / self.spacing
        np.clip(u, 0.0, (self.shape_xyz - 2).astype(float), out=u)
        p = self.origin + u * self.spacing
        for _ in range(iterations):
            d, g = self.sample_gradient(p)
            g2 = np.einsum("ij,ij->i", g, g)
            active = (np.abs(d) > tolerance) & (g2 > 1e-8)
            if not np.any(active): break
            step = np.zeros_like(p)
            step[active] = (d[active] / g2[active])[:,None] * g[active]
            length = np.linalg.norm(step, axis=1)
            cap = length > 0.08
            step[cap] *= (0.08 / length[cap])[:,None]
            p -= step
        if one: return p[0]
        return p

_FIELD = None
def load_field(path):
    global _FIELD
    if _FIELD is None or _FIELD.path != path:
        _FIELD = SDFField(path)
    return _FIELD

def reload_field(path):
    """Drop the cached/memory-mapped field after rebuilding its data file."""
    global _FIELD
    _FIELD = None
    return load_field(path)
