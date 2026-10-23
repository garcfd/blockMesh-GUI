"""Extract the zero isosurface from car_sdf.npy for the GUI."""
import json, os
import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
HERE=os.path.dirname(os.path.abspath(__file__))
field=np.load(os.path.join(HERE,"car_sdf.npy"),mmap_mode="r")
with open(os.path.join(HERE,"car_sdf.json"),encoding="utf-8") as f: m=json.load(f)
nz,ny,nx=field.shape
img=vtk.vtkImageData(); img.SetDimensions(nx,ny,nz)
img.SetOrigin(*m["origin"]); img.SetSpacing(*m["spacing"])
sc=numpy_to_vtk(np.asarray(field,dtype=np.float32).ravel(),deep=True,array_type=vtk.VTK_FLOAT)
img.GetPointData().SetScalars(sc)
iso=vtk.vtkFlyingEdges3D(); iso.SetInputData(img); iso.SetValue(0,0.0); iso.ComputeNormalsOff(); iso.Update()
pd=iso.GetOutput()
# Keep the GUI surface light enough to transfer as JSON while preserving the
# projection SDF at full resolution.
dec=vtk.vtkQuadricDecimation(); dec.SetInputData(pd); dec.SetTargetReduction(0.75); dec.Update()
pd=dec.GetOutput()
positions=vtk_to_numpy(pd.GetPoints().GetData()).astype(np.float32,copy=True)
raw=vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1,4)
if not np.all(raw[:,0]==3): raise SystemExit("zero surface contains non-triangular cells")
faces=raw[:,1:].astype(np.uint32,copy=True)
out=os.path.join(HERE,"car_sdf_surface.npz")
np.savez_compressed(out,positions=positions,faces=faces)
print(f"wrote {out}: {len(positions)} vertices, {len(faces)} triangles")
