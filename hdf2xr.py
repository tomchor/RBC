import xarray as xr
import h5py as h5
from glob import glob
import os

fs = glob("*h5")
fs.sort(key=os.path.getmtime)
f=fs[-1]
f = h5.File(f, mode="r")

das = dict()
for sva in list(f["tasks"]):
    print(sva)
    va = f["tasks"][sva]

    coords = dict()
    dims = []
    for i, d in enumerate(list(va.dims)):
        print(d)
        d = d.label
        vals = va.dims[i].items()[0][1].value
        coords[d] = vals
        dims.append(d)
    da = xr.DataArray(va[()], dims=dims, coords=coords, attrs=dict(short_name=sva))
    das[sva] = da

ds = xr.Dataset(das)
