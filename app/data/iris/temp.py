from iris import experimental
import cftime
import netCDF4
import hydroDL

import os
import iris
import numpy as np
from iris.coords import DimCoord
from iris.cube import Cube

os.environ['PROJ_LIB']


lat1 = DimCoord(np.arange(9),
                standard_name='latitude',
                units='degrees')
lon1 = DimCoord(np.arange(9),
                standard_name='longitude',
                units='degrees')
lat2 = DimCoord(np.arange(1, 9, 3),
                standard_name='latitude',
                units='degrees')
lon2 = DimCoord(np.arange(1, 9, 3),
                standard_name='longitude',
                units='degrees')
lat1.guess_bounds()
lon1.guess_bounds()
lat2.guess_bounds()
lon2.guess_bounds()
cube1 = Cube(np.random.rand(9, 9),
             dim_coords_and_dims=[(lat1, 0),
                                  (lon1, 1)])
# cube2 = Cube(np.zeros((2, 4)), dim_coords_and_dims=[(lat2, 0), (lon2, 1)])
cube2 = Cube(np.random.rand(3, 3), dim_coords_and_dims=[(lat2, 0), (lon2, 1)])

cube3 = cube1.regrid(cube2, iris.analysis.AreaWeighted(mdtol=0))

cube1.data[:3, :3]
np.mean(cube1.data[:3, :3])
cube3.data

experimental.regrid_area_weighted_rectilinear_src_and_grid(cube1, cube2)

fname = iris.sample_data_path("ostia_monthly.nc")
cube = iris.load_cube(fname)
tt = cube.coord('time')

netCDF4.num2date(tt)

for sub_cube in cube.slices_over('time'):
    print(sub_cube)

cftime.num2pydate(tt.points, tt.units.origin)
pdt = iris.time.PartialDateTime(month=4, day=3)
