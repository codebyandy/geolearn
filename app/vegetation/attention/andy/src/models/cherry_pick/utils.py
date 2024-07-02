"""
This file contains some important constants.
"""

# satellite variable names
varS = ['VV', 'VH', 'vh_vv']
varL = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'ndvi', 'ndwi', 'nirv']
varM = ["MCD43A4_b{}".format(x) for x in range(1, 8)]

# Days per satellite
bS = 8
bL = 6
bM = 10