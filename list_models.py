import netCDF4
import os
import numpy as np 

models = {
    "GLAD-M35": "GLAD-M35.r0.1-n4.nc", 
    "REVEAL": "REVEAL-viz-only.r0.0.nc", 
    "SPiRaL-1.4": "SPiRaL-1.4.Interpolated.Flattened-viz-only.r0.0.nc", 
    "TX2019slab": "TX2019slab_percent.nc", 
    "SEMUCB-WM1": "semucb-2014-ucb-vs-viz-only.r0.1.nc",
    "SEISGLOB2": "SEISGLOB2_percent.nc",
    "S362WMANI": "S362WMANI_percent.nc", 
    "SGLOBE-rani": "SGLOBE-rani-voigt_percent.nc", 
    "HMSL-S06": "HMSL-S06_percent.nc", 
    "S40RTS": "S40RTS_dvs.nc", 
    # "SPani": "spani.nc",  # deprecated 
    # "SALSA3D-v2": "salsa3d-v2-viz-only.r0.0.nc",  # deprecated 
}
variable_name = {
    "GLAD-M35": "vsv", 
    "REVEAL": "vsv", 
    "SPiRaL-1.4": "vsv", 
    "TX2019slab": "dvs",
    "SEMUCB-WM1": "vs", 
    "SEISGLOB2": "dvs", 
    "S362WMANI": "dvs", 
    "SGLOBE-rani": "dvs",
    "HMSL-S06": "dvs",
    "S40RTS": "v", 
    "SPani": "dlnvs",
    "SALSA3D-v2": "vs"
}

def main():
    cwd = os.getcwd()   # current working directory  
    dir_model = "models/" 
    print("-" * 50)
    for name, file_name in models.items():
        # file_name = url.split('/')[-1]
        file_path = os.path.join(cwd, dir_model, file_name)
        # 1. Check file existence
        if not os.path.exists(file_path):
            print(f">> Please download {name} ... ") 
        else:
            with netCDF4.Dataset(file_path, 'r') as nc_file:
                print(f"File: {file_name}")
                # Sanity check 
                dims = [f"{dim}({nc_file.dimensions[dim].size})" for dim in nc_file.dimensions]
                print(f"  Dimensions: {', '.join(dims)}")
                vars_list = list(nc_file.variables.keys())
                print(f"  Variables: {', '.join(vars_list)}")
                print(f"  Longitude range: {np.min(nc_file.variables['longitude'][:])} to {np.max(nc_file.variables['longitude'][:])}")
                print(f"  Latitude range: {np.min(nc_file.variables['latitude'][:])} to {np.max(nc_file.variables['latitude'][:])}")
                print(f"  Depth range: {np.min(nc_file.variables['depth'][:])} to {np.max(nc_file.variables['depth'][:])}")
        print("-" * 50)
    return 

if __name__ == "__main__":
    main() 