from list_models import models, variable_name 
import netCDF4
import os
import numpy as np 
import shtns
import pandas as pd 
import matplotlib.pyplot as plt 
from gmm import resampling


depth_target = 2750

nmax = 18   # spherical harmonic decomposition 


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


def main():
    cwd = os.getcwd()   # current working directory  
    dir_model = "models/" 
    d = {} # for models 
    os.makedirs("filtered_data", exist_ok=True)

    for name, file_name in models.items():
        print(f"File: {file_name}")
        file_path = os.path.join(cwd, dir_model, file_name)

        if not os.path.exists(file_path):
            print(f"Stopping loop: {file_name} not found.")
            continue

        with netCDF4.Dataset(file_path, 'r') as nc_file:
            # Depth selection
            depth = nc_file.variables['depth'][:]
            idx_depth = np.argmin(np.abs(depth - depth_target))
            depth_selected = depth[idx_depth]
            print(f"  Selected depth (closest to target): {depth_selected} km")

            # Anomaly calculation 
            lons = nc_file.variables['longitude'][:]
            lats = nc_file.variables['latitude'][:]
            if variable_name[name] in ['vsv','vs']:
                vs = nc_file.variables[variable_name[name]][idx_depth,:,:] 
                dlnvs = (vs - calculate_surface_average(lons, lats, vs)) / calculate_surface_average(lons, lats, vs) * 100
            else: 
                dlnvs = nc_file.variables[variable_name[name]][idx_depth,:,:]
            
            # SH decomposition 
            data_2d = dlnvs
            nlat, nlon = data_2d.shape
            data_ready = np.array(data_2d, dtype=np.float64, copy=True)
            sh = shtns.sht(nmax, nmax)
            sh.set_grid(nlat, nlon, flags=shtns.sht_reg_poles)
            zlm = sh.analys(data_ready)
            data_reconstruct = sh.synth(zlm)
            lons_2d, lats_2d = np.meshgrid(lons, lats)

            if len(models)==1:
                # """ (Optional) Plot for verification 
                print(nlon, nlat) 
                print(calculate_surface_average(lons, lats, dlnvs))
                print(calculate_surface_average(lons, lats, data_reconstruct))
                fig, axs = plt.subplots(2,2,figsize=[12,8]) 
                ax = axs[0,0]
                im = ax.contourf(lons_2d, lats_2d, dlnvs, levels=np.linspace(-2.5,2.5,21), cmap='RdBu', extend='both')
                cb = fig.colorbar(im, ax=ax, location='bottom')
                ax = axs[1,0]
                im = ax.contourf(lons_2d, lats_2d, data_reconstruct, levels=np.linspace(-2.5,2.5,21), cmap='RdBu', extend='both')
                cb = fig.colorbar(im, ax=ax, location='bottom')
                lon1, lat1, val1 = resampling(lons_2d, lats_2d, data_reconstruct)
                ax = axs[1,1]
                val1_mean = np.mean(val1) 
                mu2 = np.mean((val1-val1_mean)**2) # second central moment ~ variance 
                mu3 = np.mean((val1-val1_mean)**3) # third central moment ~ skewness 
                ax.hist(val1, bins=np.linspace(-2.5,2.5,51), label='%.2f'%(mu3/(mu2**1.5)))
                ax.legend()
                plt.suptitle(name+' at depth = %4d km'%depth_selected)
                plt.tight_layout()
                plt.savefig('foo.jpg')
                # """                
            else: 
                d['lon'] = lons_2d.reshape(-1) 
                d['lat'] = lats_2d.reshape(-1) 
                d['dvs'] = data_reconstruct.reshape(-1) 
                with open(f'./filtered_data/{name}_lmax{nmax}_%4d.csv'%(depth_target), "w") as f:
                    f.write("# Depth selected = %4d\n"%depth_selected)
                data = pd.DataFrame(data=d) 
                data.to_csv(f'./filtered_data/{name}_lmax{nmax}_%4d.csv'%(depth_target), mode="a", index=False) 
                d = {}

    return 

def calculate_surface_average(lons, lats, data):
    """
    計算全球表面加權平均值。
    """
    # 1. 將緯度轉為餘緯 (Colatitude) theta，並轉為弧度
    # 假設 lats 範圍是 -90 到 90
    theta = np.deg2rad(90 - lats) 
    
    # 2. 自動計算網格間距 (弧度)
    # 這裡假設是均勻網格，取第一個差值
    dtheta = np.abs(np.deg2rad(lats[1] - lats[0]))
    dphi = np.abs(np.deg2rad(lons[1] - lons[0]))
    
    # 3. 建立 2D 的 sin(theta) 權重矩陣，形狀需與 data 一致 [n_lat, n_lon]
    # 使用 np.meshgrid 或直接利用 broadcasting
    weights = np.sin(theta).reshape(-1, 1) # 轉為 [n_lat, 1] 方便與 [n_lat, n_lon] 相乘
    
    # 4. 計算加權總和與總面積
    # 面積元素 dA = R^2 * sin(theta) * dtheta * dphi (R^2 可約掉)
    total_weighted_sum = np.sum(data * weights) * dtheta * dphi
    total_area = np.sum(np.ones_like(data) * weights) * dtheta * dphi   # compare to 4*pi 

    surf_avg = total_weighted_sum / total_area
    return surf_avg

if __name__ == "__main__":
    main() 
