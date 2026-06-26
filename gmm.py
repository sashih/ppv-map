import os 
import pandas as pd 
import os
from icosphere import icosphere
from scipy.interpolate import NearestNDInterpolator
import numpy as np 
import sys 
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture

# 4 recent 
# features = ['GLAD-M35', 'REVEAL', 'SPiRaL-1.4', 'TX2019slab']

# Final result 
features = ['GLAD-M35', 'SEMUCB-WM1', 'SPiRaL-1.4', 'TX2019slab']

# All N=10 
# features = ['GLAD-M35', 'REVEAL', 'SPiRaL-1.4', 'TX2019slab', 'HMSL-S06', 'S362WMANI', 'SEISGLOB2', 'SEMUCB-WM1', 'S40RTS', 'SGLOBE-rani']

target_depth = '2750'  # Keeping it as a string to match the filename
target_lmax = '18'


n_clusters = 3  # 3 or 2 

def main():
    dir_path = "./filtered_data/"
    names = os.listdir(dir_path)
    selected_files = [
        name for name in names 
        if any(name.startswith(f"{model}_lmax{target_lmax}_") for model in features) 
        and name.endswith(f"_{target_depth}.csv")
    ]
    d = {}
    print(f"--- Found {len(selected_files)} files (lmax={target_lmax}, depth={target_depth}) ---")
    for file in sorted(selected_files):
        print(file)
        df = pd.read_csv(dir_path+file, comment="#")
        lon = df['lon'].to_numpy()
        lat = df['lat'].to_numpy()
        dvs = df['dvs'].to_numpy()
        nlon, nlat = len(np.unique(lon)), len(np.unique(lat))
        print('  ', nlon, nlat) 
        lon_2d = lon.reshape(nlat,nlon)
        lat_2d = lat.reshape(nlat,nlon)
        dvs_2d = dvs.reshape(nlat,nlon)
        # Resampling 
        lon1, lat1, val1 = resampling(lon_2d, lat_2d, dvs_2d, nu=16)
        model = file.split('_', 1)[0]
        d[model] = val1 

    data = pd.DataFrame(data=d) 
    X = data[features].copy()
    print(X)

    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', n_init=10, random_state=32)
    pred = gmm.fit_predict(X)
    resp = gmm.predict_proba(X)

    # Rank clusters by mean velocity (Slow -> Neutral -> Fast)
    means = np.mean(gmm.means_, axis=1)
    order = np.argsort(means) # Indices: [slow_idx, neutral_idx, fast_idx]
    rank = np.argsort(order)
    print(means, order, rank)



    # ------------------------------------------------------------------------------------------------
    # PATCH: PRINT IN SPECIFIED TABLE FORMAT (DYNAMIC FOR 2 OR 3 CLUSTERS)
    # ------------------------------------------------------------------------------------------------
    print("\n" + "="*80)
    print("  GMM SUMMARY TABLE (mean ± std)")
    print("="*80)
    
    # Dynamically define labels based on cluster size
    if n_clusters == 2:
        labels = ["Slow", "Fast"]
    elif n_clusters == 3:
        labels = ["Slow", "Neutral", "Fast"]
    else:
        labels = [f"Cluster_{i}" for i in range(n_clusters)]
        
    # Build dynamic header with weights
    header_parts = ["model"]
    for i, label in enumerate(labels):
        weight = gmm.weights_[order[i]] * 100
        header_parts.append(f"{label} ({weight:.1f}%)")
    
    header = " | ".join(header_parts)
    print(header)
    print("-" * len(header))
    
    # Gather statistics for each model feature
    for feat_idx, feature_name in enumerate(features):
        row_parts = [feature_name]
        for i in range(n_clusters):
            m_val = gmm.means_[order[i]][feat_idx]
            s_val = np.sqrt(gmm.covariances_[order[i]][feat_idx][feat_idx])
            row_parts.append(f"{m_val:.2f}±{s_val:.2f}")
            
        print(" | ".join(row_parts))
            
    print("="*80 + "\n")
    # ------------------------------------------------------------------------------------------------




    # ------------------------------------------------------------------------------------------------
    # output lon, lat, features, cluster assignment, and responsibility for each cluster
    output_df = pd.DataFrame(data={'lon1': lon1, 'lat1': lat1}) 
    output_df = output_df.join(X) 
    output_df['Cluster'] = pred
    # sort cluster by means and assign cluster labels accordingly, so that Cluster 0 is slow, Cluster 1 is neutral, and Cluster 2 is fast
    output_df['Cluster'] = output_df['Cluster'].replace(np.arange(n_clusters), rank)
    for i in range(n_clusters):
        output_df[f'resp_C{i}'] = resp[:, order[i]] # Add responsibilities in the order of slow, neutral, fast 
    print(output_df)

    # output_df.to_csv('gmm%1d_resp.csv'%n_clusters)
    with open('gmm%1d_resp.csv'%n_clusters, "w") as f:
        f.write("# Metadata: depth=%4d, lmax=%2d\n"%(int(target_depth), int(target_lmax)))
        f.write("# -------------------------------------------------\n") 
        output_df.to_csv(f)
    # ------------------------------------------------------------------------------------------------

    return 


def generate_evenly_distributed_grid(nu=16, ifplot=False):  
    vertices, _ = icosphere(nu=nu)
    x, y, z = vertices[:,0], vertices[:,1], vertices[:,2] 
    
    phi = np.arctan2(y, x) 
    theta = np.arctan2(np.sqrt(x**2 + y**2), z) 
    lon1 = np.rad2deg(phi)
    lat1 = 90 - np.rad2deg(theta)
    
    # if ifplot:
    #     fig, ax = plt.subplots(figsize=(10, 5)) 
    #     m = Basemap(projection="hammer", lon_0=0, resolution='c')
    #     m.drawcoastlines(color='0.7')
    #     mx, my = m(lon1, lat1) 
    #     m.scatter(mx, my, s=2, color='red', alpha=0.5) 
    #     ax.set(title=f'Evenly Distributed Grid (N={len(lat1)})')
    #     plt.show()
    return lon1, lat1 

def resampling(lon, lat, val, nu=16, dlon=0):
    lon[lon>180] = lon[lon>180]-360
    lon1, lat1 = generate_evenly_distributed_grid(nu=nu, ifplot=False) 
    lon1 += dlon
    lon1[lon1>180] = lon1[lon1>180]-360
    interp_ND = NearestNDInterpolator(np.hstack((lon.reshape(-1,1),lat.reshape(-1,1))), val.reshape(-1,1)) 
    val1 = interp_ND(lon1, lat1).reshape(-1) 
    # Sanity check for NaN values after interpolation
    # idx = np.isnan(val1)
    # print('-- number of nan: %5d'%sum(np.isnan(val1)))
    return lon1, lat1, val1 

if __name__ == "__main__":
    main() 