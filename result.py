import re
import pandas as pd 
import pygmt
import matplotlib.pyplot as plt 
import os
import numpy as np 
from scipy.interpolate import NearestNDInterpolator

# NOTE: conda activate pygmt 


def main():
    fname = "gmm3_resp.csv"
    df = pd.read_csv(fname, index_col=0, comment='#')
    print(df)
    lon1 = df['lon1'].to_numpy()
    lat1 = df['lat1'].to_numpy()
    plot_gmt(lon1, lat1, df['resp_C0'].to_numpy(), 'gmm3_slow.jpg')
    # plot_gmt(lon1, lat1, df['resp_C1'].to_numpy(), 'gmm3_neutral.jpg')
    # plot_gmt(lon1, lat1, df['resp_C2'].to_numpy(), 'gmm3_fast.jpg')


    # fname = "gmm2_resp.csv"
    # df = pd.read_csv(fname, index_col=0, comment='#')
    # print(df)
    # lon1 = df['lon1'].to_numpy()
    # lat1 = df['lat1'].to_numpy()
    # # plot_gmt(lon1, lat1, df['resp_C0'].to_numpy(), 'gmm2_slow.jpg')
    # plot_gmt(lon1, lat1, df['resp_C1'].to_numpy(), 'gmm2_fast.jpg')

    return 

def plot_gmt(lon1, lat1, val1, output_name):
    if 'slow' in output_name:
        fname_cpt = "gmt_hot.cpt"
    if 'neutral' in output_name:
        fname_cpt = "gmt_neutral.cpt"
    if 'fast' in output_name:
        fname_cpt = "gmt_cold.cpt"
    
    title = output_name.split('.')[0].replace('_', ' ').title()
    # title = title.replace('Gmm', 'GMM')
    title = re.sub(r'(Gmm)(\d+)', lambda m: m.group(1).upper() + m.group(2), title)


    lon_2d, lat_2d = np.meshgrid(np.linspace(-180,180,91), np.linspace(-90,90,46))
    interp_ND = NearestNDInterpolator(np.hstack((lon1.reshape(-1,1),lat1.reshape(-1,1))), val1.reshape(-1,1)) 
    val_2d = interp_ND(lon_2d.reshape(-1), lat_2d.reshape(-1)) 
    val_2d = val_2d.reshape(lon_2d.shape)
    df = pd.DataFrame({
        'lon': lon_2d.flatten(),
        'lat': lat_2d.flatten(),
        'val': val_2d.flatten()
    })
    grid = pygmt.xyz2grd(data=df, region="d", spacing="4.0")
    pygmt.makecpt(
        cmap=fname_cpt,
        series=[0, 1, 0.2],
        continuous=True,
    )
    fig = pygmt.Figure()
    # =============================================
    fig.grdimage(
        grid=grid,
        cmap=True, 
        projection="H180/12c",  # Winkel: "R270/12c", Hammer: "H180/12c"
        frame=["45g45", "+t"+title],
    )
    fig.coast(resolution='crude', shorelines=True)
    fig.colorbar(frame=["x+lResponsibility"]) 
    fig.savefig(output_name)
    plt.tight_layout()
    plt.close()
    return 

if __name__ == "__main__":
    main()
