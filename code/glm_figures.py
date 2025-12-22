# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:39:22 2025

@author: theo.rouhette

conda activate xesmf
cd C:\\GCAM\\Theo\\GCAM_7.2_Impacts\\python\climate_integration_metarepo\\code\\python
python glm_figures.py

"""

# Importing Needed Libraries
import os  # For navigating os
import numpy as np  # Numerical / array functions
import pandas as pd  # Data functions
import xarray as xr  # Reading and manipulating NetCDF data
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import cartopy.crs as ccrs

from pathlib import Path
fig_dir = Path("figures")
fig_dir.mkdir(parents=True, exist_ok=True)  

# PATHS
IF_PATH = "C:/GCAM/Theo/IAM-FIRE/zenodo"
inputs_dir = os.path.join(IF_PATH, f"inputs/")
outputs_dir = os.path.join(IF_PATH, f"outputs/")
figures_dir = os.path.join(IF_PATH, f"figures/")

# PARAMETERS
start_year = 2002
end_year = 2019
dates_obs = slice(f"{start_year}-01-31", f"{end_year}-12-31")
dates_obs_df = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq='YE')  # 'ME' means Month Start
dates_sim = pd.date_range(start=f"2020-01-31", end=f"2100-12-31", freq='YE')  # 'ME' means Month Start
variable = ['tas', 'pr', 'hurs', "sfcWind", "rsds"]

# INPUTS TO CREATE GLM FIGURES
landmask = xr.open_dataset(os.path.join(inputs_dir, "landseamask_no-ant.nc")).drop_vars("time").sel(time=0)
regions = xr.open_dataset(os.path.join(inputs_dir, "GFED_Regions.nc"))
figure_nc = xr.open_dataset(os.path.join(outputs_dir, "Predictions_NC_Annual.nc"))
figure_csv = figure_nc.to_dataframe().reset_index()

# Function to calculate grid cell area in km²
def grid_cell_area(lat, dlon=0.5, dlat=0.5):
    """Calculate grid cell area given latitude, assuming a 0.5° x 0.5° grid."""
    R = 6371  # Earth radius in km
    lat_rad = np.radians(lat)
    
    # Width of cell (km) varies with latitude
    cell_width = (np.pi / 180) * R * np.cos(lat_rad) * dlon  
    # Height of cell (km)
    cell_height = (np.pi / 180) * R * dlat  
    
    grid_area = cell_width * cell_height 
    
    return grid_area  # Area in km²

###############################################################################
# Process the data for figures 
###############################################################################

# Transform global dataset
figure_mean = figure_nc.sel(lat=slice(-55.25, 90)).mean(dim="time").where(landmask.mask == 1)

figure_csv = figure_csv.set_index(["time", "lat", "lon"])
figure_csv["lat_band"] = (figure_csv.index.get_level_values("lat") // 2) * 2
figure_reset = figure_csv.reset_index()
figure_reset["time_numeric"] = pd.to_datetime(figure_reset["time"]).dt.year  # if time is datetime
print("COLUMNS INITIAL:", figure_reset.columns)
figure_reset_raw = figure_reset.copy()

# Create a new variable with rounded region IDs
rounded_regions = regions['basisregions'].round()
region_ids = np.unique(rounded_regions.values)
region_ids = region_ids[~np.isnan(region_ids)]  # Remove any NaNs
region_ids = region_ids.astype(int)
region_ids = region_ids[region_ids > 0]  # Skip 0 if it means "no region"
print("Unique region IDs after rounding:", region_ids)

# # Prepare the array of region IDs present in your data
# region_ids = np.unique(ds['basisregions'].values)
region_name = {
    1: ("BONA", "Boreal North America"),
    2: ("TENA", "Temperate North America"),
    3: ("CEAM", "Central America"),
    4: ("NHSA", "Northern Hemisphere South America"),
    5: ("SHSA", "Southern Hemisphere South America"),
    6: ("EURO", "Europe"),
    7: ("MIDE", "Middle East"),
    8: ("NHAF", "Northern Hemisphere Africa"),
    9: ("SHAF", "Southern Hemisphere Africa"),
    10: ("BOAS", "Boreal Asia"),
    11: ("CEAS", "Central Asia"),
    12: ("SEAS", "Southeast Asia"),
    13: ("EQAS", "Equatorial Asia"),
    14: ("AUST", "Australia and New Zealand"),
}

fc_source = "GFED"

# Example pairs of observed vs predicted
var_pairs = [
    ("BA_area", "BA_area_pred", "Total BA per Region"),
    ("BA_area_for", "BA_area_for_pred", "Forest BA per Region"),
    ("CE_total_C", f"CE_total_C_pred_{fc_source}", "Total CE per Region"),
    ("CE_for_C", f"CE_for_C_pred_{fc_source}", "Forest CE per Region"),
]
    
# Aggregate by basisregion 
# Remove 0 regions before mapping
figure_csv = figure_csv[figure_csv["basisregions"] > 0].copy()

# Sum across space for each time and region
df_sum_time = figure_csv.groupby(["time", "basisregions"])[
    ["BA_area", "BA_area_pred", "BA_area_for", "BA_area_for_pred", 
     "CE_total_C", f"CE_total_C_pred_{fc_source}", "CE_for_C", f"CE_for_C_pred_{fc_source}"]
].sum().reset_index()

# Take the mean over time for each region
agg_df = df_sum_time.groupby("basisregions")[
    ["BA_area", "BA_area_pred", "BA_area_for", "BA_area_for_pred", 
     "CE_total_C", f"CE_total_C_pred_{fc_source}", "CE_for_C", f"CE_for_C_pred_{fc_source}"]
].mean().reset_index()

print(agg_df.head())

# Map region names
agg_df["region_name"] = agg_df["basisregions"].astype(int).map(lambda x: region_name[x][0])

# Save
save = True
if save == True:
    agg_df.to_csv(os.path.join(outputs_dir, "F3_Predictions_CSV_Regions_2001-2019_v5.1.csv"))

    
def F3_glm_figure(figure_nc, figure_csv, agg_df, save = True):

    ###########################################################################
    # FIGURE GLM 1. FINAL GLM FIGURE - NEW POST 30
    ###########################################################################

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import numpy as np
    import scipy.stats as stats
    import pymannkendall as mk
        
    
    # Create figure and GridSpec layout
    fig = plt.figure(figsize=(13, 10))
    gs = GridSpec(nrows=4, ncols=2, height_ratios=[1, 1, 1, 1], hspace=0.3)
    
    # Axes
    ax1 = fig.add_subplot(gs[0, 0])  # Total BA time series
    ax2 = fig.add_subplot(gs[0, 1])  # Total BA by regions
    
    ax3 = fig.add_subplot(gs[1, 0])  # Forest BA time series
    ax4 = fig.add_subplot(gs[1, 1])  # Forest BA by regions
    
    ax5 = fig.add_subplot(gs[2, 0])  # Total CE time series
    ax6 = fig.add_subplot(gs[2, 1])  # Total CE by regions

    ax7 = fig.add_subplot(gs[3, 0])  # Forest CE time series
    ax8 = fig.add_subplot(gs[3, 1])  # Forest CE by regions
        
    # ---------------- Panel 1: Total BA Time Series ----------------
    observed_total = figure_reset.groupby("time_numeric")["BA_area"].sum()
    pred_total = figure_reset.groupby("time_numeric")["BA_area_pred"].sum()
    
    # Numeric x-axis
    x = observed_total.index.values  # time_numeric (years)
    y_obs = observed_total.values
    y_pred = pred_total.values
    
    # Linear regression: observed vs predicted
    slope, intercept, r_value, _, _ = stats.linregress(x, y_obs)
    reg_line_obs = intercept + slope * x
    
    # Linear regression: observed vs predicted
    slope, intercept, r_value, _, _ = stats.linregress(x, y_pred)
    reg_line_pred = intercept + slope * x
    
    # Plot observed and predicted
    ax1.plot(x, y_obs, label="Observed Total BA", color='red')
    ax1.plot(x, y_pred, label="Predicted Total BA", color='blue')
    
    # Add regression line over same x-axis
    ax1.plot(x, reg_line_obs, color='red', linestyle='--')
    ax1.plot(x, reg_line_pred, color='blue', linestyle='--')
    
    ax1.set_title("(a) Total BA per Year")
    ax1.set_ylabel("Area (Mha)")
    # ax1.set_xlabel("Year")
    
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # MK test
    mk_obs = mk.original_test(y_obs)
    mk_pred = mk.original_test(y_pred)
    ax1.text(0.03, 0.15, f"Observed MK: {mk_obs.trend}", transform=ax1.transAxes,
             fontsize=8)
    ax1.text(0.03, 0.05, f"Predicted MK: {mk_pred.trend}", transform=ax1.transAxes,
             fontsize=8)
    
    # # Regression annotation
    # ax1.text(0.02, 0.8, f"Slope={slope:.2f}, Intercept={intercept:.2f}, R²={r_value**2:.2f}",
    #          transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    
    # ---------------- Panel 3: Forest BA Time Series ----------------
    obs_for = figure_reset.groupby("time_numeric")["BA_area_for"].sum()
    pred_for = figure_reset.groupby("time_numeric")["BA_area_for_pred"].sum()
    
    # Numeric x-axis
    x = obs_for.index.values  # time_numeric (years)
    y_obs = obs_for.values
    y_pred = pred_for.values
    
    # Linear regression: observed vs predicted
    slope, intercept, r_value, _, _ = stats.linregress(x, y_obs)
    reg_line_obs = intercept + slope * x
    
    # Linear regression: observed vs predicted
    slope, intercept, r_value, _, _ = stats.linregress(x, y_pred)
    reg_line_pred = intercept + slope * x
    
    # Plot observed and predicted
    ax3.plot(x, y_obs, label="Observed Forest BA", color='red')
    ax3.plot(x, y_pred, label="Predicted Forest BA", color='blue')
    
    # Add regression line over same x-axis
    ax3.plot(x, reg_line_obs, color='red', linestyle='--')
    ax3.plot(x, reg_line_pred, color='blue', linestyle='--')
    
    ax3.set_title("(c) Forest BA per Year")
    ax3.set_ylabel("Area (Mha)")
    # ax3.set_xlabel("Year")
    
    ax3.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # MK test
    mk_obs = mk.original_test(y_obs)
    mk_pred = mk.original_test(y_pred)
    ax3.text(0.03, 0.15, f"Observed MK: {mk_obs.trend}", transform=ax3.transAxes,
             fontsize=8)
    ax3.text(0.03, 0.05, f"Predicted MK: {mk_pred.trend}", transform=ax3.transAxes,
             fontsize=8)
    
    # # Regression annotation
    # ax3.text(0.02, 0.8, f"Slope={slope:.2f}, Intercept={intercept:.2f}, R²={r_value**2:.2f}",
    #          transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    
    
    # ---------------- Panel 5: Total CE Time Series ----------------
    figure_CE = figure_reset.loc[figure_reset["time"] != "2001-12-31"]
    observed_total = figure_CE.groupby("time_numeric")["CE_total_C"].sum()
    pred_total = figure_CE.groupby("time_numeric")[f"CE_total_C_pred_{fc_source}"].sum()
    
    # Numeric x-axis
    x = observed_total.index.values  # time_numeric (years)
    y_obs = observed_total.values
    y_pred = pred_total.values
    
    # Linear regression: observed vs predicted
    slope, intercept, r_value, _, _ = stats.linregress(x, y_obs)
    reg_line_obs = intercept + slope * x

    # Linear regression: observed vs predicted
    slope, intercept, r_value, _, _ = stats.linregress(x, y_pred)
    reg_line_pred = intercept + slope * x
        
    # Plot observed and predicted
    ax5.plot(x, y_obs, label="Observed Total CE", color='red')
    ax5.plot(x, y_pred, label="Predicted Total CE", color='blue')
    
    # Add regression line over same x-axis
    ax5.plot(x, reg_line_obs, color='red', linestyle='--')
    ax5.plot(x, reg_line_pred, color='blue', linestyle='--')
    
    ax5.set_title("(e) Total CE per Year")
    ax5.set_ylabel("Carbon (TgC)")
    # ax5.set_xlabel("Year")
    
    ax5.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # MK test
    mk_obs = mk.original_test(y_obs)
    mk_pred = mk.original_test(y_pred)
    ax5.text(0.03, 0.15, f"Observed MK: {mk_obs.trend}", transform=ax5.transAxes,
             fontsize=8)
    ax5.text(0.03, 0.05, f"Predicted MK: {mk_pred.trend}", transform=ax5.transAxes,
             fontsize=8)
    
    # # Regression annotation
    # ax1.text(0.02, 0.8, f"Slope={slope:.2f}, Intercept={intercept:.2f}, R²={r_value**2:.2f}",
    #          transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    
    
    
    
    # ---------------- Panel 7: Forest CE Time Series ----------------
    obs_for = figure_CE.groupby("time_numeric")["CE_for_C"].sum()
    pred_for = figure_CE.groupby("time_numeric")[f"CE_for_C_pred_{fc_source}"].sum()
    
    # Numeric x-axis
    x = obs_for.index.values  # time_numeric (years)
    y_obs = obs_for.values
    y_pred = pred_for.values
    
    # Linear regression: observed vs predicted
    slope, intercept, r_value, _, _ = stats.linregress(x, y_obs)
    reg_line_obs = intercept + slope * x

    # Linear regression: observed vs predicted
    slope, intercept, r_value, _, _ = stats.linregress(x, y_pred)
    reg_line_pred = intercept + slope * x
    
    # Plot observed and predicted
    ax7.plot(x, y_obs, label="Observed Forest CE", color='red')
    ax7.plot(x, y_pred, label="Predicted Forest CE", color='blue')
    
    # Add regression line over same x-axis
    ax7.plot(x, reg_line_obs, color='red', linestyle='--')
    ax7.plot(x, reg_line_pred, color='blue', linestyle='--')
    
    ax7.set_title("(g) Forest CE per Year")
    ax7.set_ylabel("Carbon (TgC)")
    ax7.set_xlabel("Year")
    
    ax7.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # MK test
    mk_obs = mk.original_test(y_obs)
    mk_pred = mk.original_test(y_pred)
    ax7.text(0.03, 0.15, f"Observed MK: {mk_obs.trend}", transform=ax7.transAxes,
             fontsize=8)
    ax7.text(0.03, 0.05, f"Predicted MK: {mk_pred.trend}", transform=ax7.transAxes,
             fontsize=8)
    
    # # Regression annotation
    # ax2.text(0.02, 0.8, f"Slope={slope:.2f}, Intercept={intercept:.2f}, R²={r_value**2:.2f}",
    #          transform=ax1.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

    
    
    # ---------------- Panels Regions: Regional comparisons ----------------
    region_axes = [ax2, ax4, ax6, ax8]  # predefined axes
    # Define GFED region order north → south
    region_order = [
        "BONA", "TENA", "CEAM", "NHSA", "SHSA",
        "EURO", "MIDE", "NHAF", "SHAF",
        "BOAS", "CEAS", "SEAS", "EQAS", "AUST"
    ]
    
    # Desired subplot letters
    panel_letters = ["b", "d", "f", "h"]
    for i, (obs_var, pred_var, title) in enumerate(var_pairs):
        ax = region_axes[i]  # assign ax9, ax10, ax11, ax12 in order
    
        # Aggregate per region
        region_grouped = agg_df.groupby("region_name")[[obs_var, pred_var]].sum().reindex(region_order)
    
        # Bar positions
        x = np.arange(len(region_grouped.index))
        width = 0.35
    
        # Plot observed vs predicted as side-by-side vertical bars
        ax.bar(x - width/2, region_grouped[obs_var], width, label="Observed", color="red")
        ax.bar(x + width/2, region_grouped[pred_var], width, label="Predicted", color="blue")
    
        # Formatting
        ax.set_title(f"({panel_letters[i]}) {title}")
        # ax.set_ylabel("Region")
        if i in (0, 1):
            ax.set_ylabel("Area (Mha)")
        else:
            ax.set_ylabel("Carbon (TgC)")
            
        # Only show x-axis region labels on the bottom panel (ax8)
        if i == 3:  # bottom-most regional panel
            ax.set_xticks(x)
            ax.set_xticklabels(region_grouped.index, rotation=45, ha="right")
            ax.set_xlabel("Region")
        else:
            ax.set_xticks([])  # remove x-ticks
            ax.set_xticklabels([])  # hide labels
        # ax.legend()

    import matplotlib.patches as mpatches
    
    # Define custom handles and labels
    handles = [
        mpatches.Patch(color="red", label="GFED5 Benchmark"),
        mpatches.Patch(color="blue", label="IAM-FIRE Predictions")
    ]
    
    labels = ["GFED5 Benchmark", "IAM-FIRE Predictions"]
    
    # Add one shared legend for the entire figure
    fig.legend(
        handles, labels,
        loc="lower center",         # position at bottom center
        ncol=2,                     # two columns
        bbox_to_anchor=(0.5, 0.03), # adjust spacing below figure
        frameon=False               # remove box around legend
    )
    
    plt.tight_layout()
    glm_figure=plt.gcf()    
    plt.show()
    
    if save == True:
        glm_figure.savefig(os.path.join(figures_dir, "F3_BA_Prediction_Observed_2019_FINAL_v5.1.png"))
    
def F4_glm_maps(figure_nc, figure_reset, mean = True, year = 2016, save = True):
    

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    import numpy as np
    
    # Determine the mean or the yearly dataset to map
    if mean == True: # Take the mean 
        figure_mean = figure_nc.sel(lat=slice(-55.25, 82.25)).mean(dim="time").where(landmask.mask == 1)
        figure_reset_mean = figure_reset.groupby("lat_band").mean()
    else: # Take the one year in year parameter 
        print("COLUMNS JUST BEFORE FILTER:", figure_reset.columns)
        figure_mean = figure_nc.sel(lat=slice(-55.25, 82.25)).sel(time=f"{year}-12-31").where(landmask.mask == 1)
        figure_reset_mean = figure_reset_raw.loc[figure_reset_raw["time_numeric"] == year]    
        
    # ---------------------------------------------------------------
    # Colormap setup (same as before)
    # ---------------------------------------------------------------
    colors = [
        "#e6e6ff", "#c6b3d9", "#85c6de", "#58a773",
        "#ffd966", "#f08000", "#e60000", "#67001f", "#330000"
    ]
    bounds = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, len(colors))

    # ---------------------------------------------------------------
    # Figure and GridSpec layout
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(11, 6.5))
    gs = GridSpec(
        nrows=2, ncols=3, figure=fig,
        width_ratios=[1, 1, 0.45],  # narrow right column for latitudinal profile
        height_ratios=[1, 1],
        wspace=0.05, hspace=0.12
    )

    # ---------------------------------------------------------------
    # MAP PANELS
    # ---------------------------------------------------------------
    # (a) Observed Total BA
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    im1 = figure_mean["BA_frac"].plot(
        ax=ax1, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, add_colorbar=False
    )
    ax1.coastlines(linewidth=0.4)
    ax1.set_title("(a) Observed Total BA", pad=4)
    ax1.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax1.set_yticks(np.arange(-60, 91, 30), crs=ccrs.PlateCarree())
    ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    # (b) Predicted Total BA
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    im2 = figure_mean["BA_frac_pred"].plot(
        ax=ax2, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, add_colorbar=False
    )
    ax2.coastlines(linewidth=0.4)
    ax2.set_title("(b) Predicted Total BA", pad=4)
    ax2.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    # ax2.set_yticks([])

    # LATITUDINAL PANEL – Total BA
    ax3 = fig.add_subplot(gs[0, 2])
    grouped_area = figure_reset_mean.groupby("lat_band")[["BA_area", "BA_area_pred"]].mean()
    ax3.plot(grouped_area["BA_area"], grouped_area.index, color="red", lw=1.3, label="GFED5 Benchmark")
    ax3.plot(grouped_area["BA_area_pred"], grouped_area.index, color="blue", lw=1.3, label="IAM-FIRE Predictions")
    ax3.set_title("(c) Total BA per Latitude", fontsize=10, pad=4)
    ax3.set_xlabel("Area (Mha)")
    # ax3.set_ylabel("Latitude (°)")
    # ax3.set_yticks(np.arange(-60, 91, 30))
    ax3.set_ylim(-60, 90)  # <<< aligns height with map panels
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax3.set_box_aspect(0.89)
    # <<< Remove y-axis labels and ticks
    ax3.set_yticks([])
    ax3.set_ylabel("")
    ax3.tick_params(axis='y', left=False)

    # ---------------------------------------------------------------
    # Second row — Forest BA
    # ---------------------------------------------------------------
    # (d) Observed Forest BA
    ax4 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    im4 = figure_mean["BA_frac_for"].plot(
        ax=ax4, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, add_colorbar=False
    )
    ax4.coastlines(linewidth=0.4)
    ax4.set_title("(d) Observed Forest BA", pad=4)
    ax4.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax4.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax4.set_yticks(np.arange(-60, 91, 30), crs=ccrs.PlateCarree())
    ax4.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    # (e) Predicted Forest BA
    ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    im5 = figure_mean["BA_frac_for_pred"].plot(
        ax=ax5, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, add_colorbar=False
    )
    ax5.coastlines(linewidth=0.4)
    ax5.set_title("(e) Predicted Forest BA", pad=4)
    ax5.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax5.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    # ax5.set_yticks([])

    # LATITUDINAL PANEL – Forest BA
    ax6 = fig.add_subplot(gs[1, 2])
    grouped_area_for = figure_reset_mean.groupby("lat_band")[["BA_area_for", "BA_area_for_pred"]].mean()
    ax6.plot(grouped_area_for["BA_area_for"], grouped_area_for.index, color="red", lw=1.3)
    ax6.plot(grouped_area_for["BA_area_for_pred"], grouped_area_for.index, color="blue", lw=1.3)
    ax6.set_title("(f) Forest BA per Latitude", fontsize=10, pad=4)
    ax6.set_xlabel("Area (Mha)")
    # ax6.set_yticks(np.arange(-60, 91, 30))
    ax6.set_ylim(-60, 90)  # <<< same fix here
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.legend(loc="lower right", frameon=False, fontsize=9)
        

    ax6.set_box_aspect(0.89)
    # <<< Remove y-axis labels and ticks
    ax6.set_yticks([])
    ax6.set_ylabel("")
    ax6.tick_params(axis='y', left=False)

    # ---------------------------------------------------------------
    # Shared legend for both latitudinal panels
    # ---------------------------------------------------------------
    fig.legend(
        handles=ax3.get_lines(),  # reuse handles from first latitudinal panel
        labels=["GFED5 Benchmark", "IAM-FIRE Predictions"],
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.825, 0.02)  # adjust position as needed
    )

    # ---------------------------------------------------------------
    # Shared colorbar
    # ---------------------------------------------------------------
    cbar_ax = fig.add_axes([0.20, 0.07, 0.5, 0.025])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Burned Area Fraction (0–1)", fontsize=12)

    glm_figure=plt.gcf() 
    plt.show()
    
    if save == True: 
        if mean == True: 
            glm_figure.savefig(os.path.join(figures_dir, f"F4_BA_Prediction_Observed_MAPS_LAT_MEAN_v5.1.png"))
        else:
            glm_figure.savefig(os.path.join(figures_dir, f"F4_BA_Prediction_Observed_MAPS_LAT_{year}_v5.1.png"))

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    import cartopy.crs as ccrs
    import cartopy.mpl.ticker as cticker
    import numpy as np

    # ---------------------------------------------------------------
    # Colormap setup (same as before)
    # ---------------------------------------------------------------

    # Define colors and bounds (adjust if using fraction values)
    colors = [
        "#e6e6ff", "#c6b3d9", "#85c6de", "#58a773",
        "#ffd966", "#f08000", "#e60000", "#67001f", "#330000"
    ]
    # Adjust bounds to go from 0 to 13
    # bounds = [0, 0.065, 0.13, 0.26, 0.65, 1.3, 2.6, 6.5, 9.1, 13.0]
    bounds = [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 7, 10]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, len(colors))


    # ---------------------------------------------------------------
    # Figure and GridSpec layout
    # ---------------------------------------------------------------
    fig = plt.figure(figsize=(11, 6.5))
    gs = GridSpec(
        nrows=2, ncols=3, figure=fig,
        width_ratios=[1, 1, 0.45],  # narrow right column for latitudinal profile
        height_ratios=[1, 1],
        wspace=0.05, hspace=0.12
    )

    # ---------------------------------------------------------------
    # MAP PANELS
    # ---------------------------------------------------------------
    # (a) Observed Total BA
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    im1 = figure_mean["CE_total_C"].plot(
        ax=ax1, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, add_colorbar=False
    )
    ax1.coastlines(linewidth=0.4)
    ax1.set_title("(g) Observed Total CE", pad=4)
    ax1.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax1.set_yticks(np.arange(-60, 91, 30), crs=ccrs.PlateCarree())
    ax1.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    # (b) Predicted Total BA
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    im2 = figure_mean[f"CE_total_C_pred_{fc_source}"].plot(
        ax=ax2, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, add_colorbar=False
    )
    ax2.coastlines(linewidth=0.4)
    ax2.set_title("(h) Predicted Total CE", pad=4)
    ax2.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax2.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    # ax2.set_yticks([])

    # LATITUDINAL PANEL – Total BA
    ax3 = fig.add_subplot(gs[0, 2])
    grouped_area = figure_reset_mean.groupby("lat_band")[["CE_total_C", f"CE_total_C_pred_{fc_source}"]].mean()
    ax3.plot(grouped_area["CE_total_C"], grouped_area.index, color="red", lw=1.3, label="GFED5 Benchmark")
    ax3.plot(grouped_area[f"CE_total_C_pred_{fc_source}"], grouped_area.index, color="blue", lw=1.3, label="IAM-FIRE Predictions")
    ax3.set_title("(i) Total CE per Latitude", fontsize=10, pad=4)
    ax3.set_xlabel("Carbon Emissions (TgC)")
    # ax3.set_ylabel("Latitude (°)")
    # ax3.set_yticks(np.arange(-60, 91, 30))
    ax3.set_ylim(-60, 90)  # <<< aligns height with map panels
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    ax3.set_box_aspect(0.89)
    # <<< Remove y-axis labels and ticks
    ax3.set_yticks([])
    ax3.set_ylabel("")
    ax3.tick_params(axis='y', left=False)

    # ---------------------------------------------------------------
    # Second row — Forest BA
    # ---------------------------------------------------------------
    # (d) Observed Forest BA
    ax4 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    im4 = figure_mean["CE_for_C"].plot(
        ax=ax4, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, add_colorbar=False
    )
    ax4.coastlines(linewidth=0.4)
    ax4.set_title("(j) Observed Forest CE", pad=4)
    ax4.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax4.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax4.set_yticks(np.arange(-60, 91, 30), crs=ccrs.PlateCarree())
    ax4.yaxis.set_major_formatter(cticker.LatitudeFormatter())

    # (e) Predicted Forest BA
    ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    im5 = figure_mean[f"CE_for_C_pred_{fc_source}"].plot(
        ax=ax5, x="lon", y="lat", transform=ccrs.PlateCarree(),
        cmap=cmap, norm=norm, add_colorbar=False
    )
    ax5.coastlines(linewidth=0.4)
    ax5.set_title("(k) Predicted Forest CE", pad=4)
    ax5.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    ax5.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    # ax5.set_yticks([])

    # LATITUDINAL PANEL – Forest BA
    ax6 = fig.add_subplot(gs[1, 2])
    grouped_area_for = figure_reset_mean.groupby("lat_band")[[f"CE_for_C", f"CE_for_C_pred_{fc_source}"]].mean()
    ax6.plot(grouped_area_for["CE_for_C"], grouped_area_for.index, color="red", lw=1.3)
    ax6.plot(grouped_area_for[f"CE_for_C_pred_{fc_source}"], grouped_area_for.index, color="blue", lw=1.3)
    ax6.set_title("(l) Forest CE per Latitude", fontsize=10, pad=4)
    ax6.set_xlabel("Carbon Emissions (TgC)")
    # ax6.set_yticks(np.arange(-60, 91, 30))
    ax6.set_ylim(-60, 90)  # <<< same fix here
    ax6.spines["top"].set_visible(False)
    ax6.spines["right"].set_visible(False)
    ax6.legend(loc="lower right", frameon=False, fontsize=9)
        

    ax6.set_box_aspect(0.89)
    # <<< Remove y-axis labels and ticks
    ax6.set_yticks([])
    ax6.set_ylabel("")
    ax6.tick_params(axis='y', left=False)

    # ---------------------------------------------------------------
    # Shared legend for both latitudinal panels
    # ---------------------------------------------------------------
    fig.legend(
        handles=ax3.get_lines(),  # reuse handles from first latitudinal panel
        labels=["GFED5 Benchmark", "IAM-FIRE Predictions"],
        loc="lower center",
        ncol=1,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.825, 0.02)  # adjust position as needed
    )

    # ---------------------------------------------------------------
    # Shared colorbar
    # ---------------------------------------------------------------
    cbar_ax = fig.add_axes([0.20, 0.07, 0.5, 0.025])
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Carbon Emissions (TgC)", fontsize=12)

    # # Colorbar for Predicted Total CE (panel b)
    # cbar_ax_b = fig.add_axes([0.91, 0.56, 0.015, 0.33])  # [left, bottom, width, height]
    # cbar_b = fig.colorbar(im2, cax=cbar_ax_b, orientation='vertical')
    # cbar_b.set_label("Total CE (TgC)", fontsize=10)
    # cbar_b.ax.tick_params(labelsize=8)

    # # Colorbar for Predicted Forest CE (panel d)
    # cbar_ax_d = fig.add_axes([0.91, 0.10, 0.015, 0.33])  # [left, bottom, width, height]
    # cbar_d = fig.colorbar(im5, cax=cbar_ax_d, orientation='vertical')
    # cbar_d.set_label("Forest CE (TgC)", fontsize=10)
    # cbar_d.ax.tick_params(labelsize=8)

    glm_figure=plt.gcf() 
    plt.show()
            
    if save == True: 
        if mean == True: 
            glm_figure.savefig(os.path.join(figures_dir, f"F4_CE_Prediction_Observed_MAPS_LAT_MEAN_v5.1.png"))
        else:
            glm_figure.savefig(os.path.join(figures_dir, f"F4_CE_Prediction_Observed_MAPS_LAT_{year}_v5.1.png"))

def SM2_forest_proportion(figure_mean, save = True):
    ###########################################################################
    # FIGURE GLM SM. FOREST PROPORTION MAP
    ##########################################################################
    # List of land-use variables
    import cartopy.feature as cfeature
    print(figure_mean)
    forest_prop = figure_mean["forest_proportion"].transpose("lat", "lon")
    
    # Fix longitudes (convert 0–360 → -180–180 if necessary)
    if forest_prop.lon.max() > 180:
        forest_prop = forest_prop.assign_coords(lon=(((forest_prop.lon + 180) % 360) - 180))
        forest_prop = forest_prop.sortby("lon")
    
    # Plot with Cartopy
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    forest_prop.plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap="Greens", vmin=0, vmax=1,
        # cbar_kwargs={"label": "Fraction of forest cover"}
        cbar_kwargs={
            "label": "Fraction of forest cover",
            "shrink": 0.70,       # ← reduce height to 75%
            "aspect": 25          # ← optional: make it thicker/thinner
        }
    )
    
    # Add land and ocean for better contrast
    ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", alpha=0.2)
    
    # Add continent outlines and borders
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="black")
    
    # Title and gridlines
    ax.set_title("Forest Proportion of Total BA (GFEDv5)", fontsize=16, fontweight="bold")
    ax.gridlines(draw_labels=False, linewidth=0.3, linestyle="--", color="gray", alpha=0.5)
    
    plt.tight_layout()
    prop_figure=plt.gcf() 
    plt.show()
    
    # Save and show
    if save == True:
        output_path = os.path.join(figures_dir, "SM2_Forest_proportion_countries_v5.1.png")
        prop_figure.savefig(output_path)

def SM4_fuel_consumption(figure_nc, save = True):

    variables = ["for", "nonf"]  # adjust to your available data
    labels = {"for": "Forest", "nonf": "Non-forest"}    

    # Define custom colors and bins
    colors = [
        "#e6e6ff",  # 0
        "#c6b3d9",  # 50
        "#85c6de",  # 100
        "#58a773",  # 200
        "#ffd966",  # 500
        "#f08000",  # 1000
        "#e60000",  # 2000
        "#67001f",  # 5000
        "#330000"   # 10000+
    ]
    bounds = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
    
    # Create colormap and normalization
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Create vertically stacked subplots
    n_vars = len(variables)
    fig, axes = plt.subplots(
        n_vars, 1,
        figsize=(10, 3.5 * n_vars),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )
    
    if n_vars == 1:
        axes = [axes]
    
    # Loop through each variable and plot
    for i, variable in enumerate(variables):
        ax = axes[i]
    
        # Compute time mean
        data_mean = figure_nc[f"FC_{variable}_GFED"].mean(dim="time")
    
        # Plot with shared color scale
        im = data_mean.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
            add_labels=False
        )
    
        # Add coastlines and borders
        ax.coastlines(linewidth=0.6)
        # ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.set_global()
        ax.set_title(f"{labels[variable]}", fontsize=13, fontweight='bold', loc='left')
    
    # Add one shared colorbar at the bottom
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        im, cax=cbar_ax, orientation='horizontal',
        boundaries=bounds, ticks=bounds,
        label='Fuel consumption (g C m⁻² burned)'
    )
    
    # Add main title and adjust layout
    # fig.suptitle("Mean Fuel Consumption (GFED5.1, 2001–2015)", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # Save the figure
    if save == True:
        output_path = os.path.join(figures_dir, "SM4_Fuel_Consumption_GFED5.1_VerticalPanels.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

def SM89_BA_historic_trends_reg(df_sum_time, variable = "BA_area", save = True):
    
    
    # Ensure time is datetime
    df_sum_time["time"] = pd.to_datetime(df_sum_time["time"])
    
    # Sort the dataframe just in case
    df = df_sum_time.sort_values(["basisregions", "time"])
    
    # Unique regions
    regions = df["basisregions"].unique()
    regions.sort()
    
    # Determine number of subplots
    n_regions = len(regions)
    ncols = 4   # adjust if you want fewer columns
    nrows = int(np.ceil(n_regions / ncols))
    
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(20, 3 * nrows),
        sharex=True,
        sharey=False
    )
    
    axes = axes.flatten()
    
    for i, region in enumerate(regions):
        ax = axes[i]
        sub = df[df["basisregions"] == region]
        
        short_name, long_name = region_name[region]
        
        # Observed
        ax.plot(
            sub["time"],
            sub[f"{variable}"],
            marker="o",
            linewidth=2,
            label="Observed",
            alpha=0.9
        )
        
        # Predicted
        ax.plot(
            sub["time"],
            sub[f"{variable}_pred"],
            marker="s",
            linewidth=2,
            linestyle="--",
            label="Predicted",
            alpha=0.9
        )
        
        ax.set_title(f"{region} — {short_name}\n{long_name}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("f{variable} (Mha)")
    
    # Turn off any empty axes
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    # Add one legend for all panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower right",
        ncol=2,
        fontsize=12
    )
    
    fig.suptitle("Observed vs Predicted Burned Area Through Time by Region", fontsize=18, y=1.02)
    plt.tight_layout()
    reg_figure=plt.gcf() 
    plt.show()
    
    # Save and show
    if save == True:
        output_path = os.path.join(figures_dir, f"SM8_Regional_Historic_{variable}_v5.1.png")
        reg_figure.savefig(output_path)
    
def SMX_correlation_maps(figure_nc, save = True):
    
    figure_mean = figure_nc.sel(lat=slice(-55.25, 90)).mean(dim="time").where(landmask.mask == 1)

    
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import scipy.stats as stats
    import math
    import os

    # ---- DATA PREP ----
    # We must keep the time dimension to compute correlations over time
    ds = figure_nc.sel(lat=slice(-55.25, 90)).where(landmask.mask == 1)
    # figure_nc["grazing_pressure"].mean(dim="time").plot()
    # figure_nc["TPI"].mean(dim="time").plot()

    # figure_nc["grazing_pressure"].notnull().sum(dim="time").plot() 
    
    predictors = [
        "vpd_log",
        "ndd_log",
        "ndd_season_log",
        "sfcWind_log",
        "pr_sum_month_log",
        "GPP_log",
        # "TPI",
        # "VRM",
        "grassland",
        "shrubland",
        "cropland",
        "forest",
        "HDI"
        # "population",
        # "grazing_pressure",
        # "pop_density",
        # "grazing_pressure"
    ]

    # ---- FIGURE LAYOUT ----
    n = len(predictors)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig = plt.figure(figsize=(5*ncols, 4*nrows))
    proj = ccrs.Robinson()

    # ---- LOOP OVER PREDICTORS ----
    for i, var in enumerate(predictors, 1):

        # Compute Pearson correlation over time
        # r = cov(x,y) / (std(x)*std(y)) computed for each (lat,lon)
        ba = ds["BA_area"]
        x  = ds[var]

        # xr.corr works beautifully across a dimension
        rmap = xr.corr(ba, x, dim="time")

        # ---- PLOT PANEL ----
        ax = fig.add_subplot(nrows, ncols, i, projection=proj)
        ax.set_global()
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)

        # Filled map
        pcm = rmap.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap="coolwarm",
            vmin=-1, vmax=1,
            add_colorbar=False
        )

        ax.set_title(f"corr(BA_area, {var})", fontsize=10)

    # ---- GLOBAL COLORBAR ----
    cbar_ax = fig.add_axes([0.25, 0.06, 0.5, 0.015])
    fig.colorbar(pcm, cax=cbar_ax, orientation="horizontal", label="Pearson r")

    fig.suptitle("Pearson Correlation with BA_area (computed over time)", fontsize=18, y=0.95)
    plt.tight_layout(rect=[0,0.08,1,0.94])
    
    corr_figure=plt.gcf() 


    # ---- SAVE ----
    if save:
        output_path = os.path.join(figures_dir, "SM9_Correlation_Maps_BA_v5.1.png")
        corr_figure.savefig(output_path, dpi=300)
        print("Saved to:", output_path)

    plt.show()

    


if __name__ == "__main__":
    
    print("Launching the GLM figure")
    F3_glm_figure(figure_nc, figure_csv, agg_df, save = True)
    F4_glm_maps(figure_nc, figure_csv, mean = True, year = 2003, save = True)
    F4_glm_maps(figure_nc, figure_csv, mean = False, year = 2003, save = True)
    F4_glm_maps(figure_nc, figure_csv, mean = False, year = 2016, save = True)

    SM2_forest_proportion(figure_mean, save = True)
    SM4_fuel_consumption(figure_nc, save = True)
    SM89_BA_historic_trends_reg(df_sum_time, variable = "BA_area", save = True)
    SM89_BA_historic_trends_reg(df_sum_time, variable = "BA_area_for", save = True)
    
    SMX_correlation_maps(figure_nc, save = True)
    

