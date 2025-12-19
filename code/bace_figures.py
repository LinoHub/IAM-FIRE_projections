# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:39:22 2025

@author: theo.rouhette

conda activate xesmf
cd C:\\GCAM\\Theo\\GCAM_7.2_Impacts\\python\climate_integration_metarepo\\code\\python
python glm_model_dem2.py


"""

# Importing Needed Libraries
import os  # For navigating os
import sys  # Getting system details
import numpy as np
import pandas as pd  # Data functions
import xarray as xr  # Reading and manipulating NetCDF data
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pymannkendall as mk
from scipy.stats import linregress


# PATHS
IF_PATH = "C:/GCAM/Theo/IAM-FIRE/zenodo"
inputs_dir = os.path.join(IF_PATH, f"inputs/")
outputs_dir = os.path.join(IF_PATH, f"outputs/")
figures_dir = os.path.join(IF_PATH, f"figures/")

# # DEBUG
# outputs_dir = "C:/GCAM/Theo/IAM-FIRE/output/fire_impacts"

# INPUTS TO CREATE BACE FIGURES
landmask = xr.open_dataset(os.path.join(inputs_dir, "landseamask_no-ant.nc")).drop_vars("time").sel(time=0)
regions = xr.open_dataset(os.path.join(inputs_dir, "GFED_Regions.nc"))
regions_csv = regions.to_dataframe().reset_index()

# time_series = pd.read_csv(os.path.join(outputs_dir, "BA_CE_Prediction_AllScen_ESA_v5.1.csv")) # SUE THE ORIGINAL ONE TO GET FULL CELLS ... 
time_series = pd.read_csv(os.path.join(outputs_dir, "BA_CE_Prediction_AllScen.csv")) # SUE THE ORIGINAL ONE TO GET FULL CELLS ... 
time_series_reg = pd.read_csv(os.path.join(outputs_dir, "BA_CE_Prediction_AllScen_Reg.csv")) 

# CONSTANTS
scenarios = ["SSP1-2p6o", "SSP2-4p5", "SSP3-6p6", "SSP5-7p6"]
esms = ["MPI-ESM1-2-LR", "CanESM5"]    
start_hist = 2002
start_sim = 2020
end = 2100
dates_LUC = list(range(start_hist, start_sim, 1)) + list(range(start_sim, end + 1, 5))
colors = {
    "Historic": "black",
    "SSP1-2p6o": "tab:green",
    "SSP2-4p5": "tab:blue",
    "SSP3-6p6": "tab:red",
    "SSP5-7p6": "tab:orange"
}
# Prepare the array of region IDs present in your data
region_ids = np.unique(time_series_reg['basisregions'].values)
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


def generate_colors(scenarios, esms, palette_name="Set2"):
 # Create a list of all unique sources
    unique_sources = [f"{scenario} - {esm}" for scenario in scenarios for esm in esms] + ["Historic"]
    
    n_colors = len(unique_sources)
    
    # Choose the palette
    if isinstance(palette_name, list):  # If a custom list of colors is provided
        palette = palette_name
    else:
        palette = sns.color_palette(palette_name, n_colors)

    # Make sure palette has enough colors
    if len(palette) < n_colors:
        raise ValueError(f"Palette '{palette_name}' does not have enough distinct colors for {n_colors} sources.")
    
    # Map each source to a color
    colors = {source: color for source, color in zip(unique_sources, palette)}
    
    return colors

def generate_colors_ssp(scenarios, palette_name="Set2"):
 # Create a list of all unique sources
    unique_sources = [f"{scenario}" for scenario in scenarios] + ["Historic"]
    
    n_colors = len(unique_sources)
    
    # Choose the palette
    if isinstance(palette_name, list):  # If a custom list of colors is provided
        palette = palette_name
    else:
        palette = sns.color_palette(palette_name, n_colors)

    # Make sure palette has enough colors
    if len(palette) < n_colors:
        raise ValueError(f"Palette '{palette_name}' does not have enough distinct colors for {n_colors} sources.")
    
    # Map each source to a color
    colors = {source: color for source, color in zip(unique_sources, palette)}
    
    return colors

def add_trend(ax, x, y, label, color):
    # Drop NaNs
    mask = np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(y) < 5:
        return None
    
    mk_result = mk.original_test(y)
    slope, intercept, r, p, stderr = linregress(x, y)
    print(f"{label}: MK trend={mk_result.trend}, MK p={mk_result.p:.3f}, slope={slope:.4f}")
    
    # --- Dynamic positioning logic ---
    # default: bottom-left
    x_text, y_text, ha, va = 0.03, 0.90, "left", "top"
    
    # # place in top-left for specific axes
    # if ax in [axs[1,0], axs[1,1]]:   # ← your special case
    #     x_text, y_text, ha, va = 0.03, 0.90, "left", "top"
    
    # Add small text annotation on the plot
    ax.text(
        x_text,
        y_text - 0.05 * list(colors.keys()).index(label),  # small vertical offset per scenario
        # f"{label}: {mk_result.trend} (p={mk_result.p:.3f})",
        f"{label}: {mk_result.trend}",
        transform=ax.transAxes,
        fontsize=8,
        color=color
    )
    
    return slope, mk_result

        
###########################################################################
# FIGURE DATAFRAMES. MME AND RANGES 
###########################################################################

# Format the time-series 
time_series = time_series.set_index("time")
time_series.index = pd.to_datetime(time_series.index, format="mixed").year
time_series["grass-shrub"] = time_series["grassland"] + time_series["shrubland"]

# Filter 2001 since GFED FC values start at 2002
time_series = time_series.loc[time_series.index.get_level_values("time") != 2001]
time_series = time_series.loc[time_series.BA_area_pred > 0]

variables_mme = ["BA_area_pred", "BA_area_for_pred", "BA_area_nonf_pred",
                 "BA_area", "BA_area_for", "BA_area_nonf",
                 "CE_total_C_pred_GFED", "CE_for_C_pred_GFED", "CE_nonf_C_pred_GFED",
                 "CE_total_C", "CE_for_C", "CE_nonf_C",
                 "FC_total_GFED_proj", "FC_for_GFED_proj", "FC_nonf_GFED_proj",
                 "FC_total_GFED_proj_fixedBA", "FC_for_GFED_proj_fixedBA", "FC_nonf_GFED_proj_fixedBA",
                 "FC_total_GFED", "FC_for_GFED", "FC_nonf_GFED", 
                 "FC_total_GFED_fixedBA", "FC_for_GFED_fixedBA", "FC_nonf_GFED_fixedBA", 
                 "vpd", "ndd", "pr_30d_sum", "sfcWind", "GPP"]
time_series["Scenario"] = time_series["Scenario"].fillna("Historic")    
# Compute df mean ± std ---
time_series_mme = []
for variable in variables_mme: 

    df = (
        time_series.groupby(["time", "Scenario", "basisregions"])
          .agg(
              mean=(f"{variable}", "mean"),
              std=(f"{variable}", "std"),
              vmin=(f"{variable}", "min"),
              vmax=(f"{variable}", "max")
          )
          .reset_index()
    )

    # Add lower/upper
    df[f"{variable}_mean"] = df["mean"]
    # df[f"{variable}_lower"] = df["mean"] - df["std"]
    # df[f"{variable}_upper"] = df["mean"] + df["std"]
    df[f"{variable}_lower"] = df["vmin"]          # range lower bound
    df[f"{variable}_upper"] = df["vmax"]          # range upper bound
    
    # Keep only relevant columns
    df = df[["time", "Scenario", "basisregions",
             f"{variable}_mean", f"{variable}_lower", f"{variable}_upper"]]
    
    time_series_mme.append(df)

# Concatenate all variables into one DataFrame
time_series_mme = pd.concat(time_series_mme, axis=1)
    
# Drop duplicate index columns from concat
time_series_mme = time_series_mme.loc[:, ~time_series_mme.columns.duplicated()]

# Data frame with non-ESM dependent values
time_series_ssp = time_series[~time_series["Source"].str.contains("CanESM5")]
time_series_ssp["Source"] = time_series_ssp["Source"].str.split(" - ").str[0]
time_series_ssp.index =  pd.to_datetime(time_series_ssp.index.astype(str), format="%Y")
time_series_ssp = time_series_ssp[time_series_ssp.index.year.isin(dates_LUC)]

# Data frame with regional values
time_series_reg = time_series_reg.set_index("time")
time_series_reg.index = pd.to_datetime(time_series_reg.index, format="mixed").year
time_series_reg = time_series_reg.loc[time_series_reg.index.get_level_values("time") != 2001]
time_series_reg = time_series_reg.loc[time_series_reg.BA_area_pred > 0]
time_series_reg["grass-shrub"] = time_series_reg["grassland"] + time_series_reg["shrubland"]
print(time_series_reg) # time as index, column for Scenario, Source, basisregions, etc
        

    
    
def F5_global_BAFCCE_trend(time_series_mme):
    
    ###########################################################################
    # FIGURE 1.B GLOBAL TEMPORAL TREND (GRAPH) -- BA + FC + CE -- WITH SCENARIO MME
    ###########################################################################

    # Initialize the figure
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(12, 8))
    
    # # Generate the colors dictionary
    # colors = generate_colors_ssp(scenarios)
    
    # Check if "Source" column exists and print unique values
    if "Scenario" in time_series_mme.columns:
        print("Unique Sources:", time_series["Scenario"].unique())
    else:
        print("Column 'Source' not found in time_series!")
    
    # Define subplot titles (row, col): title
    subplot_titles = {
        (0, 0): "(a) Total BA",
        (0, 1): "(b) Forest BA",
        (1, 0): "(c) Total FC",
        (1, 1): "(d) Forest FC",
        (2, 0): "(e) Total CE",
        (2, 1): "(d) Forest CE",
    }
    
    # Assign titles to subplots
    for (i, j), title in subplot_titles.items():
        axs[i, j].set_title(title)
    
    # Collect all legend handles/labels
    legend_handles = []
    legend_labels = []
    
    # Plot each variable with color based on 'Source'
    for source, data in time_series_mme.groupby("Scenario"):
        print(f"Plotting {source}...")  # Debugging output
        # data['BA_frac_pred'].plot(ax=axs[0, 0], color=colors.get(source, "gray"), label=f"Predicted BA (Frac) ({source})")
        # data['BA_frac'].plot(ax=axs[0, 0], linestyle="dashed", color=colors.get(source, "gray"), label=f"Observed BA (Frac) ({source})")
        
        # color = colors.get(source, "gray")
        # # Same plotting logic as before...
        # line, = data['BA_area_pred'].plot(ax=axs[0, 0], color=color, label=f"{source}")
        # legend_handles.append(line)
        # legend_labels.append(line.get_label())
                
        # BA PANELS
        if source == "Historic": 
            data.set_index("time")['BA_area_pred_mean'].plot(ax=axs[0, 0], color=colors.get(source, "gray"), label=f"{source} - Predicted")
            data.set_index("time")['BA_area_mean'].plot(ax=axs[0, 0], linestyle="dashed", color=colors.get(source, "gray"), label=f"{source} - Observed")
        else:
            data.set_index("time")['BA_area_pred_mean'].plot(ax=axs[0, 0], color=colors.get(source, "gray"), label=f"{source}")
            axs[0,0].fill_between(
                data["time"],
                data["BA_area_pred_lower"],
                data["BA_area_pred_upper"],
                color=colors.get(source, "gray"),
                alpha=0.2
            )
            axs[0,0].set_ylabel("Area (Mha)")
        add_trend(axs[0,0], data["time"].values, data["BA_area_pred_mean"].values, source, colors.get(source, "gray"))
                
        # Forest Burned Area
        if source == "Historic": 
            data.set_index("time")['BA_area_for_pred_mean'].plot(ax=axs[0, 1], color=colors.get(source, "gray"), label=f"{source} - Predicted")
            data.set_index("time")['BA_area_for_mean'].plot(ax=axs[0, 1], linestyle="dashed", color=colors.get(source, "gray"), label=f"{source} - Observed")
        else:
            data.set_index("time")['BA_area_for_pred_mean'].plot(ax=axs[0, 1], color=colors.get(source, "gray"), label=f"{source}")
            axs[0,1].fill_between(
                data["time"],
                data["BA_area_for_pred_lower"],
                data["BA_area_for_pred_upper"],
                color=colors.get(source, "gray"),
                alpha=0.2
            )
            axs[0,1].set_ylabel("Area (Mha)")
        add_trend(axs[0,1], data["time"].values, data["BA_area_for_pred_mean"].values, source, colors.get(source, "gray"))


        # fc_start = np.datetime64("2020-12-31")
        data_fc = data[data["time"] >= 2020]        
        
        # Total FC
        if source == "Historic": 
            data_fc.set_index("time")['FC_total_GFED_fixedBA_mean'].plot(ax=axs[1, 0], color=colors.get(source, "gray"), label=f"{source} - Predicted")
            # data_fc.set_index("time")['FC_total_GFED_fixedBA_mean'].plot(ax=axs[1, 0], linestyle="dashed", color=colors.get(source, "gray"), label=f"{source} - Observed")
        else:
            data_fc.set_index("time")['FC_total_GFED_proj_fixedBA_mean'].plot(ax=axs[1, 0], color=colors.get(source, "gray"), label=f"{source}")
            axs[1,0].fill_between(
                data_fc["time"],
                data_fc["FC_total_GFED_proj_fixedBA_lower"],
                data_fc["FC_total_GFED_proj_fixedBA_upper"],
                color=colors.get(source, "gray"),
                alpha=0.2
            )
            axs[1,0].set_ylabel("Combusted Fuel (gC.m-2)")
        add_trend(axs[1,0], data_fc["time"].values, data_fc["FC_total_GFED_fixedBA_mean"].values, source, colors.get(source, "gray"))
        add_trend(axs[1,0], data_fc["time"].values, data_fc["FC_total_GFED_proj_fixedBA_mean"].values, source, colors.get(source, "gray"))
        
        # axs[1,0].set_xlim(pd.Timestamp("2020-12-31"), None)
        
        # Forest FC
        if source == "Historic": 
            data_fc.set_index("time")['FC_for_GFED_fixedBA_mean'].plot(ax=axs[1, 1], color=colors.get(source, "gray"), label=f"{source} - Predicted")
            # data_fc.set_index("time")['CE_for_C_mean'].plot(ax=axs[1, 1], linestyle="dashed", color=colors.get(source, "gray"), label=f"{source} - Observed")
        else:
            data_fc.set_index("time")['FC_for_GFED_proj_fixedBA_mean'].plot(ax=axs[1, 1], color=colors.get(source, "gray"), label=f"{source}")
            axs[1,1].fill_between(
                data_fc["time"],
                data_fc["FC_for_GFED_proj_fixedBA_lower"],
                data_fc["FC_for_GFED_proj_fixedBA_upper"],
                color=colors.get(source, "gray"),
                alpha=0.2
            )
            axs[1,1].set_ylabel("Combusted Fuel (gC.m-2)")
        add_trend(axs[1,1], data_fc["time"].values, data_fc["FC_for_GFED_fixedBA_mean"].values, source, colors.get(source, "gray"))
        add_trend(axs[1,1], data_fc["time"].values, data_fc["FC_for_GFED_proj_fixedBA_mean"].values, source, colors.get(source, "gray"))

        # axs[1,1].set_xlim(pd.Timestamp("2020-12-31"), None)

            
        # Total Carbon Emissions
        if source == "Historic": 
            data.set_index("time")['CE_total_C_pred_GFED_mean'].plot(ax=axs[2, 0], color=colors.get(source, "gray"), label=f"{source} - Predicted")
            data.set_index("time")['CE_total_C_mean'].plot(ax=axs[2, 0], linestyle="dashed", color=colors.get(source, "gray"), label=f"{source} - Observed")
        else:
            data.set_index("time")['CE_total_C_pred_GFED_mean'].plot(ax=axs[2, 0], color=colors.get(source, "gray"), label=f"{source}")
            axs[2,0].fill_between(
                data["time"],
                data["CE_total_C_pred_GFED_lower"],
                data["CE_total_C_pred_GFED_upper"],
                color=colors.get(source, "gray"),
                alpha=0.2
            )
            axs[2,0].set_ylabel("Carbon Emission (PgC)")
        add_trend(axs[2,0], data["time"].values, data["CE_total_C_pred_GFED_mean"].values, source, colors.get(source, "gray"))

        
        # Forest Carbon Emissions
        if source == "Historic": 
            data.set_index("time")['CE_for_C_pred_GFED_mean'].plot(ax=axs[2, 1], color=colors.get(source, "gray"), label=f"{source} - Predicted")
            data.set_index("time")['CE_for_C_mean'].plot(ax=axs[2, 1], linestyle="dashed", color=colors.get(source, "gray"), label=f"{source} - Observed")
        else:
            data.set_index("time")['CE_for_C_pred_GFED_mean'].plot(ax=axs[2, 1], color=colors.get(source, "gray"), label=f"{source}")
            axs[2,1].fill_between(
                data["time"],
                data["CE_for_C_pred_GFED_lower"],
                data["CE_for_C_pred_GFED_upper"],
                color=colors.get(source, "gray"),
                alpha=0.2
            )
            axs[2,1].set_ylabel("Carbon Emission (PgC)")

        add_trend(axs[2,1], data["time"].values, data["CE_for_C_pred_GFED_mean"].values, source, colors.get(source, "gray"))
        
        

       

    # Collect unique legend entries across all subplots
    legend_handles = []
    legend_labels = []
    seen = set()
    
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in seen:
                legend_handles.append(h)
                legend_labels.append(l)
                seen.add(l)
    
    # Clear legends from individual plots
    for ax in axs.flatten():
        ax.legend().remove()
    
    # Add single legend at the bottom center
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower center',
        ncol=6,  # Adjust based on number of entries
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01)
    )
    
    # Adjust layout to make space for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    pred_time=plt.gcf()
    plt.show()
    
    
    
    if "MME" in esms:
        pred_time.savefig(os.path.join(figures_dir,f"F5_BA_CE_Prediction_AllScen_Trend_MME_BAFCCE_LOCAL_v5.1_FC.png"))
    else:
        pred_time.savefig(os.path.join(figures_dir,f"F5_BA_CE_Prediction_AllScen_Trend_BAFCCE_LOCAL_v5.1_FC.png"))
    
    
    trend_results = []

    for source, data in time_series_mme.groupby("Scenario"):
        for var in ["BA_area_pred_mean", 'BA_area_for_pred_mean', 
                    # 'FC_total_GFED_proj_mean', 'FC_for_GFED_proj_mean',
                    'FC_total_GFED_proj_fixedBA_mean', 'FC_for_GFED_proj_fixedBA_mean',
                    "CE_total_C_pred_GFED_mean", 'CE_for_C_pred_GFED_mean']:
            print(var)
            # if source == "Historic" and var in ['FC_total_GFED_proj_mean', 'FC_for_GFED_proj_mean']:
            #     pass
            if source == "Historic" and var in ['FC_total_GFED_proj_fixedBA_mean', 'FC_for_GFED_proj_fixedBA_mean']:
                pass
            else: 
                y = data[var].values
                x = np.arange(len(y))
                mk_result = mk.original_test(y)
                slope, intercept, r, p, stderr = linregress(x, y)
                trend_results.append({
                    "Scenario": source,
                    "Variable": var,
                    "Slope": slope,
                    "p_value": p,
                    "MK_trend": mk_result.trend,
                    "MK_p": mk_result.p
            })
    
    trend_df = pd.DataFrame(trend_results)
    print(trend_df)
    trend_df.to_csv(os.path.join(outputs_dir,f"F5_BA_CE_Prediction_AllScen_Trend_MK.csv"))  

def F6_global_regions_heatmap(time_series_reg):

    import numpy as np   
    import pandas as pd

    trend_results_regions = []
    variables = ["BA_area_pred", "BA_area_for_pred",
                 "CE_total_C_pred_GFED", "CE_for_C_pred_GFED"]
    
    for variable in variables:
        
        # 1️⃣ Aggregate the data exactly like in the plotting loop
        df_agg = (
            time_series_reg.groupby(["time", "Scenario", "basisregions"])
                      .agg(mean=(variable, "mean"),
                           std=(variable, "std"),
                           vmin=(variable, "min"),
                           vmax=(variable, "max"))
                      .reset_index()
        )
        
        # df_agg["lower"] = df_agg["mean"] - df_agg["std"]
        # df_agg["upper"] = df_agg["mean"] + df_agg["std"]
        df_agg["lower"] = df_agg["vmin"] 
        df_agg["upper"] = df_agg["vmax"] 
        
        
        df_agg["region_code"] = df_agg["basisregions"].map(lambda x: region_name[x][0])
        df_agg["region_full"] = df_agg["basisregions"].map(lambda x: region_name[x][1])
        
        # 2️⃣ Compute trend per scenario × region on the aggregated mean
        for scenario, df_scenario in df_agg.groupby("Scenario"):
            for region_id, df_region in df_scenario.groupby("basisregions"):
                
                y = df_region["mean"].values
                x = df_region["time"].values.astype(int)  # ensures slope is per year
                
                # Skip series with too few points
                mask = np.isfinite(y)
                if mask.sum() < 5:
                    continue
                y = y[mask]
                x = x[mask]
                
                # Mann-Kendall test
                mk_result = mk.original_test(y)
                
                # Linear regression
                slope, intercept, r, p, stderr = linregress(x, y)
                
                # get BA at 2020
                base_2020 = df_region.loc[df_region["time"] == 2020, "mean"]
                if not base_2020.empty and base_2020.values[0] > 0:
                    base_2020_value = base_2020.values[0]
                    pct_change = (slope * (2100 - 2020) / base_2020_value) * 100
                else:
                    pct_change = np.nan
                
                trend_results_regions.append({
                    "Scenario": scenario,
                    "Region": region_id,
                    "Region_Full": df_region["region_full"].iloc[0],
                    "Variable": variable,
                    "Slope": slope,
                    "p_value": p,
                    "MK_trend": mk_result.trend,
                    "MK_p": mk_result.p,
                    "Base_2020": base_2020_value if not base_2020.empty else np.nan,
                    "Pct_Change_2020_2100": pct_change
                })
    
    # Convert to DataFrame
    trend_df_regions = pd.DataFrame(trend_results_regions)
    print(trend_df_regions.head())
    trend_df_regions.to_csv(os.path.join(outputs_dir,f"F6_BA_CE_Prediction_AllScen_Trend_REGIONS_MK.csv"))  

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    # # Assuming your CSV is named 'trend_df_regions.csv'
    # trend_df_regions = pd.read_csv("trend_df_regions.csv")
    
    
    # Ensure correct ordering
    # --- Setup ---
    region_order = list(region_name.keys())
    region_labels = [region_name[i][0] for i in region_order]  # short codes (BONA, TENA, ...)
    
    scenario_order = ["SSP1-2p6o", "SSP2-4p5", "SSP3-6p6", "SSP5-7p6"][:4]
    variable_order = ["BA_area_pred", "BA_area_for_pred",
                     "CE_total_C_pred_GFED", "CE_for_C_pred_GFED"]
    # Custom labels for the variables
    variable_labels = ["(a) Total BA", "(b) Forest BA",
                       "(c) Total CE", "(d) Forest CE"]
    
    # Map MK trend symbol for annotation
    trend_symbol = {"increasing": "↑", "decreasing": "↓", "no trend": "–"}
    
    trend_df_regions["MK_symbol"] = trend_df_regions["MK_trend"].map(trend_symbol)
    
    
    # --- CREATE SUBPLOTS BY VARIABLE, NOT BY SCENARIO ---
    n_vars = len(variable_order)
    fig, axes = plt.subplots(1, n_vars, figsize=(3.5 * n_vars, 8))
    
    # Common colormap centered at 0
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    cmap = plt.cm.RdYlBu_r
    
    for i, variable in enumerate(variable_order):
        
        # i = 0 
        # scenario = "SSP1-2p6o"
        ax = axes[i]
        sub = trend_df_regions[trend_df_regions["Variable"] == variable]
    
        # Pivot to get Region × Variable matrix of slopes
        pivot = sub.pivot_table(index="Region", columns="Scenario", values="Pct_Change_2020_2100") # OPTION 1
        # pivot = sub.pivot_table(index="Region", columns="Scenario", values="Slope") # OPTION 2
        pivot = pivot.reindex(index=region_order, columns=scenario_order)
    
        # Pivot for MK symbols (no aggregation, just reshape)
        ann = sub.pivot(index="Region", columns="Scenario", values="MK_symbol")
        ann = ann.reindex(index=region_order, columns=scenario_order)
    
        # --- Compute variable-specific limits ---
        # vmax_var = np.nanmax(np.abs(pivot.values))  # get largest absolute magnitude
        # vmin_var = -vmax_var                        # keep symmetric color range
        vmax_var = 100
        vmin_var = -100
        
        # Add colorbar only for last subplot
        cbar_flag = True if i == len(variable_order) - 1 else False
    
        sns.heatmap(
            pivot,
            ax=ax,
            cmap=cmap,
            center=0,
            vmin=vmin_var, vmax=vmax_var,         
            annot=ann,
            fmt="s",
            annot_kws={"fontsize": 11, "color": "black"},
            cbar=True,
            # cbar_kws={"label": "Percent Change (2020-2100)"},
            # cbar=cbar_flag,
            cbar_kws={"label": "Percent Change (2020–2100)"} if cbar_flag else None,
            # cbar_kws={"label": "Slope (2020–2100)"} if cbar_flag else None,
            yticklabels=False  # we’ll add custom ones below
        )
        
        ax.set_title(variable_labels[i], fontsize=14, pad=10)
        # ax.set_xlabel("Scenario", fontsize=12)
        if i == 0:
            ax.set_ylabel("Region", fontsize=12)
            ax.set_yticks(np.arange(len(region_labels)) + 0.5)
            ax.set_yticklabels(region_labels, rotation=0)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        print(region_labels)
        print(pivot.index)
    
    plt.suptitle("Regional Trends by Scenario (Percent Change ± MK Trend)", fontsize=18, y=1.02)
    # plt.suptitle("Regional Trends by Scenario (Slope ± MK Trend)", fontsize=18, y=1.02)
    plt.tight_layout()
    region_figure=plt.gcf()

    plt.show()
    region_figure.savefig(os.path.join(figures_dir,f"F6_HeatMap_Regional_Total_BA_CE_SLP_LOCAL.png"))

def F7_SM16_fact_decomp_drivers(scenarios, esms):

###########################################################################
# FACTORIAL DECOMPOSITION  -- FIX ALL BUT ONE FREE ( DRIVERS / REVERSE )
###########################################################################
   
   ###########################################################################
   # MME DATAFRAME DRIVERS - GLOBAL
    fact_decomp_drivers = []
    for i, (scenario, esm) in enumerate([(s, e) for s in scenarios for e in esms]):
       
       # scenario = "SSP1-2p6o"
       # esm = "CanESM5"
       
       # Load and aggregate
       fname = f"BA_Factorial_Decomp_Drivers_Agg_{scenario}_{esm}_v5.1.csv"
       df = pd.read_csv(os.path.join(outputs_dir, fname))
       df = df.set_index("time").groupby(level="time").sum() # IF AGG WORKS .. 

       # df = df.set_index(["lat", "lon", "time"])
       # df = df.groupby(level="time").sum()

       
       df_2020 = df.loc[df.index.get_level_values("time") == "2020-12-31"]
       df = df.loc[df.index.get_level_values("time") == "2100-12-31"]
       df["Scen - ESM"] = f"{scenario} - {esm}"
       df["Scenario"] = f"{scenario}"
       df["ESM"] = f"{esm}"
       
       # Select BA columns
       ba_cols = [c for c in df.columns if "BA_Free" in c or c == "BA_area_pred"]
   
       for col in ba_cols:
           df[f"{col}_percent"] = 100 * (df[f"{col}"] - df["BA_area_pred"]) / df["BA_area_pred"] # % change relative to baseline (2100)
           df[f"{col}_abs"] = df[f"{col}"].values - df_2020["BA_area_pred"].values # Absolute difference (Mha) relative to baseyear (2015)
       
       fact_decomp_drivers.append(df)
   
    fact_drivers = pd.concat(fact_decomp_drivers)
   
   # Compute df mean ± std ---
    fact_drivers_mme = []
   
    for variable in ba_cols: 
       df = (
           fact_drivers.groupby(["time", "Scenario"])
             .agg(
                 mean=(f"{variable}_percent", "mean"),
                 std=(f"{variable}_percent", "std"),
                 mean_abs=(f"{variable}_abs", "mean"),
                 std_abs=(f"{variable}_abs", "std"),
                 vmax_abs=(f"{variable}_abs", "max"),
                 vmin_abs=(f"{variable}_abs", "min"),
             )
             .reset_index()
       )
       
       # Add lower/upper
       df[f"{variable}_percent_mean"] = df["mean"]
       df[f"{variable}_percent_lower"] = df["mean"] - df["std"]
       df[f"{variable}_percent_upper"] = df["mean"] + df["std"]
       df[f"{variable}_percent_std"] = df["std"]
       
       df[f"{variable}_abs_mean"] = df["mean_abs"]
       # df[f"{variable}_abs_lower"] = df["mean_abs"] - df["std_abs"]
       # df[f"{variable}_abs_upper"] = df["mean_abs"] + df["std_abs"]
       df[f"{variable}_abs_lower"] = df["vmin_abs"] 
       df[f"{variable}_abs_upper"] = df["vmax_abs"] 
       
       df[f"{variable}_abs_std"] = df["std_abs"]
       
       
       
       # Keep only relevant columns
       df = df[["time", "Scenario", ""
                f"{variable}_percent_mean", f"{variable}_percent_lower", f"{variable}_percent_upper", f"{variable}_percent_std",
                f"{variable}_abs_mean", f"{variable}_abs_lower", f"{variable}_abs_upper", f"{variable}_abs_std"]]
       
       fact_drivers_mme.append(df)
   
   # Concatenate all variables into one DataFrame
    fact_drivers_mme = pd.concat(fact_drivers_mme, axis=1)
   
   # Drop duplicate index columns from concat
    fact_drivers_mme = fact_drivers_mme.loc[:, ~fact_drivers_mme.columns.duplicated()]
   
    fact_drivers_mme.to_csv(os.path.join(outputs_dir, "F7_Factor_Decomp_Drivers_CSV.csv"))
   # fact_drivers_mme = pd.read_csv(os.path.join(results_dir, "figures_v3_2019/F7_Factor_Decomp_Drivers_CSV.csv"))
   
    def plot_barh_with_errors(df_mean, df_lower, df_upper, title="", save=True):
       """
       Plot grouped horizontal bar chart with asymmetric error bars.

       Parameters
       ----------
       df_mean : DataFrame
           Mean values, shape (nrows, ncols).
       df_lower : DataFrame
           Lower bounds, same shape as df_mean.
       df_upper : DataFrame
           Upper bounds, same shape as df_mean.
       title : str
           Plot title.
       """

       means = df_mean.values
       lower = df_lower.values
       upper = df_upper.values

       err_neg = means - lower
       err_pos = upper - means

       nrows, ncols = means.shape
       y = np.arange(nrows)

       fig, ax = plt.subplots(figsize=(10, 6))
       cmap = plt.get_cmap("tab20")
       bar_height = 0.8 / ncols

       for i, col in enumerate(df_mean.columns):
           offset = (i - (ncols - 1) / 2) * bar_height
           y_pos = y + offset
           ax.barh(
               y_pos,
               means[:, i],
               xerr=[err_neg[:, i], err_pos[:, i]],
               height=bar_height * 0.9,
               color=cmap(i),
               label=(col.replace("_abs_mean", "")
                         .replace("BA_Free_", "Transient ")
                         .replace("BA_area_pred", "All transient (default BA)")),

               capsize=3,
               ecolor="black",
           )

       ax.set_yticks(y)
       ax.set_yticklabels(df_mean.index)
       ax.set_xlabel("Absolute change (Mha) relative to 2020")
       # ax.set_ylabel("Scenario - ESM")
       ax.set_title(title)
       ax.legend(title="Driver", bbox_to_anchor=(1, 0.3), loc="best")
       plt.tight_layout()
       fact_figure = plt.gcf()
       plt.show()
       
       if save == True:
           fact_figure.savefig(os.path.join(figures_dir,f"F7_BA_Factor_Decomp_Drivers_Barplot_MME_LOCAL.png"))

    # Use the function to create the MME figure with fact_limiting_mme
    cols_mme = [
        "BA_area_pred_abs_mean",
        "BA_Free_socioeconomic_abs_mean",
        "BA_Free_land_use_abs_mean",
        "BA_Free_vegetation_abs_mean",
        "BA_Free_climate_abs_mean"]
    cols_lower = [c.replace("_mean", "_lower") for c in cols_mme]
    cols_upper = [c.replace("_mean", "_upper") for c in cols_mme]
    
    df_mean  = fact_drivers_mme.set_index("Scenario")[cols_mme]
    df_lower = fact_drivers_mme.set_index("Scenario")[cols_lower]
    df_upper = fact_drivers_mme.set_index("Scenario")[cols_upper]
    
    plot_barh_with_errors(df_mean, df_lower, df_upper,
                          title="Factorial Decomposition by Drivers",
                          save = True)
   
   
   
    ###########################################################################
    # REGIONAL SCALE
    ###########################################################################

   
   
    ###########################################################################
    # MME DATAFRAME DRIVERS - REGIONAL
    fact_decomp_drivers_reg = []
      
    for i, (scenario, esm) in enumerate([(s, e) for s in scenarios for e in esms]):
       
       # scenario = "SSP1-2p6o"
       # esm = "CanESM5"
       
       # Load and aggregate
       # fname = f"BA_Factorial_Decomp_{scenario}_{esm}.csv"
       fname = f"BA_Factorial_Decomp_Drivers_Agg_{scenario}_{esm}_v5.1.csv"
       df = pd.read_csv(os.path.join(outputs_dir, fname))
       df = df.set_index(["time", "basisregions"])
       
       # df = df.set_index(["lat", "lon", "time", "basisregions"])
       # df = df.groupby(level=["time", "basisregions"]).sum()

       
       df_2020 = df.loc[df.index.get_level_values("time") == "2020-12-31"]
       df = df.loc[df.index.get_level_values("time") == "2100-12-31"]
       df["Scen - ESM"] = f"{scenario} - {esm}"
       df["Scenario"] = f"{scenario}"
       df["ESM"] = f"{esm}"
       # df["Region_id"]= f"{basisregions}"
       
       # Select BA columns
       ba_cols = [c for c in df.columns if "BA_Free" in c or c == "BA_area_pred"]
   
       for col in ba_cols:
           df[f"{col}_percent"] = 100 * (df[f"{col}"] - df["BA_area_pred"]) / df["BA_area_pred"] # % change relative to baseline (2100)
           df[f"{col}_abs"] = df[f"{col}"].values - df_2020["BA_area_pred"].values # Absolute difference (Mha) relative to baseyear (2015)
       
       fact_decomp_drivers_reg.append(df)
   
    fact_drivers_reg = pd.concat(fact_decomp_drivers_reg)
   
   # Compute df mean ± std ---
    fact_drivers_reg_mme = []
   
    for variable in ba_cols: 
       df = (
           fact_drivers_reg.groupby(["time", "Scenario", "basisregions"])
             .agg(
                 mean=(f"{variable}_percent", "mean"),
                 std=(f"{variable}_percent", "std"),
                 mean_abs=(f"{variable}_abs", "mean"),
                 std_abs=(f"{variable}_abs", "std"),
                 vmax_abs=(f"{variable}_abs", "max"),
                 vmin_abs=(f"{variable}_abs", "min"),
                 
             )
             .reset_index()
       )
       
       # Add lower/upper
       df[f"{variable}_percent_mean"] = df["mean"]
       df[f"{variable}_percent_lower"] = df["mean"] - df["std"]
       df[f"{variable}_percent_upper"] = df["mean"] + df["std"]
       df[f"{variable}_percent_std"] = df["std"]
       df[f"{variable}_abs_mean"] = df["mean_abs"]
       # df[f"{variable}_abs_lower"] = df["mean_abs"] - df["std_abs"]
       # df[f"{variable}_abs_upper"] = df["mean_abs"] + df["std_abs"]
       df[f"{variable}_abs_lower"] = df["vmin_abs"] 
       df[f"{variable}_abs_upper"] = df["vmax_abs"] 

       df[f"{variable}_abs_std"] =df["std_abs"]
       
       
       # Keep only relevant columns
       df = df[["time", "Scenario", "basisregions",
                f"{variable}_percent_mean", f"{variable}_percent_lower", f"{variable}_percent_upper", f"{variable}_percent_std",
                f"{variable}_abs_mean", f"{variable}_abs_lower", f"{variable}_abs_upper", f"{variable}_abs_std"]]
       
       fact_drivers_reg_mme.append(df)
   
    # Concatenate all variables into one DataFrame
    fact_drivers_reg_mme = pd.concat(fact_drivers_reg_mme, axis=1)
   
    # Drop duplicate index columns from concat
    fact_drivers_reg_mme = fact_drivers_reg_mme.loc[:, ~fact_drivers_reg_mme.columns.duplicated()]
   
    fact_drivers_reg_mme.to_csv(os.path.join(outputs_dir, "F7_Factor_Decomp_Drivers_Regions_CSV.csv"))
   
   ###########################################################################
   # REGIONS
   
    def plot_barh_with_errors_by_region(df, title_prefix="", save=True):
        """
        Plot grouped horizontal bar charts with asymmetric error bars per region.
        2 columns × N rows grid.
        """
        
        # 🔹 Region name mapping
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
        
        # df = fact_drivers_reg_mme
    
        # 🔹 Filter out region 0
        df = df[df["basisregions"] != 0]
    
        # Identify all unique regions
        regions = df["basisregions"].unique()
        n_regions = len(regions)
        ncols = 3
        nrows = int(np.ceil(n_regions / ncols))
    
        # Define which columns to use
        cols_mme = [
            "BA_area_pred_abs_mean",
            "BA_Free_socioeconomic_abs_mean",
            "BA_Free_land_use_abs_mean",
            "BA_Free_vegetation_abs_mean",
            "BA_Free_climate_abs_mean"
        ]
        cols_lower = [c.replace("_mean", "_lower") for c in cols_mme]
        cols_upper = [c.replace("_mean", "_upper") for c in cols_mme]
    
        # --- Create subplots (not sharing x-axis) ---
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(14, nrows * 3),
            sharex=False
        )
        axes = axes.flatten()
        
        cmap = plt.get_cmap("tab20")
        bar_height = 0.8 / len(cols_mme)
        
        for i, region in enumerate(regions):
            ax = axes[i]
            sub = df[df["basisregions"] == region]
        
            # Prepare data
            df_mean = sub.set_index("Scenario")[cols_mme]
            df_lower = sub.set_index("Scenario")[cols_lower]
            df_upper = sub.set_index("Scenario")[cols_upper]
        
            means = df_mean.values
            lower = df_lower.values
            upper = df_upper.values
        
            err_neg = means - lower
            err_pos = upper - means
        
            nrows_sub, ncols_sub = means.shape
            y = np.arange(nrows_sub)
        
            # Plot bars for this region
            for j, col in enumerate(df_mean.columns):
                offset = (j - (ncols_sub - 1) / 2) * bar_height
                y_pos = y + offset
                ax.barh(
                    y_pos,
                    means[:, j],
                    xerr=[err_neg[:, j], err_pos[:, j]],
                    height=bar_height * 0.9,
                    color=cmap(j),
                    label=(col.replace("_abs_mean", "")
                              .replace("BA_Free_", "Transient ")
                              .replace("BA_area_pred", "All transient (default BA)")),
                    capsize=3,
                    ecolor="black",
                )
            # --- Use full region name as title ---
            region_label = region_name.get(region, ("UNK", f"Region {region}"))[1]
            ax.set_title(region_label, fontsize=11)
        
            ax.set_yticks(y)
            ax.set_yticklabels(df_mean.index, fontsize=9)
            # ax.set_title(f"Region {region}", fontsize=11)
            ax.axvline(0, color="black", lw=0.8)
        
            if i % 2 == 0:
                ax.set_ylabel("Scenario", fontsize=10)
            else:
                ax.set_ylabel("")
        
            ax.set_xlabel("Absolute change (Mha) relative to 2020")
        
            # Allow independent scales per region (default behavior of sharex=False)
            ax.autoscale(enable=True, axis='x', tight=False)
        
        # Hide unused panels if any
        for ax in axes[len(regions):]:
            ax.set_visible(False)
        
        # Common legend
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Driver",
                   bbox_to_anchor=(0.65, 0.1), loc="center left")
        
        fig.suptitle(f"{title_prefix} — Factorial Decomposition by Drivers",
                     fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 0.88, 0.96])
        
        if save:
            fig.savefig(os.path.join(figures_dir,
                        "SM16_BA_Factor_Decomp_Drivers_Barplot_REGIONAL.png"),
                        dpi=300, bbox_inches="tight")
        
        plt.show()
        
        # if save == True:
        #     fact_figure.savefig(os.path.join(results_dir,f"figures/BA_Factor_Decomp_Drivers_Barplot_MME_LOCAL.png"))
    
    plot_barh_with_errors_by_region(fact_drivers_reg_mme, title_prefix="(a)", save=True)
    
def SM11_14_global_regions(time_series_reg):

    
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import linregress

        
    time_series_reg["Scenario"] = time_series_reg["Scenario"].fillna("Historic")
    time_series_reg = time_series_reg.loc[time_series_reg["basisregions"] != 0]
    
    ######################################################################
    # BURNED AREAS
    ######################################################################

    
    variables = ["BA_area_pred", "BA_area_for_pred"]
    # variables = ["CE_total_C_pred_GFED", "CE_for_C_pred_GFED", "CE_nonf_C_pred_GFED"]
    
    for fig_num, variable in enumerate(variables, start=11):
    
        # variable = "CE_total_C_pred_GFED"
        
        # Compute df mean ± std ---
        df = (
            time_series_reg.groupby(["time", "Scenario", "basisregions"])
              .agg(
                  mean=(f"{variable}", "mean"),
                  std=(f"{variable}", "std"),
                  vmin=(f"{variable}", "min"),
                  vmax=(f"{variable}", "max")
              )
              .reset_index()
        )
        
        # df["lower"] = df["mean"] - df["std"]
        # df["upper"] = df["mean"] + df["std"]
        df["lower"] = df["vmin"] 
        df["upper"] = df["vmax"] 
        
        
        
        # # Ensure 'time' is numeric
        # df["time_numeric"] = pd.to_datetime(df["time"]).dt.year  # if time is datetime
    
        
        # Replace basisregions numbers with codes (or full names)
        df["region_code"] = df["basisregions"].map(lambda x: region_name[x][0])
        df["region_full"] = df["basisregions"].map(lambda x: region_name[x][1])
    
        
        # --- Plot with matplotlib ---
        regions = df["region_code"].unique()
        n_regions = len(regions)
        
        ncols = 4
        nrows = int((n_regions + ncols - 1) / ncols)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3*nrows), sharex=True)
        axes = axes.flatten()
        
        colors = {
            "Historic": "black",
            "SSP1-2p6o": "tab:green",
            "SSP2-4p5": "tab:blue",
            "SSP3-6p6": "tab:red",
            "SSP5-7p6": "tab:orange"
        }
        
        # colors = generate_colors_ssp(scenarios)
        
        for i, region in enumerate(regions):
            ax = axes[i]
            sub = df[df["region_code"] == region]
        
            for scenario, group in sub.groupby("Scenario"):
                c = colors.get(scenario, "gray")
        
                # Shaded ±1σ
                ax.fill_between(group["time"], group["lower"], group["upper"], color=c, alpha=0.2)
        
                # Mean line
                ax.plot(group["time"], group["mean"], color=c, label=f"{scenario}", linewidth=2)
                        
                add_trend(ax, group["time"].values, group[f"mean"].values, scenario, colors.get(scenario, "gray"))

            ax.set_title(f"{region} – {sub['region_full'].iloc[0]}")
            ax.set_ylabel("Burned Area (Mha)")
        # Hide unused subplots
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
        
        fig.text(0.5, 0.04, "Year", ha="center")
        fig.suptitle("Burned Area (Predicted) – Ensemble Mean ± 1σ", fontsize=16)
        # Put legend outside
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title="Scenario", loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4)
        
        plt.tight_layout(rect=[0,0.05,1,0.95])
        region_figure=plt.gcf()
        plt.show()
        
    
        # Save figure with dynamic SM7, SM8, etc.
        filename = f"SM{fig_num}_{variable}_Regional_Total_BA_MME_LOCAL.png"
        
        if "MME" in esms:
            region_figure.savefig(os.path.join(figures_dir, f"{filename}.png"))
        else:
            region_figure.savefig(os.path.join(figures_dir, f"{filename}.png"))
        
            
        ######################################################################
        # CARBON EMISSIONS 
        ######################################################################
        
        variables = ["CE_total_C_pred_GFED", "CE_for_C_pred_GFED"]
        for fig_num, variable in enumerate(variables, start=13):
        
            # variable = "CE_total_C_pred_GFED"
            
            # Compute df mean ± std ---
            df = (
                time_series_reg.groupby(["time", "Scenario", "basisregions"])
                  .agg(
                      mean=(f"{variable}", "mean"),
                      std=(f"{variable}", "std"),
                      vmin=(f"{variable}", "min"),
                      vmax=(f"{variable}", "max")
                  )
                  .reset_index()
            )
            
            # df["lower"] = df["mean"] - df["std"]
            # df["upper"] = df["mean"] + df["std"]
            df["lower"] = df["vmin"] 
            df["upper"] = df["vmax"] 
            
            
            
            # # Ensure 'time' is numeric
            # df["time_numeric"] = pd.to_datetime(df["time"]).dt.year  # if time is datetime
        
            
            # Replace basisregions numbers with codes (or full names)
            df["region_code"] = df["basisregions"].map(lambda x: region_name[x][0])
            df["region_full"] = df["basisregions"].map(lambda x: region_name[x][1])
        
            
            # --- Plot with matplotlib ---
            regions = df["region_code"].unique()
            n_regions = len(regions)
            
            ncols = 4
            nrows = int((n_regions + ncols - 1) / ncols)
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3*nrows), sharex=True)
            axes = axes.flatten()
            
            colors = {
                "Historic": "black",
                "SSP1-2p6o": "tab:green",
                "SSP2-4p5": "tab:blue",
                "SSP3-6p6": "tab:red",
                "SSP5-7p6": "tab:orange"
            }
            
            # colors = generate_colors_ssp(scenarios)
            
            for i, region in enumerate(regions):
                ax = axes[i]
                sub = df[df["region_code"] == region]
            
                for scenario, group in sub.groupby("Scenario"):
                    c = colors.get(scenario, "gray")
            
                    # Shaded ±1σ
                    ax.fill_between(group["time"], group["lower"], group["upper"], color=c, alpha=0.2)
            
                    # Mean line
                    ax.plot(group["time"], group["mean"], color=c, label=f"{scenario}", linewidth=2)
                                
                    add_trend(ax, group["time"].values, group[f"mean"].values, scenario, colors.get(scenario, "gray"))

                ax.set_title(f"{region} – {sub['region_full'].iloc[0]}")
                ax.set_ylabel("Carbon Emissions (TgC)")
            # Hide unused subplots
            for j in range(i+1, len(axes)):
                fig.delaxes(axes[j])
            
            fig.text(0.5, 0.04, "Year", ha="center")
            fig.suptitle("Carbon Emissions (Predicted) – Ensemble Mean ± 1σ", fontsize=16)
            # Put legend outside
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, title="Scenario", loc="upper center", bbox_to_anchor=(0.5, 0), ncol=4)
            
            plt.tight_layout(rect=[0,0.05,1,0.95])
            region_figure=plt.gcf()
            plt.show()
            
        
            # Save figure with dynamic SM7, SM8, etc.
            filename = f"SM{fig_num}_{variable}_Regional_Total_CE_MME_LOCAL.png"
            
            if "MME" in esms:
                region_figure.savefig(os.path.join(figures_dir,f"{filename}.png"))
            else:
                region_figure.savefig(os.path.join(figures_dir,f"{filename}.png"))
            
            print("Figure 1.E (Maps) for Dominant Driver Map Processed and Saved")
    
def SM15_global_all_drivers(time_series_mme):
    
    ###########################################################################
    # FIGURE 1.E GLOBAL TEMPORAL TREND (GRAPH) ESM-DEPENDENT DRIVERS ONLY
    ###########################################################################
    
    # Initialize the figure
    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(12, 12))
    
    # # Generate the colors dictionary
    # colors = generate_colors_ssp(scenarios, esms)
    
    # Check if "Source" column exists and print unique values
    if "Source" in time_series_mme.columns:
        print("Unique Sources:", time_series_mme["Source"].unique())
    else:
        print("Column 'Source' not found in time_series!")
    
    # Define subplot titles (row, col): title
    subplot_titles = {
        (0, 0): "(a) Vapor Pressure Deficit (VPD)",
        (0, 1): "(b) Number of Dry Days (NDD)",
        (0, 2): "(c) Wind Speed",
        (1, 0): "(d) Precipitation 30-Day Rolling Mean",
        (1, 1): "(e) Gross Primary Productivity (GPP)",
        (1, 2): "(f) Human Development Index (HDI)",
        (2, 0): "(g) Cropland",
        (2, 1): "(i) Forests",
        (2, 2): "(j) Grass & Shrubs"
        
    }
    
    # Assign titles to subplots
    for (i, j), title in subplot_titles.items():
        axs[i, j].set_title(title)
    
    # Collect all legend handles/labels
    legend_handles = []
    legend_labels = []
    
    # Plot each variable with color based on 'Source'
    for source, data in time_series_mme.groupby("Scenario"):
        print(f"Plotting {source}...")  # Debugging output
        
        # Ensure the index is datetime for proper plotting
        data = data.set_index("time")
    
        # --- (a) Vapor Pressure Deficit (VPD) ---
        data["vpd_mean"].plot(ax=axs[0, 0], color=colors.get(source, "gray"), label=f"{source}")
        axs[0, 0].fill_between(
            data.index,
            data["vpd_lower"],
            data["vpd_upper"],
            color=colors.get(source, "gray"),
            alpha=0.2
        )
        axs[0,0].set_ylabel("Pascals (kPa)")

    
        # --- (b) Number of Dry Days (NDD) ---
        data["ndd_mean"].plot(ax=axs[0, 1], color=colors.get(source, "gray"), label=f"{source}")
        axs[0, 1].fill_between(
            data.index,
            data["ndd_lower"],
            data["ndd_upper"],
            color=colors.get(source, "gray"),
            alpha=0.2
        )
        axs[0,1].set_ylabel("Days per Months (0-31)")

    
        # --- (c) Wind Speed (sfcWind) ---
        data["sfcWind_mean"].plot(ax=axs[0, 2], color=colors.get(source, "gray"), label=f"{source}")
        axs[0, 2].fill_between(
            data.index,
            data["sfcWind_lower"],
            data["sfcWind_upper"],
            color=colors.get(source, "gray"),
            alpha=0.2
        )
        axs[0,2].set_ylabel("Wind Speed (m/s)")

    
        # --- (d) Precipitation 30-Day Rolling Mean (pr_30d_sum) ---
        data["pr_30d_sum_mean"].plot(ax=axs[1, 0], color=colors.get(source, "gray"), label=f"{source}")
        axs[1, 0].fill_between(
            data.index,
            data["pr_30d_sum_lower"],
            data["pr_30d_sum_upper"],
            color=colors.get(source, "gray"),
            alpha=0.2
        )
        axs[1,0].set_ylabel("Millimeters (mm)")

    
        # --- (e) Gross Primary Productivity (GPP) ---
        data["GPP_mean"].plot(ax=axs[1, 1], color=colors.get(source, "gray"), label=f"{source}")
        axs[1, 1].fill_between(
            data.index,
            data["GPP_lower"],
            data["GPP_upper"],
            color=colors.get(source, "gray"),
            alpha=0.2
        )
        axs[1,1].set_ylabel("Biomass Flux (gC.m2.year)")
    
        # time_series_ssp['HDI'].plot(ax=axs[2, 1], color=colors.get(source, "gray"), label=f"{source}")
        for source, data_hdi in time_series_ssp.groupby("Scenario"):
            color = colors.get(source, "gray")
            # data_hdi = data_hdi.set_index("time")
            data_hdi["HDI"].plot(
                ax=axs[1, 2],
                color=color,
                label=f"{source}"
            )
            axs[1,2].set_ylabel("Index (0-1)")
        # time_series_ssp['HDI'].plot(ax=axs[2, 1], color=colors.get(source, "gray"), label=f"{source}")
        for source, data_hdi in time_series_ssp.groupby("Scenario"):
            color = colors.get(source, "gray")
            # data_hdi = data_hdi.set_index("time")
            data_hdi["cropland"].plot(
                ax=axs[2, 0],
                color=color,
                label=f"{source}"
            )
            axs[2,0].set_ylabel("Area (Mha)")

        # time_series_ssp['HDI'].plot(ax=axs[2, 1], color=colors.get(source, "gray"), label=f"{source}")
        for source, data_hdi in time_series_ssp.groupby("Scenario"):
            color = colors.get(source, "gray")
            # data_hdi = data_hdi.set_index("time")
            data_hdi["forest"].plot(
                ax=axs[2, 1],
                color=color,
                label=f"{source}"
            )
            axs[2,1].set_ylabel("Area (Mha)")

        # time_series_ssp['HDI'].plot(ax=axs[2, 1], color=colors.get(source, "gray"), label=f"{source}")
        for source, data_hdi in time_series_ssp.groupby("Scenario"):
            color = colors.get(source, "gray")
            # data_hdi = data_hdi.set_index("time")
            data_hdi["grass-shrub"].plot(
                ax=axs[2, 2],
                color=color,
                label=f"{source}"
            )
            axs[2,2].set_ylabel("Area (Mha)")


    # Collect unique legend entries across all subplots
    legend_handles = []
    legend_labels = []
    seen = set()
    
    for ax in axs.flatten():
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l not in seen:
                legend_handles.append(h)
                legend_labels.append(l)
                seen.add(l)
    
    # Clear legends from individual plots
    for ax in axs.flatten():
        ax.legend().remove()
    
    # Add single legend at the bottom center
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='lower center',
        ncol=5,  # Adjust based on number of entries
        fontsize=9,
        bbox_to_anchor=(0.5, -0.01)
    )
    
    # Adjust layout to make space for legend
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    pred_time=plt.gcf()
    plt.show()
    pred_time.savefig(os.path.join(figures_dir,f"SM15_Predictors_AllScen_Trend_MME.png"))

def SM1718_fact_decomp_limiting(scenarios, esms):
    
    
   ###########################################################################
   # DATAFRAME FOR FACTORIAL DECOMPOSITION  
   ###########################################################################
   
   # MME LIMITING FACTORS 
    fact_decomp_limit = []
    for i, (scenario, esm) in enumerate([(s, e) for s in scenarios for e in esms]):
       
       # scenario = "SSP1-2p6o"
       # esm = "CanESM5"
       # Load and aggregate
       # fname = f"BA_Factorial_Decomp_{scenario}_{esm}.csv"
       fname = f"BA_Factorial_Decomp_LimitingF_Agg_{scenario}_{esm}_v5.1.csv"
       df = pd.read_csv(os.path.join(outputs_dir, fname))
       df = df.set_index("time").groupby(level="time").sum() # IF AGG WORKS .. 

       # df = df.set_index(["lat", "lon", "time"])
       # df = df.groupby(level="time").sum()
       
       
       
       df_2020 = df.loc[df.index.get_level_values("time") == "2020-12-31"]
       df = df.loc[df.index.get_level_values("time") == "2100-12-31"]
       df["Scen - ESM"] = f"{scenario} - {esm}"
       df["Scenario"] = f"{scenario}"
       df["ESM"] = f"{esm}"
       
       # Select BA columns
       ba_cols = [c for c in df.columns if "BA_Fixed" in c or c == "BA_area_pred"]
   
       for col in ba_cols:
           df[f"{col}_percent"] = 100 * (df[f"{col}"] - df["BA_area_pred"]) / df["BA_area_pred"] # % change relative to baseline (2100)
           # df[f"{col}_abs"] = df[f"{col}"] - df["BA_area_pred"] # Absolute difference (Mha) relative to baseyear (2020)
           df[f"{col}_abs"] = df[f"{col}"].values - df_2020["BA_area_pred"].values # Absolute difference (Mha) relative to baseyear (2020)

       
       fact_decomp_limit.append(df)
   
    fact_limiting = pd.concat(fact_decomp_limit)
   
   # Compute df mean ± std ---
    fact_limiting_mme = []
   
    for variable in ba_cols: 
       df = (
           fact_limiting.groupby(["time", "Scenario"])
             .agg(
                 mean=(f"{variable}_percent", "mean"),
                 std=(f"{variable}_percent", "std"),
                 mean_abs=(f"{variable}_abs", "mean"),
                 std_abs=(f"{variable}_abs", "std"),
                 vmax_abs=(f"{variable}_abs", "max"),
                 vmin_abs=(f"{variable}_abs", "min"),

                 
             )
             .reset_index()
       )
       
       # Add lower/upper
       df[f"{variable}_percent_mean"] = df["mean"]
       df[f"{variable}_percent_lower"] = df["mean"] - df["std"]
       df[f"{variable}_percent_upper"] = df["mean"] + df["std"]
       df[f"{variable}_abs_mean"] = df["mean_abs"]
       # df[f"{variable}_abs_lower"] = df["mean_abs"] - df["std_abs"]
       # df[f"{variable}_abs_upper"] = df["mean_abs"] + df["std_abs"]
       df[f"{variable}_abs_lower"] = df["vmin_abs"] 
       df[f"{variable}_abs_upper"] = df["vmax_abs"] 

       
       
       # Keep only relevant columns
       df = df[["time", "Scenario",
                f"{variable}_percent_mean", f"{variable}_percent_lower", f"{variable}_percent_upper",
                f"{variable}_abs_mean", f"{variable}_abs_lower", f"{variable}_abs_upper"]]
       
       fact_limiting_mme.append(df)
   
   # Concatenate all variables into one DataFrame
    fact_limiting_mme = pd.concat(fact_limiting_mme, axis=1)
   
   # Drop duplicate index columns from concat
    fact_limiting_mme = fact_limiting_mme.loc[:, ~fact_limiting_mme.columns.duplicated()]
   
    fact_limiting_mme.to_csv(os.path.join(outputs_dir, "SM17_Factor_Decomp_Limiting_CSV.csv"))

    def plot_barh_with_errors_LF(df_mean, df_lower, df_upper, title="", save=True):
        """
        Plot grouped horizontal bar chart with asymmetric error bars.
    
        Parameters
        ----------
        df_mean : DataFrame
            Mean values, shape (nrows, ncols).
        df_lower : DataFrame
            Lower bounds, same shape as df_mean.
        df_upper : DataFrame
            Upper bounds, same shape as df_mean.
        title : str
            Plot title.
        """
    
        means = df_mean.values
        lower = df_lower.values
        upper = df_upper.values
    
        err_neg = means - lower
        err_pos = upper - means
    
        nrows, ncols = means.shape
        y = np.arange(nrows)
    
        fig, ax = plt.subplots(figsize=(10, 6))
        cmap = plt.get_cmap("tab20")
        bar_height = 0.8 / ncols
    
        for i, col in enumerate(df_mean.columns):
            offset = (i - (ncols - 1) / 2) * bar_height
            y_pos = y + offset
            ax.barh(
                y_pos,
                means[:, i],
                xerr=[err_neg[:, i], err_pos[:, i]],
                height=bar_height * 0.9,
                color=cmap(i),
                label=(col.replace("_abs_mean", "")
                          .replace("BA_Fixed_", "Fixed ")
                          .replace("BA_area_pred", "Default BA")),
    
                capsize=3,
                ecolor="black",
            )
    
        ax.set_yticks(y)
        ax.set_yticklabels(df_mean.index)
        ax.set_xlabel("Absolute change (Mha) relative to 2020")
        # ax.set_ylabel("Scenario - ESM")
        ax.set_title(title)
        ax.legend(title="Limiting Factor", bbox_to_anchor=(1, 0.3), loc="best")
        plt.tight_layout()
        fact_figure = plt.gcf()
        plt.show()
        
        if save == True:
            fact_figure.savefig(os.path.join(figures_dir,f"SM17_BA_Factor_Decomp_Limited_Barplot_MME_LOCAL.png"))
    
     # Use the function to create the MME figure with fact_limiting_mme
    cols_mme = [
         "BA_area_pred_abs_mean",
         "BA_Fixed_socioeconomic_abs_mean",
         "BA_Fixed_land_use_abs_mean",
         "BA_Fixed_vegetation_abs_mean",
         "BA_Fixed_climate_abs_mean"]
    cols_lower = [c.replace("_mean", "_lower") for c in cols_mme]
    cols_upper = [c.replace("_mean", "_upper") for c in cols_mme]
     
    df_mean  = fact_limiting_mme.set_index("Scenario")[cols_mme]
    df_lower = fact_limiting_mme.set_index("Scenario")[cols_lower]
    df_upper = fact_limiting_mme.set_index("Scenario")[cols_upper]
     
    plot_barh_with_errors_LF(df_mean, df_lower, df_upper,
                           title="Factorial Decomposition by Limiting Factors",
                           save = True)
     
        
    
    
   ###########################################################################
   # MME DATAFRAME DRIVERS - REGIONAL
    fact_decomp_limiting_reg = []
   
   
    for i, (scenario, esm) in enumerate([(s, e) for s in scenarios for e in esms]):
       
       # scenario = "SSP1-2p6o"
       # esm = "CanESM5"
       
       # Load and aggregate
       fname = f"BA_Factorial_Decomp_LimitingF_Agg_{scenario}_{esm}_v5.1.csv"
       # fname = f"BA_Factorial_Decomp_Reverse_{scenario}_{esm}.csv"
       df = pd.read_csv(os.path.join(outputs_dir, fname))
       df = df.set_index(["time", "basisregions"])

       # df = df.set_index(["lat", "lon", "time", "basisregions"])
       # df = df.groupby(level=["time", "basisregions"]).sum()

       
       df_2020 = df.loc[df.index.get_level_values("time") == "2020-12-31"]
       df = df.loc[df.index.get_level_values("time") == "2100-12-31"]
       df["Scen - ESM"] = f"{scenario} - {esm}"
       df["Scenario"] = f"{scenario}"
       df["ESM"] = f"{esm}"
       # df["Region_id"]= f"{basisregions}"
       
       # Select BA columns
       ba_cols = [c for c in df.columns if "BA_Fixed" in c or c == "BA_area_pred"]
   
       for col in ba_cols:
           df[f"{col}_percent"] = 100 * (df[f"{col}"] - df["BA_area_pred"]) / df["BA_area_pred"] # % change relative to baseline (2100)
           df[f"{col}_abs"] = df[f"{col}"].values - df_2020["BA_area_pred"].values # Absolute difference (Mha) relative to baseyear (2020)
           
       
       fact_decomp_limiting_reg.append(df)
   
    fact_limiting_reg = pd.concat(fact_decomp_limiting_reg)
   
   # Compute df mean ± std ---
    fact_limiting_reg_mme = []
   
    for variable in ba_cols: 
       df = (
           fact_limiting_reg.groupby(["time", "Scenario", "basisregions"])
             .agg(
                 mean=(f"{variable}_percent", "mean"),
                 std=(f"{variable}_percent", "std"),
                 mean_abs=(f"{variable}_abs", "mean"),
                 std_abs=(f"{variable}_abs", "std"),
                 vmax_abs=(f"{variable}_abs", "max"),
                 vmin_abs=(f"{variable}_abs", "min"),

                 
             )
             .reset_index()
       )
       
       # Add lower/upper
       df[f"{variable}_percent_mean"] = df["mean"]
       df[f"{variable}_percent_lower"] = df["mean"] - df["std"]
       df[f"{variable}_percent_upper"] = df["mean"] + df["std"]
       df[f"{variable}_abs_mean"] = df["mean_abs"]
       df[f"{variable}_abs_lower"] = df["mean_abs"] - df["std_abs"]
       df[f"{variable}_abs_upper"] = df["mean_abs"] + df["std_abs"]
       df[f"{variable}_abs_lower"] = df["vmin_abs"] 
       df[f"{variable}_abs_upper"] = df["vmax_abs"] 

       
       
       # Keep only relevant columns
       df = df[["time", "Scenario", "basisregions",
                f"{variable}_percent_mean", f"{variable}_percent_lower", f"{variable}_percent_upper",
                f"{variable}_abs_mean", f"{variable}_abs_lower", f"{variable}_abs_upper"]]
       
       fact_limiting_reg_mme.append(df)
   
   # Concatenate all variables into one DataFrame
    fact_limiting_reg_mme = pd.concat(fact_limiting_reg_mme, axis=1)
   
   # Drop duplicate index columns from concat
    fact_limiting_reg_mme = fact_limiting_reg_mme.loc[:, ~fact_limiting_reg_mme.columns.duplicated()]
   
    fact_limiting_reg_mme.to_csv(os.path.join(outputs_dir, "SM18_Factor_Decomp_Limiting_Regions_CSV.csv"))
     
    def plot_barh_with_errors_LF_by_region(df, title_prefix="", save=True):
         """
         Plot grouped horizontal bar charts with asymmetric error bars per region.
         2 columns × N rows grid.
         """
         
         # 🔹 Region name mapping
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
         
         # df = fact_drivers_reg_mme
     
         # 🔹 Filter out region 0
         df = df[df["basisregions"] != 0]
     
         # Identify all unique regions
         regions = df["basisregions"].unique()
         n_regions = len(regions)
         ncols = 3
         nrows = int(np.ceil(n_regions / ncols))
     
         # Define which columns to use
         cols_mme = [
             "BA_area_pred_abs_mean",
             "BA_Fixed_socioeconomic_abs_mean",
             "BA_Fixed_land_use_abs_mean",
             "BA_Fixed_vegetation_abs_mean",
             "BA_Fixed_climate_abs_mean"
         ]
         cols_lower = [c.replace("_mean", "_lower") for c in cols_mme]
         cols_upper = [c.replace("_mean", "_upper") for c in cols_mme]
     
         # --- Create subplots (not sharing x-axis) ---
         fig, axes = plt.subplots(
             nrows=nrows, ncols=ncols, figsize=(14, nrows * 3),
             sharex=False
         )
         axes = axes.flatten()
         
         cmap = plt.get_cmap("tab20")
         bar_height = 0.8 / len(cols_mme)
         
         for i, region in enumerate(regions):
             ax = axes[i]
             sub = df[df["basisregions"] == region]
         
             # Prepare data
             df_mean = sub.set_index("Scenario")[cols_mme]
             df_lower = sub.set_index("Scenario")[cols_lower]
             df_upper = sub.set_index("Scenario")[cols_upper]
         
             means = df_mean.values
             lower = df_lower.values
             upper = df_upper.values
         
             err_neg = means - lower
             err_pos = upper - means
         
             nrows_sub, ncols_sub = means.shape
             y = np.arange(nrows_sub)
         
             # Plot bars for this region
             for j, col in enumerate(df_mean.columns):
                 offset = (j - (ncols_sub - 1) / 2) * bar_height
                 y_pos = y + offset
                 ax.barh(
                     y_pos,
                     means[:, j],
                     xerr=[err_neg[:, j], err_pos[:, j]],
                     height=bar_height * 0.9,
                     color=cmap(j),
                     label=(col.replace("_abs_mean", "")
                               .replace("BA_Fixed_", "Fixed ")
                               .replace("BA_area_pred", "Default BA")),
                     capsize=3,
                     ecolor="black",
                 )
             # --- Use full region name as title ---
             region_label = region_name.get(region, ("UNK", f"Region {region}"))[1]
             ax.set_title(region_label, fontsize=11)
         
             ax.set_yticks(y)
             ax.set_yticklabels(df_mean.index, fontsize=9)
             # ax.set_title(f"Region {region}", fontsize=11)
             ax.axvline(0, color="black", lw=0.8)
         
             if i % 2 == 0:
                 ax.set_ylabel("Scenario", fontsize=10)
             else:
                 ax.set_ylabel("")
         
             ax.set_xlabel("Absolute change (Mha) relative to 2020")
         
             # Allow independent scales per region (default behavior of sharex=False)
             ax.autoscale(enable=True, axis='x', tight=False)
         
         # Hide unused panels if any
         for ax in axes[len(regions):]:
             ax.set_visible(False)
         
         # Common legend
         handles, labels = axes[0].get_legend_handles_labels()
         fig.legend(handles, labels, title="Limiting Factor",
                    bbox_to_anchor=(0.65, 0.1), loc="center left")
         
         fig.suptitle(f"{title_prefix} — Factorial Decomposition by Limiting Factors",
                      fontsize=16, y=0.98)
         plt.tight_layout(rect=[0, 0, 0.88, 0.96])
         
         if save:
             fig.savefig(os.path.join(figures_dir,
                         "SM18_BA_Factor_Decomp_Limiting_Barplot_REGIONAL.png"),
                         dpi=300, bbox_inches="tight")
         
         plt.show()
         
         # if save == True:
         #     fact_figure.savefig(os.path.join(figures_dir,f"figures/BA_Factor_Decomp_Drivers_Barplot_MME_LOCAL.png"))
     
    plot_barh_with_errors_LF_by_region(fact_limiting_reg_mme, title_prefix="(a)", save=True)


def remove_nas(x):
    return x[~pd.isnull(x)]
    
if __name__ == "__main__":
    
    F5_global_BAFCCE_trend(time_series_mme)
    F6_global_regions_heatmap(time_series_reg)
    
    SM11_14_global_regions(time_series_reg)
    SM15_global_all_drivers(time_series_mme)

    F7_SM16_fact_decomp_drivers(scenarios, esms)
    SM1718_fact_decomp_limiting(scenarios, esms)

    
    
    
    
    
    