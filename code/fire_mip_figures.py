# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:08:21 2025

@author: theo.rouhette
"""

# Packages Required to Browse and Analyze CMIP6 Data
import numpy as np
import pandas as pd
import xarray as xr
import os
import dask.array as da
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
from scipy.stats import linregress

# PATHS
IF_PATH = "C:\\GCAM\\Theo\\zenodo\\"
inputs_dir = os.path.join(IF_PATH, f"inputs/")
outputs_dir = os.path.join(IF_PATH, f"outputs/")
figures_dir = os.path.join(IF_PATH, f"figures/")

# SCENARIOS & ESMS
scenarios = ["SSP1-2p6o", "SSP2-4p5", "SSP3-6p6", "SSP5-7p6"]
scenarios = ["SSP1-2.6", "SSP2-4.5", "SSP3-6.6", "SSP5-7.6"]
esms = ["MPI-ESM1-2-LR", "CanESM5"]   

# INPUTS
iamfire_results = pd.read_csv(os.path.join(outputs_dir, "BA_CE_Prediction_AllScen.csv"))
# Define the mapping dictionary
name_mapping = {
    "SSP1-2p6o": "SSP1-2.6",
    "SSP2-4p5": "SSP2-4.5",
    "SSP3-6p6": "SSP3-6.6",
    "SSP5-7p6": "SSP5-7.6"
}

colors = {
    "Historic": "black",
    "SSP1-2.6": "tab:green",
    "SSP2-4.5": "tab:blue",
    "SSP3-6.6": "tab:red",
    "SSP5-7.6": "tab:orange"
}

# Apply to the first DataFrame
iamfire_results['Scenario'] = iamfire_results['Scenario'].replace(name_mapping)
for old_name, new_name in name_mapping.items():
    iamfire_results["Source"] = iamfire_results["Source"].str.replace(old_name, new_name, regex=False)
BA_CMIP6 = pd.read_csv(os.path.join(inputs_dir, "BA_CMIP6.csv"))
BA_FireMIP = pd.read_csv(os.path.join(inputs_dir, "BA_FireMIP.csv"))


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

def grid_cell_area(lat, dlon=0.5, dlat=0.5):
    """Calculate grid cell area given latitude, assuming a 0.5° x 0.5° grid."""
    R = 6371  # Earth radius in km
    lat_rad = np.radians(lat)
    
    # Width of cell (km) varies with latitude
    cell_width = (np.pi / 180) * R * np.cos(lat_rad) * dlon  
    # Height of cell (km)
    cell_height = (np.pi / 180) * R * dlat  
    
    return cell_width * cell_height  # Area in km²


def master_csv(BA_CMIP6, BA_FireMIP, iamfire_results):

    BA_FireMIP["Model"] = "FireMIP"
    BA_FireMIP['time'] = pd.to_datetime(BA_FireMIP['time'])
    BA_FireMIP["year"] = BA_FireMIP["time"].dt.year
    BA_FireMIP = BA_FireMIP[BA_FireMIP["year"] > 1999] 
    BA_FireMIP["ESM"] = BA_FireMIP["Fire Model"] + "-" + BA_FireMIP["ESM"] 

    BA_CMIP6["Model"] = "CMIP6"
    BA_CMIP6['time'] = pd.to_datetime(BA_CMIP6['time'])
    BA_CMIP6["year"] = BA_CMIP6["time"].dt.year
    BA_CMIP6 = BA_CMIP6[(BA_CMIP6["ESM"] != "EC-Earth3-CC")]
    BA_CMIP6 = BA_CMIP6.drop(["grid_area", "burntFractionAll"], axis=1)
        
    # --- 1. Split Source into Scenario & ESM ---
    iamfire_results[["Experiment", "ESM"]] = iamfire_results["Source"].str.split(" - ", n=1, expand=True)
    iamfire_results["Model"] = "IAM-FIRE"
    iamfire_results['time'] = pd.to_datetime(iamfire_results['time'])
    iamfire_results["year"] = iamfire_results["time"].dt.year
    iamfire_results = iamfire_results.loc[iamfire_results["Grid_area"] > 0]    
    iamfire_results = iamfire_results[["time", "ESM", "Experiment", "Source", "Model", "year", "BA_area_pred"]]
    
    ba_dfs = [BA_CMIP6, BA_FireMIP, iamfire_results] 
    df = pd.concat(ba_dfs)

    # Handles cases like "ssp126", "ssp245", "ssp370", "ssp3-6p6", "ssp5-7p6", "ssp585"
    def extract_rcp(exp):
        exp = exp.lower()
        if ("126" in exp) or ("2.6" in exp):
            return 2.6
        elif ("245" in exp) or ("4.5" in exp):
            return 4.5
        elif "370" in exp:
            return 7.0
        elif "3-6.6" in exp:
            return 6.6
        elif "5-7.6" in exp:
            return 7.6
        elif "585" in exp:
            return 8.5
        else:
            return np.nan

    df["RCP"] = df["Experiment"].apply(extract_rcp)

    # DF will now serve for figure 1 main time series panel of F7 
    df.to_csv(os.path.join(outputs_dir, "F9_BA_Global_Trends.csv"))
    
    # SUmmaries will be a CSV used for the scatterplots of F7 
    # --- Keep only years 2020–2100 ---
    df_sp = df[(df["year"] >= 2020) & (df["year"] <= 2100)]
    # df_sp = df[(df["year"] >= 2002) & (df["year"] <= 2100)]
    # --- Compute slope and final BA for each model+scenario ---
    summaries = []
    for (model, esm, exp), group in df_sp.groupby(["Model", "ESM", "Experiment"]):
        if group.empty or group["BA_area_pred"].isna().all():
            continue
        g = group.dropna(subset=["BA_area_pred"]).sort_values("year")
        if len(g) < 5:
            continue
        # slope in Mha per year
        slope, intercept, r, p, se = linregress(g["year"], g["BA_area_pred"])
        # value in 2100
        ba_2020 = g.loc[g["year"] == 2020, "BA_area_pred"].mean()
        ba_2100 = g.loc[g["year"] == 2100, "BA_area_pred"].mean()
        summaries.append({
            "Model": model,
            "Source": esm,
            "Experiment": exp,
            "RCP": extract_rcp(exp),
            "Total_BA_2020": ba_2020,
            "Total_BA_2100": ba_2100,
            "Slope_2020_2100": slope,
            "r_value": r,
            "p_value": p
        })

    summary_df = pd.DataFrame(summaries)

    out_file = os.path.join(outputs_dir, "F9_BA_Summary_RCP_Slopes.csv")
    summary_df.to_csv(out_file, index=False)
        
    print(f"Scenario summary saved to {out_file}")
    
    # Extact the slopes of Historic period for CMIP6 and FireMIP
    
    # 1. Define Historical Period (Adjust based on your data availability)
    start_hist = 2002
    end_hist   = 2016 # CMIP6 historical usually ends in 2014
    
    # 2. Filter for Historical data only
    # We use .str.lower() to catch both "Historical" and "historical"
    df_hist = df[(df["year"] >= start_hist) & (df["year"] <= end_hist)]
    df_hist = df_hist[df_hist["Experiment"].str.lower().str.contains("historic")]
    
    hist_summaries = []
    
    # 3. Group by Model and ESM
    for (model, esm), group in df_hist.groupby(["Model", "ESM"]):
        g = group.dropna(subset=["BA_area_pred"]).sort_values("year")
        
        # Ensure we have enough data points for a meaningful slope
        if len(g) >= 5:
            slope, intercept, r, p, se = linregress(g["year"], g["BA_area_pred"])
            
            hist_summaries.append({
                "Model": model,
                "ESM": esm,
                "Period": f"{start_hist}-{end_hist}",
                "Hist_Slope_Mha_yr": slope,
                "R_value": r,
                "P_value": p,
                "Mean_BA_Hist": g["BA_area_pred"].mean()
            })
    
    # 4. Create the final DataFrame
    df_hist_slopes = pd.DataFrame(hist_summaries)
    
    # 5. Display the average slope per Model group
    print("Average Historical Slopes by Model Type:")
    print(df_hist_slopes.groupby("Model")["Hist_Slope_Mha_yr"].mean())
    
    # Save to CSV
    df_hist_slopes.to_csv(os.path.join(outputs_dir, "F9_Historical_BA_Slopes.csv"), index=False)
    
    
    return df
    return summary_df
    
    
def F9_cmip6_figure_final(global_df, slopes_df):
    
    ###########################################################################
    # --- CLEAN INPUTS ---
    ###########################################################################

    global_df = global_df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Compute ensemble mean ± std for time series
    ensemble = (
        global_df.groupby(["time", "Model", "Experiment"])
          .agg(
              mean=("BA_area_pred", "mean"),
              std=("BA_area_pred", "std"),
              vmax=("BA_area_pred", "max"),
              vmin=("BA_area_pred", "min")
          )
          .reset_index()
    )

    ensemble["lower_std"] = ensemble["mean"] - ensemble["std"]
    ensemble["upper_std"] = ensemble["mean"] + ensemble["std"]
    ensemble["lower_min"] = ensemble["vmin"]
    ensemble["upper_max"] = ensemble["vmax"]

    ensemble["time_numeric"] = pd.to_datetime(ensemble["time"]).dt.year
    ensemble.to_csv(os.path.join(outputs_dir, "F9_BA_Ensemble.csv"))

    # Clean summary dataframe
    sum_df = slopes_df.dropna(subset=["RCP", "Total_BA_2100", "Slope_2020_2100"]).copy()
    sum_df["RCP"] = sum_df["RCP"].astype(float)

    df_iam = sum_df[sum_df["Model"] == "IAM-FIRE"].copy()
    df_cmip6 = sum_df[sum_df["Model"] == "CMIP6"].copy()
    df_firemip = sum_df[sum_df["Model"] == "FireMIP"].copy()
    # Create new columns for cleaner grouping
    def extract_impact_model(source_name):
        source_name = source_name.lower()
        if 'elm-eca' in source_name:
            return 'ELM-ECA'
        elif 'classic' in source_name:
            return 'CLASSIC'
        elif 'visit' in source_name:
            return 'VISIT'
        else:
            # Fallback: take the first part
            return source_name.split('-')[0].upper()
    
    df_firemip['Impact_Model'] = df_firemip['Source'].apply(extract_impact_model)
    
    # Clean up GCM_Forcing by removing the Impact_Model string and the leading hyphen
    df_firemip['GCM_Forcing'] = df_firemip.apply(
        lambda row: row['Source'].lower().replace(row['Impact_Model'].lower(), '').lstrip('-'), 
        axis=1
    ).str.upper()

    ###########################################################################
    # --- FIGURE SETUP ---
    ###########################################################################

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(3, 3, figsize=(14, 14))

    colors = {
        "historical": "black", "Historic": "black",
        "ssp126": "tab:green", "SSP1-2.6": "tab:green",
        "ssp245": "tab:blue", "SSP2-4.5": "tab:blue",
        "ssp370": "tab:red", "SSP3-6.6": "tab:red",
        "ssp585":  "orange", "SSP5-7.6": "orange"
    }

    ###########################################################################
    # --- (a) IAM-FIRE TEMPORAL TREND ---
    ###########################################################################

    axA = axes[0,0]
    sub_iam = ensemble[ensemble["Model"] == "IAM-FIRE"]

    for exp, subdf in sub_iam.groupby("Experiment"):

        c = colors.get(exp, "gray")

        axA.fill_between(subdf["time_numeric"], subdf["lower_min"], subdf["upper_max"],
                         color=c, alpha=0.2)

        axA.plot(subdf["time_numeric"], subdf["mean"], color=c, linewidth=2, label=exp)

        slope, intercept, *_ = linregress(subdf["time_numeric"], subdf["mean"])
        axA.plot(subdf["time_numeric"],
                 intercept + slope * subdf["time_numeric"],
                 color=c, linestyle="--")
    axA.set_ylim(250, 950)
    axA.set_title("(a) IAM-FIRE – Global Burned Area Trend")
    axA.set_ylabel("Burned Area (Mha)")
    axA.legend()

    ###########################################################################
    # --- (b) CMIP6 TEMPORAL TREND ---
    ###########################################################################

    axB = axes[0,1]
    sub_cmip6 = ensemble[ensemble["Model"] == "CMIP6"]

    for exp, subdf in sub_cmip6.groupby("Experiment"):

        c = colors.get(exp, "gray")

        axB.fill_between(subdf["time_numeric"], subdf["lower_min"], subdf["upper_max"],
                         color=c, alpha=0.2)

        axB.plot(subdf["time_numeric"], subdf["mean"], color=c, linewidth=2, label=exp)

        slope, intercept, *_ = linregress(subdf["time_numeric"], subdf["mean"])
        axB.plot(subdf["time_numeric"],
                 intercept + slope * subdf["time_numeric"],
                 color=c, linestyle="--")
    axB.set_ylim(250, 950)
    axB.set_title("(b) CMIP6 – Global Burned Area Trend")
    axB.set_ylabel("")

    axB.legend()
    
    ###########################################################################
    # --- (c) FireMIP TEMPORAL TREND ---
    ###########################################################################

    axC = axes[0,2]
    sub_firemip = ensemble[ensemble["Model"] == "FireMIP"]

    for exp, subdf in sub_firemip.groupby("Experiment"):

        c = colors.get(exp, "gray")

        axC.fill_between(subdf["time_numeric"], subdf["lower_min"], subdf["upper_max"],
                         color=c, alpha=0.2)

        axC.plot(subdf["time_numeric"], subdf["mean"], color=c, linewidth=2, label=exp)

        slope, intercept, *_ = linregress(subdf["time_numeric"], subdf["mean"])
        axC.plot(subdf["time_numeric"],
                 intercept + slope * subdf["time_numeric"],
                 color=c, linestyle="--")
    axC.set_ylim(250, 950)
    axC.set_title("(c) FireMIP – Global Burned Area Trend")
    axC.set_ylabel("")

    axC.legend()

    ###########################################################################
    # --- HELPER: annotate regression slope ---
    ###########################################################################

    def annotate_slope(ax, x, y, ypos=0.85):
        if len(x) > 1:
            slope, intercept, r, p, stderr = linregress(x, y)
            sign = "+" if slope >= 0 else ""
            ax.text(
                0.05, ypos,
                f"slope = {sign}{slope:.2f} Mha/W m²\n(p = {p:.3f})",
                transform=ax.transAxes,
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.6)
            )

    ###########################################################################
    # --- (d) IAM-FIRE TOTAL BA vs RCP ---
    ###########################################################################

    axD = axes[1,0]

    sns.scatterplot(data=df_iam, x="RCP", y="Total_BA_2100",
                    hue="Source", s=100, edgecolor="black", ax=axD)

    sns.regplot(data=df_iam, x="RCP", y="Total_BA_2100",
                scatter=False, color="black", ax=axD)

    annotate_slope(axD, df_iam["RCP"], df_iam["Total_BA_2100"])

    axD.set_title("(d) IAM-FIRE – Burned Area in 2100 vs RCP")
    axD.set_ylabel("Burned Area (Mha)")
    axD.set_ylim(250, 850)
    axD.legend(title="IAM-FIRE", ncol=2, loc='lower center', fontsize='small')

    ###########################################################################
    # --- (e) CMIP6 TOTAL BA vs RCP ---
    ###########################################################################

    axE = axes[1,1]

    sns.scatterplot(data=df_cmip6, x="RCP", y="Total_BA_2100",
                    hue="Source", s=100, edgecolor="black", ax=axE)

    sns.regplot(data=df_cmip6, x="RCP", y="Total_BA_2100",
                scatter=False, color="black", ax=axE)

    annotate_slope(axE, df_cmip6["RCP"], df_cmip6["Total_BA_2100"])

    axE.set_title("(e) CMIP6 – Burned Area in 2100 vs RCP")
    axE.set_ylabel("")
    axE.set_ylim(250, 850)
    axE.legend(title="CMIP6", ncol=2, loc='lower center', fontsize='small')
    
    ###########################################################################
    # --- (f) FireMIP TOTAL BA vs RCP (Grouped) ---
    ###########################################################################
    axF = axes[1,2]

    # Use 'Impact_Model' for color and 'GCM_Forcing' for marker style (optional)
    sns.scatterplot(data=df_firemip, x="RCP", y="Total_BA_2100",
                    hue="Impact_Model", style="GCM_Forcing", 
                    s=100, edgecolor="black", ax=axF)

    sns.regplot(data=df_firemip, x="RCP", y="Total_BA_2100",
                scatter=False, color="black", ax=axF)

    annotate_slope(axF, df_firemip["RCP"], df_firemip["Total_BA_2100"])

    axF.set_title("(f) FireMIP – Grouped by Impact Model")
    axF.set_ylabel("")
    axF.set_ylim(250, 850)
    
    # Legend is now much smaller as it only shows 3-4 Impact Models
    axF.legend(title="FireMIP", ncol=2, loc='lower center', fontsize='small')

    ###########################################################################
    # --- (g) IAM-FIRE SLOPE BA vs RCP ---
    ###########################################################################

    axG = axes[2,0]

    sns.scatterplot(data=df_iam, x="RCP", y="Slope_2020_2100",
                    hue="Source", s=100, edgecolor="black", ax=axG)

    sns.regplot(data=df_iam, x="RCP", y="Slope_2020_2100",
                scatter=False, color="black", ax=axG)

    annotate_slope(axG, df_iam["RCP"], df_iam["Slope_2020_2100"])

    axG.set_title("(g) IAM-FIRE – BA Slope (2020–2100) vs RCP")
    axG.set_ylabel("Slope of Burned Area (Mha/year)")
    axG.set_ylim(-5, 5)
    axG.legend(title="IAM-FIRE", ncol=2, loc='lower center', fontsize='small')
    
    ###########################################################################
    # --- (h) CMIP6 SLOPE BA vs RCP ---
    ###########################################################################

    axH = axes[2,1]

    sns.scatterplot(data=df_cmip6, x="RCP", y="Slope_2020_2100",
                    hue="Source", s=100, edgecolor="black", ax=axH)

    sns.regplot(data=df_cmip6, x="RCP", y="Slope_2020_2100",
                scatter=False, color="black", ax=axH)

    annotate_slope(axH, df_cmip6["RCP"], df_cmip6["Slope_2020_2100"])

    axH.set_title("(h) CMIP6 – BA Slope (2020–2100) vs RCP")
    axH.set_ylabel("")
    axH.set_ylim(-5, 5)
    axH.legend(title="CMIP6", ncol=2, loc='lower center', fontsize='small')
    
    ###########################################################################
    # --- (i) FireMIP SLOPE BA vs RCP (Grouped) ---
    ###########################################################################
    axI = axes[2,2]

    sns.scatterplot(data=df_firemip, x="RCP", y="Slope_2020_2100",
                    hue="Impact_Model", style="GCM_Forcing", 
                    s=120, edgecolor="black", ax=axI)

    sns.regplot(data=df_firemip, x="RCP", y="Slope_2020_2100",
                scatter=False, color="black", ax=axI)

    annotate_slope(axI, df_firemip["RCP"], df_firemip["Slope_2020_2100"])

    axI.set_title("(i) FireMIP – BA Slope Grouped")
    axI.set_ylabel("")
    axI.set_ylim(-5, 5)
    
    # Place legend in two columns if you still include GCMs, 
    # or one column if just using Impact_Model
    axI.legend(title="FireMIP", ncol=2, loc='lower center', fontsize='small')

    ###########################################################################
    # --- SAVE ---
    ###########################################################################
    plt.show()

    plt.tight_layout()
    out_file = os.path.join(figures_dir, "F9_BA_Compare.png")
    fig.savefig(out_file, dpi=300)

    print("Final MIPs vs. IAMFIRE figure saved to:", out_file)


if __name__ == "__main__":
    
    # Set up the experiments and ESMs name
    experiments = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
    esms = ["CESM2", "CESM2-WACCM", "CMCC-CM2-SR5", "EC-Earth3-CC", "NorESM2-LM"]
    
    # Run the Master CSV function
    print("Creating CMIP6 vs. IAMFIRE figures")
    master_csv(BA_CMIP6, BA_FireMIP, iamfire_results)    
    global_df = pd.read_csv(os.path.join(outputs_dir, "F9_BA_Global_Trends.csv"))
    slopes_df = pd.read_csv(os.path.join(outputs_dir, "F9_BA_Summary_RCP_Slopes.csv"))
    
    # Computes the mean for all numeric columns, grouping by everything except the 'Source'
    mme_df = slopes_df.groupby(["Model", "Experiment", "RCP"]).mean(numeric_only=True).reset_index()
    
    # Run the Figure 8 function
    F9_cmip6_figure_final(global_df, slopes_df)

    
    