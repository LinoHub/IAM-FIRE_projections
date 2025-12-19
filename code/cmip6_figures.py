# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 12:08:21 2025

@author: theo.rouhette

conda activate xesmf
cd C:\\GCAM\\Theo\\GCAM_7.2_Impacts\\python\climate_integration_metarepo\\code\\python
python cmip6_data.py


"""

# Packages Required to Browse and Analyze CMIP6 Data
import numpy as np
import pandas as pd
import xarray as xr
import os
import dask.array as da
import matplotlib.pyplot as plt
from intake_esgf import ESGFCatalog
import matplotlib.pyplot as plt
import seaborn as sns
import pymannkendall as mk
from scipy.stats import linregress

# CONSTANTS
scenarios = ["SSP1-2p6o", "SSP2-4p5", "SSP3-6p6", "SSP5-7p6"]
esms = ["MPI-ESM1-2-LR", "CanESM5"]   

# PATHS
IF_PATH = "C:/GCAM/Theo/IAM-FIRE/zenodo"
inputs_dir = os.path.join(IF_PATH, f"inputs/")
outputs_dir = os.path.join(IF_PATH, f"outputs/")
figures_dir = os.path.join(IF_PATH, f"figures/")

# INPUTS
iamfire_results = pd.read_csv(os.path.join(outputs_dir, "BA_CE_Prediction_AllScen.csv"))
BA_CMIP6 = pd.read_csv(os.path.join(inputs_dir, "BA_CMIP6.csv"))

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



###############################################################################
# BURNED AREA CMIP6
###############################################################################


def normalize_burnt_fraction(ba, esm_name=None):
    """Return burnt fraction in [0, 1], fixing % units and capping outliers.
    Logs how many values were clipped. Optionally applies extra rules for a given ESM.
    """    
    
    # # DEBUG
    # da = ba
    
    units = (ba.attrs.get("units") or "").lower()

    # # 1) Normalize units
    # if "%" in units or "percent" in units:
    #     da = da / 100.0
    # else:
    #     # Heuristic fallback if metadata is unhelpful
    #     # If max > 1, it's almost surely percent
    #     da_max = da.max(skipna=True)
    #     try:
    #         # works for dask and numpy
    #         da_max_val = float(da_max.compute() if hasattr(da_max, "compute") else da_max)
    #     except Exception:
    #         da_max_val = float(da_max)

    #     if da_max_val > 1.0:
    #         da = da / 100.0

    # 2) Guardrails: Assign 1 for all fraction > 1
    # (do this BEFORE converting to area and BEFORE resampling)
    gt1_count = (ba > 1).sum()

    try:
        gt1_count_val = int(gt1_count.compute() if hasattr(gt1_count, "compute") else gt1_count)
    except Exception:
        gt1_count_val = int(gt1_count)

    if gt1_count_val:
        print(f"   Cleaning burntFractionAll: >1={gt1_count_val} (capping).")

    ba = xr.where(ba > 1, 1, ba)

    # # 3) Optional: model-specific tweaks
    # if esm_name and esm_name.startswith("CNRM"):
    #     # If CNRM sometimes peaks ~1.22, the clip above handles it.
    #     # You could also emit a dedicated notice:
    #     print("   Note: Applied >1 cap for CNRM; check variable metadata (cell_methods) if available.")

    return ba

def cmip6_compile(experiments, esms):

    # # DEBUG
    # experiments = ["ssp126"]
    # exp = "ssp126"
    # esm = "CMCC-CM2-SR5"
    # esms = ["CESM2"]
    
    cat = ESGFCatalog()
    print(cat)  # <-- nothing to see here yet
    cat.variable_info("air temperature surface")
    cat.variable_info("burntFractionAll")
    from intake_esgf.base import NoSearchResults
    
    cmip6_df = []

    for exp in experiments: 
        experiment_ba = []   
        experiment_ba_df = []
        for esm in esms: 
            
            print(f"Processing {esm} – {exp}")
            
            try:
                cat.search(
                    experiment_id=exp,
                    source_id=esm,
                    frequency="mon",
                    variable_id="burntFractionAll",
                ) 
            except NoSearchResults: 
                print(f"⚠️ No results for {esm} – {exp}, skipping...")
                continue
    
            # only gets here if search succeeds
            cat_subset = cat.remove_ensembles()
    
            dsd = cat_subset.to_dataset_dict(ignore_facets='table_id', add_measures=False)
            
            # There can be multiple matches (e.g. Amon, Lmon etc.), take the first
            for name, ds in dsd.items():
                print(f"  → dataset: {name}")
    
                # --- 3. Extract variable
                if "burntFractionAll" not in ds.variables:
                    print("    burntFractionAll not found in dataset variables")
                    continue
    
                ba = ds["burntFractionAll"]
                # ds_file = os.path.join(OUTPUT_PATH, f"BA_DS_{esm}_{exp}.nc")
                # print(f"    Saving to {ds_file}")
                # ds.to_netcdf(ds_file)
                
                # Normalize the fraction 
                if esm == "CNRM-ESM2-1":
                    ba = normalize_burnt_fraction(ba, esm_name=esm)
                else:
                    pass
                
                ba = ba.to_dataset()
                    
                # --- 4. Transform the dataframe (PENDING)
                print(ba.coords)
                lat = ba["lat"].values
                lon = ba["lon"].values 
    
                if lon.max() > 180:
                    ba.coords['lon'] = (ba['lon'] + 180) % 360 - 180
                    ba = ba.sortby(ba.lon)
                else:
                    pass
                
                dlat = abs(lat[1] - lat[0])
                dlon = abs(lon[1] - lon[0])
                ba["grid_area"] = xr.apply_ufunc(grid_cell_area, ba["lat"], kwargs={"dlon": dlon, "dlat": dlat},  vectorize=True)
                
                val = ba["burntFractionAll"].max()
                import dask.array as da
                # ba["burntFractionAll"].plot().hist()
                if isinstance(val.data, da.Array):
                    result = val.compute().item()
                else:
                    result = val.item()
                
                print("Max burnt fraction:", result)            
                
                
                
                
                if val > 2:
                    ba["burntFractionAll"] = ba["burntFractionAll"] / 100
                else:
                    pass
                
                # ba["burntFractionAll"].mean(dim="time").plot()
    
                ba["BA_area_pred"] = ba["grid_area"] * ba["burntFractionAll"]
                ba["BA_area_pred"] = ba["BA_area_pred"] / 10000 # From km2 to Mha
                ba = ba.resample(time="YE").sum()
                # ba["burntArea"].sum(dim=["lat", "lon"]).plot()
                # ba["burntArea"].sum(dim="time").plot()
                
                # --- 4b. Trim time to 2100 max (avoid overflow when converting to datetime64)
                # ba = ba.sel(time=[t for t in ba.time.values if t.year <= 2100])
                ba = ba.sel(time=ba.time.dt.year <= 2100) # Deals with numpy.datetime64 as well as Python and pandas (in theory)
                
                # --- 5. Convert to common time calendar 
                try:
                    ba = ba.convert_calendar("proleptic_gregorian", use_cftime=False)
                except Exception as e:
                    print("Calendar conversion failed, forcing to_datetimeindex:", e)
                    ba["time"] = ba.indexes["time"].to_datetimeindex()
        
                if "type" in ba.coords:
                    ba = ba.drop_vars("type")
        
                # --- 6. Save as NetCDF and append to experiment dict
                ba["Experiment"] = f"{exp}"                
                ba["ESM"] = f"{esm}"
                ba["Source"] = f"{exp} (CMIP6) - {esm}"
                experiment_ba.append(ba)
                
                # --- 7. Save as CSV and append to experiment dict
                ba_df = ba.to_dataframe()
                ba_df = ba_df.loc[ba_df.index.get_level_values("time") >= "2000-01-01"]
                ba_df = ba_df.groupby(["time", "Experiment", "ESM", "Source"]).sum()
                experiment_ba_df.append(ba_df)
                
                out_file = os.path.join(outputs_dir, f"BA_{esm}_{exp}_2210.nc")
                print(f"    Saving to {out_file}")
                ba.to_netcdf(out_file)
    
        # master_nc = xr.concat(experiment_ba, dim = "Source", compat="override") 
        # master_nc.to_netcdf(os.path.join(OUTPUT_PATH,f"BA_ALL_{exp}.nc"))
        master_df = pd.concat(experiment_ba_df) 
        master_df.to_csv(os.path.join(outputs_dir, f"BA_ALL_{exp}_2210.csv"))
        print(f"    Saving final file for {exp}")
        cmip6_df.append(master_df)
        
    BA_CMIP6 = pd.concat(cmip6_df)
    BA_CMIP6.to_csv(os.path.join(inputs_dir, "BA_CMIP6.csv"))
    print(f"    Saving final file for {experiments} and {esms}")

def cmip6_master_csv(BA_CMIP6, iamfire_results):


    BA_CMIP6["Model"] = "CMIP6"
    BA_CMIP6['time'] = pd.to_datetime(BA_CMIP6['time'])
    BA_CMIP6["year"] = BA_CMIP6["time"].dt.year
    BA_CMIP6 = BA_CMIP6[(BA_CMIP6["ESM"] != "EC-Earth3-CC")]
    
    
    # BA_CMIP6 = BA_CMIP6.rename(columns={"burntArea": "BA_area_pred"})
    
    # --- 1. Split Source into Scenario & ESM ---
    iamfire_results[["Experiment", "ESM"]] = iamfire_results["Source"].str.split(" - ", n=1, expand=True)
    iamfire_results["Model"] = "IAM-FIRE"
    iamfire_results['time'] = pd.to_datetime(iamfire_results['time'])
    iamfire_results["year"] = iamfire_results["time"].dt.year
    iamfire_results = iamfire_results.loc[iamfire_results["Grid_area"] > 0]    
    
    ba_dfs = [BA_CMIP6, iamfire_results] 
    df = pd.concat(ba_dfs)
    # df = figure_df.rename(columns={"Source": "ESM"})
    # figure_df = pd.concat([df.reset_index() for df in ba_dfs], ignore_index=True)

    # Handles cases like "ssp126", "ssp245", "ssp370", "ssp3-6p6", "ssp5-7p6", "ssp585"
    def extract_rcp(exp):
        exp = exp.lower()
        if ("126" in exp) or ("2p6" in exp):
            return 2.6
        elif ("245" in exp) or ("4p5" in exp):
            return 4.5
        elif "370" in exp:
            return 7.0
        elif "3-6p6" in exp:
            return 6.6
        elif "5-7p6" in exp:
            return 7.6
        elif "585" in exp:
            return 8.5
        else:
            return np.nan

    df["RCP"] = df["Experiment"].apply(extract_rcp)

    # DF will now serve for figure 1 main time series panel of F7 
    df.to_csv(os.path.join(outputs_dir, "F8_BA_Global_Trends.csv"))
    
    # SUmmaries will be a CSV used for the scatterplots of F7 
    # --- Keep only years 2020–2100 ---
    df_sp = df[(df["year"] >= 2020) & (df["year"] <= 2100)]

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

    out_file = os.path.join(outputs_dir, "F8_BA_Summary_RCP_Slopes.csv")
    summary_df.to_csv(out_file, index=False)
        
    print(f"Scenario summary saved to {out_file}")
    return df
    return summary_df
    
def F8_cmip6_figure_final(global_df, slopes_df):
    
    ###########################################################################
    # --- CLEAN INPUTS ---
    ###########################################################################

    df = global_df.drop(columns=["Unnamed: 0"], errors="ignore")

    # Compute ensemble mean ± std for time series
    ensemble = (
        df.groupby(["time", "Model", "Experiment"])
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

    # Clean summary dataframe
    sum_df = slopes_df.dropna(subset=["RCP", "Total_BA_2100", "Slope_2020_2100"]).copy()
    sum_df["RCP"] = sum_df["RCP"].astype(float)

    df_iam = sum_df[sum_df["Model"] == "IAM-FIRE"].copy()
    df_cmip6 = sum_df[sum_df["Model"] == "CMIP6"].copy()

    ###########################################################################
    # --- FIGURE SETUP ---
    ###########################################################################

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    colors = {
        "historical": "black", "Historic": "black",
        "ssp126": "tab:blue", "SSP1-2p6o": "tab:blue",
        "ssp245": "tab:orange", "SSP2-4p5": "tab:orange",
        "ssp370": "tab:red", "SSP3-6p6": "tab:red",
        "ssp585":  "olive", "SSP5-7p6": "olive"
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

    axB.set_title("(b) CMIP6 – Global Burned Area Trend")
    axB.set_ylabel("")

    axB.legend()

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
    # --- (c) IAM-FIRE TOTAL BA vs RCP ---
    ###########################################################################

    axC = axes[1,0]

    sns.scatterplot(data=df_iam, x="RCP", y="Total_BA_2100",
                    hue="Source", s=100, edgecolor="black", ax=axC)

    sns.regplot(data=df_iam, x="RCP", y="Total_BA_2100",
                scatter=False, color="black", ax=axC)

    annotate_slope(axC, df_iam["RCP"], df_iam["Total_BA_2100"])

    axC.set_title("(c) IAM-FIRE – Burned Area in 2100 vs RCP")
    axC.set_ylabel("Burned Area (Mha)")
    axC.set_ylim(250, 850)
    axC.legend(title="IAM-FIRE")

    ###########################################################################
    # --- (d) CMIP6 TOTAL BA vs RCP ---
    ###########################################################################

    axD = axes[1,1]

    sns.scatterplot(data=df_cmip6, x="RCP", y="Total_BA_2100",
                    hue="Source", s=100, edgecolor="black", ax=axD)

    sns.regplot(data=df_cmip6, x="RCP", y="Total_BA_2100",
                scatter=False, color="black", ax=axD)

    annotate_slope(axD, df_cmip6["RCP"], df_cmip6["Total_BA_2100"])

    axD.set_title("(d) CMIP6 – Burned Area in 2100 vs RCP")
    axD.set_ylabel("")
    axD.set_ylim(250, 850)
    axD.legend(title="CMIP6")

    ###########################################################################
    # --- SAVE ---
    ###########################################################################

    plt.tight_layout()
    out_file = os.path.join(figures_dir, "F8_BA_Compare_CMIP6_4panel.png")
    fig.savefig(out_file, dpi=300)
    plt.show()

    print("Final CMIP6 vs. IAMFIRE figure saved to:", out_file)


if __name__ == "__main__":
    
    # Set up the experiments and ESMs name
    experiments = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
    esms = ["CESM2", "CESM2-WACCM", "CMCC-CM2-SR5", "EC-Earth3-CC", "NorESM2-LM"]
    
    # Run the CMIP6 Compile function
    # print("Compiling CMIP6 data")
    # cmip6_compile(experiments, esms)
    
    # Run the Master CSV function
    print("Creating CMIP6 vs. IAMFIRE figures")
    cmip6_master_csv(BA_CMIP6, iamfire_results)    
    global_df = pd.read_csv(os.path.join(outputs_dir, "F8_BA_Global_Trends.csv"))
    slopes_df = pd.read_csv(os.path.join(outputs_dir, "F8_BA_Summary_RCP_Slopes.csv"))
    
    # Run the Figure 8 function
    F8_cmip6_figure_final(global_df, slopes_df)

    
    