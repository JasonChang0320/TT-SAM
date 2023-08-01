import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

sta_path = "data/station_information"
prediction_2016 = pd.read_csv(
    "predict/acc predict pga 1999_2019/model 2 5 sec prediction.csv"
)
station_info = pd.read_csv(f"{sta_path}/TSMIPstations_new.csv")
# residual in meinong earthquake
meinong_earthquake = prediction_2016.query("EQ_ID ==24784.0")
src_crs = ccrs.PlateCarree()
fig, ax_map = plt.subplots(subplot_kw={"projection": src_crs}, figsize=(7, 7))
ax_map.coastlines("10m")
scatter = ax_map.scatter(
    meinong_earthquake["longitude"],
    meinong_earthquake["latitude"],
    edgecolors="k",
    linewidth=1,
    marker="o",
    s=15,
    zorder=3,
    c=meinong_earthquake["predict"] - meinong_earthquake["answer"],
    cmap="seismic",
    alpha=0.5
)
cb = plt.colorbar(scatter)
cb.set_label("predict-answer log10(PGA (m/s^2))")

ax_map.set_title("5 sec 2016 Meinong earthquake residual in prediction")
# fig.savefig(
#     "./predict/acc predict pga 1999_2019/model 2 meinong intensity map/5 sec 2016 Meinong earthquake residual in prediction",
#     dpi=300,
# )
