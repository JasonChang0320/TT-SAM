import matplotlib.pyplot as plt
import pandas as pd

Afile_path="data/Afile"
origin_catalog=pd.read_csv(f"{Afile_path}/2012-2020 catalog (no 2020_7-9).csv")
catalog=pd.read_csv(f"{Afile_path}/final catalog.csv")

validation_year=2018
fig,ax=plt.subplots()
ax.hist(catalog["magnitude"],bins=30,ec='black',label="train")
year_filter=(catalog["year"]==validation_year)
ax.hist(catalog[year_filter]["magnitude"],bins=30,ec='black',label="validation")
ax.set_yscale("log")
ax.set_xlabel("Magnitude")
ax.set_ylabel("number of events")
ax.set_title(f"2012-2020 TSIMP data: validate on {validation_year}")
ax.legend()

