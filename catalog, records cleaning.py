import matplotlib.pyplot as plt
import pandas as pd

# from obspy.signal.trigger import ar_pick

# from read_tsmip import read_tsmip, trace_pick_plot

Afile_path="data/Afile"
##############clean broken data##############

traces_catalog=pd.read_csv(f"{Afile_path}/1991-2020 traces.csv")
acc_filter=((traces_catalog["pga_z"]==0) | (traces_catalog["pga_ns"]==0) | (traces_catalog["pga_ew"]==0) |
            (traces_catalog["pga_z"]<2.5) | (traces_catalog["pga_ns"]<2.5) | (traces_catalog["pga_ew"]<2.5) |
            (traces_catalog["pga_z"]>1300) | (traces_catalog["pga_ns"]>1300) | (traces_catalog["pga_ew"]>1300))
broken_traces=traces_catalog[acc_filter]
broken_traces.to_csv(f"{Afile_path}/1991-2020 broken traces.csv",index=False)

traces_catalog.drop(traces_catalog[acc_filter].index,inplace=True)
traces_catalog.to_csv(f"{Afile_path}/1991-2020 traces no broken data.csv",index=False)

##############find double event traces##############
catalog=pd.read_csv(f"{Afile_path}/1991-2020 catalog.csv")
traces_ljoin_catalog=pd.merge(catalog[["EQ_ID","year","month","day","hour","minute","second"]],traces_catalog
                            ,on="EQ_ID")

double_traces_catalog = pd.DataFrame()
for year in range(1991,2021):
    for month in range(1,13):
        time_filter=((traces_ljoin_catalog["year"]==year) & (traces_ljoin_catalog["month"]==month))
        tmp_catalog=traces_ljoin_catalog[time_filter]
        file_name_num=tmp_catalog["file_name"].value_counts()
        double_event=file_name_num[file_name_num>1]
        same_filename_filter=(tmp_catalog["file_name"].isin(double_event.index))
        double_traces=tmp_catalog[same_filename_filter]
        double_traces_catalog=pd.concat([double_traces_catalog,double_traces])
double_traces_catalog.to_csv(f"{Afile_path}/1991-2020 double traces.csv",index=False)

#clean trace double event
traces_catalog=pd.read_csv(f"{Afile_path}/1991-2020 traces no broken data.csv")
catalog=pd.read_csv(f"{Afile_path}/1991-2020 catalog.csv")
traces_catalog_merge=pd.merge(catalog[["EQ_ID","year","month","day","hour","minute","second"]],traces_catalog
                            ,on="EQ_ID")

double_event=pd.read_csv(f"{Afile_path}/1991-2020 double traces.csv")

final_traces_catalog=pd.concat([traces_catalog_merge,double_event,double_event]).drop_duplicates(keep=False)
final_traces_catalog.to_csv(f"{Afile_path}/1991-2020 traces (no broken data, double event).csv",index=False)

##############plot double events trace##############

# counts_file_times=double_traces_catalog[["file_name","year","month"]].value_counts()
# error_file={"year":[],"month":[],"file":[],"eq_num":[],"reason":[]}
# for (file_name,year,month),eq_num in zip(counts_file_times.index,counts_file_times):
#     if len(str(month))<2:
#         month="0"+str(month)
#     path=f"data/waveform/{year}/{month}"
#     file_name=file_name.strip()
#     try:
#         trace=read_tsmip(f"{path}/{file_name}.txt")
#         print("read_file ok")

#     except Exception as reason:
#         print(file_name,f"year:{year},month:{month}, {reason}")
#         error_file["year"].append(year)
#         error_file["month"].append(month)
#         error_file["file"].append(file_name)
#         error_file["reason"].append(reason)
#         error_file["eq_num"].append(eq_num)
#         continue
#     sampling_rate=trace[0].stats.sampling_rate
#     try:
#         p_pick,s_pick=ar_pick(trace[0],trace[1],trace[2],
#                             samp_rate=sampling_rate,
#                             f1=1, #Frequency of the lower bandpass window
#                             f2=20, #Frequency of the upper bandpass window
#                             lta_p=1, #Length of LTA for the P arrival in seconds
#                             sta_p=0.1, #Length of STA for the P arrival in seconds
#                             lta_s=4.0, #Length of LTA for the S arrival in seconds
#                             sta_s=1.0, #Length of STA for the P arrival in seconds
#                             m_p=2, #Number of AR coefficients for the P arrival
#                             m_s=8, #Number of AR coefficients for the S arrival
#                             l_p=0.1,
#                             l_s=0.2,
#                             s_pick=True)
#     except Exception as reason:
#         print(file_name,f"year:{year},month:{month}, {reason}")
#         error_file["year"].append(year)
#         error_file["month"].append(month)
#         error_file["file"].append(file_name)
#         error_file["reason"].append(reason)
#         error_file["eq_num"].append(eq_num)
#         continue
#     fig,ax=plt.subplots(3,1)
#     ax[0].set_title(f"station: {trace[0].stats.station}, start time: {trace[0].stats.starttime}")
#     ax[1].set_title(f"eqrthquake_number: {eq_num}")
#     for component in range(len(trace)):
#         ax[component].plot(trace[component],"k")
#         ymin,ymax=ax[component].get_ylim()
#         ax[component].vlines(p_pick*sampling_rate,ymin,ymax,"r",label="P pick")
#         ax[component].vlines(s_pick*sampling_rate,ymin,ymax,"g",label="S pick")
#     ax[0].legend()
#     fig.tight_layout()
#     output_path="data/waveform/double event picking"
#     fig.savefig(f"{output_path}/{file_name}.png")
#     plt.close()

# error_file_df=pd.DataFrame(error_file)
# # error_file_df.to_csv(f"{Afile_path}/double event error.csv",index=False)

# #pick again error file
# error_file_df=pd.read_csv(f"{Afile_path}/double event error_new.csv")
# cant_picking_filter=((error_file_df["year"]!=2020) & (error_file_df["month"]!="07") & (error_file_df["month"]!="08") & (error_file_df["month"]!="09"))
# cant_picking_file=error_file_df[cant_picking_filter].reset_index(drop=True)

# for i in range(len(cant_picking_file)):
#     year=cant_picking_file["year"][i]
#     month=cant_picking_file["month"][i]
#     if len(str(month))<2:
#         month="0"+str(month)
#     file_name=cant_picking_file["file"][i]
#     eq_num=cant_picking_file["eq_num"][i]

#     path=f"data/waveform/{year}/{month}"

#     trace=read_tsmip(f"{path}/{file_name}.txt")
#     trace_pick_plot(trace,file_name,eq_num=eq_num,output_path="data/waveform/double event picking")

