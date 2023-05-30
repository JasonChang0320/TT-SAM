import pandas as pd
import matplotlib.pyplot as plt

traces=pd.read_csv("2009_2019_target_traces.csv")
events=pd.read_csv("2009_2019_target_catalog.csv")

traces.quality_control.value_counts().plot(kind='pie', autopct='%.1f%%')
labels = traces.quality_control.unique()
plt.legend(labels=labels)

#trace為Y
y_filter=(traces["quality_control"]=="y")

ok_traces=traces[y_filter]



#本來有4級的traces但是壞掉了 需要把其他traces替除
intensity_filter=(ok_traces["intensity"]>=4)
include_intensity_4=ok_traces[intensity_filter]["EQ_ID"].unique().tolist()
ok_traces_filter=(ok_traces["EQ_ID"].isin(include_intensity_4))
ok_traces=ok_traces[ok_traces_filter]

ok_traces.to_csv("2009_2019_ok_traces.csv",index=False)


fig,ax=plt.subplots()
ax.hist(traces["intensity"],bins=16,edgecolor="gray")
ax.hist(ok_traces["intensity"],bins=16,edgecolor="gray")
plt.yscale("log")

#把events也替除
ok_event_filter=(events["EQ_ID"].isin(include_intensity_4))
ok_events=events[ok_event_filter]
ok_events.to_csv("2009_2019_ok_events.csv",index=False)
fig,ax=plt.subplots()
ax.hist(events["magnitude"],bins=28,edgecolor="gray")
ax.hist(ok_events["magnitude"],bins=28,edgecolor="gray")
plt.yscale("log")