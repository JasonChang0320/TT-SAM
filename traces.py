from read_tsmip import read_tsmip
from obspy.signal.trigger import classic_sta_lta, plot_trigger,ar_pick
import matplotlib.pyplot as plt
import scipy.signal as ss


year=2012
month="01"
file_name="01801600.CVA.txt" #normal case
file_name="D3700402.SMT.txt" #double event example big bring small
file_name="11801000.CVA.txt" #double event example small bring big
path=f"data/waveform/{year}/{month}"

trace=read_tsmip(f"{path}/{file_name}")

#resample 200Hz to 100Hz
trace.resample(100,window='hann')
sampling_rate=trace[0].stats.sampling_rate

# #classic sta_lta
# cft = classic_sta_lta(trace[0], int(5 * sampling_rate), int(10 * sampling_rate))
# plot_trigger(trace[0], cft, 1.5, 0.5)

#AR-AIC method
#output is second!!
p_pick,s_pick=ar_pick(trace[0],trace[1],trace[2],
                        samp_rate=sampling_rate,
                        f1=1, #Frequency of the lower bandpass window
                        f2=20, #Frequency of the upper bandpass window
                        lta_p=1, #Length of LTA for the P arrival in seconds
                        sta_p=0.1, #Length of STA for the P arrival in seconds
                        lta_s=4.0, #Length of LTA for the S arrival in seconds
                        sta_s=1.0, #Length of STA for the P arrival in seconds
                        m_p=2, #Number of AR coefficients for the P arrival
                        m_s=8, #Number of AR coefficients for the S arrival
                        l_p=0.1,
                        l_s=0.2,
                        s_pick=True)
#resample 200Hz to 100Hz
trace.resample(100,window='hann')

fig,ax=plt.subplots(3,1)
ax[0].set_title(f"station: {trace[0].stats.station}, start time: {trace[0].stats.starttime}")
for component in range(len(trace)):
    ax[component].plot(trace[component],"k")
    ymin,ymax=ax[component].get_ylim()
    ax[component].vlines(p_pick*sampling_rate,ymin,ymax,"r")
    ax[component].vlines(s_pick*sampling_rate,ymin,ymax,"g")

