import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix

path="./predict/random sec updated dataset and new data generator/ok model prediction"
mask_after_sec=10
trigger_station_threshold=1
data=pd.read_csv(f"{path}/model2 7 9 {mask_after_sec} sec {trigger_station_threshold} triggered station prediction.csv")


predict_pga=(data["predict"])
real_pga=data["answer"]

PGA_threshold=np.log10(9.8*np.array([0.01,0.02,0.025,0.05,0.08,0.1])) # g*9.8 = m/s^2
performance_score={"pga_threshold (g)":[],"confusion matrix":[],"accuracy":[],
                   "precision":[],"recall":[],"f1_score":[]}
for pga_threshold in PGA_threshold:
    predict_logic=np.where(predict_pga>pga_threshold,1,0)
    real_logic=np.where(real_pga>pga_threshold,1,0)

    matrix=confusion_matrix(real_logic, predict_logic,labels=[1,0])
    accuracy=np.sum(np.diag(matrix))/np.sum(matrix) # (TP+TN)/all
    precision=matrix[0][0]/np.sum(matrix,axis=0)[0] # TP/(TP+FP)
    recall=matrix[0][0]/np.sum(matrix,axis=1)[0] # TP/(TP+FP)
    F1_score=2/((1/precision)+(1/recall))
    performance_score["pga_threshold (g)"].append(np.round((10**pga_threshold)/9.8,3)) # m/s^2 / 9.8 = g
    performance_score["confusion matrix"].append(matrix)
    performance_score["accuracy"].append(accuracy)
    performance_score["precision"].append(precision)
    performance_score["recall"].append(recall)
    performance_score["f1_score"].append(F1_score)
    sn.set(rc={"figure.figsize":(8, 5)},font_scale=1.2) # for label size
    fig,ax = plt.subplots()
    sn.heatmap(matrix,ax=ax,
                xticklabels=["Predict True","Predict False"],\
                yticklabels=["Actual True","Actual False"],\
                fmt="g",\
                annot=True,\
                annot_kws={"size": 16},\
                cmap="Reds") # font size    
    ax.set_title(f"EEW {mask_after_sec} sec confusion matrix, PGA threshold: {np.round((10**pga_threshold)/9.8,3)} (g)")
    fig.savefig(f"{path}/confusion matrix/{mask_after_sec} sec pga_threshold {np.round((10**pga_threshold)/9.8,3)} g.png")
    predict_table=pd.DataFrame(performance_score)
    predict_table.to_csv(f"{path}/confusion matrix/{mask_after_sec} sec confustion matrix table.csv",index=False)

#pga threshold performance
axis_fontsize=30
Fig,axes=plt.subplots(1,3,figsize=(14,5))
for i,column in enumerate(["precision","recall","f1_score"]):
    x_axis=[str(label) for label in performance_score["pga_threshold (g)"]]
    axes[i].bar(x_axis,performance_score[f"{column}"])
    axes[i].set_title(f"{column}",fontsize=axis_fontsize)
    axes[i].tick_params(axis='both', which='major', labelsize=axis_fontsize-15)
    axes[i].set_ylim(0,1)
axes[1].set_xlabel("PGA threshold (g)",fontsize=axis_fontsize-12)
Fig.savefig(f"{path}/confusion matrix/{mask_after_sec} sec F1 score.png")

#pga threshold by time plot
sec3_table=pd.read_csv(f"{path}/confusion matrix/3 sec confustion matrix table.csv")
sec5_table=pd.read_csv(f"{path}/confusion matrix/5 sec confustion matrix table.csv")
sec7_table=pd.read_csv(f"{path}/confusion matrix/7 sec confustion matrix table.csv")
sec10_table=pd.read_csv(f"{path}/confusion matrix/10 sec confustion matrix table.csv")

fig,ax=plt.subplots(2,2,figsize=(14,14))
PGA_threshold=[0.01,0.02,0.025,0.08]
plot_index=[[0,0],[0,1],[1,0],[1,1]]
for index,pga_threshold in zip(plot_index,PGA_threshold):
    Accuracy=[];Precision=[];Recall=[];F1_score=[]
    for table in [sec3_table,sec5_table,sec7_table,sec10_table]:
        tmp_table=table[table["pga_threshold (g)"]==pga_threshold]
        Accuracy.append(tmp_table["accuracy"].values[0])
        Precision.append(tmp_table["precision"].values[0])
        Recall.append(tmp_table["recall"].values[0])
        F1_score.append(tmp_table["f1_score"].values[0])
    ax[index[0],index[1]].plot(Accuracy,marker="o",label="accuracy")
    ax[index[0],index[1]].plot(Precision,marker="o",label="precision")
    ax[index[0],index[1]].plot(Recall,marker="o",label="recall")
    ax[index[0],index[1]].plot(F1_score,marker="o",label="F1 score")
    ax[index[0],index[1]].set_title(f"PGA threshold: {pga_threshold} (g)")
    ax[index[0],index[1]].set_xticks([0,1,2,3])
    ax[index[0],index[1]].set_xticklabels(["3","5","7","10"])
    ax[index[0],index[1]].set_xlabel("After first triggered station (sec)")
    ax[index[0],index[1]].set_ylim(0.2,1)
    ax[index[0],index[1]].legend()
fig.savefig(f"{path}/confusion matrix/different pga threshold performance by time.png")

