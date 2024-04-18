# Taiwan Transformer Shaking Alert Model (TT-SAM)

This study has referenced the Transformer Earthquake Alerting Model (TEAM), a deep learning earthquake early warning (EEW) framework. We optimized the model using seismic data from Taiwan to develop the Taiwan Transformer Shaking Alert Model (TT-SAM), and it could rapidly calculate the seismic intensity to provide longer warning time.


## Data Preprocess

![image](data_preprocess/images/workflow.png)

## Model architecture
![image](images/TEAM-Taiwan_model_architecture.png)

## Model Performance

We use 2016 seismic data to evaluate model performance.

Seismic intensity threshold is from Central Weather Administration.

Background color represents model predicted intensity.

### 2016 Meinong Earthquake

![image](images/Meinong_event.gif)

### 2016 Taitung Offshore Earthquake
![image](images/Taitung_offshore_event.gif)

## Reference
Münchmeyer et al.,2021 (https://academic.oup.com/gji/article/225/1/646/6047414)