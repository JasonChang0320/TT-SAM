# Data Preprocess
There are four components you need to prepare first
1. **Event catalog**
2. **Traces catalog**
3. **Seismic waveform**
4. **Vs30 table for TSMIP station**

Please follow the step:

`1_afile_to_catalog.py`

`2_catalog_records_cleaning.py`

...

`13_cut_waveform_to_hdf5.py`

At each step you will get an `.csv` output, it will be input in next step script.

After finished these steps, you will get an `.hdf5` file include all of the information you gave before.

## Preprocess Workflow
![image](images/workflow.png)

### Others

We used 3D velocity model to shift P-wave arrival, follow: Huang et al., 2014

Paper link: 

https://www.sciencedirect.com/science/article/pii/S0012821X14000995

All of the works are in `tracer_demo`.

Vs30 dataset was contributed by Kuo et al., 2012 and Lee et al., 2008

Paper link:

https://www.sciencedirect.com/science/article/pii/S0013795212000397

http://tao.cgu.org.tw/index.php/articles/archive/geophysics/item/799-2008196671pt
