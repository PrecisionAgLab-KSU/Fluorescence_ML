
(1) Signal denoising and outlier removal
Folder: Signal_denoising_outlier
Data description: 
ardec0710_excel.xlsx contains sample fluorescenec data (Multiplex3) over several plots.
These are raw signal.

Fluo_waveletTransform_signalDenoise.py performs wavelet transform based signal denoising for each plot individually. 

Data_cleaning_IQR.py performs data ommision (based on FRF_R threshold) and outlier removal using IQR method.

