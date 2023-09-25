
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 10:48:49 2021

@author: Dipankar
"""
# ----------------------------------------------------------------------------
  # Copyright (C) 2023 by PrecisionAG, Agronomy, KSU
 
  # This program is free software; you can redistribute it and/or modify it
  # under the terms of the GNU General Public License as published by the Free
  # Software Foundation; either version 3 of the License, or (at your option)
  # any later version.
  # This program is distributed in the hope that it will be useful, but WITHOUT
  # ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  # FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
  # more details.
 
  # You should have received a copy of the GNU General Public License along
  # with this program; if not, see http://www.gnu.org/licenses/
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.ensemble import RandomForestRegressor



########################

Y1 = pd.read_excel('Ardec12+Iliff12_Regression_Data.xlsx', sheet_name='trainv6', na_values = ['no info', '.','None','#VALUE!', '#DIV/0!'],skiprows=[0],header=None)
Ym=Y1.values
Ym0=np.float64(Ym[:,11])#N%
Ym1=np.float64(Ym[:,12])#Nuptake
Ym2=np.float64(Ym[:,13])#Biomass
#Y=np.column_stack((Ym0,Ym2))
Y=Ym0

## Predictors--WT denoised
NBI_R = np.float64(Ym[:,3])
NBI_G = np.float64(Ym[:,4])
NBI_B = np.float64(Ym[:,5])
NBI1 = np.float64(Ym[:,6])
CHL = np.float64(Ym[:,7])
CHL1 = np.float64(Ym[:,8])
FLAV = np.float64(Ym[:,9])


# # Predictors--raw
# NBI_R = np.float64(Ym[:,12])
# NBI_G = np.float64(Ym[:,13])
# NBI_B = np.float64(Ym[:,14])
# NBI1 = np.float64(Ym[:,15])
# CHL = np.float64(Ym[:,16])
# CHL1 = np.float64(Ym[:,17])
# FLAV = np.float64(Ym[:,18])


X=np.column_stack((NBI_R, NBI_G, NBI_B, NBI1, CHL, CHL1, FLAV))


###################MSVR########################################################################


gpr = RandomForestRegressor(n_estimators=500, criterion='mse', max_depth=None,bootstrap=True,
                                                         min_impurity_decrease=0.005,
                                                          min_samples_leaf=4,random_state=10)


gpr.fit(X,Y)


y_out1 = gpr.predict(X)

######################################3
##PAI estimation and error
#rmse estimation
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_value1 = rmse(np.array(Ym0), np.array(y_out1))
#Correlation coefficient 
corrr_value=np.corrcoef(np.array(Ym0), np.array(y_out1))
rr_value1= corrr_value[0,1]
mae_value1 = mean_absolute_error(Ym0,y_out1)


########################################################################################################
###Load validation data
#replacing 'no info' and '.' i.e. blank space or 'None' string with NaN
cornval = pd.read_excel('Ardec12+Iliff12_Regression_Data.xlsx', sheet_name='testv6', na_values = ['no info', '.','None','#VALUE!', '#DIV/0!'],skiprows=[0],header=None)
vald=cornval.dropna(subset=[1])                                                                                                                            
valdm=vald.values

valN=np.float64(valdm[:,11])#N%
valNup=np.float64(valdm[:,12])#Nuptake
valbiom=np.float64(valdm[:,13])#Biomass
valY = valN



## Predictors--WT denoised
vNBI_R = np.float64(valdm[:,3])
vNBI_G = np.float64(valdm[:,4])
vNBI_B = np.float64(valdm[:,5])
vNBI1 = np.float64(valdm[:,6])
vCHL = np.float64(valdm[:,7])
vCHL1 = np.float64(valdm[:,8])
vFLAV = np.float64(valdm[:,9])


# # Predictors--raw
# vNBI_R = np.float64(valdm[:,12])
# vNBI_G = np.float64(valdm[:,13])
# vNBI_B = np.float64(valdm[:,14])
# vNBI1 = np.float64(valdm[:,15])
# vCHL = np.float64(valdm[:,16])
# vCHL1 = np.float64(valdm[:,17])
# vFLAV = np.float64(valdm[:,18])


valX=np.column_stack((vNBI_R, vNBI_G, vNBI_B, vNBI1, vCHL, vCHL1, vFLAV))


# Predictfor validation data
y_out = gpr.predict(valX);

######################################3
##PAI estimation and error
#rmse estimation
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmse_value2 = rmse(np.array(valN), np.array(y_out))
#Correlation coefficient 
corrr_value=np.corrcoef(np.array(valN), np.array(y_out))
rr_value2= corrr_value[0,1]
mae_value2 = mean_absolute_error(valN,y_out)
# #Plotting
# plt.plot(valN,y_out, 'go')
# #plt.xlim([0, 6])
# #plt.ylim([0, 6])
# plt.xlabel("Observed N content ($\%$)")
# plt.ylabel("Estimated N content ($\%$)")
# plt.plot([2, 5], [2, 5], 'k:')
# plt.annotate('r = %.2f'%rr_value2, xy=(2.05, 4.9))#round off upto 3decimals
# plt.annotate('RMSE = %.2f'%rmse_value2, xy=(2.05, 4.6))
# plt.annotate('MAE = %.2f'%mae_value2, xy=(2.05, 4.3))
# matplotlib.rcParams.update({'font.size': 16})
# plt.yticks(np.arange(2, 5.2, .5))
# plt.xticks(np.arange(2, 5.2, .5))
# plt.gca().set_aspect('equal', adjustable='box')
# plt.savefig('NTest_scatter.png',bbox_inches="tight",dpi=450)
# plt.show()



#------------------------------------------------------------------------------
######################################
#Train data plotting
matplotlib.rcParams.update({'font.size': 14})
plt.plot(Ym0,y_out1, 'ro', markersize=5, markerfacecolor='None', markeredgecolor='r', label = "Train")
plt.plot(valN,y_out, 'gd', label = "Test")
plt.xlabel("Observed N content ($\%$)")
plt.ylabel("Estimated N content ($\%$)")
y_ticks = np.arange(1.5, 5, 0.5)
plt.yticks(y_ticks)
x_ticks = np.arange(1.5, 5, 0.5)
plt.xticks(x_ticks)
plt.plot([1.5, 4.5], [1.5, 4.5], 'k:', label = "1:1 line")
plt.gca().set_aspect('equal', adjustable='box')
matplotlib.rcParams.update({'font.size': 13})
plt.annotate('r = %.2f (Train)/'%rr_value1, xy=(1.51, 4.5))#round off upto 2decimals
plt.annotate('%.2f (Test)'%rr_value2, xy=(3.15, 4.5))#round off upto 2decimals
plt.annotate('RMSE = %.2f (Train)/'%rmse_value1, xy=(1.51, 4.3))
plt.annotate('%.2f (Test)'%rmse_value2, xy=(2.35, 4.1))
plt.annotate('MAE = %.2f (Train)/'%mae_value1, xy=(1.51, 3.9))
plt.annotate('%.2f (Test)'%mae_value2, xy=(2.22, 3.7))
plt.legend(loc='lower right')
plt.savefig('Regression_N_V6_RFR.png',bbox_inches="tight",dpi=450)
plt.show()
