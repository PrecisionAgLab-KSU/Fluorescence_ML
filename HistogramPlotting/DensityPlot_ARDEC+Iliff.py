# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:27:06 2021

@author: dmandal
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

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
#plt.rcParams.update({'figure.figsize':(10,5), 'figure.dpi':300})
import matplotlib
matplotlib.rcParams.update({'font.size': 18})

# # Import data
# df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/diamonds.csv')
# x1 = df.loc[df.cut=='Ideal', 'depth']
# x2 = df.loc[df.cut=='Fair', 'depth']
# x3 = df.loc[df.cut=='Good', 'depth']

# # Plot
# kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})

# plt.figure(figsize=(10,7), dpi= 80)
# sns.distplot(x1, color="dodgerblue", label="Compact", **kwargs)
# sns.distplot(x2, color="orange", label="SUV", **kwargs)
# sns.distplot(x3, color="deeppink", label="minivan", **kwargs)
# plt.xlim(50,75)
# plt.legend();


Y1 = pd.read_excel('Ardec12+Iliff12_Regression_Data.xlsx', sheet_name='trainv6', na_values = ['no info', '.','None','#VALUE!', '#DIV/0!'],skiprows=[0],header=None)
Ym=Y1.values
Ym0=np.float64(Ym[:,11])#N%
Ym1=np.float64(Ym[:,12])#Nuptake
Ym2=np.float64(Ym[:,13])#Biomass
#Y=np.column_stack((Ym0,Ym2))

##--------------------------------------------------
Y11 = pd.read_excel('Ardec12+Iliff12_Regression_Data.xlsx', sheet_name='testv6', na_values = ['no info', '.','None','#VALUE!', '#DIV/0!'],skiprows=[0],header=None)
Ym16=Y11.values
Ym01=np.float64(Ym16[:,11])#N%
Ym11=np.float64(Ym16[:,12])#Nuptake
Ym21=np.float64(Ym16[:,13])#Biomass
#Y=np.column_stack((Ym0,Ym2))





##Plotting density curves------------------------------------------------------
# Plot
kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2, 'shade': True})
#matplotlib.rcParams.update({'font.size': 14})
#plt.figure(figsize=(10,5), dpi= 450)
sns.distplot(Ym0,  rug=True,  hist=False, color="dodgerblue", label="Train-ARDEC12+Iliff12", **kwargs)
sns.distplot(Ym01, rug=True,  hist=False, color="orange", label="Test-ARDEC12+Iliff12", **kwargs)
plt.xlabel('N (%)')
plt.ylabel('Density')
plt.legend(prop={'size': 14})
plt.savefig('Density_Train_Test_ARDEC12+Iliff_N.png',bbox_inches="tight",dpi=300)
plt.show()





##--------------------------------------------------------------------------------------------------------
## Biom
sns.distplot(Ym2,  rug=True,  hist=False, color="dodgerblue", label="Train-ARDEC12+Iliff12", **kwargs)
sns.distplot(Ym21, rug=True,  hist=False, color="orange", label="Test-ARDEC12+Iliff12", **kwargs)
plt.xlabel('Biomass (g)')
plt.ylabel('Density')
plt.legend(prop={'size': 14})
plt.savefig('Density_Train_Test_ARDEC12+Iliff_Biom.png',bbox_inches="tight",dpi=300)
plt.show()





##--------------------------------------------------------------------------------------------------------
## Nuptake
sns.distplot(Ym1,  rug=True,  hist=False, color="dodgerblue", label="Train-ARDEC12+Iliff12", **kwargs)
sns.distplot(Ym11, rug=True,  hist=False, color="orange", label="Test-ARDEC12+Iliff12", **kwargs)
plt.xlabel('N uptake (g)')
plt.ylabel('Density')
plt.legend(prop={'size': 14})
plt.savefig('Density_Train_Test_ARDEC12+Iliff_Nuptake.png',bbox_inches="tight",dpi=300)
plt.show()


