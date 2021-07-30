#Karna Gowda
#0.0.1

import bmgdata as bd
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def find_outliers(data):
    #returns true for elements more than 1.5 interquartile ranges above the upper quartile.
    
    q75 = np.quantile(data,0.75)
    q25 = np.quantile(data,0.25)
    is_outlier = data > q75 + (q75-q25)*1.5
    return is_outlier

def remove_bubbles(data_in,data_540,data_900):
    #find wells with bubbles in the no2no3 measurements
    
    bub_in  = find_outliers(data_in['900'])
    bub_900 = find_outliers(data_900)

    #replace values at 540 nm
    data_out = data_in.copy()
    for idx in bub_in.index[bub_in == True].tolist():
        replacement = data_540.loc[idx]
        data_out['540'].loc[idx] = np.median(replacement[~bub_900.loc[idx]]) #replace with median of values that don't have bubbles
    return data_out

def read_griess(meta_fn,data_fn=None,data_540_fn=None,data_900_fn=None):
    #returns absorbance data at 540 nm
    #corrects for bubbles if well scan measurements are provided
    #there are three valid cases that this function works for:
    #1) A file name is provided for data_fn. Usually this is for importing no2 data.
    #2) File names are provided for data_fn and associated well scan files data_540_fn and data_900_fn. Usually this is for importing old no2no3 measurements using both endpoint and well scan data.
    #3) File names are provided for only well scan files data_540_fn and data_900_fn. This is the prefered measurement data for future experiments.
    
    meta       = pd.read_csv(meta_fn,index_col=0).dropna(how='all')  #import metadata
    
    if (data_540_fn != None) & (data_900_fn != None): 
            data_540 = bd.read_abs_wellscan(data_540_fn) #import well scan data (540 nm)
            data_540 = data_540[data_540.index.isin(meta.index)] #remove indices with no values in the metadata
            data_900 = bd.read_abs_wellscan(data_900_fn) #import well scan data (900 nm)
            data_900 = data_900[data_900.index.isin(meta.index)] #remove indices with no values in the metadata
            
    if data_fn != None: #case 1 or 2
        data_out    = bd.read_abs_endpoint(data_fn) #import data
        data_out    = data_out[data_out.index.isin(meta.index)] #remove indices with no values in the metadata
        #correct for bubbles
        if data_540_fn != None: #case 2
            data_out = remove_bubbles(data_out,data_540,data_900) #correct for bubbles in measurement.
        data_out = data_out['540'] #pick out 540 nm measurement
    elif (data_540_fn != None) & (data_900_fn != None): #case 3
        bub_900 = find_outliers(data_900)
        for row in data_540.index:
            if sum(np.isnan(bub_900.loc[row])) < 4:
                data_540.loc[row][bub_900.loc[row]] = np.NaN #Set bubbles to NaN
            else:
                data_540.loc[row] = data_540.loc[row] - data_900.loc[row]
        data_out = data_540.median(axis=1)
        data_out = data_out.rename("540")
        
    return data_out

def fit_griess(meta_fn,no2_fn,no2no3_fn=None,no2no3_540_fn=None,no2no3_900_fn=None):
    meta = pd.read_csv(meta_fn,index_col=0).dropna(how='all')
    meta = meta[['NO2','NO3']] #keep only the NO2/NO3 concentration columns
    no2 = read_griess(meta_fn,no2_fn)
    no2no3 = read_griess(meta_fn,data_fn = no2no3_fn,data_540_fn=no2no3_540_fn,data_900_fn=no2no3_900_fn)
    
    #remove nan values
    nan_idx = meta['NO2'].index[meta['NO2'].apply(np.isnan)].union(meta['NO3'].index[meta['NO3'].apply(np.isnan)])
    meta = meta.drop(nan_idx)
    no2 = no2.drop(nan_idx)
    no2no3 = no2no3.drop(nan_idx)
    
    #subtract blank values
    blank_idx = meta.index[(meta['NO2']==0) & (meta['NO3']==0)].tolist()
    no2_blank = no2.loc[blank_idx].median()
    no2no3_blank = no2no3.loc[blank_idx].median()
    no2 = no2 - no2_blank
    no2no3 = no2no3 - no2no3_blank

    #identify pure no2 and pure no3 samples
    no2_std_idx = meta.index[(meta['NO2']>0) & (meta['NO3']==0)].tolist()
    no3_std_idx = meta.index[(meta['NO2']==0) & (meta['NO3']>0)].tolist()

    #no2 standard curve, griess measurement
    y = no2.loc[no2_std_idx].values.reshape(-1, 1)
    x = meta['NO2'].loc[no2_std_idx].values.reshape(-1, 1)

    #Fit a quadratic model to the nitrite concentration
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    reg = LinearRegression().fit(x, y)
    g_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    #no2 and no3 standard curves, vcl3 measurement
    y1 = no2no3.loc[no2_std_idx].values
    y2 = no2no3.loc[no3_std_idx].values
    x1 = meta.loc[no2_std_idx].values
    x2 = meta.loc[no3_std_idx].values

    #Fit a quadratic model to the sum of nitrate and nitrite concentrations
    x = np.append(np.sum(x1,axis=1),np.sum(x2,axis=1)).reshape(-1,1)
    x = np.append(x, x**2,axis=1) #make x and x^2 the independent variables
    y = np.append(y1,y2).reshape(-1,1)
    reg = LinearRegression().fit(x, y)
    v_fit = [reg.intercept_[0], reg.coef_[0][0], reg.coef_[0][1]]

    fit = [[no2_blank,no2no3_blank], g_fit, v_fit]
    
    return fit

def invert_griess(no2,fit,no2no3=None):
    #Returns inferred concentrations
    NO2 = ((-fit[1][1] + np.sqrt(fit[1][1]**2 - 4*(fit[1][0]-no2)*fit[1][2]))/2/fit[1][2]).rename("NO2"); #solve quadratic formula
    NO2[NO2<0] = 0.0
    if isinstance(no2no3,pd.Series):
        NO3 = (((-fit[2][1] + np.sqrt(fit[2][1]**2 - 4*(fit[2][0]-no2no3)*fit[2][2]))/2/fit[2][2]) - NO2).rename("NO3"); #solve quadratic formula
        NO3[NO3<0] = 0.0
    else:
        NO3 = NO2.copy().rename("NO3")
        NO3[NO3!=0] = 0.0

    data_out = pd.DataFrame()
    data_out = data_out.append([NO2,NO3]).transpose()
    return data_out