#Karna Gowda
#0.0.1
import csv
import numpy as np
import pandas as pd

def search_rows_str(data,search_str,nth=1):
    ctr = 1
    for i in range(0,len(data)):
        if any(search_str in string for string in data[i]):
            if ctr == nth:
                search_idx = i
                break
            else:
                ctr = ctr + 1;
    return search_idx
    
def find_first_well(data):
    letters = ["A","B","C","D","E","F","G","H"]
    numbers = ["01","02","03","04","05","06","07","08","09","10","11","12"]
    wells = []
    for letter in letters:
        for number in numbers:
            wells.append(letter+number)
            
    start_idx = 0
    for i in range(0,len(data)):
        try:
            if data[i][0][0:3] in wells:
                start_idx = i
                break
        except:
            pass
            
    return start_idx
        
def read_abs_endpoint(file_name):
    with open(file_name,encoding='latin_1') as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))

    #Find row containing number of wavelengths.
    idx = search_rows_str(data,"No. of Channels / Multichromatics:")
    row = data[idx].copy()
    n_wl = [int(i) for i in row[0].split() if i.isdigit()][0] #extract number of wavelengths

    #Find row(s) containing wavelength range information.
    idx = search_rows_str(data,"nm")
    row = data[idx].copy()
    if "..." in row[0]:
        wl_range = [int(i) for i in row[0].replace('...',' ').replace('nm','').split() if i.isdigit()] #extract wavelength range
        wl = np.linspace(wl_range[0],wl_range[1],n_wl).astype('int')
    else:
        wl = np.zeros(n_wl).astype('int')
        for j in range(0,n_wl):
            idx = search_rows_str(data,"nm",nth=j+1)
            row = data[idx].copy()
            wl[j] = [int(i) for i in row[0].replace('nm','').split() if i.isdigit()][0]

    #Find first data row and fill in data starting at that row.   
    start_idx = find_first_well(data)
    for idx in range(start_idx,len(data)-1):
        row = data[idx].copy()
    
        if idx == start_idx:
            temp_exists = any("T" in string for string in row[-1]) #determine whether there is a temperature field
            row_label = [row[0][0:3]]
        else:
            row_label.append(row[0][0:3])

        row[0] = row[0][5:] #strip out plate index numbering
    
        if temp_exists:
            row[-1] = row[-1][7:]
    
        try:
            if 'data_array' in locals():
                data_array = np.vstack((data_array,np.asarray(row, dtype=np.float64, order='C')))
            else:
                data_array = np.asarray(row, dtype=np.float64, order='C')
        except:
            row_label.pop(-1) #if row contains unparseable data, remove label

    #Turn into data frame.
    if temp_exists:
        df = pd.DataFrame(data_array,columns = np.append(wl,'T'),index = row_label)
    else:
        df = pd.DataFrame(data_array,columns = wl.astype('str'),index = row_label)
    
    return df
    
def read_abs_wellscan(file_name):
    with open(file_name,encoding='latin_1') as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))

    #Find first data row and fill in data starting at that row.   
    start_idx = find_first_well(data)
    row_label = []
    for idx in range(start_idx,len(data)-1,5):
        row = data[idx].copy()
        row_label.append(row[0][0:3])
        
        #assumes a 2x2 well scan
        row = data[idx+2].copy()
        row.append(data[idx+3][0])
        row.append(data[idx+3][1])
        
        try:
            if 'data_array' in locals():
                data_array = np.vstack((data_array,np.asarray(row, dtype=np.float64, order='C')))
            else:
                data_array = np.asarray(row, dtype=np.float64, order='C')
        except:
            row_label.pop(-1) #if row contains unparseable data, remove label

    #Turn into data frame.
    df = pd.DataFrame(data_array,index = row_label)
    
    return df