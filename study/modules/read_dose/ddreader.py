#dose data reader(ddreader)

import numpy as np
import pandas as pd

#=============================================================================
#                                For Geant4
#=============================================================================
def GetDoseDepth_z(filename, z = 150, thin_z = 0.2):
    
    data_origin = pd.read_csv(filename)

    data_dose = data_origin[data_origin['dose']>0].drop(columns = ["ix","iy"])

    dose_sum = np.zeros(z)
    list_z = np.zeros(z)
    for i in range(z):
        data_z = data_dose[data_dose['iz'] == i].sum()
        dose_sum[i] = data_z['dose'] 
        list_z[i] = thin_z * i

    return list_z, dose_sum

#-----------------------------------------------------------------------------
def GetDoseDepth_z_center(filename, z = 150, center = 31, thin_z = 0.2):
    
    data_origin = pd.read_csv(filename)

    data_dose = data_origin[data_origin['dose'] > 0]
    data_dose = data_dose[data_dose['ix'] == center]
    data_dose = data_dose[data_dose['iy'] == center].drop(columns = ["ix","iy"])
    dose_sum = np.zeros(z)
    list_z = np.zeros(z)
    
    for i in range(z):
        data_z = data_dose[data_dose['iz'] == i]
        dose_sum[i] = data_z['dose'] 
        list_z[i] = thin_z * i

    return list_z, dose_sum

#-----------------------------------------------------------------------------
def GetDoseProf(filename, z = 0):
    data_origin = pd.read_csv(filename)

    data_dose = data_origin[data_origin["iz"] == z].drop(columns = ["ix", "iy", "iz"])
    
    dose_prof = np.array(data_dose).reshape(61,61)
    
    return dose_prof

#=============================================================================
#                                For EGSnrc++
#=============================================================================
def GetDosedepth_z_EGS(filename):
    data = np.loadtxt(filename, skiprows=4)
    data = data[0]
    dose_z = np.zeros(150, dtype="double")
    for i in range(150):
        dose = 0
        for j in range(61 * 61):
            idx = j + i * 61 * 61
            dose += data[idx] * 5 * 10
        dose_z[i] = dose / (1.601 * 10**(-14))
    return dose_z

#-----------------------------------------------------------------------------
def GetDoseDepth_z_center_EGS(filename):
    data = np.loadtxt(filename, skiprows=4)
    data = data[0]
    dose_z = np.zeros(150, dtype="double")
    for i in range(150):
        dose = 0
        for j in range(61 * 61):
            if j == 1922:
                idx = j + i * 61 * 61
                dose += data[idx] * 5 * 10
            dose_z[i] = dose / (1.602 * 10**(-14))
    return dose_z

#-----------------------------------------------------------------------------
def GetDoseProf_EGS(filename, z = 0):
    data = np.loadtxt(filename, skiprows=4)
    data = data[0]
    data = data.reshape(150,61,61)
    
    data = data[z] * (5 * 10) / (1.602 * 10**(-14))
    
    return data
#-----------------------------------------------------------------------------