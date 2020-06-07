## import modules here 
import pandas as pd
import numpy as np
import helper


################### Question 1 ###################

def binary_list(array, original_array):
    """
    In this function we treat the list as a binary digital number
    The elements that are not 'ALL' in the list are recognized as 0
    The elements that are 'ALL' or '*' in the list are considered as 
    """
    i = len(array) - 1
    while i >= 0:
        if array[i] != 'ALL':
            array[i] = 'ALL'
            return True
        else:
            array[i] = original_array[i]
        i -= 1

    return False

def single_tuple_opt(df):

    value = df.iloc[0,-1]
    # Remove the last dim
    org_dimensions = df.iloc[0, : -1]
    #tuple to a numpy array
    array = np.zeros(df.values.shape[1] - 1, dtype = object)
    org = np.copy(org_dimensions.values)

    for i in range(len(array)):
        array[i] = org[i]
    re = []
    re.append(df.values[0])

    while binary_list(array, org):       
        row = np.zeros(df.values.shape[1], dtype = object)
        for i in range(len(array)):
            row[i] = array[i]
        row[-1] = value
        re.append(row)

    return re

def buc_rec_optimized(df):# do not change the heading of the function
    df_fill = pd.DataFrame(columns = df.columns)
    dic = {}
    index = 0
    for i in range(df.values.shape[0]):
        rows = single_tuple_opt(df.iloc[i:i+1, :])    
        for row in rows:
            value = row[-1]
            key = ""
            for j in range(len(row) - 1):
                key += str(row[j])
                if j < len(row) - 2:
                    key += '~'
            if key in dic:
                dic[key] += int(value)
            else:
                dic[key] = int(value)

    index = 0
    for key, value in sorted(dic.items()):
        str_list = key.split('~')
        new_row = np.zeros((len(str_list) + 1), dtype = object)
        
        for i in range(len(str_list)):
                new_row[i] = str_list[i]

        new_row[-1] = value
        df_fill.loc[index] = new_row
        index += 1
            
    return df_fill   

