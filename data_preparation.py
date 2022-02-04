import pandas as pd
import os


def __make_data_file(data_type, cat_col, data, ones, zeros):
    if data_type == 'training':
        factor = 0.6
    elif data_type == 'testing' or data_type == 'trimming':
        factor = 0.2
    else:
        print(f'ERROR: data_type {data_type} doesn\'t exist')
    lst = []
    n = int(factor * ones)
    i = 0
    while n > 0:
        if data[cat_col][i] == 1:
            lst.append(data.loc[i])
            data.loc[i, cat_col] = 2
            n = n - 1
        i = i + 1
    n = int(factor * zeros)
    i = 0
    while n > 0:
        if data[cat_col][i] == 0:
            lst.append(data.loc[i])
            data.loc[i, cat_col] = 2
            n = n - 1
        i = i + 1
    generated_data = pd.DataFrame(lst, columns=range(58))
    print(f'==========={data_type}===========\n{generated_data}\n')
    filename = f'data/{data_type}.data'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    generated_data.to_csv(f'data/{data_type}.data', header=False, index=False)
    return data


def make_files(datafile_path, cat_col):
    data_original = pd.read_csv(f'{datafile_path}', header=None)
    data = data_original.sample(frac=1).reset_index(drop=True) 
    ones = data[cat_col].sum()
    zeros = len(data[cat_col]) - ones
    print(f'Data after shuffling:\n{data}\n')
    data = __make_data_file('training', cat_col, data, ones, zeros)
    data = __make_data_file('testing', cat_col, data, ones, zeros)
    data = __make_data_file('trimming', cat_col, data, ones, zeros)
