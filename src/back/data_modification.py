import pandas as pd
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))


def modify_dataset_country(data, mode=True):
    country_path = os.path.join(project_root, 'data', 'country_codes.csv')
    country_codes = pd.read_csv(country_path)
    del country_path
    
    if mode == True:
        result = country_codes.loc[country_codes.iloc[:, 1] == data, country_codes.columns[0]]
        if not result.empty:
            return result.iloc[0]
        else:
            return None
    else:
        new_data = pd.DataFrame()
        for i in data:
            i[1]= country_codes.loc[country_codes.iloc[:, 0] == i[1], country_codes.columns[1]]
            new_data = new_data.append(i)
        return new_data
    
def modify_dataset_epi(data:pd.DataFrame):
    epi_path = os.path.join(project_root, 'data', 'epi_codes.csv')
    epi_codes = pd.read_csv(epi_path)
    del epi_path

    

