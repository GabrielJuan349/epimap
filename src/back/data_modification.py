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
            # Create a copy of i as a list to modify
            row = list(i)
            # Look up the country name
            result = country_codes.loc[country_codes.iloc[:, 0] == row[1], country_codes.columns[1]]
            if not result.empty:
                row[1] = result.iloc[0]
            new_data = pd.concat([new_data, pd.DataFrame([row])], ignore_index=True)
        return new_data
    
def modify_dataset_epi(data:pd.DataFrame):
    epi_path = os.path.join(project_root, 'data', 'epi_codes.csv')
    epi_codes = pd.read_csv(epi_path)
    del epi_path
    
    # Crear una copia del DataFrame de entrada para no modificar el original
    modified_data = data.copy()
    
    # Crear un diccionario de mapeo desde el código (columna 0) al valor de la columna 1
    code_to_detail = dict(zip(epi_codes.iloc[:, 0], epi_codes.iloc[:, 1]))
    
    # Reemplazar los valores en la columna 2 del DataFrame de entrada
    # con los valores correspondientes de la columna 1 de epi_codes
    modified_data.iloc[:, 2] = modified_data.iloc[:, 2].map(code_to_detail)
    
    return modified_data





