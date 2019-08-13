import pandas as pd
import re
import base64
import datetime

import boto3
from io import BytesIO, StringIO
s3 = boto3.resource("s3")
bucket = s3.Bucket("notebook-research")

def load_owners():
    df = s3_to_df('data_final/owners.csv')
    try:
        return df.reset_index(drop=True)
   
    except Exception:
        print('Could not open owners.csv')
        return None
        

def load_notebooks(sample = 1):
    df = s3_to_df('data_final/notebooks.csv')
    try:
        return df.reset_index(drop=True)
    except Exception:
        print('Could not open notebooks.csv')
        return None        

def load_repos(sample = 1):
    df = s3_to_df('data_final/repos.csv')
    try:
        return df.reset_index(drop=True)
    except Exception:
        print('Could not open repos.csv')
        return None

def load_cells(columns=None):
    if columns:
        df = s3_to_df(
                'data_final/cells.csv', 
                usecols = columns
            )
    else:
        df = s3_to_df(
                'data_final/cells.csv', 
            )
        
    try:
        print('Data opened, processing.')
        return process_cells(df.reset_index(drop=True))
    except Exception:
        print('Could not open cells.csv')
        return None
        
def process_cells(cells):
    def string_to_list(df, column):
        if type(df[column][0]) == str:
            df[column] = [[] if c == '[]' 
                          else c[1:-1].replace('"','').split(', ') 
                          for c in df[column]]
        df[column] = [[s.replace('\\n','').replace("'","").strip() 
            for s in c] for c in df[column]]


    for column in ['comments','error_names',
                   'classes','display_data_keys',
                  'error_values','execute_result_keys','functions']:
        if column in cells.columns:
            print('Cleaning {0}...'.format(column))
            string_to_list(cells, column)
    
    def code_string_to_list(df, column):
        patterns = ["', '", "', \"", "\", '", "\", \""]
        df[column] = [[row[2:-2].split('\\n')[0]] + ([c[4:] for c in row[2:-2].split('\\n')[1:] if c.strip() not in patterns] 
                                                    if len(row[2:-2]) > 1 else []) for row in df[column]]
       
    if 'code' in cells.columns:
        print('Cleaning code...')
        code_string_to_list(cells, 'code')

    def string_to_list_of_lists(df, column):
        print(type(df))
        print(len(df))
        df[column] = [[] if len(df[column][i]) == 2 else [h.replace("'","").split(', ') 
          for h in df[column][i][2:-2].replace('(','[').replace(')',']').split('], [')] 
          for i in range(len(df))]

    for column in ['headings','imports','links','markdown']:
        if column in cells.columns:
            print('Cleaning {0}...'.format(column))
            string_to_list_of_lists(cells, column)
            if column == 'imports':
                cells['imports'] = [[[im[0], im[1].split('#')[0].strip()] if len(im) == 2 else [] for im in imports] for imports in cells.imports]
    
    return cells


def s3_to_df(path, usecols = None):
    try:
        df_obj = s3.Object("notebook-research", path)
        if usecols:
            return pd.read_csv(BytesIO(df_obj.get()["Body"].read()), header = 0, usecols = usecols)
        else:
            return pd.read_csv(BytesIO(df_obj.get()["Body"].read()), header = 0)
    except Exception as e:
        print(e)
        return None
    
def df_to_s3(data_frame, path):
    obj = s3.Object("notebook-research", path)
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, index = False)
    obj.put(Body=csv_buffer.getvalue())

def list_s3_dir(path):
    list_dir = set([])
    for obj in bucket.objects.filter(Prefix = path):
        list_dir.add(obj.key.split("/")[1])
    return list_dir


#### Only used in PrepareData.ipynb ####
def get_single_df(df_name, i):
    csvs = ['csv/'+ f for f in list_s3_dir('csv/{0}__{1}'.format(df_name, i))]
    dfs = []
    for f in csvs:
        df = s3_to_df(f)
        if (
            (df_name == 'cell1' and 'file' in df.columns)
            or df_name != 'cell1'
        ):
            dfs.append(df)
        print('Done with',f)
    return pd.concat(dfs)

def combine_notebooks():
    start = datetime.datetime.now()
    for i in range(10):
        start_i = datetime.datetime.now()
        nb_i = get_single_df('notebooks2',i)
        mid_i = datetime.datetime.now()
        print('notebooks_{0} loaded in'.format(i),mid_i - start_i)
        nb_i.to_csv('notebooks_{0}.csv'.format(i), index = False)
        end_i = datetime.datetime.now() 
        print('notebooks_{0} in csv in'.format(i), end_i-mid_i)
    
    end = datetime.datetime.now()
    print('DONE! Now all in csv.', end - start)
    
def combine_cells(start_i = 0):
    start = datetime.datetime.now()
    for i in range(start_i, 10):
        start_i = datetime.datetime.now()
        cells_i = get_single_df('cells1',i)
        mid_i = datetime.datetime.now()
        print('cells_{0} loaded in'.format(i),mid_i - start_i)
        cells_i.to_csv('cells_{0}.csv'.format(i), index = False)
        end_i = datetime.datetime.now()
        print('cells_{0} in csv in'.format(i), end_i - mid_i)
     
    end = datetime.datetime.now()
    print('DONE! Now all in csv.', end - start)
    
def combine_repos():
    start = datetime.datetime.now()
    for i in range(10):
        start_i = datetime.datetime.now()
        repos_i = s3_to_df('csv/repos2_{0}.csv'.format(i))
        mid_i = datetime.datetime.now()
        print('repos_{0} loaded in'.format(i),mid_i - start_i)
        repos_i.to_csv('repos_{0}.csv'.format(i), index = False)
        end_i = datetime.datetime.now()
        print('repos_{0} in csv in'.format(i), end_i - mid_i)
        
    end = datetime.datetime.now()
    print('DONE! Now all in csv.', end - start)
    
def combine_owners():
    start = datetime.datetime.now()
    for i in range(10):
        start_i = datetime.datetime.now()
        repos_i = s3_to_df('csv/owners2_{0}.csv'.format(i))
        mid_i = datetime.datetime.now()
        print('owners_{0} loaded in'.format(i),mid_i - start_i)
        repos_i.to_csv('owners_{0}.csv'.format(i), index = False)
        end_i = datetime.datetime.now()
        print('owners_{0} in csv in'.format(i), end_i - mid_i)
        
    end = datetime.datetime.now()
    print('DONE! Now all in csv.', end - start)