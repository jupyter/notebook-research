import pandas as pd
import re
import base64
import datetime
import pickle

import boto3
from io import BytesIO, StringIO
s3 = boto3.resource("s3")
bucket = s3.Bucket("notebook-research")

flatten = lambda l: [item for sublist in l for item in sublist]

### Methods to load full CSVs

def load_owners():
    start = datetime.datetime.now()
    owners = pd.read_csv('data_final/owners_final.csv')
    end = datetime.datetime.now()
    print('Owners loaded in', end - start)
    return owners.drop(
        columns = [
            c for c in owners.columns 
            if c.startswith('Unnamed')
        ])
        

def load_notebooks(type = None):
    if type == "vis":
        start = datetime.datetime.now()
        f = open('analysis_data/notebooks_vis.df', 'rb')
        notebooks = pickle.load(f)
        f.close()
        end = datetime.datetime.now()
        print('Notebooks (vis) loaded in',end - start)
        return notebooks.drop(
        columns = [
            c for c in notebooks.columns 
            if c.startswith('Unnamed')
        ])
        
    else:
        start = datetime.datetime.now()
        notebooks = pd.read_csv('data_final/notebooks_final.csv')
        end = datetime.datetime.now()
        print('Notebooks loaded in', end - start)
        return notebooks.drop(
        columns = [
            c for c in notebooks.columns 
            if c.startswith('Unnamed')
        ])
        

def load_repos():
    start = datetime.datetime.now()
    repos = pd.read_csv('data_final/repos_final.csv')
    end = datetime.datetime.now()
    print('Repos loaded in', end - start)
    return repos.drop(
        columns = [
            c for c in repos.columns 
            if c.startswith('Unnamed')
        ])

### Methods to load aggregations computed in analysis notebooks

def load_code():
    start = datetime.datetime.now()
    
    with open('analysis_data/nb_code.df', 'rb') as f:
        nb_code = pickle.load(f)
    
    end = datetime.datetime.now()
    print('Code loaded in', end - start)
    return nb_code

def load_nb_imports(code = False):
    if code:
        start = datetime.datetime.now()
        
        print('opening code')
        with open('analysis_data/nb_code.df', 'rb') as f:
            nb_code_df = pickle.load(f)
        print('code opened')
    
        print('opening imports')
        with open('analysis_data/nb_imports.df', 'rb') as f:
            nb_imports_df = pickle.load(f)
        t1 = datetime.datetime.now()
        print('imports opened')
        print('Opened data frame in', t1 - start)
        
        # get processed imports column
        print('combining dataframes')
        nb_imports_code_df = nb_code_df.merge(
            nb_imports_df[['file','imports']],
            on = 'file'
        )
        print('dataframes combined')
        # delete nb_imports_df to save memory
        del nb_imports_df
        t2 = datetime.datetime.now()
        print('Processed imports in', t2 - t1)
        
        end = datetime.datetime.now()
        print('Notebook imports with code loaded in', end - start)
        
        return nb_imports_code_df
    else:
        start = datetime.datetime.now()

        f = open('analysis_data/nb_imports.df', 'rb')
        nb_imports_df = pickle.load(f)
        f.close()

        end = datetime.datetime.now()
        print('Notebook imports loaded in', end - start)
    
        return nb_imports_df

    
def code_string_to_list(df, column, printing = True):
    start = datetime.datetime.now()
    patterns = '\\\', \\\'|\\\', "|", \\\'|", "'
    c = []
    for i, row in df.iterrows():
        c.append(re.split(patterns, row[column][2:-2]))
        if i%100000 == 0 and printing:
            print(i, datetime.datetime.now() - start)
    df[column] = c
    end = datetime.datetime.now()
    if printing:
        print(end - start)
    
    
def string_to_list_of_lists(df, column):
    start = datetime.datetime.now()
    patterns = '\\\', \\\'|\\\', "|", \\\'|", "'
    c = []
    for i, row in df.iterrows():
        c.append(
            [
                [word.replace("'","") for word in 
                    single_import.split('\', \'')
                 ] 
                for single_import in re.split('\], \[', row[column][2:-2])
            ]
        )
        if i%500000 == 0:
            print(i, datetime.datetime.now() - start)
    
    df[column] = c
    end = datetime.datetime.now()
    
def load_errors():
    start = datetime.datetime.now()

    f = open('analysis_data/error.df', 'rb')
    errors_df = pickle.load(f)
    f.close

    end = datetime.datetime.now()
    print('Errors loaded in', end - start)
    return errors_df


def load_cell_types():
    start = datetime.datetime.now()

    f = open('analysis_data/cell_types.df', 'rb')
    cell_types_df = pickle.load(f)
    f.close()
    end = datetime.datetime.now()
    print('Cell types loaded in', end - start)
    return cell_types_df


def load_cell_order():
    start = datetime.datetime.now()
   
    f = open('analysis_data/cell_order.df', 'rb')
    cell_order_df = pickle.load(f)
    f.close()
    
    end = datetime.datetime.now()
    print('Cell order loaded in', end - start)
    return cell_order_df

def load_output():
    start = datetime.datetime.now()
   
    f = open('analysis_data/cell_output.df', 'rb')
    output_df = pickle.load(f)
    f.close()
    
    
    end = datetime.datetime.now()
    print('Outputs loaded in', end - start)
    return output_df


def load_statuses():
    start = datetime.datetime.now()
    
    f = open('analysis_data/statuses.df', 'rb')
    statuses_df = pickle.load(f)
    f.close()
    
    end = datetime.datetime.now()
    print('Statuses loaded in', end - start)
    return statuses_df


def load_cell_stats():
    start = datetime.datetime.now()

    f = open('analysis_data/cell_stats.df','rb')
    cell_stats_df = pickle.load(f)
    f.close()

    end = datetime.datetime.now()
    print('Cell stats loaded in', end - start)
    return cell_stats_df

def load_collab_status():
    start = datetime.datetime.now()
    
    f = open('analysis_data/collab_status.df','rb')
    collab_status_df = pickle.load(f)
    f.close()
    
    end = datetime.datetime.now()
    print('Collaboration statuses loaded in', end - start)
    return collab_status_df

def load_special():
    start = datetime.datetime.now()

    f = open('analysis_data/special_functions.df', 'rb')
    special_df = pickle.load(f)
    f.close()

    end = datetime.datetime.now()
    print('Special functions loaded in', end - start)
    return special_df
    
def load_vis_uses():
    start = datetime.datetime.now()

    f = open('analysis_data/all_vis_uses.list', 'rb')
    all_vis_uses = pickle.load(f)
    f.close()

    end = datetime.datetime.now()
    print('Visualization counts loaded in', end - start)
    return all_vis_uses
    
def load_framework_uses():
    start = datetime.datetime.now()

    f = open('analysis_data/framework_uses.df','rb')
    framework_uses_df = pickle.load(f)
    f.close()
    
    end = datetime.datetime.now()
    print('Framework uses loaded in', end - start)
    return framework_uses_df

def load_function_defs():
    start = datetime.datetime.now()

    f = open('analysis_data/function_defs.df', 'rb')
    function_defs_df = pickle.load(f)
    f.close()
    
    end = datetime.datetime.now()
    print('Function definitions loaded in', end - start)
    return function_defs_df

def load_function_use():
    start = datetime.datetime.now()

    f = open('analysis_data/nb_function_use.df', 'rb')
    function_use_df = pickle.load(f)
    f.close()

    end = datetime.datetime.now()
    print('Function uses loaded in', end - start)
    return function_use_df

def load_lines():
    start = datetime.datetime.now()

    f = open('analysis_data/lines_per_code_cell.list', 'rb')
    lines_per_code_cell = pickle.load(f)
    f.close()

    end = datetime.datetime.now()
    print('Lines per code cell loaded in', end - start)
    return lines_per_code_cell

def load_objects():
    start = datetime.datetime.now()

    f = open('analysis_data/all_objects.df', 'rb')
    all_objects_df = pickle.load(f)
    f.close()

    end = datetime.datetime.now()
    print('Objects loaded in', end - start)
    return all_objects_df

def load_edu_status():
    start = datetime.datetime.now()

    f = open('analysis_data/repo_edu_status.df','rb')
    repo_edu_status = pickle.load(f)
    f.close()
    
    end = datetime.datetime.now()
    print('Educational status loaded in', end - start)
    return repo_edu_status

### Methods to process csv columns

def string_to_list(df, column):
    """ 
    Converts a string column back to a list 
    (needed because list columns are converted to strings in CSV)
    
    e.g. ['import pandas as pd','for i in range(10)','\tprint(i)']
    """
    try:
        if type(df[column].iloc[0]) == str and len(df[column]) > 0:
            df[column] = [[] if c == '[]' 
                    else c[1:-1].replace('"','').split(', ') 
                    for c in df[column]
            ]
        df[column] = [[s.replace('\\n','').replace("'","").strip() 
                for s in c] for c in df[column]
        ]
    except Exception:
        print(type(df))

### Methods for interaction with S3 bucket

def s3_to_df(path, usecols = None):
    """ Method to open csv from S3 bucket as a dataframe """
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
    """ Method to put a dataframe into S3 bucket as a csv """
    obj = s3.Object("notebook-research", path)
    csv_buffer = StringIO()
    data_frame.to_csv(csv_buffer, index = False)
    obj.put(Body=csv_buffer.getvalue())

def list_s3_dir(path):
    """ Method to list contents of a directory in the S3 bucket """
    list_dir = set([])
    for obj in bucket.objects.filter(Prefix = path):
        list_dir.add(obj.key.split("/")[1])
    return list_dir