import pandas as pd
import re
import base64
import boto3

s3 = boto3.resource('s3')


def s3_to_df(path):
    df_obj = s3.Object('notebook-research', path)
    return pd.read_csv(BytesIO(df_obj.get()['Body'].read()), header = 0)

def load(lang = None, local=False):
#     if local:
#         cells = pd.read_csv('../csv/cells2.csv')
#         notebooks = pd.read_csv('../csv/notebooks3.csv')
#         owners = pd.read_csv('../csv/owners2.csv')
#         readmes = pd.read_csv('../csv/readmes1.csv')
#         repos = pd.read_csv('../csv/repos3.csv')
#     else:
#         cells = s3_to_df('csv/cells2.csv')
#         notebooks = s3_to_df('csv/notebooks3.csv')
#         owners = s3_to_df('csv/owners2.csv')
#         readmes = s3_to_df('csv/readmes1.csv')
#         repos = s3_to_df('csv/repos3.csv')

## Uncomment section above and DELETE THIS ####################
    cells = pd.read_csv('../save/cells2.csv')
    notebooks = pd.read_csv('../save/notebooks3.csv')
    owners = pd.read_csv('../save/owners2.csv')
    readmes = pd.read_csv('../save/readmes1.csv')
    repos = pd.read_csv('../save/repos3.csv')    
    cells = cells.rename(columns = {'comments':'num_comments'}).rename(columns={'comments_words':'comments'})
###############################################################

    def string_to_list(df, column):
        if type(df[column][0]) == str:
            df[column] = [[] if c == '[]' 
                          else c[1:-1].replace('"','').split(', ') 
                          for c in df[column]]
        df[column] = [[s.replace('\\n','').replace("'","").strip() 
            for s in c] for c in df[column]]

    string_to_list(cells,'comments')
    string_to_list(cells,'error_names')
    string_to_list(cells, 'classes')
    string_to_list(cells, 'display_data_keys')
    string_to_list(cells, 'error_values')
    string_to_list(cells, 'execute_result_keys')
    string_to_list(cells, 'functions')
    
    def code_string_to_list(df, column):
        patterns = ["', '", "', \"", "\", '", "\", \""]
        df[column] = [[row[2:-2].split('\\n')[0]] + ([c[4:] for c in row[2:-2].split('\\n')[1:] if c.strip() not in patterns] 
                                                    if len(row[2:-2]) > 1 else []) for row in df[column]]
        
    code_string_to_list(cells, 'code')

    def string_to_list_of_lists(df, column):
        df[column] = [[] if len(df[column][i]) == 2 else [h.replace("'","").split(', ') 
          for h in df[column][i][2:-2].replace('(','[').replace(')',']').split('], [')] 
          for i in range(len(df))]

    string_to_list_of_lists(cells, 'headings')
    string_to_list_of_lists(cells, 'imports')
    cells['imports'] = [[[im[0], im[1].split('#')[0].strip()] if len(im) == 2 else [] for im in imports] for imports in cells.imports]
    string_to_list_of_lists(cells, 'links')
    string_to_list_of_lists(cells, 'markdown')
    
    content = []
    for i in range(len(readmes)):
        try:
            content.append(str(base64.b64decode(readmes.content[i]))[2:-1])
        except Exception:
            content.append('')
    
    readmes['content'] = content

    if lang == 'python':
        notebooks = notebooks[notebooks.lang_name == 'python']
        cells = cells[cells.file.isin(notebooks.file)]
        repos = repos[repos.repo_id.isin(notebooks.repo_id)]
        readmes = readmes[readmes.readme_id.isin(notebooks.repo_id)]
        owners = owners[owners.owner_id.isin(repos.owner_id)]
        
    return notebooks, cells, owners, readmes, repos

