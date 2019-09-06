import math
import load_data
import pickle
import pandas as pd
import numpy as np
import datetime


from collections import deque
import scipy.stats as st
import ast
import astpretty
import re

def main():
    # Used first in Organization.ipynb
    print('\nCell Output')
    get_cell_output()
    print('\nCell Stats')
    get_cell_stats()
    print('\nCell Order')
    get_cell_order()
    print('\nCell Types')
    get_cell_types()
    print('\nComments')
    get_comments()
    
    # Used first in Packages.ipynb
    print('\nGet Imports')
    get_nb_imports()
    print('\nGet Code')
    get_nb_code()
    
    print('\nGetting nb_imports_code_df')
    nb_imports_code_df = load_data.load_nb_imports(code = True)
    print('\nnb_imports_code_df loaded')
    cell_types_df = load_data.load_cell_types()
    print('\ncell_types loaded')
    cell_stats_df = load_data.load_cell_stats()
    print('\ncell_stats loaded')
    cell_info_code_df = cell_types_df.merge(
        cell_stats_df, on = 'file'
    ).merge(
        nb_imports_code_df.rename(columns={'code':'code_list'}), on = 'file'
    )
    print('\ndfs combined')
    
    #Used first in APIs.ipynb
    print('\nGet Objects')
    get_all_objects(cell_info_code_df)
    print('\nGet Lines Per Code Cell')
    get_lines_per_code_cell(cell_info_code_df)
    print('\nGet Function Definitions')
    get_function_defs(cell_info_code_df)
    print('\nGet Function Uses')
    get_function_use(cell_info_code_df)
    print('\nSeparate User-defined functions from not user-defined')
    add_user_funcs()
    
    # Used first in Struggles.ipynb
    print('\nGet Erros')
    get_errors()
    print('\nGet Statuses')
    get_statuses()

    
    # Used first in Visualizations.ipynb
    print('\nGet Visualization Uses')
    get_vis_uses(nb_imports_code_df)
    print('\nAdd Visualization Uses to Notebooks')
    get_vis_uses_nb(nb_imports_code_df)
    
    # Used first in Models.ipynb
    print('\nGet Framework Uses')
    get_framework_uses(nb_imports_code_df)

    print('\nGet Magic')
    get_magic()
    

def get_magic():
    df_chunks = pd.read_csv(
        'data_final/cells_final.csv', 
        header = 0, 
        usecols = ['file','cell_id','code'], 
        chunksize = 10000
    )

    def aggregate_special_lines(list_of_lines_of_code):
        return [
            l
            for l in load_data.flatten([l.split('\n') for l in list_of_lines_of_code if str(l) != 'nan'])
            if l.startswith('%') or '!' in l or
            l.startswith('?') or l.endswith('?')
        ]


    special_dfs = []
    i = 0

    start = datetime.datetime.now()
    i = len(special_dfs)
    for chunk in df_chunks:
        df = chunk.groupby('file')['code'].aggregate(
            aggregate_special_lines
        ).reset_index()

        special_dfs.append(df)
        if i%1000 == 0:
            print(i, datetime.datetime.now() - start)
        i+=1

    end = datetime.datetime.now()
    print('Chunks done in', end - start)

    start = datetime.datetime.now()
    special_df = pd.concat(
        special_dfs,
        sort = False
    ).reset_index(drop = True).groupby('file')['code'].aggregate(
        load_data.flatten
    ).reset_index()
    end = datetime.datetime.now()
    print('Combined in', end - start)


    start = datetime.datetime.now()
    f = open('analysis_data/special_functions.df', 'wb')
    pickle.dump(special_df, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved in', end - start)

    
def get_nb_code():
    start = datetime.datetime.now()
    df_chunks = pd.read_csv(
        'data_final/cells_final.csv', 
        header = 0, usecols = ['file','code','cell_type'], 
        chunksize=10000
    )

    # 25 minutes
    start = datetime.datetime.now()
    i = 0
    code_dfs = []

    for chunk in df_chunks:
        code_dfs.append(
           chunk[chunk.cell_type == 'code'].groupby('file')['code'].aggregate(lambda x: list(x)).reset_index()
        )
        if i%1000 == 0:
            print(i, datetime.datetime.now() - start)
        i += 1

    end = datetime.datetime.now()
    print('Chunks', end - start)

    code_df = pd.concat(code_dfs, sort = False).reset_index(drop=True)

    start = datetime.datetime.now()

    code_df = code_df.groupby('file')['code'].aggregate(load_data.flatten).reset_index()

    end = datetime.datetime.now()
    print('Combined', end - start)
    

    print('now saving')
    start = datetime.datetime.now()

    try:
        f = open('analysis_data/nb_code.df', 'wb')
        pickle.dump(code_df, f)
        f.close()
        print('saved to pickle')
    except:
        try:
            f = open('analysis_data/nb_code.df', 'wb')
            pickle.dump(code_df, f)
            f.close()
            print('saved to pickle')
        except:
            try:
                f = open('analysis_data/nb_code.df', 'wb')
                pickle.dump(code_df, f)
                f.close()
                print('saved to pickle')
            except:
                code_df.to_csv('analysis_data/nb_code.csv', index = False)
                print('saved to csv')

    end = datetime.datetime.now()
    print(end - start)
    
def get_all_objects(cell_info_code_df):
    # 1.5 hours
    start = datetime.datetime.now()
    all_objects = []
    unprocessed = 0
    target_types = [ast.Name,ast.Tuple,
           ast.Attribute, ast.Subscript,
           ast.List
    ]
    for i, row in cell_info_code_df.iterrows():
        o = {
            'file': row['file'],
            'objects': []
        }
        try:
            all_code = '\n'.join([
                c for c in '\n'.join([l for l in row.code_list if type(l) == str]).split('\n') 
                if (c != '' and not c.strip().startswith('%') and 
                    not c.strip().startswith('?') and not c.strip().startswith('!')
                )
            ])
            tree = ast.parse(all_code)
        except Exception as e:
            all_objects.append(o)
            unprocessed += 1
            if i%200000 == 0:
                print(i, datetime.datetime.now() - start, unprocessed, 'unprocessed')
            continue

        for t in tree.body:
            if type(t) == ast.Assign:
                value_type = type(t.value)
                for target in t.targets:
                    if type(target) in [ast.Tuple, ast.List]:
                        for node in ast.walk(target):
                            if type(node) == ast.Name:
                                o['objects'].append((node.id, value_type))
                    else:
                        if type(target) == ast.Name:
                            for node in ast.walk(target):
                                if type(node) == ast.Name:
                                    o['objects'].append((node.id, value_type))

        all_objects.append(o)
        if i%200000 == 0:
            print(i, datetime.datetime.now() - start)

    end = datetime.datetime.now()
    print('Found objects', end - start)
    
    all_objects_df = pd.DataFrame(all_objects)
    
    # 14 seconds
    start = datetime.datetime.now()

    f = open('analysis_data/all_objects.df', 'wb')
    pickle.dump(all_objects_df, f)
    f.close()

    end = datetime.datetime.now()
    print('Saved', end - start)

def get_lines_per_code_cell(cell_info_code_df):
    # 12.5 minutes
    start = datetime.datetime.now()

    lines_per_code_cell = [
        row['lines_of_code'] / row['code'] 
        for i, row in cell_info_code_df.iterrows()
        if row['code'] != 0
    ]

    end = datetime.datetime.now()
    print('Calculated', end - start)
    
    # 0.2 seconds
    start = datetime.datetime.now()

    f = open('analysis_data/lines_per_code_cell.list', 'wb')
    pickle.dump(lines_per_code_cell, f)
    f.close()

    end = datetime.datetime.now()
    print('Saved',end - start)
    
def get_function_use(cell_info_code_df):
    '''
    Get all function calls from a python file

    The MIT License (MIT)
    Copyright (c) 2016 Suhas S G <jargnar@gmail.com>
    '''
    class FuncCallVisitor(ast.NodeVisitor):
        def __init__(self):
            self._name = deque()

        @property
        def name(self):
            return '.'.join(self._name)

        @name.deleter
        def name(self):
            self._name.clear()

        def visit_Name(self, node):
            self._name.appendleft(node.id)

        def visit_Attribute(self, node):
            try:
                self._name.appendleft(node.attr)
                self._name.appendleft(node.value.id)
            except AttributeError:
                self.generic_visit(node)


    def get_func_calls(tree):
        func_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                callvisitor = FuncCallVisitor()
                callvisitor.visit(node.func)
                func_calls.append((callvisitor.name, [type(a) for a in node.args]))

        return func_calls
    
    # 1 hour 45 minutes
    start = datetime.datetime.now()
    function_use = {
        'functions': [],
        'parameters': [],
        'file': []
    }
    unprocessed = 0
    for i, row in cell_info_code_df.iterrows():
        nb_funcs = []
        nb_params = []
        try:
            all_code = '\n'.join([c for c in '\n'.join([l for l in row.code_list if str(l) != 'nan']).split('\n') if (c != '' and 
                  str(c) != 'nan' and not c.strip().startswith('%') and not c.strip().startswith('?') and 
                  not c.strip().startswith('!'))])
            tree = ast.parse(all_code)
        except:
            unprocessed += 1
            if i%200000 == 0:
                print(i, datetime.datetime.now() - start)
            continue

        for t in tree.body:
            try:
                for f in get_func_calls(t):
                    if f[0] not in nb_funcs:
                        nb_funcs.append(f[0])
                        nb_params.append(len(f[1]))
            except:
                unprocessed += 1
                if i%200000 == 0:
                    print(i, datetime.datetime.now() - start)
                continue

        function_use['functions'].append(nb_funcs)
        function_use['parameters'].append(nb_params)
        function_use['file'].append(row['file'])

        if i%200000 == 0:
            print(i, datetime.datetime.now() - start)

    end = datetime.datetime.now()
    print('Gone through for function uses', end - start)
                                     
    function_use_df = pd.DataFrame(function_use)
    # 48 seconds
    start = datetime.datetime.now()

    f = open('analysis_data/nb_function_use.df', 'wb')
    pickle.dump(function_use_df, f)
    f.close()

    end = datetime.datetime.now()
    print('Saved', end - start)

    
def get_function_defs(cell_info_code_df):
    start = datetime.datetime.now()
    unprocessed = 0
    function_defs = {
        'function': [],
        'parameters':[],
        'file': []
    }

    for i, row in cell_info_code_df.iterrows():
        try:
            all_code = '\n'.join([c for c in '\n'.join([l for l in row.code_list if str(l) != 'nan']).split('\n') if (c != '' and 
                  str(c) != 'nan' and not c.strip().startswith('%') and not c.strip().startswith('?') and 
                  not c.strip().startswith('!'))])
            tree = ast.parse(all_code)
        except:
            unprocessed += 1
            if i%200000 == 0:
                print(i, datetime.datetime.now() - start)
            continue

        for t in tree.body:
            if type(t) == ast.FunctionDef:
                name = t.name
                num_args = 0
                for a in ast.walk(t.args):
                    if type(a) == ast.arg:
                        num_args += 1
                function_defs['function'].append(name)
                function_defs['parameters'].append(num_args)
                function_defs['file'].append(row.file)

            elif type(t) == ast.ClassDef:
                name = t.name
                num_args = 0
                for b in t.body:
                    if type(b) == ast.FunctionDef and b.name == '__init__':
                        for a in ast.walk(b.args):
                            if type(a) == ast.arg and a.arg != 'self':
                                num_args += 1
                    elif type(b) == ast.FunctionDef:
                        name_b = name+"."+b.name
                        num_args_b = 0
                        for a in ast.walk(b.args):
                            if type(a) == ast.arg and a.arg != 'self':
                                num_args_b += 1
                        function_defs['function'].append(name_b)
                        function_defs['parameters'].append(num_args_b)
                        function_defs['file'].append(row.file)

                function_defs['function'].append(name) 
                function_defs['parameters'].append(num_args)
                function_defs['file'].append(row.file)
        if i%200000 == 0:
            print(i, datetime.datetime.now() - start)

    end = datetime.datetime.now()
    print('Through cell_info_code for functions', end - start) 
    
    start = datetime.datetime.now()
    function_defs_df = pd.DataFrame(function_defs)
    f = open('analysis_data/function_defs.df', 'wb')
    pickle.dump(function_defs_df, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved', end - start)
    
def add_user_funcs():

    notebooks = load_data.load_notebooks()
    cell_stats_df = load_data.load_cell_stats()
    cell_types_df = load_data.load_cell_types()
    function_defs_df = load_data.load_function_defs()
    function_use_df = load_data.load_function_use()
    
    print('grouping...')
    start = datetime.datetime.now()
    function_defs_nb_df = function_defs_df.groupby('file')['function'].aggregate(lambda x: list(x)).reset_index().rename(columns={'function':'function_defs'})
    end = datetime.datetime.now()
    print('...grouped', end - start)
    
    print('merging...')
    start = datetime.datetime.now()
    functions_df = function_use_df.merge(function_defs_nb_df, on = 'file', how = 'left')
    functions_df.function_defs.loc[functions_df.function_defs.isna()] = [[]]*sum(functions_df.function_defs.isna())
    end = datetime.datetime.now()
    print('...merged', end - start)
    
    start = datetime.datetime.now()
    all_def = []
    all_not = []
    for i, row in functions_df.iterrows():
        def_uses = [f for f in row.functions if f in row.function_defs]
        not_uses = [f for f in row.functions if f not in row.function_defs]
        
        all_def.append(def_uses)
        all_not.append(not_uses)

        if i%100000 == 0 or i == 100 or i == 1000:
            print(i, datetime.datetime.now() - start)

    end = datetime.datetime.now()
    print(end - start)

    function_use_df['user_def'] = all_def
    function_use_df['not_user_def'] = all_not
    t = datetime.datetime.now()
    print('Added to df', t - end)


    f = open('analysis_data/nb_function_use.df', 'wb')
    pickle.dump(function_use_df, f)
    f.close()
    print('Saved', datetime.datetime.now() - t)
    print('DONE')

    
def get_errors():
    df_chunks = pd.read_csv(
        'data_final/cells_final.csv', 
        header = 0, usecols = ['file','num_error','error_names','cell_id'], 
        chunksize=10000
    )
    
    # 25 minutes
    start = datetime.datetime.now()
    error_dfs = []
    i = 0
    for chunk in df_chunks:
        try:
            load_data.string_to_list(chunk, 'error_names')
            error_dfs.append(
                chunk.groupby('file')['num_error'].aggregate(['sum','count']).reset_index().merge(
                    chunk.groupby('file')['error_names'].aggregate(load_data.flatten).reset_index(),
                    on = 'file'
                )
            )
        except Exception:
            print(i, type(chunk))
        if i%1000 == 0:
            print(i, datetime.datetime.now() - start)
        i+=1
    end = datetime.datetime.now()
    print('Chunks', end - start)
    
    error_df = pd.concat(error_dfs, sort = False).reset_index(drop=True)
    
    start = datetime.datetime.now()

    error_df = error_df.groupby('file')['count'].sum().reset_index().merge(
                    error_df.groupby('file')['error_names'].aggregate(load_data.flatten).reset_index(),
                    on = 'file'
                )

    end = datetime.datetime.now()
    print('Combined', end - start)
    
    # 5 seconds
    start = datetime.datetime.now()

    f = open('analysis_data/error.df', 'wb')
    pickle.dump(error_df, f)
    f.close

    end = datetime.datetime.now()
    print('Saved', end - start)
    

def get_vis_uses_nb(nb_imports_code_df):
    notebooks = pd.read_csv('data_final/notebooks_final.csv')
    repos = pd.read_csv('data_final/repos_final.csv')
    DATE_CHOICE = 'pushed_at'
    
    vis = ['matplotlib','altair','seaborn',
           'ggplot','bokeh','pygal','plotly',
           'geoplotlib','gleam','missingno',
           'leather']
    
    start = datetime.datetime.now()
    for v in vis:
        nb_imports_code_df[v] = [v in 
             [i[0].split('.')[0] for i in imports]
             for imports in nb_imports_code_df.imports
        ]
        print(v, datetime.datetime.now() - start)
    end = datetime.datetime.now()
    print('Got uses', end - start)
    
    start = datetime.datetime.now()
    notebooks = notebooks.merge(
    nb_imports_code_df[['file'] + vis], on = 'file'
        ).merge(repos[['repo_id',DATE_CHOICE]], on = 'repo_id')
    notebooks[DATE_CHOICE] = pd.to_datetime(notebooks[DATE_CHOICE])
    notebooks['year'] = [c.year for c in notebooks[DATE_CHOICE]]
    notebooks['month'] = [c.month for c in notebooks[DATE_CHOICE]]
    end = datetime.datetime.now()
    print('Added to nbs', end - start)
    
    start = datetime.datetime.now()
    f = open('analysis_data/notebooks_vis.df', 'wb')
    pickle.dump(notebooks, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved', end - start)
    
def get_vis_uses(nb_imports_code_df):    
    def get_uses(imports, code):
        vis = ['matplotlib','altair','seaborn',
           'ggplot','bokeh','pygal','plotly',
           'geoplotlib','gleam','missingno',
           'leather']
        uses = []
        for im in imports:
            for v in vis:
                if im[0].startswith(v):
                    # look for im in code
                    for line in '\n'.join([c for c in row.code if type(c) == str]).split('\n'):
                        f = re.findall(
                            '(?<![a-zA-Z._])'+im[1]+'\.[a-zA-Z._]{1,}|(?<![a-zA-Z._])'+im[1]+'\s',
                            str(line)
                        )
                        if len(f) > 0:
                            uses += [use.strip().replace(im[1],im[0]) for use in f]
        return uses
    
    # 2 hours
    start = datetime.datetime.now()

    all_vis_uses = []
    for i, row in nb_imports_code_df.iterrows():
        all_vis_uses += get_uses(
            row.imports, 
            row.code
        )
        if i%100000==0:
            print(i, datetime.datetime.now() - start)

    end = datetime.datetime.now()
    print('Gone through code for visualization uses', end - start)
    
    # 30 minutes
    start = datetime.datetime.now()

    f = open('analysis_data/all_vis_uses.list', 'wb')
    pickle.dump(all_vis_uses, f)
    f.close()

    end = datetime.datetime.now()
    print('Saved', end - start)

def get_framework_uses(nb_imports_code_df):
    # 22 minutes    
    def get_uses(row):
        frameworks = ['tensorflow', 'sklearn', 
                  'keras', 'theano', 'mxnet', 
                  'caffe', 'pytorch', 'cntk']
        uses = {}
        for im in row.imports:
            for f in frameworks:
                if im[0].startswith(f):
                    if f not in uses:
                        uses[f] = set([])
                    as_alias = im[1]
                    for line in '\n'.join([c for c in row.code if type(c) == str]).split('\n'):
                        line = line.split('#')[0]
                        if as_alias in line and 'import' not in line:
                            use = as_alias+as_alias.join(line.split(as_alias)[1:])
                            use = re.split('[()\[\]=\s]', use)[0]
                            if use == as_alias or use.startswith(as_alias+'.'):
                                use = use.replace(as_alias, im[0])
                                uses[f].add(use)
        return uses

    framework_uses = {
        'file': [],
        'uses': []
    }

    start = datetime.datetime.now()
    for i, row in nb_imports_code_df[len(framework_uses):].iterrows():
        framework_uses['file'].append(row.file)
        try:
            framework_uses['uses'].append(get_uses(row))
        except:
            framework_uses['uses'].append({})
        if i%100000 == 0:
            print(i, datetime.datetime.now() - start)
    end = datetime.datetime.now()
    print('Gone through code for uses', end - start)
    
    start = datetime.datetime.now()
    framework_uses_df = pd.DataFrame(framework_uses)
    f = open('analysis_data/framework_uses.df','wb')
    pickle.dump(framework_uses_df, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved', end - start)
    
def get_nb_imports():
    df_chunks = pd.read_csv(
        'data_final/cells_final.csv', 
        header = 0, usecols = ['file','imports','cell_id'], 
        chunksize=10000
    )
    
    def string_to_list_of_lists(df, column):
        df[column] = [[] if len(df[column][i]) == 2 else [h.replace("'","").split(', ') 
          for h in df[column][i][2:-2].replace('(','[').replace(')',']').split('], [')] 
          for i in range(len(df))]
        
    def agg_imports(list_of_lists):
        overall = []
        for li in list_of_lists:
            for l in li:
                if type(l) == list and len(l) != 0:
                    overall.append(l)
        return overall

    nb_imports = []
    # 1 hour
    start = datetime.datetime.now()
    i = 0
    for chunk in df_chunks:
        try:
            string_to_list_of_lists(chunk.reset_index(drop=True), 'imports')

            chunk_nb_imports = chunk.groupby('file')['imports']\
                .aggregate(agg_imports).reset_index()

            nb_imports.append(chunk_nb_imports)
        except Exception:
            print(type(chunk))

        if i%1000 == 0:
            print(i, datetime.datetime.now() - start) # prints up to 15000
        i+=1
    end = datetime.datetime.now()
    print('Chunks', end - start)
    
    # 2.5 minutes
    start = datetime.datetime.now()

    nb_imports_df = pd.concat(nb_imports).reset_index(drop=True).groupby('file')['imports'].aggregate(load_data.flatten).reset_index()

    end = datetime.datetime.now()
    print('Combined', end - start)
    
    # 1 minute
    start = datetime.datetime.now()

    f = open('analysis_data/nb_imports.df', 'wb')
    pickle.dump(nb_imports_df, f)
    f.close()

    end = datetime.datetime.now()
    print('Saved', end - start)

    
def get_cell_output():
    df_chunks = pd.read_csv(
    'data_final/cells_final.csv', 
        header = 0, usecols = [
            'file','num_execute_result',
            'execute_result_keys','num_display_data',
            'num_stream','cell_id'
        ], 
        chunksize=10000
    )
    
    # 27 minutes
    start = datetime.datetime.now()
    output_dfs = []

    def agg_execute_result_keys(list_of_strs):
        vis_count = 0
        for v_output in [
            'application/vnd.vegalite.v2+json', 
            'application/vnd.vegalite.v3+json', 
            'image/png'
        ]:
            vis_count += ' '.join(list_of_strs).count(v_output)
        return vis_count

    i = 0
    for chunk in df_chunks:

        df = chunk.groupby('file')[[
            'num_execute_result','num_display_data','num_stream'
        ]].sum().reset_index().merge(
            chunk.groupby('file')['cell_id'].count().reset_index().rename(
                columns = {'cell_id':'num_cells'}
            ),
            on = 'file'
        ).merge(
            chunk.groupby('file')['execute_result_keys'].aggregate(
                agg_execute_result_keys
            ).reset_index().rename(columns = {'execute_result_keys':'num_vis_out'}),
            on = 'file'
        )

        output_dfs.append(df)

        if i%1000 == 0:
            print(i, datetime.datetime.now() - start)
        i += 1

    end = datetime.datetime.now()
    print('Chunks done', end - start)
    
    # 33 seconds
    start = datetime.datetime.now()
    output_df = pd.concat(output_dfs).groupby('file')[[
        'num_execute_result','num_display_data','num_stream','num_cells','num_vis_out'
    ]].sum().reset_index()
    end = datetime.datetime.now()
    print('Combined', end - start)
    
    # 3 seconds
    start = datetime.datetime.now()
    f = open('analysis_data/cell_output.df', 'wb')
    pickle.dump(output_df, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved',end - start)
    
def get_comments():
    df_chunks = pd.read_csv(
    'data_final/cells_final.csv', 
        header = 0, usecols = [
            'file','num_comments'
        ], 
        chunksize=10000
    )

    # 27 minutes
    start = datetime.datetime.now()
    comments_dfs = []

    i = 0
    for chunk in df_chunks:

        df = chunk.groupby('file')[[
            'num_comments'
        ]].sum().reset_index()

        comments_dfs.append(df)

        if i % 1000 == 0:
            print(i, datetime.datetime.now() - start)
        i += 1


    end = datetime.datetime.now()
    print('Chunks done', end - start)
    
    # 33 seconds
    start = datetime.datetime.now()
    comments_df = pd.concat(comments_dfs).groupby('file')[[
        'num_comments'
    ]].sum().reset_index()
    end = datetime.datetime.now()
    print('Combined', end - start)

    # 3 seconds
    start = datetime.datetime.now()
    with open('analysis_data/comments.df', 'wb') as f:
        pickle.dump(comments_df, f)
    end = datetime.datetime.now()
    print('Saved',end - start)
    
def get_cell_stats():
    df_chunks = pd.read_csv(
        'data_final/cells_final.csv', 
        header = 0, 
        usecols = ['file','cell_id','lines_of_code','num_words'], 
        chunksize=10000
    )
    
    start = datetime.datetime.now()

    i = 0
    stat_dfs = []
    for chunk in df_chunks:
        chunk = chunk[~chunk.cell_id.isna()].reset_index(drop=True)

        stats = chunk.groupby('file')[
            "file", "lines_of_code", "num_words"
        ].sum().reset_index().merge(
            chunk.groupby('file')['cell_id'].count().reset_index().rename(
                columns = {'cell_id':'num_cells'}
            ),
            on = 'file'
        )

        stat_dfs.append(pd.DataFrame(stats))

        if i%1000 == 0:
            print(i, datetime.datetime.now() - start)
        i+=1
    
    end = datetime.datetime.now()
    print('Chunks done', end - start)
    
    start = datetime.datetime.now()
    stat_df = pd.concat(stat_dfs).groupby('file')[
        'file','lines_of_code','num_words','num_cells'
    ].sum().reset_index()
    end = datetime.datetime.now()
    print('Combined', end - start)
    
    start = datetime.datetime.now()
    f = open('analysis_data/cell_stats.df', 'wb')
    pickle.dump(stat_df, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved', end - start)
    
    
def get_cell_order():
    df_chunks = pd.read_csv(
        'data_final/cells_final.csv', 
        header = 0, usecols = ['file','execution_count','cell_type','cell_id'], 
        chunksize=10000
    )
    
    execution_dfs = []
    i = 0
    
    # 40 min
    start = datetime.datetime.now()

    i = len(execution_dfs)
    for chunk in df_chunks:
        chunk = chunk[~chunk.cell_id.isna()].reset_index(drop=True)
        chunk_data = {
            'file': [],
            'execution': [],
            'cell_types': [],
        }
        file = ''
        chunk_data = chunk.groupby('file')[[
            'execution_count','cell_type'
        ]].aggregate(lambda x: list(x))

        execution_dfs.append(pd.DataFrame(chunk_data))

        if i%1000 == 0:
            print(i, datetime.datetime.now() - start)
        i+=1

    end = datetime.datetime.now()
    print('Chunks done', end - start)
    
    # 8 minutes
    start = datetime.datetime.now()

    cell_order_df = pd.concat([e.reset_index() for e in execution_dfs], sort = False)
    cell_order_df = cell_order_df.groupby('file')[['execution_count','cell_type']].aggregate(load_data.flatten).reset_index()

    end = datetime.datetime.now()
    print('Combined', end - start)
    
    start = datetime.datetime.now()

    cell_order_df['in_order'] = [
        is_monotonic_inc(order) 
        for order in cell_order_df.execution_count
    ]
    
    end = datetime.datetime.now()
    print('Added order', end - start)
    
    start = datetime.datetime.now()
    f = open('analysis_data/cell_order.df', 'wb')
    pickle.dump(cell_order_df, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved final', end - start)

def is_monotonic_inc(int_list):
    current = -1
    for i in int_list:
        try:
            i = float(i)
        except:
            continue
        if i != math.nan and i < current:
            return False
        if i != math.nan:
            current = i
    return True
    
def get_cell_types():
    df_chunks = pd.read_csv(
        'data_final/cells_final.csv', 
        header = 0, usecols = ['file','cell_id','cell_type'], 
        chunksize=10000
    )
    
    # 13 minutes
    start = datetime.datetime.now()
    cell_type_dfs = []
    i = 0
    for chunk in df_chunks:
        chunk = chunk[~chunk.cell_id.isna()].reset_index(drop=True)

        chunk_cell_types = chunk.groupby(['file', 'cell_type'])['cell_id'].count().reset_index()
        c = chunk_cell_types.pivot(
            index = 'file', columns = 'cell_type', values = 'cell_id'
        ).reset_index().fillna(0)

        if len(c) > 0:
            if 'heading' in c.columns and 'markdown' in c.columns:
                c['markdown'] = c['markdown'] + c['heading']
            elif 'heading' in c.columns:
                c['markdown'] = c['header']

            if 'markdown' not in c.columns:
                c['markdown'] = [0]*len(c)
            if 'heading' not in c.columns:
                c['heading'] = [0]*len(c)
            if 'code' not in c.columns:
                c['code'] = [0]*len(c)

            c = c[['file','code','markdown']]

            cell_type_dfs.append(c)

        if i%1000 == 0:
            print(i, datetime.datetime.now() - start)
        i += 1

    end = datetime.datetime.now()
    print('Chunks done', end - start)
    
    # 25 seconds
    start = datetime.datetime.now()
    cell_types_df = pd.concat(
        cell_type_dfs, sort = False
    ).groupby('file')[['code','markdown']].sum().reset_index()
    end = datetime.datetime.now()
    print('Concatenated', end - start)
    
    # 2 seconds
    start = datetime.datetime.now()
    f = open('analysis_data/cell_types.df', 'wb')
    pickle.dump(cell_types_df, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved progress to pickle', end - start)
    
    # 13 minutes
    start = datetime.datetime.now()

    ratio_mc = []
    for i, row in cell_types_df.iterrows():
        ratio_mc.append(
            row['markdown'] / row['code'] 
            if row['code'] != 0 
            else math.inf 
        )
        if i%100000 == 0:
            print(i, datetime.datetime.now() - start)
    
    cell_types_df['ratio_mc'] = ratio_mc
            
    end = datetime.datetime.now()
    print('Added ratio of markdown to code', end - start)
    
    start = datetime.datetime.now()
    f = open('analysis_data/cell_types.df', 'wb')
    pickle.dump(cell_types_df, f)
    f.close()
    end = datetime.datetime.now()
    print('Saved final to pickle', end - start)

def get_statuses():
    flatten = lambda l: [item for sublist in l for item in sublist]

    def string_to_list_of_lists(df, column):
            df[column] = [[] if len(df[column].iloc[i]) == 2 else [h.replace("'","").split(', ') 
              for h in df[column].iloc[i][2:-2].replace('(','[').replace(')',']').split('], [')] 
              for i in range(len(df))]

    print('opening notebooks')
    n = load_data.load_notebooks()
    n_py = n[n['lang_name'] == 'python']
    n_py3 = n_py[[str(l).startswith('3') for l in n_py['python_version']]]
    print('notebooks final open and python3 row selected')
    print(len(n_py3), 'files')

    print('opening code')
    s = datetime.datetime.now()
    with open('analysis_data/nb_code.df','rb') as f:
        code = pickle.load(f)
    code = code[code.file.isin(n_py3.file)]

    print('opening imports')
    with open('analysis_data/nb_imports.df','rb') as f:
        imports = pickle.load(f)

    code = code.merge(imports, on = 'file')
    print('code and imports opened and combined', datetime.datetime.now() - s)
    print(len(code), 'cells')

    print('aggregating code')
    s = datetime.datetime.now()
    print(len(code), 'files')
    print('code aggregated', datetime.datetime.now() - s)
    
    
    def get_status(file, code, imports, variables, functions):
        status = {
            'file': file,
            'import': 0,
            'variable': 0,
            'function': 0,
            'syntax': True,
            'import_done': [],
            'variables_done': [],
            'functions_done': [],
            'imports_not_used':set([im[1] for im in imports]),
            'variables_not_used':set(variables),
            'functions_not_used':set(functions)
        }

        import_aliases = [im[1] for im in imports]

        try:
            tree = ast.parse(code)
            
            update_status(tree.body, status)
            status['imports_not_used'] = len(status['imports_not_used'])
            status['variables_not_used'] = len(status['variables_not_used'])
            status['functions_not_used'] = len(status['functions_not_used'])
            status.pop('import_done')
            status.pop('variables_done')
            status.pop('functions_done')

            return status
            
        except Exception as e:
            status['syntax'] = False
            status['imports_not_used'] = len(status['imports_not_used'])
            status['variables_not_used'] = len(status['variables_not_used'])
            status['functions_not_used'] = len(status['functions_not_used'])
            status.pop('import_done')
            status.pop('variables_done')
            status.pop('functions_done')
            return status

    def update_status(t_list, status):
        for t in t_list:

            new_vars_this_t = []
            new_funcs_this_t = []
            func_calls_this_t = []

            if type(t) == ast.For:
                target = t.target
                if type(target) == ast.Name:
                    status['variables_done'].append(target.id)
                    new_vars_this_t.append(target.id)
                elif type(target) == ast.Tuple:
                    for w in ast.walk(t):
                        if type(w) == ast.Name:
                            status['variables_done'].append(w.id)
                            new_vars_this_t.append(w.id)
                update_status(t.body, status)

            elif type(t) == ast.While:
                update_status(t.body, status)

            elif type(t) == ast.Assign:        
                for target in t.targets:
                    for w in ast.walk(target):
                        if type(w) == ast.Name and w.id in variables:
                            status['variables_done'].append(w.id)
                            new_vars_this_t.append(w.id)

            elif type(t) == ast.With:
                for item in t.items:
                    if type(item) == ast.withitem and item.optional_vars:
                        for w in ast.walk(item.optional_vars):
                            if type(w) == ast.Name and w.id in variables:
                                status['variables_done'].append(w.id)
                                new_vars_this_t.append(w.id)
                update_status(t.body, status)


            elif type(t) in [ast.Import, ast.ImportFrom]:
                for w in ast.walk(t):
                    if type(w) == ast.alias:
                        if w.asname == None:
                            status['import_done'].append(w.name)
                        else:
                            status['import_done'].append(w.asname)

            else:
                for w in ast.walk(t):
                    if type(w) == ast.FunctionDef:
                        new_funcs_this_t.append(w.name)
                    elif type(w) == ast.Call:
                        for func_node in ast.walk(w):
                            if w.func and type(w.func) == ast.Name and w.func.id in functions:
                                func_calls_this_t.append(w.func.id)
                                if w.func.id in status['functions_not_used']:
                                    status['functions_not_used'].remove(w.func.id)
                            elif w.func and type(w.func) == ast.Attribute and w.func.attr in functions:
                                func_calls_this_t.append(w.func.attr)
                                if w.func.attr in status['functions_not_used']:
                                    status['functions_not_used'].remove(w.func.attr)

                    elif type(w) == ast.Name:
                        if w.id in status['imports_not_used']:
                            status['imports_not_used'].remove(w.id)
                            if w.id not in status['import_done']:
                                status['import'] += 1
                        elif (w.id in status['variables_not_used'] and 
                              w.id not in new_vars_this_t and
                              type(t) not in [ast.FunctionDef, ast.ClassDef]
                        ):
                            status['variables_not_used'].remove(w.id)
                            if w.id not in status['variables_done']:
                                status['variable'] += 1

                status['functions_done'] += new_funcs_this_t
                if len(set(func_calls_this_t) - set(status['functions_done'])) > 0:
                    status['function'] += 1

    def get_all_vars(code):
        variables = set([])

        try:
            tree = ast.parse(code)
        except Exception as e:
            return variables

        for t in tree.body:
            if type(t) == ast.Assign:        
                for target in t.targets:
                    for w in ast.walk(target):
                        if type(w) == ast.Name:
                            variables.add(w.id)

        return variables

    def get_user_funcs(code):
        funcs = []

        try:
            tree = ast.parse(code)
        except Exception as e:
            return variables

        for t in tree.body:
            if type(t) == ast.FunctionDef:        
                funcs.append(t.name)

            if type(t) == ast.Assign and type(t.value) == ast.Lambda:
                for target in t.targets:
                    try:
                        funcs.append(target.id)
                    except:
                        pass

            if type(t) == ast.ClassDef:
                class_name = t.name
                funcs.append(class_name)
                for w in ast.walk(t):
                    if type(w) == ast.FunctionDef:
                        function_name = w.name
                        funcs.append(class_name +'.' + function_name)

        return funcs

    print('getting statuses')
    start = datetime.datetime.now()
    statuses = []
    for i, row in code.iterrows():
        row_code = '\n'.join([c for c in '\n'.join([l for l in row.code if str(l) != 'nan']).split('\n') if (c != '' and 
                  str(c) != 'nan' and not c.strip().startswith('%') and not c.strip().startswith('?') and 
                  not c.strip().startswith('!'))])

        variables = get_all_vars(row_code)
        functions = get_user_funcs(row_code)
        statuses.append(get_status(row.file, row_code, row.imports, variables, functions))
        if i%10000 == 0:
            print(i, datetime.datetime.now() - start)
    end = datetime.datetime.now()
    print(end - start)

    statuses_df = pd.DataFrame(statuses)

    print('saving file')
    s = datetime.datetime.now()
    f = open('analysis_data/statuses.df', 'wb')
    pickle.dump(statuses_df, f)
    f.close()
    print('saved', datetime.datetime.now() - s)

    print('Done!')

if __name__ == '__main__':
    main()
