import pandas as pd
import numpy as np
import pickle
import re
import os
import math
import ast
import load_data
import datetime

def main():
    get_all_processed_code()
    print('\n---\n')
    get_km_orders()
    print('\n---\n')
    get_lda_orders()


def get_all_processed_code():
    print('Starting at', datetime.datetime.now())

    # 4 minutes
    notebooks_temp = load_data.load_notebooks()
    code_temp = load_data.load_code()
    nb_imports_temp = load_data.load_nb_imports()

    notebooks = notebooks_temp.copy()[
        (notebooks_temp.lang_name == 'python')&
        ([str(n).startswith('3') for n in notebooks_temp.lang_version])
    ].reset_index(drop=True)

    print("{0:,} ({1}%) of notebooks were written in Python 3. The remaining {2}% have been removed.".format(
        len(notebooks),
        round(100*len(notebooks)/len(notebooks_temp), 2),
        round(100 - 100*len(notebooks)/len(notebooks_temp), 2)
    ))

    code = code_temp.copy()[code_temp.file.isin(notebooks.file)]
    nb_imports = nb_imports_temp.copy()[nb_imports_temp.file.isin(notebooks.file)]

    del notebooks_temp
    del code_temp
    del nb_imports_temp

    packages = ['pandas','numpy','matplotlib','seaborn','sklearn','keras','tensorflow']

    # 45 seconds
    start = datetime.datetime.now()
    for p in packages:
        if p not in nb_imports.columns:
            nb_imports[p] = [1 if p in [i[0] for i in im] else 0 for im in nb_imports.imports]
            print(p, datetime.datetime.now() - start)
    end = datetime.datetime.now()
    print(end - start)

    nb_imports_data_sci = nb_imports.copy()[
        nb_imports.pandas + nb_imports.numpy + nb_imports.matplotlib + \
        nb_imports.seaborn + nb_imports.sklearn + nb_imports.keras + nb_imports.tensorflow >= 2
    ]

    print("{0}% of python 3 notebooks are data science / machine learning notebooks.".format(
        round(100*len(nb_imports_data_sci) / len(nb_imports), 2)
    ))

    notebooks_data_sci = notebooks.copy()[notebooks.file.isin(nb_imports_data_sci.file)]
    code_data_sci = code.copy()[code.file.isin(nb_imports_data_sci.file)]

    code_data_sci.to_csv('analysis_data/time/code_data_sci.csv', index = False)

    del notebooks
    del code
    del nb_imports

    stop = list('abcdefghijklmnopqrstuvwxyz') # one letter variables

    def process_code(row):
        all_processed_code = []
        for code in row['code']:
            try:
                tree = ast.parse(code)
            except Exception:
                all_processed_code.append('') 
                continue

            processed_code = ''
            if type(code) != str:
                all_processed_code.append(processed_code)
                continue

            num_lines = len(code.splitlines())

            im_done = []
            for im in row['imports']:
                if im[0]+'..'+im[1] not in im_done:
                    try:
                        if im[1] != '*':
                            code = re.sub('(?<![a-zA-Z_])'+im[0]+'(?![a-zA-Z_])',' PLACEHOLDER ', code)
                            code = re.sub('(?<![a-zA-Z_])'+im[1]+'(?![a-zA-Z_])',' '+im[0]+' ', code)
                            code = re.sub(' PLACEHOLDER ',im[0], code)
                    except:
                        pass
                    im_done.append(im[0]+'..'+im[1])


            # Imports have already been counted in ast, don't want to count packages in import statements
            code = re.sub('(from [a-zA-Z_\.]+ import [\(\)a-zA-Z_\.,\s]+( as [a-zA-Z_\.]+)?)|(import [\(\)a-zA-Z_\.,\s]+( as [a-zA-Z_\.]+)?)', ' ', code)

            for t in tree.body:
                if type(t) in [ast.Import, ast.ImportFrom]:
                    processed_code += ' <import>'
                elif type(t) == ast.FunctionDef:
                    processed_code += ' <function_def>'

                for i in ast.walk(t):
                    if type(i) == ast.Name and i.id.split('.')[0] not in [im[0] for im in row['imports']]:
                        code = re.sub('(?<![a-zA-Z_])'+i.id+'(?![a-zA-Z_])', ' ', code)

            code = re.sub('(?<![a-zA-Z_])".+"(?![a-zA-Z_])',' ', code)
            code = re.sub("(?<![a-zA-Z_])'.+'(?![a-zA-Z_])",' ', code)

            logic = ['<=','>=','!=','<','>','==','&','|']
            for l in logic:
                code = code.replace(l, ' PLACEHOLDER ')
            code = code.replace(' PLACEHOLDER ', ' <logic> ')
            logic_words = ['if','else','elif','or','and','not','True','False']
            for l in logic_words:
                code = re.sub('(?<![a-zA-Z_])'+l+'(?![a-zA-Z_])',' <logic> ', code)

            operations = ['**','//','*=','/=','+=','-=','*','/','-','+','%']
            for o in operations:
                code = code.replace(o, ' ')

            loops = ['for','while','in','range','iterrows','break','continue']
            for l in loops:
                code = re.sub('(?<![a-zA-Z_])'+l+'(?![a-zA-Z_])',' <loop> ', code)

            symbols = ['^%()']    
            for sym in symbols:
                code = code.replace(sym, ' '+sym+' ')

            for f in ['tensorflow', 'sklearn', 
                    'keras', 'theano', 'mxnet', 
                    'caffe', 'pytorch', 'cntk','torch']:
                code = re.sub('(?<![a-zA-Z_])'+f+'(?![a-zA-Z_])', ' <framework> ', code)


            for v in ['matplotlib','altair','seaborn',
                   'ggplot','bokeh','pygal','plotly',
                   'geoplotlib','gleam','missingno',
                   'leather']:
                code = re.sub('(?<![a-zA-Z_])'+v+'(?![a-zA-Z_])', ' <visualization> ', code)

            updated_code = []
            for line in code.splitlines():
                updated_code.append(line.split('#')[0])
            code = '\n'.join(updated_code)

            code = re.sub('(?<![a-zA-Z_])[0-9.]{1,}(?![a-zA-Z_])',' ', code)

            remove = [':',',','(',')','=','[',']','"',"'",'\\','$','{','}',';','.','print','self']
            for r in remove:
                code = code.replace(r,' ')

            code = re.sub('\s+',' ',code).strip()

            processed_code = processed_code + ' ' + ' '.join([c for c in code.split(' ') if len(c) > 1])

            all_processed_code.append(processed_code)

        return all_processed_code



    code_imports_data_sci = code_data_sci.merge(nb_imports_data_sci[['file','imports']], on = 'file')

    del code_data_sci
    del nb_imports_data_sci

    if 'all_processed_code.list' in os.listdir('analysis_data/time'):
        with open(':/time/all_processed_code.list','rb') as f:
                all_processed_code = pickle.load(f)
        print("Opened processed code for {0}/{1} notebooks.".format(len(all_processed_code), len(code_imports_data_sci)))

    else:
        all_processed_code = []

    start = datetime.datetime.now()
    for i, row in code_imports_data_sci[len(all_processed_code):].iterrows():
        try:
            all_processed_code.append(process_code(row))

        except Exception as e:
            print(e)
            all_processed_code.append([])
            continue

        if i%1000 == 0:
            print(i, datetime.datetime.now() - start)
        if i%10000 == 0:
            with open('analysis_data/time/all_processed_code.list','wb') as f:
                pickle.dump(all_processed_code, f)

    with open('analysis_data/time/all_processed_code.list','wb') as f:
        pickle.dump(all_processed_code, f)
    end = datetime.datetime.now()
    print(end - start)

def get_lda_orders():
    
    all_processed_code = load_data.load_all_processed_code()
    print("Processed code for {0} notebooks has been opened.".format(len(all_processed_code)))
    
    print('Opening LDA Model')
    with open('analysis_data/time/lda8.model','rb') as f:
        lda_model = pickle.load(f)
    with open("analysis_data/time/lda_tfidf_corpus","rb") as f:
        corpus_tfidf = pickle.load(f)
    with open("analysis_data/time/lda_dict","rb") as f:
        dictionary = pickle.load(f)
    print("Found and opened")
    
    num_to_group = {
        0: 'Data',
        1: 'Visualization',
        2: '',
        3: '',
        4: 'Imports',
        5: 'Data',
        6: 'Machine Learning',
        7: 'Logic & Loops'
    }
    
    lda_distrib = {
        'place': [],
        'task': [],
        'weight': []
     }
    lda_total = []
    num_inprogress = 0
            

    i = 0
    start = datetime.datetime.now()
    print("Starting")
    for processed_code in all_processed_code[i:]:
        j = 0
        for cell in processed_code:
            place = round(j/len(processed_code), 1)
            scores = sorted(lda_model[dictionary.doc2bow(cell.strip().split(' '))], key = lambda x: -1*x[1])
            for score in scores:
                lda_distrib['place'].append(place)
                lda_distrib['task'].append(num_to_group[score[0]])
                lda_distrib['weight'].append(score[1])
            lda_total.append(place)
            j += 1
        if i%10000 == 0:
            print('{0}/{1}'.format(i, len(all_processed_code)),  datetime.datetime.now() - start)
        if i%100000 == 0:
            print("saving...", end = '')
            distrib_df = pd.DataFrame(lda_distrib).groupby(
                ['place','task']
            )['weight'].sum().reset_index().merge(
                pd.Series(lda_total).value_counts().reset_index().rename(
                    columns = {'index':'place', 0:'total'}
                ), 
                on = 'place'
            ).rename(columns = {'weight':'count'})

            distrib_df['prop'] = distrib_df['count']/distrib_df['total']
            try:
                distrib_df.to_csv('analysis_data/time/lda_distrib_saving2.csv', index = False)
                os.rename('analysis_data/time/lda_distrib_saving2.csv','analysis_data/time/lda_distrib2.csv')
            except Exception as e:
                print(e)
                break
                
            print('...saved')
        i += 1
    end = datetime.datetime.now()
    print(end - start)
        
    print("saving...", end = '')
    start = datetime.datetime.now()
    distrib_df = pd.DataFrame(lda_distrib).groupby(
        ['place','task']
    )['weight'].sum().reset_index().merge(
        pd.Series(lda_total).value_counts().reset_index().rename(
            columns = {'index':'place', 0:'total'}
        ), 
        on = 'place'
    ).rename(columns = {'weight':'count'})

    distrib_df['prop'] = distrib_df['count']/distrib_df['total']
    try:
        distrib_df.to_csv('analysis_data/time/lda_distrib_saving2.csv', index = False)
        os.rename('analysis_data/time/lda_distrib_saving2.csv','analysis_data/time/lda_distrib2.csv')
    except Exception as e:
        print(e)

    end = datetime.datetime.now()
    print("...saved", end - start)
    
def get_km_orders():
    if 'all_processed_code.list' not in os.listdir('analysis_data/time'):
         time.get_all_processed_code()

    all_processed_code = load_data.load_all_processed_code()
    print("Processed code for {0} notebooks has been opened.".format(len(all_processed_code)))
    
    with open('analysis_data/time/km.model','rb') as f:
        kmeans = pickle.load(f)
    with open('analysis_data/time/km.vectorizer','rb') as f:
        vectorizer = pickle.load(f)
    print("K Means model opened")
        
        
    # 6 hours
    num_to_group = {
        -1: '',
        0: 'Data',
        1: 'Logic & Loops',
        2: 'Other',
        3: 'Visualization',
        4: 'Imports',
        5: 'Machine Learning',
        6: ''
    }


    if "kmeans_orders.list" in os.listdir("analysis_data/time"):
        print('Opening K Means Orders')
        with open("analysis_data/time/kmeans_orders.list","rb") as f:
            kmeans_orders = pickle.load(f)
        print("Found and opened")
    else:

        if "kmeans_orders_inprogress.list" in os.listdir("analysis_data/time"):
            with open("analysis_data/time/kmeans_orders_inprogress.list","rb") as f:
                kmeans_orders = pickle.load(f)
            print("Found and opened for {0} notebooks.".format(len(kmeans_orders)))
        else:
            kmeans_orders = []

        i = len(kmeans_orders)
        start = datetime.datetime.now()
        for c in all_processed_code[i:]:
            kmeans_orders.append([
                num_to_group[i] for i in [kmeans.predict(
                vectorizer.transform([p])
                )[0] if p != '' and type(p) == str else -1 for p in c]
            ])
            if i%10000 == 0:
                print('{0}/{1}'.format(i, len(all_processed_code)), datetime.datetime.now() - start)
                with open("analysis_data/time/kmeans_orders_inprogress.list","wb") as f:
                    pickle.dump(kmeans_orders, f)
            i += 1
        print()
        end = datetime.datetime.now()
        print(end - start)

        start = datetime.datetime.now()
        with open("analysis_data/time/kmeans_orders.list","wb") as f:
            pickle.dump(kmeans_orders, f)
        end = datetime.datetime.now()
        print("\nSaved", end - start)
        
if __name__ == '__main__':
    main()