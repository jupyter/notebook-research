import ipywidgets as widgets
from ipywidgets import interact, interact_manual
from IPython.core.display import Markdown
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import load_data

import pandas as pd

def data():
    # 2 minutes
    notebooks = load_data.load_notebooks()
    repos = load_data.load_repos()
    owners = load_data.load_owners()
    nb_imports = load_data.load_nb_imports()

    repos['pushed_at'] = pd.to_datetime(repos['pushed_at'])
    repos['created_at'] = pd.to_datetime(repos['created_at'])
    repos['updated_at'] = pd.to_datetime(repos['updated_at'])

    # Query was completed July 15, 2019. Times marked as after that are only
    # if the repository was pushed to after we queried
    repos = repos[repos.pushed_at < pd.Timestamp('07-15-2019', tz='UTC')]

    df = notebooks.merge(
        repos[(set(repos.columns) - set(notebooks.columns)).union(set(['repo_id','owner_id']))], 
        on = ['repo_id', 'owner_id']
    ).merge(
        owners, on = 'owner_id'
    ).merge(
        nb_imports[(set(nb_imports.columns) - set(notebooks.columns)).union(set(['file']))], 
        on = 'file'
    ).drop_duplicates(subset = ['file']).reset_index(drop=True)

    df = df[df.lang_name == 'python']


    # 5.5 minutes
    for p in ['matplotlib','altair','seaborn',
           'ggplot','bokeh','pygal','plotly',
           'geoplotlib','gleam','missingno',
           'leather','cntk',
            'sklearn','tensorflow','keras','torch', 'mxnet','pytorch']:
        if p not in df.columns:
            df[p] = [p in [i[0].split('.')[0] for i in im] for im in df.imports]


    # 1 minute
    errors_temp = load_data.load_errors()
    cell_stats_temp = load_data.load_cell_stats()
    cell_order_temp = load_data.load_cell_order()
    output_temp = load_data.load_output()
    statuses_temp = load_data.load_statuses()
    cell_stats_temp = load_data.load_cell_stats()
    collab_status_temp = load_data.load_collab_status()
    special_temp = load_data.load_special()
    framework_uses_temp = load_data.load_framework_uses()
    edu_status_temp = load_data.load_edu_status()

    errors = errors_temp.copy()[errors_temp.file.isin(df.file)]
    cell_stats = cell_stats_temp.copy()[cell_stats_temp.file.isin(df.file)]
    cell_order = cell_order_temp.copy()[cell_order_temp.file.isin(df.file)]
    output = output_temp.copy()[output_temp.file.isin(df.file)]
    statuses = statuses_temp.copy()[statuses_temp.file.isin(df.file)]
    cell_stats = cell_stats_temp.copy()[cell_stats_temp.file.isin(df.file)]
    collab_status = collab_status_temp.copy()[collab_status_temp.repo_id.isin(df.repo_id)]
    special = special_temp.copy()[special_temp.file.isin(df.file)]
    framework_uses = framework_uses_temp.copy()[framework_uses_temp.file.isin(df.file)]
    edu_status = edu_status_temp.copy()[edu_status_temp.repo_id.isin(df.repo_id)]

    errors['num_errors'] = [len(e) for e in errors['error_names']]
    cell_stats['ratio_wl'] = cell_stats['num_words']/cell_stats['lines_of_code']
    cell_stats = cell_stats[cell_stats['ratio_wl'] < math.inf]
    
    summary = df.merge(
            cell_stats[['file', 'lines_of_code','num_words']], on = 'file')[[
            'num_cells','forks_count','open_issues_count',
            'stargazers_count','subscribers_count','watchers_count',
            'lines_of_code','num_words'
        ]].aggregate(['mean','median','min','max'])

    data_frames = {
        'df': df,
        'repos': repos,
        'errors': errors,
        'cell_stats': cell_stats,
        'output': output,
        'statuses': statuses,
        'cell_order': cell_order,
        'cell_stats': cell_stats,
        'collab_status': collab_status,
        'special': special,
        'framework_uses': framework_uses,
        'edu_status': edu_status,
        'summary': summary
    }
    
    return data_frames


def interactive(data_frames):
    query = {
        'date_choice': widgets.Text(),
        'start_date': widgets.Text(),
        'end_date': widgets.Text(),
        'min_stargazers': 0,
        'min_watchers': 0,
        'min_issues': 0,
        'min_forks': 0,
        'uses_ml': False,
        'uses_vis': False,
        'user': True,
        'org': True,
        'edu': True,
        'not_edu': True,
        'google_colab': False
    }


    def get_date_choice(choice = widgets.RadioButtons(
        options = [('Created','created_at'), ('Pushed','pushed_at'), ('Updated','updated_at')],
        value = 'pushed_at',
        description = 'Date choice:',
        style = dict(description_width='initial')
    )):
        query['date_choice'].value = choice

    def get_start_date(date = widgets.DatePicker(
        description='Start date:',
        disabled=False,
        value=data_frames['repos']['pushed_at'].min(),
        style=dict(description_width='initial')
    )):

        query['start_date'].value = str(date)

    def get_end_date(date = widgets.DatePicker(
        description='End date:',
        disabled=False,
        value=data_frames['repos']['pushed_at'].max(),
        style=dict(description_width='initial')
    )):

        query['end_date'].value = str(date)

    def get_min_stargazers(stargazers = widgets.BoundedIntText(
        value = 0,
        min = 0,
        max = data_frames['repos'].stargazers_count.max(),
        step = 1,
        description = 'Minimum stargazers:',
        style=dict(description_width='initial')
    )):
        query['min_stargazers'] = stargazers

    def get_min_watchers(watchers = widgets.BoundedIntText(
        value = 0,
        min = 0,
        max = data_frames['repos'].watchers_count.max(),
        step = 1,
        description = 'Minimum watchers:',
        style=dict(description_width='initial')
    )):
        query['min_watchers'] = watchers

    def get_min_forks(forks = widgets.BoundedIntText(
        value = 0,
        min = 0,
        max = data_frames['repos'].forks_count.max(),
        step = 1,
        description = 'Minimum forks:',
        style=dict(description_width='initial')
    )):
        query['min_forks'] = forks

    def get_min_issues(issues = widgets.BoundedIntText(
        value = 0,
        min = 0,
        max = data_frames['repos'].open_issues_count.max(),
        step = 1,
        description = 'Minimum issues:',
        style=dict(description_width='initial')
    )):
        query['min_issues'] = issues

    def get_ml_use(ml = widgets.Checkbox(
        value = False,
        description = 'Uses a Machine Learning Framework?',
        style=dict(description_width='initial')
    )):
        query['uses_ml'] = ml

    def get_vis_use(vis = widgets.Checkbox(
        value = False,
        description = 'Uses a Visualization Package?',
        style=dict(description_width='initial')
    )):
        query['uses_vis'] = vis

    def get_user(user = widgets.Checkbox(
        value = True, 
        description = 'Individual Users',
        style = dict(description_width='initial')
    )):
        query['user'] = user

    def get_org(org = widgets.Checkbox(
        value = True, 
        description = 'Organizations',
        style = dict(description_width='initial')
    )):
        query['org'] = org

    def get_edu(edu = widgets.Checkbox(
        value = True, 
        description = 'Educational Owners',
        style = dict(description_width='initial')
    )):
        query['edu'] = edu

    def get_not_edu(not_edu = widgets.Checkbox(
        value = True, 
        description = 'Not Educational Owners',
        style = dict(description_width='initial')
    )):
        query['not_edu'] = not_edu

    def get_google_colab(google_colab = widgets.Checkbox(
        value = False,
        description = 'Google Colaboratory',
        style = dict(description_width='initial')
    )):
        query['google_colab'] = google_colab

    display(Markdown('### Notebook Basics'))
    interact(get_date_choice)
    interact(get_start_date)
    interact(get_end_date)

    print()
    display(Markdown('### Collaboration'))
    interact(get_min_stargazers)
    interact(get_min_watchers)
    interact(get_min_forks)
    interact(get_min_issues)

    print()
    display(Markdown('### Owners'))
    interact(get_user)
    interact(get_org)
    interact(get_edu)
    interact(get_not_edu)

    print()
    display(Markdown('### Uses'))
    interact(get_ml_use)
    interact(get_vis_use)

    print()
    display(Markdown('### Platforms'))
    interact(get_google_colab)

    print()
    return query

def subset(data_frames, query):
    print("Subsetting to Python notebooks {0} between {1} and {2}.".format(
        query['date_choice'].value.split('_')[0],
        str(pd.Timestamp(query['start_date'].value, tz='UTC')).split()[0],
        str(pd.Timestamp(query['end_date'].value, tz='UTC')).split()[0]
    ))

    if query['min_stargazers']+query['min_watchers']+query['min_forks']+query['min_issues'] > 0:
        print("Limiting to those in repositories with at least {0} stargazers, {1} watchers, {2} forks, and {3} issues.".format(
            query['min_stargazers'],
            query['min_watchers'],
            query['min_forks'],
            query['min_issues']
        ))


    owner_types = []
    owner_labels = []
    if query['org']:
        owner_types.append('Organization')
        owner_labels.append('organizations')
    if query['user']:
        owner_types.append('User')
        owner_labels.append('individual users')

    edu_types = []
    edu_labels = []
    if query['edu']:
        edu_types.append(True)
        edu_labels.append('educational')
    if query['not_edu']:
        edu_types.append(False)
        edu_labels.append('not educational')

    if len(edu_labels) < 2 or len(owner_labels) < 2:
        print("Only looking at notebooks created by {0} {1}.".format(
            " or ".join(edu_labels),
            " or ".join(owner_labels)
        ))

    uses_labels = []
    if query['uses_ml']:
        uses_labels.append('at least one machine learning framework')
    if query['uses_vis']:
        uses_labels.append('at least one visualization package')

    if len(uses_labels) > 0:
        print("Notebooks considered use {0}.".format(
            ' and '.join(uses_labels) if len(uses_labels) <= 2 else
            ', '.join(uses_labels[:-1]) + ', and ' + uses_labels[-1]
        ))

    if query['google_colab']:
        print("Only looking at notebooks made using Google Colaboratory.")

    df_subset = data_frames['df'][
        (data_frames['df'].lang_name == 'python')&
        (data_frames['df'][query['date_choice'].value] < pd.Timestamp(query['end_date'].value, tz='UTC'))&
        (data_frames['df'][query['date_choice'].value] > pd.Timestamp(query['start_date'].value, tz='UTC'))&
        (data_frames['df'].stargazers_count >= query['min_stargazers'])&
        (data_frames['df'].watchers_count >= query['min_watchers'])&
        (data_frames['df'].forks_count >= query['min_forks'])&
        (data_frames['df'].open_issues_count >= query['min_issues'])&
        (data_frames['df'].type.isin(owner_types))
    ]

    if query['google_colab']:
        df_subset = df_subset[df_subset.google_collab == True]

    if len(edu_types) == 1:
        df_subset = df_subset[(df_subset.repo_id.isin(data_frames['edu_status'][
            data_frames['edu_status'].edu.isin(edu_types)
        ].repo_id))]

    if query['uses_ml']:
        df_subset = df_subset[
            df_subset['tensorflow'] |
            df_subset['keras'] | 
            df_subset['sklearn'] | 
            df_subset['torch'] | 
            df_subset['mxnet'] | 
            df_subset['pytorch'] 
        ]


    if query['uses_vis']:
        df_subset = df_subset[
            df_subset['matplotlib'] |
            df_subset['seaborn'] | 
            df_subset['plotly'] | 
            df_subset['bokeh'] | 
            df_subset['altair']
        ]

    print("{0:,} ({1}%) notebooks fit your criteria.".format(
        len(df_subset),
        round(100*len(df_subset) / len(data_frames['df']), 2)
    ))
    
    errors_subset          = data_frames['errors'].copy()[data_frames['errors'].file.isin(df_subset.file)]
    cell_stats_subset      = data_frames['cell_stats'].copy()[data_frames['cell_stats'].file.isin(df_subset.file)]
    cell_order_subset      = data_frames['cell_order'].copy()[data_frames['cell_order'].file.isin(df_subset.file)]
    output_subset          = data_frames['output'].copy()[data_frames['output'].file.isin(df_subset.file)]
    statuses_subset        = data_frames['statuses'].copy()[data_frames['statuses'].file.isin(df_subset.file)]
    cell_stats_subset      = data_frames['cell_stats'].copy()[data_frames['cell_stats'].file.isin(df_subset.file)]
    collab_status_subset   = data_frames['collab_status'].copy()[data_frames['collab_status'].repo_id.isin(df_subset.repo_id)]
    special_subset         = data_frames['special'].copy()[data_frames['special'].file.isin(df_subset.file)]
    framework_uses_subset  = data_frames['framework_uses'].copy()[data_frames['framework_uses'].file.isin(df_subset.file)]
    edu_status_subset      = data_frames['edu_status'].copy()[data_frames['edu_status'].repo_id.isin(df_subset.repo_id)]
    repos_subset           = data_frames['repos'].copy()[data_frames['repos'].repo_id.isin(df_subset.repo_id)]

    summary_subset = df_subset.merge(
            cell_stats_subset[['file', 'lines_of_code','num_words']], on = 'file')[[
            'num_cells','forks_count','open_issues_count',
            'stargazers_count','subscribers_count','watchers_count',
            'lines_of_code','num_words'
        ]].aggregate(['mean','median','min','max'])
    
    data_frames_sub = {
        'df': df_subset,
        'repos': repos_subset,
        'errors': errors_subset,
        'cell_stats': cell_stats_subset,
        'output': output_subset,
        'statuses': statuses_subset,
        'cell_stats': cell_stats_subset,
        'cell_order': cell_order_subset,
        'collab_status': collab_status_subset,
        'special': special_subset,
        'framework_uses': framework_uses_subset,
        'edu_status': edu_status_subset,
        'summary': summary_subset
    }
    
    return data_frames_sub


def report_comparisons(data_frames_sub, data_frames):
    def compare(func, title):
        print()
        display(Markdown("### {0}".format(title)))
        func(data_frames_sub, 'these')
        display(Markdown("##### Compare to all:"))
        func(data_frames, 'all')

    def compare_plots(func, plot_title):
        fig = plt.figure(figsize = (14, 4))
        plt.subplot(1,2,1)
        func(data_frames_sub, plot_title)
        plt.subplot(1,2,2)
        func(data_frames, 'Compare to All: '+plot_title)
        plt.show()


    ### Summary
    def summary(data, what):
        display(round(data['summary'], 2))
    compare(summary, 'Summary Statistics')

    ### Package Use
    def package_plot(data, title):
        imports = pd.Series(load_data.flatten([
            set([i[0].split('.')[0] for i in im]) for im in data['df'].imports
        ]))
        imports_counts = imports.value_counts().reset_index().rename(
            columns = {'index':'im',0:'num'}
        )
        imports_counts['prop'] = imports_counts['num']/len(data['df'])
        x = imports_counts.im[:20]
        x_pos = np.arange(len(x))
        y = imports_counts.prop[:20]
        plt.bar(x_pos, y, color = 'teal')
        plt.xticks(x_pos, x, rotation = 70)
        plt.title(title)
        plt.ylabel('Proportion of Notebooks')
    display(Markdown('### Package Use'))
    compare_plots(package_plot,'Top 20 Packages Used')


    ### Framework Use
    def framework(data, what):
        print("{0}% of {1} notebooks use at least one framework.".format(
            round(100*sum(data['df'].sklearn|data['df'].tensorflow|\
                          data['df'].keras|data['df'].torch|data['df'].mxnet|\
                          data['df'].pytorch)/len(data['df']),2),
            what
        ))
    def framework_plot(data, title):
        x = ['sklearn','tensorflow','keras','torch', 'mxnet','pytorch']
        x_pos = np.arange(len(x))
        y = []
        for f in x:
            y.append(round(sum(data['df'][f])/len(data['df']), 2))
        plt.bar(x_pos, y, color = 'teal')
        plt.xticks(x_pos, x, rotation = 70)
        plt.title(title)
        plt.ylabel('Proportion of Notebooks')
    compare(framework, 'Framework Use')
    compare_plots(framework_plot, 'Framework Use')


    ### Visualization Package Use
    def visualization(data, what):
        print("{0}% of {1} notebooks use at least one visualization package.".format(
            round(100*sum((data['df'].matplotlib|data['df'].seaborn|
                          data['df'].plotly|data['df'].bokeh|data['df'].altair))/len(data['df']),2),
            what
        ))
    def visualization_plot(data, title):
        x = ['matplotlib','seaborn','plotly', 'bokeh', 'altair']
        x_pos = np.arange(len(x))
        y = []
        for f in x:
            y.append(round(sum(data['df'][f])/len(data['df']), 2))
        plt.bar(x_pos, y, color = 'teal')
        plt.xticks(x_pos, x, rotation = 70)
        plt.title(title)
        plt.ylabel('Proportion of Notebooks')

    compare(visualization, 'Visualization Package Use')
    compare_plots(visualization_plot, 'Visualization Package Use')
    
    ### Errors
    def errors(data, what):
        display(round(
            data['errors']['num_errors'].aggregate(
                ['mean','median','min','max']
            ), 2
        ))
    compare(errors, 'Number of Errors per Notebook')


    ### Markdown
    def markdown(data, what):
        display(round(data['cell_stats']['ratio_wl'].aggregate(['mean','median','min','max']), 2))
    compare(markdown, 'Ratio of Markdown to Code')


    ### Execution
    def execution_plot(data, title):
        status_props = (data['statuses'][data['statuses']['syntax']][['function','import','variable']].sum()/
                  sum(data['statuses']['syntax'])).values
        
        x = ['cells','function definitions','package imports','variable definitions']
        x_pos = np.arange(len(x))
        y = [
            1 - sum(data['cell_order']['in_order'])/len(data['cell_order']),
            status_props[0], status_props[1], status_props[2]
        ]
        plt.bar(x_pos, y, color = 'teal')
        plt.xticks(x_pos, x, rotation = 70)
        plt.title(title)
        plt.ylabel('Proportion of Notebooks\nout of order')
        
    def execution(data, what):
        print("{0}% of {1} notebooks have cells run in order.".format(
            round(100*sum(data['cell_order']['in_order'])/len(data['cell_order']), 2),
            what
        ))

        print("{0}% of {1} notebooks have at least one output, {2}% of which are run in order.".format(
            round(100*len(data['output'][
                    (data['output']['num_execute_result'] + 
                    data['output']['num_display_data'] + 
                    data['output']['num_stream']) > 0
                ])/len(data['output']), 2),
            what,
            round(100*sum(data['cell_order'][
                data['cell_order'].file.isin(data['output'][
                    data['output']['num_execute_result'] + 
                    data['output']['num_display_data'] + 
                    data['output']['num_stream'] > 0
                ].file)]['in_order'])/len(data['cell_order'][
                data['cell_order'].file.isin(data['output'][
                    data['output']['num_execute_result'] + 
                    data['output']['num_display_data'] + 
                    data['output']['num_stream'] > 0
                ].file)]), 2
            ))
        )

        print("\n{0}% of {1} notebooks were able to be parsed with Python AST.".format(
            round(100*sum(data['statuses']['syntax'])/len(data['statuses']), 2),
            what
        ))
        status_props = round(
            100*((data['statuses'][data['statuses']['syntax']][['function','import','variable']].sum()/
                  sum(data['statuses']['syntax']))), 
            2
        ).values
        print("Of these, {0}% had a function used before it was defined, {1}% had a package used before it was imported, and {2}% used a variable before it was defined.".format(
            status_props[0], status_props[1], status_props[2]
        ))
    compare(execution, 'Execution Order')
    compare_plots(execution_plot, 'Execution Order')


    ### Collaboration
    def collaboration_plot(data, title):
        x = ['collaborative','watched','isolated']
        x_pos = np.arange(len(x))
        y = []
        if 'collaborative' in data['collab_status'].collab.values:
            y.append(data['collab_status'].collab.value_counts().collaborative / len(data['collab_status']))
        else:
            y.append(0)
            
        if 'watched' in data['collab_status'].collab.values:
            y.append(data['collab_status'].collab.value_counts().watched / len(data['collab_status']))
        else:
            y.append(0)
            
        if 'isolated' in data['collab_status'].collab.values:
            y.append(data['collab_status'].collab.value_counts().isolated / len(data['collab_status']))
        else:
            y.append(0)
            
        plt.bar(x_pos, y, color = 'teal')
        plt.xticks(x_pos, x, rotation = 70)
        plt.title(title)
        plt.ylabel('Proportion of Repositories')
        
    def collaboration(data, what):
        print("{0}% of {1} repositories are collaborative, containing {2}% of {3} notebooks.".format(
            round(100*data['collab_status'].collab.value_counts().collaborative / len(data['collab_status']),2),
            what,
            round(100*data['df'].merge(data['collab_status'], on = 'repo_id').collab.value_counts()['collaborative'] /
                 len(data['df'].merge(data['collab_status'], on = 'repo_id')), 2),
            what
        ))
        
    compare(collaboration, 'Collaboration')
    compare_plots(collaboration_plot, 'Collaboration')


    ### Educational
    def educational(data, what):
        print("{0}% of {1} repos are educational, holding {2}% of {3} notebooks".format(
            round(100*data['edu_status'].edu.value_counts()[True] / len(data['edu_status']), 2),
            what,
            round((100*data['df'].merge(data['edu_status'], on = 'repo_id').edu.value_counts()[True] / 
                   len(data['df'].merge(data['edu_status'], on = 'repo_id'))), 
                  2
            ),
            what
        ))
    compare(educational, 'Educational Status')