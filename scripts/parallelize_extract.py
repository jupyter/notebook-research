import sys
import argparse
import os
import time

import pandas as pd
import numpy as np
import pickle

from consts import PATH, s3

from funcs import s3_to_df, df_to_s3, list_s3_dir

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_const", 
        dest="local", const=True, default=False, 
        help="Stores results locally instead of using S3."
    )
    args = parser.parse_args()
    local = args.local

    # Open DataFrames.
    if local:
        try:
            notebooks = pd.read_csv('../csv/notebooks1.csv')
            owners = pd.read_csv('../csv/owners1.csv')
            repos = pd.read_csv('../csv/repos1.csv')
        except Exception:
            print("notebooks1.csv, repos1.csv, owners1.csv not found.",
                "Please run query_git.py with --process first and try again.")
            sys.exit(0)
    else:
        try:
            notebooks = s3_to_df("csv/notebooks1.csv")
            owners = s3_to_df("csv/owners1.csv") 
            repos = s3_to_df("csv/repos1.csv")
        except Exception:
            print("notebooks1.csv, repos1.csv, owners1.csv not found.",
                "Please run query_git.py with --process first and try again.")
            sys.exit(0)

    current_csvs = list_s3_dir('csv/')
    subset = []
    for i in range(10):
        subset += [
            'notebooks1_{0}.csv'.format(i),
            'owners1_{0}.csv'.format(i),
            'repos1_{0}.csv'.format(i)
        ]
    if not set(subset).issubset(current_csvs):
        # Randomize notebook distribution among workers.
        print('Subsets not found')
        print(current_csvs)
        # split_csvs(notebooks, repos, owners, local)
    else:
        print('Subsets found, adding to them')
        update_split_csvs(notebooks, repos, owners, local)

    # Format commands.
    extract_commands = [
        ("nohup python3 -u extract_data.py --worker {0}{1}"
        " >> extract_{2}.log &").format(i, ('--local' if local else ''), i) 
        for i in range(10)
    ]
    
    for command in extract_commands:
        print(command)
        os.system(command)
        time.sleep(15)

def update_split_csvs(notebooks, repos, owners, local):
    all_notebook_files = []
    all_notebooks = []
    for i in range(10):
        notebooks_i = s3_to_df('csv/notebooks1_{0}.csv'.format(i))
        all_notebook_files += list(notebooks_i.file)
        all_notebooks.append(notebooks_i)
    
    # Isolate new notebooks and shuffle
    new_notebooks = notebooks[~notebooks.file.isin(all_notebook_files)]
    print('There are {0} new notebooks to distribute.'.format(len(new_notebooks)))
    new_notebooks = new_notebooks.sample(frac = 1).reset_index(drop = True)

    # Split up and add to existing csvs
    partition_new_notebooks = np.array_split(
        new_notebooks,
        10
    )
    for i in range(10):
        new_notebooks_i = partition_new_notebooks[i]
        old_notebooks_i = all_notebooks[i]
        notebooks_i = pd.concat([old_notebooks_i, new_notebooks_i])
        if local:
            notebooks_i.to_csv('../csv/notebooks1b_{0}.csv'.format(i))
        else:
            df_to_s3(notebooks_i, 'csv/notebooks1b_{0}.csv'.format(i))

        repos_i = repos[
            repos.repo_id.isin(notebooks_i.repo_id)
        ].reset_index(drop=True)
        if local:
            repos_i.to_csv('../csv/repos1b_{0}.csv'.format(i))
        else:
            df_to_s3(repos_i, 'csv/repos1b_{0}.csv'.format(i))
        
        owners_i = owners[
            owners.owner_id.isin(notebooks_i.owner_id)
        ].reset_index(drop=True)
        if local:
            owners_i.to_csv('../csv/owners1b_{0}.csv'.format(i))
        else:
            df_to_s3(owners_i, 'csv/owners1b_{0}.csv'.format(i))




def split_csvs(notebooks, repos, owners, local):
    # Shuffle repositories.
    repos = repos.sample(frac=1).reset_index(drop=True)

    # Randomly assign repos and the notbooks/owners that go with them.
    partition_repos = np.array_split(
        repos,
        10
    )
    for i in range(10):
        repos_i = partition_repos[i]
        if local:
            repos_i.to_csv('../csv/repos1b_{0}.csv'.format(i))
        else:
            df_to_s3(repos_i, 'csv/repos1b_{0}.csv'.format(i))
        
        notebooks_i =  notebooks[
            notebooks.repo_id.isin(repos_i["repo_id"])
        ].reset_index(drop=True)
        if local:
            notebooks_i.to_csv('../csv/notebooks1b_{0}.csv'.format(i))
        else:
            df_to_s3(notebooks_i, 'csv/notebooks1b_{0}.csv'.format(i))
        
        owners_i = owners[
            owners.owner_id.isin(repos_i["owner_id"])
        ]
        if local:
            owners_i.to_csv('../csv/owners1b_{0}.csv'.format(i))
        else:
            df_to_s3(owners_i, 'csv/owners1b_{0}.csv'.format(i))


if __name__ == "__main__":
    main()