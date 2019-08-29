import os
import sys
import random
import pickle
import argparse
import time

import pandas as pd
import numpy as np
import boto3
from io import BytesIO


from consts import (
    PATH,
    NUM_WORKERS,
    s3, 
    bucket
)

from funcs import (
    debug_print,
    s3_to_df
)

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_const", 
        dest="local", const=True, default=False, 
        help="Saves output locally instead of in S3.")
    args = parser.parse_args()
    local = args.local

    # Open DataFrames.
    try:
        if local:
            notebooks = pd.read_csv("{0}/notebooks1.csv".format(PATH))
            repos = pd.read_csv("{0}/repos1.csv".format(PATH))
            owners = pd.read_csv("{0}/owners1.csv".format(PATH))
        else:
            notebooks = s3_to_df("csv/notebooks1.csv")
            owners = s3_to_df("csv/owners1.csv") 
            repos = s3_to_df("csv/repos1.csv")

        print("notebooks1.csv, repos1.csv, owners1.csv found and opened.")

    except Exception:
        print("notebooks1.csv, repos1.csv, owners1.csv not found.",
              "Please run query_git.py first and try again.")
        sys.exit(0)

    # Randomize notebook distribution among workers.
    already_done = parallelize_download(notebooks, repos, owners, local)
        
    # List files already downloaded, used in download.py
    if not already_done:
        current_files = set([])
        for obj in bucket.objects.filter(Prefix = 'notebooks/'):
            current_files.add(obj.key.split("/")[1])
        
        obj = s3.Object("notebook-research", "current_notebooks.pickle")
        obj.put(Body = bytes(pickle.dumps(current_files)))

        print("saved current files")

    # Format commands.
    download_commands = [
        ("nohup python3 -u download.py --worker {0}{1}"
        " > download_{2}.log &").format(
            i, 
            (' --local' if local else ''),
            i
        ) 
        for i in range(NUM_WORKERS)
    ]

    for command in download_commands:
        print(command)
        os.system(command)
        time.sleep(10)

def parallelize_download(notebooks, repos, owners, local):
    # Open existing partitions if they are present.
    try:
        if local:
            f = open("download_partitions.pickle", "rb")
            partitions = pickle.load(f)
            f.close()

        else:
            partitions = []
            for i in range(NUM_WORKERS):
                obj = s3.Object("notebook-research", "download_partitions_{0}.pickle".format(i))
                partitions.append(pickle.load(BytesIO(obj.get()['Body'].read())))

        print("Paritions opened")

        # List already partitioned notebooks
        notebooks_partitioned = []
        for partition in partitions:
            notebooks_partitioned += list(partition['notebooks']['file'])
        debug_print("{0} notebooks have already been partitioned.".format(len(notebooks_partitioned)))

        # Isolate notebooks not yet partitioned
        notebooks_new = notebooks[~notebooks.file.isin(notebooks_partitioned)]
        if len(notebooks_new) == 0:
            print("All notebooks have already been partitioned.")
            return True

    except Exception as e:
        print(e)
        # All notebooks are new
        notebooks_new = notebooks
        partitions = []
        for i in range(NUM_WORKERS):
            partitions.append({
                "id": i,
                "notebooks": [],
                "repos": [],
                "owners": []
            })

    # Shuffle new notebooks
    notebooks_new = notebooks_new.sample(frac=1).reset_index(drop=True)

    # Randomly assign notebooks and the repos/owners that go with them.
    partition_notebooks = np.array_split(
        notebooks_new,
        NUM_WORKERS
    )
    for i in range(NUM_WORKERS):
        p = partitions[i]

        # Add new notebooks, repos, and owners to partitions
        if len(p["notebooks"]) > 0:
            p["notebooks"] = pd.concat([
                p["notebooks"],             # existing notebooks
                partition_notebooks[i]      # new notebooks
            ])
        else:
            p["notebooks"] = partition_notebooks[i]

        if len(p["repos"]) > 0:
            p["repos"] = pd.concat([
                p["repos"],
                repos[
                    repos.repo_id.isin(partition_notebooks[i]["repo_id"])
                ].reset_index(drop=True)
            ])
        else:
            p["repos"] = repos[
                repos.repo_id.isin(partition_notebooks[i]["repo_id"])
            ].reset_index(drop=True)

        if len(p["owners"]) > 0:
            p["owners"] = pd.concat([
                p["owners"],
                owners[
                    owners.owner_id.isin(partition_notebooks[i]["owner_id"])
                ].reset_index(drop=True)
            ])
        else:
            p["owners"] = owners[
                owners.owner_id.isin(partition_notebooks[i]["owner_id"])
            ].reset_index(drop=True)  

        print('done with', i)

    # Save partition data.
    print('saving...')
    if local:
        f = open("download_partitions.pickle", "wb")
        pickle.dump(partitions, f)
        f.close()
    else:
        for i in range(len(partitions)):
            obj = s3.Object("notebook-research", "download_partitions_{0}.pickle".format(i))
            obj.put(Body = bytes(pickle.dumps(partitions[i])))
    print('...saved')
    return False
    

if __name__ == "__main__":
    main()