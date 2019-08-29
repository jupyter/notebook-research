"""
Script to processes basic data from all query 
files to notebooks1.csv. After notebooks1.csv is created, files 
can be downloaded with download.py.
"""

import time
import os
import datetime
import json
import sys

import argparse
import requests
import pandas as pd

from consts import (
    URL,
    COUNT_TRIGGER,
    BREAK,
    JSON_PATH,
    PATH,
    HEADERS,
    TOKENS,
    NUM_WORKERS,
    s3
)

from funcs import (
    debug_print,
    write_to_log,
    df_to_s3,
    s3_to_df,
    list_s3_dir
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update", action="store_const", 
        dest="updating", const=True, default=False, 
        help=(
            "Search notebooks that have been added "
            + "or updated since last search, along with new "
            + "notebooks"
        )
    )
    parser.add_argument(
        "--local", action="store_const", 
        dest="local", const=True, default=False, 
        help="Saves output locally instead of in S3."
    )
    parser.add_argument(
        "--needed", metavar = "num_needed", type = int
    )
    args = parser.parse_args()
    updating = args.updating
    local = args.local
    num_needed = args.needed


    if num_needed == None:
        print('test')
        num_needed = 0
        for i in range(NUM_WORKERS):
            try:
                with open('num_needed_{0}.save'.format(i),'r') as f:
                    num_needed += int(f.readlines()[0])
            except:
                print("Parallelize_query.py was not completed.")
                print("Please complete query and try again.")
                sys.exit(0)
    

    clean_metadata(num_needed, updating, local)

    debug_print(
        "Notebooks1, Owners1, and Repos1 were created and saved. "
    )

def clean_metadata(num_needed, updating, local):
    """ 
    Extract information from metadata JSON files and save to CSVs. 
    Equivalent to Adam's 1_nb_metadata_cleaning.ipynb.
    """

    try:
        if local:
            pass
        else:
            notebooks_done = s3_to_df("csv/notebooks1.csv")
            owners_done = s3_to_df("csv/owners1.csv")
            repos_done = s3_to_df("csv/repos1.csv")

        notebook_files_done = set(notebooks_done.file)
        owner_ids_done = set(owners_done.owner_id)
        repo_ids_done = set(repos_done.repo_id)

        print('Metadata already processed for {0} notebooks, {1} owners, and {2} repos.'.format(
            len(notebook_files_done),
            len(owner_ids_done),
            len(repo_ids_done)
        ))
        

    except:
        notebook_files_done = []
        owner_ids_done = []
        repo_ids_done = []
        
        print("Metadata not processed for any files.")

    # Get all query files.
    if local:
        nb_search_files = os.listdir(JSON_PATH)
    else:
        nb_search_files = list_s3_dir('json/')
   
    # Sort query files by size then by page number.
    nb_search_files = sorted(
        nb_search_files, 
        key = lambda x: (
            int(x.split("_")[2].split("..")[0]),
            int(x.split("_")[3][1:].split(".")[0])
        )
    )

    debug_print("We have {0} query files.".format(len(nb_search_files)))

    notebooks = {}
    repos = {}
    owners = {}
    
    for j, json_file_name in enumerate(nb_search_files):
        # Keep track of progress.
        if (j+1) % COUNT_TRIGGER/100 == 0 or j+1 == len(nb_search_files):
            debug_print("{0} / {1} data files processed".format(
                j+1, len(nb_search_files)
            ))
        
        file_components = json_file_name.replace(".json","").split("_")
        filesize = file_components[2]
        query_page = int(file_components[3][1:])
                
        if local:
            with open(JSON_PATH+json_file_name, "r") as json_file:
                # Parse file name to get size and query page.
                file_dict = json.load(json_file)
        else:
            obj = s3.Object(
                "notebook-research",
                "json/{0}".format(json_file_name)
            )
            file_dict = json.loads(obj.get()["Body"].read().decode("UTF-8"))

        # Report missing data.
        if "incomplete_results" in file_dict:
            if file_dict["incomplete_results"] == True:
                msg = "{0} has incomplete results".format(json_file_name)
                write_to_log("../logs/nb_metadata_cleaning_log.txt", msg)
        
        days_since = file_dict["days_since"]
        if "items" in file_dict:
            if len(file_dict["items"]) == 0:
                msg = "{0} has 0 items".format(json_file_name)
                write_to_log("../logs/nb_metadata_cleaning_log.txt", msg)
            
            else:
                # Save data for each item.
                for i in range(len(file_dict["items"])):
                    item = file_dict["items"][i]
                    item_repo = item["repository"]
                    repo_id = item_repo["id"]
                    owner_id = item_repo["owner"]["id"]
                    
                    # Don"t save forked notebooks.
                    if item_repo["fork"]:
                        continue

                    # Full path is unique for each file.
                    name = "{0}/{1}/{2}".format(
                        item_repo["owner"]["login"], 
                        item_repo["name"], 
                        item["path"]
                    ).replace("/","..")
                
                    if name not in notebook_files_done:
                        notebook = {
                            "file": name,
                            "html_url": item["html_url"],
                            "name" : item["name"],
                            "path": item["path"],
                            "repo_id": repo_id,
                            "owner_id": owner_id,
                            "filesize": filesize,
                            "query_page": query_page,
                            "days_since": days_since
                        }
                        notebooks[name] = notebook

                    if repo_id not in repos and repo_id not in repo_ids_done:
                        repo = {
                            "repo_name": item_repo["name"],
                            "owner_id": owner_id,
                            "repo_description": item_repo["description"],
                            "repo_fork": item_repo["fork"],
                            "repo_html_url": item_repo["html_url"],
                            "repo_private": item_repo["private"],
                        }
                        repos[repo_id] = repo

                    if owner_id not in owners and owner_id not in owner_ids_done:
                        owner = {
                            "owner_html_url": item_repo["owner"]["html_url"],
                            "owner_login": item_repo["owner"]["login"],
                        }
                        owners[owner_id] = owner  

                    # If updating we dont always need the full page.
                    if updating and len(notebooks) == num_needed:
                        break                  
        else:
            msg = "{0} has no items object".format(json_file_name)
            write_to_log("../logs/nb_metadata_cleaning_log.txt", msg) 
    
        if updating and len(notebooks) == num_needed:
            break    

    # Display status
    debug_print(("\nAfter processing all query files, "
                "we have {0} new notebooks.").format(len(notebooks)))
    debug_print("Written by {0} owners.".format(len(owners)))
    debug_print("Held in {0} repositories.".format(len(repos)))

    # Translate dictionaries to DataFrames and save to CSV.
    # Ordered by days since, if duplicates keep the most recent 
    # (i.e. keep last, which was found more days since 1-1-19).
    notebooks_df = pd.DataFrame(notebooks).transpose()\
        .sort_values(by=["days_since","file"]).drop_duplicates(
            subset =["file"], 
            keep="last"
        )
    owners_df = pd.DataFrame(owners).transpose().reset_index().rename(
        columns = {"index":"owner_id"}, index = str
    )
    repos_df = pd.DataFrame(repos).transpose().reset_index().rename(
        columns = {"index":"repo_id"}, index = str
    )

    if local:
        pd.concat([notebooks_df, notebooks_done]).to_csv("{0}/notebooks1.csv".format(PATH), index = False)
        pd.concat([owners_df, owners_done]).to_csv("{0}/owners1.csv".format(PATH), index = False)
        pd.concat([repos_df, repos_done]).to_csv("{0}/repos1.csv".format(PATH), index = False)
    else:
        df_to_s3(pd.concat([notebooks_df, notebooks_done]), "csv/notebooks1.csv")
        df_to_s3(pd.concat([owners_df, owners_done]), "csv/owners1.csv")
        df_to_s3(pd.concat([repos_df, repos_done]), "csv/repos1.csv")

if __name__ == '__main__':
    main()