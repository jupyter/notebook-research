"""
Script to extract data from downloaded Jupyter notebooks.

Saves important information about notebooks and repos
to CSV files.
"""

import json
import os
import datetime
import re
import sys
import ast

import argparse
import pandas as pd
import numpy as np
import boto3
from io import BytesIO

from consts import (
    PATH,
    BREAK,
    COUNT_TRIGGER,
    TOKENS,
    s3
)

from funcs import (
    write_to_log,
    debug_print,
    df_to_s3,
    s3_to_df,
    list_s3_dir, 
)

S3_PATH = "csv"

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_const", 
        dest="local", const=True, default=False, 
        help="Stores results locally instead of using S3."
    )
    parser.add_argument(
        "--worker", metavar="N", type=int, 
        help=("GITHUB_TOKEN assigned to these sizes (workers "
            "sorted in alphabetical order: {0})."
            ).format(list(TOKENS.keys()))
    )
    args = parser.parse_args()
    local = args.local
    worker = args.worker

    # If running in parallel, mark csv files with the worker number.
    global EXTENSION 
    EXTENSION = '_{0}'.format(worker) if worker != None else ''
    print('EXTENSION', EXTENSION)

    start = datetime.datetime.now()

    # List of saved CSV files.
    if local:
        current_csvs = set(os.listdir(PATH))
    else:
        current_csvs = list_s3_dir(S3_PATH)

    # Open basic data from json files (files created in query_git.py).
    if set([
        "notebooks1{0}.csv".format(EXTENSION),
        "repos1{0}.csv".format(EXTENSION),
        "owners1{0}.csv".format(EXTENSION)
    ]).issubset(current_csvs):
        notebooks1 = get_df("notebooks1{0}.csv".format(EXTENSION),local)
        owners1 = get_df("owners1{0}.csv".format(EXTENSION), local)
        repos1 = get_df("repos1{0}.csv".format(EXTENSION), local)
    else:
        debug_print("Notebooks1, Owners1, and Repos1 were not found.")
        sys.exit(0)

    debug_print("Notebooks1, Owners1, and Repos1 were found and opened."+BREAK)

    ### Add information for repositories and owners. ##################
    save = False
    if not set(["owners2{0}.csv".format(EXTENSION), "repos2{0}.csv".format(EXTENSION)]).issubset(current_csvs):
        owners2, repos2 = update_owners_repos(owners1, repos1, local)
        save = True
    else:
        try:
            owners2_old = get_df("owners2{0}.csv".format(EXTENSION), local)
            repos2_old = get_df("repos2{0}.csv".format(EXTENSION), local)
            debug_print("Found and opened data for {0} owners and {1} repos.".format(
                len(owners2_old), len(repos2_old)
            ))
        except:
            owners2_old = []
            repos2_old = []

        if len(owners2_old) > 0 and len(repos2_old) > 0:
            owners1_new = owners1[~owners1.owner_id.isin(owners2_old.owner_id)]
            repos1_new = repos1[~repos1.repo_id.isin(repos2_old.repo_id)]
        else:
            owners1_new = owners1
            repos1_new = repos1

        debug_print("Collecting data for {0} owners and {1} repos.".format(
            len(owners1_new), len(repos1_new)
        ))

        if len(owners1_new) > 0 and len(repos1_new) > 0:
            owners2_new, repos2_new = update_owners_repos(owners1_new, repos1_new, local)

            if len(owners2_new) > 0 and len(repos2_new) > 0:
                owners2 = pd.concat([owners2_old, owners2_new]).drop_duplicates(subset = 'owner_id')
                repos2 = pd.concat([repos2_old, repos2_new]).drop_duplicates(subset = 'repo_id')
            else:
                owners2 = owners2_old
                repos2 = repos2_old
        else:
            owners2 = owners2_old
            repos2 = repos2_old
    
    ## Save
    if save:
        debug_print("Saving combined data for {0} owners and {1} repos".format(
            len(owners2), len(repos2)
        ))
        if local:
            owners2.to_csv("{0}/owners2{1}.csv".format(PATH, EXTENSION), index = False)
            repos2.to_csv("{0}/repos2{1}.csv".format(PATH, EXTENSION), index = False)
        else:
            df_to_s3(owners2, "{0}/owners2{1}.csv".format(S3_PATH, EXTENSION))
            df_to_s3(repos2, "{0}/repos2{1}.csv".format(S3_PATH, EXTENSION))
        debug_print("Owners2 and Repos2 were created and saved.\n"+BREAK)
        
    
    ## Add data on cells within each notebook. #######################
    if not set(["notebooks2{0}.csv".format(EXTENSION)]).issubset(current_csvs):
        print("Notebooks2 not found, creating from scratch.")
        get_all_nb_cells(notebooks1, local, 0)
    else:
        # Get existing data.
        try:
            notebooks2_old = get_df("notebooks2{0}.csv".format(EXTENSION), local)
            debug_print("Found and opened notebook data for {0} notebooks.".format(
                len(notebooks2_old)
            ))
        except Exception as e:
            notebooks2_old = []
            print("Notebooks2 could not be opened, creating from scratch.")
            print(type(e), e)
       

        # Isolate rows of notebooks1 corresponding to new notebooks
        if len(notebooks2_old) > 0:
            notebooks1_new = notebooks1[~notebooks1.file.isin(notebooks2_old.file)]
        else:
            notebooks1_new = notebooks1

        debug_print("Collecting data for {0} notebooks.".format(
            len(notebooks1_new)
        ))

        # If there are new notebooks, add cell data
        if len(notebooks1_new) > 0:
            get_all_nb_cells(notebooks1_new, local, len(notebooks2_old))

        del notebooks2_old
           
    # Check time and report status.
    end = datetime.datetime.now()
    debug_print("TOTAL TIME: {0}".format(end - start))


def get_df(file_name, local):
    """ Returns specified DataFrame from local storage or S3 """

    if local:
        df = pd.read_csv("{0}/{1}".format(PATH, file_name))
    else:
        df = s3_to_df("{0}/{1}".format(S3_PATH, file_name))
    return df


# Equivalent to 5_repo_metadata_cleaning.ipynb
def update_owners_repos(owners, repos, local):
    """ Add information on Owners and Repos"""
    
    new_repo_info = {}
    new_owner_info = {}
    repo_ids = list(repos.repo_id)
    missing = 0
    forked = 0
    moved = 0

    for i, repo_id in enumerate(repo_ids):
        repo_json = None

        # Keep track of progress.
        if i % COUNT_TRIGGER == 0:
            debug_print("{0} / {1} repo data files processed".format(i, len(repo_ids)))
        
        try:
            obj = s3.Object("notebook-research","repos/repo_{0}.json".format(repo_id))
            repo_json = json.loads(obj.get()["Body"].read().decode("UTF-8"))
        
        except Exception:
            missing += 1
            # Report missed files.
            msg = "Repo {0} metadata did not process.".format(repo_id)
            write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)
            continue

        if repo_json != None:
            if "message" in repo_json and (
                repo_json["message"] == "Not Found" or
                repo_json["message"] == "Bad credentials"
            ):
                # Report missed files.
                missing += 1
                msg = "Repo {0} metadata file did not download well.".format(repo_id)

                # Move bad file
                s3.Object(
                    'notebook-research',
                    'repos_bad/repo_{0}.json'.format(repo_id)
                ).copy_from(CopySource='notebook-research/repos/repo_{0}.json'.format(repo_id))
                s3.Object(
                    'notebook-research',
                    'repos/repo_{0}.json'.format(repo_id)
                ).delete()
                moved += 1

                write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)
                continue

            if "owner" in repo_json:
                owner_id = repo_json["owner"]["id"]  
            else:
                # Report missed files.
                msg = "Repo {0} metadata file not complete.".format(repo_id)
                write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)
                continue

            if not repo_json["fork"]:   
                # Add repo info.
                repo_info = {
                    "repo_id": repo_id,
                    "language": repo_json["language"],
                    "forks_count": repo_json["forks_count"],
                    "stargazers_count": repo_json["stargazers_count"],
                    "watchers_count": repo_json["watchers_count"],
                    "subscribers_count": repo_json["subscribers_count"],
                    "size": repo_json["size"],
                    "open_issues_count": repo_json["open_issues_count"],
                    "has_issues": repo_json["has_issues"],
                    "has_wiki": repo_json["has_wiki"],
                    "has_pages": repo_json["has_pages"],
                    "has_downloads": repo_json["has_downloads"],
                    "pushed_at": repo_json["pushed_at"],
                    "created_at": repo_json["created_at"],
                    "updated_at": repo_json["updated_at"]
                }
                if repo_id not in new_repo_info:
                    new_repo_info[repo_id] = repo_info

                # Add owner info  
                owner_info = {
                    "owner_id": owner_id,
                    "type": repo_json["owner"]["type"]
                }
                if owner_id not in new_owner_info:
                    new_owner_info[owner_id] = owner_info
            else:
                forked += 1
        else:
            missing += 1

    # Display status.
    debug_print("We have {0} new repos.".format(len(new_repo_info)))
    debug_print("Couldn't process {0} files.".format(missing))
    debug_print("{0} new repos were forked.".format(forked))
    debug_print("{0} files had to be moved".format(moved))

    # Translate dictionaries to DataFrames.
    if len(new_owner_info) > 0 and len(new_repo_info)> 0:
        updated_owners = owners.merge(pd.DataFrame(new_owner_info).transpose().reset_index(drop=True), on = "owner_id")
        updated_repos = repos.merge(pd.DataFrame(new_repo_info).transpose().reset_index(drop=True), on = "repo_id")
    else:
        updated_owners = []
        updated_repos = []
    
    return updated_owners, updated_repos

# Equivalent to 6_compute_nb_data.ipynb    
def get_all_nb_cells(notebooks, local, done):
    """ Get cell and notebook data for each notebook. """
    new_nb_info = {}
    all_cells_info = {}
    missing = []
    
    for count, row in notebooks.iterrows():
        # Track progress.
        file_name = row["file"]
        data = None
        if count % COUNT_TRIGGER == 0 or count == len(notebooks) - 1:
            print("{0} / {1} notebooks processed for cell data".format(count, len(notebooks)+done))
            
            # Save data and reset. (In chunks to avoid MemoryError).
            if count > 0:
                # Transform data to DataFrame.
                notebooks_temp = pd.DataFrame(new_nb_info).transpose()
                cells_temp = pd.DataFrame(all_cells_info).transpose().reset_index(drop=True)

                # Save data to CSV.
                try:
                    if local:
                        notebooks_temp.to_csv("{0}/notebooks2_{1}_{2}.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        cells_temp.to_csv("{0}/cells1_{1}_{2}.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                    else:
                        df_to_s3(notebooks_temp, "{0}/notebooks2_{1}_{2}.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(cells_temp, "{0}/cells1_{1}_{2}.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))

                except MemoryError:
                    # Split data into 3 sections and try saving again.
                    n1 = notebooks_temp.iloc[:len(notebooks_temp)//4]
                    n2 = notebooks_temp.iloc[len(notebooks_temp)//4:2*len(notebooks_temp)//4]
                    n3 = notebooks_temp.iloc[2*len(notebooks_temp)//4:3*len(notebooks_temp)//4]
                    n4 = notebooks_temp.iloc[3*len(notebooks_temp)//4:]
                    
                    c1 = cells_temp.iloc[:len(cells_temp)//8]
                    c2 = cells_temp.iloc[len(cells_temp)//8:2*len(cells_temp)//8]
                    c3 = cells_temp.iloc[2*len(cells_temp)//8:3*len(cells_temp)//8]
                    c4 = cells_temp.iloc[3*len(cells_temp)//8:4*len(cells_temp)//8]
                    c5 = cells_temp.iloc[4*len(cells_temp)//8:5*len(cells_temp)//8]
                    c6 = cells_temp.iloc[5*len(cells_temp)//8:6*len(cells_temp)//8]
                    c7 = cells_temp.iloc[6*len(cells_temp)//8:7*len(cells_temp)//8]
                    c8 = cells_temp.iloc[7*len(cells_temp)//8:]
                    
                    if local:
                        n1.to_csv("{0}/notebooks2_{1}_{2}_1.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        n2.to_csv("{0}/notebooks2_{1}_{2}_2.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        n3.to_csv("{0}/notebooks2_{1}_{2}_3.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        n4.to_csv("{0}/notebooks2_{1}_{2}_4.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        
                        c1.to_csv("{0}/cells1_{1}_{2}_1.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        c2.to_csv("{0}/cells1_{1}_{2}_2.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        c3.to_csv("{0}/cells1_{1}_{2}_3.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        c4.to_csv("{0}/cells1_{1}_{2}_4.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        c5.to_csv("{0}/cells1_{1}_{2}_5.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        c6.to_csv("{0}/cells1_{1}_{2}_6.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        c7.to_csv("{0}/cells1_{1}_{2}_7.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                        c8.to_csv("{0}/cells1_{1}_{2}_8.csv".format(PATH, EXTENSION, count/COUNT_TRIGGER), index = False)
                    else:
                        df_to_s3(n1, "{0}/notebooks2_{1}_{2}_1.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(n2, "{0}/notebooks2_{1}_{2}_2.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(n3, "{0}/notebooks2_{1}_{2}_3.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(n4, "{0}/notebooks2_{1}_{2}_4.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        
                        df_to_s3(c1, "{0}/cells1_{1}_{2}_1.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(c2, "{0}/cells1_{1}_{2}_2.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(c3, "{0}/cells1_{1}_{2}_3.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(c4, "{0}/cells1_{1}_{2}_4.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(c5, "{0}/cells1_{1}_{2}_5.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(c6, "{0}/cells1_{1}_{2}_6.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(c7, "{0}/cells1_{1}_{2}_7.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))
                        df_to_s3(c8, "{0}/cells1_{1}_{2}_8.csv".format(S3_PATH, EXTENSION, count/COUNT_TRIGGER))

                # Empty current dictionaries.
                new_nb_info = {}
                all_cells_info = {}
                print("CSVs saved")

        # Initialize row of data.
        nb_info = {
            "file": file_name,
            "google_collab": False,
            "nbformat": "",
            "nbformat_minor": "",
            "num_cells": 0,
            "kernel_lang": "",
            "kernel_name": "",
            "lang_name": "",
            "lang_version": ""
        }

        # Open notebooks as json.
        try:
            obj = s3.Object("notebook-research","notebooks/{0}".format(file_name))
            data = json.loads(obj.get()["Body"].read().decode("UTF-8"))
        except Exception:
            # Report missed files.
            msg = "Notebook {0} did not open.".format(file_name)
            write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)
            missing.append(file_name)
            
            # Add row with missing values.
            if file_name not in new_nb_info:
                new_nb_info[file_name]= nb_info
            
            continue

        # If data was able to load as JSON, extract information.
        if data and isinstance(data, dict):
            keys = data.keys()
            
            # Get nb top level format metadata.
            if "nbformat" in keys:
                nb_info["nbformat"] = data["nbformat"]
            if "nbformat_minor" in keys:
                nb_info["nbformat_minor"] = data["nbformat_minor"]
            
            # Get info from the metadata dictionary.
            if (
                "metadata" in keys 
                and data["metadata"] != None
                and isinstance(data["metadata"], dict)
            ):
                metadata_keys = data["metadata"].keys()

                # Access language data.
                if (
                    "kernelspec" in metadata_keys 
                    and data["metadata"]["kernelspec"] != None
                    and isinstance(data["metadata"]["kernelspec"], dict)
                ):
                    kernel_keys = data["metadata"]["kernelspec"].keys()
                    
                    # If Google colab notebook, only Python 2.7 or 3.6 are possible.
                    if "colab" in metadata_keys:
                        nb_info["google_collab"] = True
                        if (
                            "name" in kernel_keys 
                            and "display_name" in kernel_keys
                        ):
                            nb_info["kernel_lang"] = data["metadata"]["kernelspec"]["name"]
                            nb_info["kernel_name"] = data["metadata"]["kernelspec"]["display_name"]
                            if nb_info["kernel_lang"] == "python3":
                                nb_info["lang_name"] = "python"
                                nb_info["lang_version"] = "3.6"
                            elif nb_info["kernel_lang"] == "python2":
                                nb_info["lang_name"] = "python"
                                nb_info["lang_version"] = "2.7"

                    # Not Google colab, access kernel language and display name.
                    else:
                        if "language" in kernel_keys:
                            nb_info["kernel_lang"] = data["metadata"]["kernelspec"]["language"]
                        if "display_name" in kernel_keys:
                            nb_info["kernel_name"] = data["metadata"]["kernelspec"]["display_name"]

                # Access language info.
                if (
                    "language_info" in metadata_keys 
                    and "colab" not in metadata_keys
                ):
                    lang_keys = data["metadata"]["language_info"].keys()
                    if "name" in lang_keys and "colab" not in metadata_keys:
                        nb_info["lang_name"] = data["metadata"]["language_info"]["name"]
                    if "version" in lang_keys and "colab" not in metadata_keys:
                        nb_info["lang_version"] = data["metadata"]["language_info"]["version"]
                elif "language" in metadata_keys:
                    nb_info["lang_name"] = data["metadata"]["language"]

            
            # Get information about individual cells.
            cells_info = {}
            if "cells" in keys:
                nb_info["num_cells"] = len(data["cells"]) 
                cell_id = 0                
                for cell in data["cells"]:
                    cell_info, nb_language = get_single_cell(cell_id, file_name, cell, nb_info["lang_name"])
                    
                    if nb_info["lang_name"] == "":
                        nb_info["lang_name"] = nb_language.lower()
                    
                    if (file_name, cell_id) not in cells_info:
                        cells_info[(file_name, cell_id)] = cell_info
                    
                    cell_id += 1
                
            elif "worksheets" in keys:
                cell_id = 0
                for w in data["worksheets"]:
                    for cell in w["cells"]:
                        cell_info, nb_language = get_single_cell(cell_id, file_name, cell, nb_info["lang_name"])
                        
                        if nb_info["lang_name"] == "":
                            nb_info["lang_name"] = nb_language.lower()

                        if (file_name, cell_id) not in cells_info:
                            cells_info[(file_name, cell_id)] = cell_info
                        
                        cell_id += 1
                
        all_cells_info.update(cells_info)

        if file_name not in new_nb_info:
            new_nb_info[file_name]= nb_info

    debug_print("{0} notebooks are missing cell data.".format(len(missing)))
    return new_nb_info, all_cells_info

def get_single_cell(cell_id, file_name, cell, nb_language):
    """ Get data for a single cell in a notebook. """
    nbformat_3_mimes = ["text", "latex", "png", "jpeg", "svg", "html", "javascript", "json", "pdf", "metadata"]
    
    # Initialize row of data.
    cell_info = {
        "file": file_name,
        "cell_id": cell_id,
        "cell_type": "",
        "num_words": 0,
        "markdown": [],
        "headings": [],
        "lines_of_code": 0,
        "code": [],
        "imports": [],
        "parsed_ast": False,
        "links": [],
        "num_execute_result": 0,
        "execute_result_keys": [],
        "execution_count": None,
        "num_error": 0,
        "error_names": [],
        "error_values": [],
        "num_stream": 0,
        "num_display_data": 0,
        "display_data_keys": [],
        "num_functions": 0,
        "functions": [],
        "num_classes": 0,
        "classes": [],
        "comments": [],
        "num_comments": 0
    }
    
    if isinstance(cell, dict): 
        cell_keys = cell.keys()
    else:
        cell_keys = []
    
    if "cell_type" in cell_keys:
        cell_info["cell_type"] = cell["cell_type"]
    
    # Add data on markdown and raw cells.
    if cell_info["cell_type"] in ["raw","markdown","heading"]:
        if "source" in cell_keys:
            if isinstance(cell["source"], list):
                for l in cell["source"]:
                    words = len(l.split())
                    cell_info["num_words"] += words
            elif isinstance(cell["source"], str):
                cell_info["num_words"] += len(cell["source"].split())

    # Add additional data on markdown cells.   
    if cell_info["cell_type"] in ["markdown","heading"]:
        
        # Get the lines of markdown.
        if "source" in cell_keys:
            if isinstance(cell["source"], list):
                cell_info["markdown"] += cell["source"]
            elif isinstance(cell["source"], str):
                cell_info["markdown"] += cell["source"].splitlines()
        
        # Track headings in heading cells.
        if ("level" in cell_keys and
            "source" in cell_keys and
            len(cell_info["markdown"]) > 0
        ):
            cell_info["headings"].append([cell["level"], cell["source"]])
        
        for l in cell_info["markdown"]:
            # Get headings in markdown cells.
            words = l.split()
            if len(words) > 1 and words[0].strip() in ["#", "##", "###", "####", "#####", "######"]:
                cell_info["headings"].append([len(words[0].strip()), " ".join(words[1:])])


            # Get links
            urls = re.findall(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", l)
            all_links = re.findall(r"\[([^]]+)]\(\s*(\S*)\)", l)
            for al in all_links:
                if al[:4] != "http":
                    urls.append(al)
            for u in urls:
                cell_info["links"].append(u)
    
    # Add data for code cells.
    if cell_info["cell_type"] == "code":
        if "execution_count" in cell_keys:
            cell_info["execution_count"] = cell["execution_count"]

        if "source" in cell_keys:
            if isinstance(cell["source"], list):
                cell_info["lines_of_code"] += len(cell["source"])
                cell_info["code"] += cell["source"]
            elif isinstance(cell["source"], str):
                cell_info["lines_of_code"] += len(cell["source"].splitlines())
                cell_info["code"] += cell["source"].split("\n")
            
        elif "input" in cell_keys:
            if isinstance(cell["input"], list):
                cell_info["lines_of_code"] = len(cell["input"])
                cell_info["code"] += cell["input"]
            elif isinstance(cell["input"], str):
                cell_info["lines_of_code"] = len(cell["input"].splitlines())
                cell_info["code"] += cell["input"].split("\n")

        if nb_language == "" and "language" in cell_keys:
            nb_language = cell["language"].lower()
            
        # Parse imports, functions, comments, and classes (language dependent).
        if nb_language == "python":
            cell_info = parse_py(cell_info)
        elif nb_language == "r":
            cell_info = parse_r(cell_info)
        elif nb_language == "julia":
            cell_info = parse_ju(cell_info) 

    cell_info["code"] = '\n'.join(cell_info["code"])       

    # Get data on cell output.        
    if "outputs" in cell_keys:
        for o in cell["outputs"]:

            if isinstance(o, dict):
                output_keys = o.keys()
            else:
                output_keys = []

            if "output_type" in output_keys:
                if o["output_type"] in ["execute_result","pyout"]:
                    cell_info["num_execute_result"] += 1
                    if "data" in output_keys and isinstance(o["data"], dict):
                        data_keys = o["data"].keys()
                        for k in data_keys:
                            cell_info["execute_result_keys"].append(k)
                    else:
                        for k in output_keys:
                            if k in nbformat_3_mimes:
                                cell_info["execute_result_keys"].append(k)

                elif o["output_type"] in ["error","pyerr"]:
                    cell_info["num_error"] += 1
                    if "ename" in output_keys:
                        cell_info["error_names"].append(o["ename"])
                    if "evalue" in output_keys:
                        cell_info["error_values"].append(o["evalue"])

                elif o["output_type"] == "stream":
                    cell_info["num_stream"] += 1

                elif o["output_type"] == "display_data":
                    cell_info["num_display_data"] += 1
                    if "data" in output_keys:
                        if isinstance(o["data"], dict):
                            data_keys = o["data"].keys()
                            for k in data_keys:
                                cell_info["display_data_keys"].append(k)
                    else:
                        for k in output_keys:
                            if k in nbformat_3_mimes:
                                cell_info["display_data_keys"].append(k)
                                
    return cell_info, nb_language

def parse_py_ast(list_of_code):
    """ Parse imports, functions, and classes for Python code cells. """
   
    imports = []
    functions = []
    classes = []
    parsed_ast = False
    
    # Remove magic lines because AST doesnt understand magic.
    code = "\n".join([c for c in list_of_code if
        not c.startswith("%") and
        not c.startswith("!") and 
        not c.startswith("?")
    ]).replace(";"," ")            

    def search_tree(tree):
        for t in tree.body:
            if type(t) == ast.ClassDef:
                class_name = t.name
                methods = []
                attributes = []
                for b in t.body:
                    if type(b) == ast.FunctionDef:
                        methods.append(b.name)
                    elif type(b) == ast.Assign:
                        try:
                            attributes.append(b.targets[0].id)
                        except Exception:
                            attributes.append("")
                classes.append([class_name, len(methods), len(attributes)])
            
            elif type(t) in [ast.ImportFrom, ast.Import]:
                for name in t.names:
                    n = name.name
                    asname = name.asname
                    if asname == None:
                        asname = n

                    if type(t) == ast.ImportFrom:
                        module = t.module
                        if module != None and n != None:
                            n = module+"."+n

                    imports.append([n, asname])
            
            elif type(t) == ast.FunctionDef:
                name = t.name
                args = [arg.arg for arg in ast.walk(t.args) if type(arg) == ast.arg]
                functions.append([name, args])

            elif type(t) == ast.Assign and type(t.value) == ast.Lambda:
                try:
                    name = t.targets[0].id
                except Exception:
                    name = ""
                args = [arg.arg for arg in ast.walk(t.value.args) if type(arg) == ast.arg]
                functions.append([name, args])

    # Try to parse the entire cell.
    try:
        tree = ast.parse(code)
        parsed_ast = True
        search_tree(tree)
                    
    # Exception due to syntax error in code. Try line by line.
    # We could still miss a multi-line import, but better than nothing.                
    except Exception:
        for c in code.split("\n"):
            try:
                tree = ast.parse(c)
                parsed_ast = True
                search_tree(tree)
                        
            except Exception:
                continue
        
    return imports, functions, classes, parsed_ast

def parse_py(cell_info):
    """ Parse imports, functions, classes, and comments for Python code cells. """
    
    imports, functions, classes, parsed_ast = parse_py_ast(cell_info["code"])
    cell_info["imports"] = imports
    cell_info["parsed_ast"] = parsed_ast
    cell_info["functions"] = functions
    cell_info["classes"] = classes
    cell_info["num_imports"] = len(imports)
    cell_info["num_functions"] = len(functions)
    cell_info["num_classes"] = len(classes)

    # Split "code" into "comments" and "code".
    in_multiline = False
    for l in cell_info["code"]:
        l = l.strip()
        has_sep = False
        for sep in [""""","""""]:
            if sep in l:
                has_sep = True
                if not in_multiline:
                    comment = l.split(sep)[1:]
                    cell_info["num_comments"] += 1
                    comment = comment[0].strip()
                    if comment != "":
                        cell_info["comments"].append(comment)
                    if len(l.split(sep)) == 2:
                        in_multiline = True
                else:
                    comment = l.split(sep)[0].strip()
                    if len(l.split(sep)) > 2:
                        # Starts a new comment on same line.
                        # Add both comments, still in multiline.
                        second_comment = l.split(sep)[2]
                        cell_info["num_comments"] += 1
                        if comment != "":
                            cell_info["comments"].append(comment)
                        if second_comment != "":
                            cell_info["comments"].append(second_comment)
                    else:
                        if comment != "":
                            cell_info["comments"].append(comment)
                        in_multiline = False
        if not has_sep and in_multiline:
            cell_info["comments"].append(l)

        if "#" in l:
            cell_info["num_comments"] += 1
            cell_info["comments"].append(" ".join(l.split("#")[1:]).strip())

    return cell_info

def parse_r_imports(list_of_code):
    """ Parse imports for R code cells """
    imports = []
    for code in list_of_code:
        code = code.replace("\\n","").replace("\n","")
        im = None
        if "library(" in code and not code.startswith("#"):
            im = code.split("library(")[1].split(")")[0]
            imports.append((im, ""))
            
        if "require(" in code and not code.startswith("#"):
            im = code.split("require(")[1].split(")")[0]
            imports.append((im,""))
            
    return imports

def parse_r(cell_info):
    """ Parse imports, functions, and comments for R code cells. """
    
    cell_info["imports"] = parse_r_imports(cell_info["code"])

    for l in cell_info["code"]:
        l = l.lstrip()

        if "function" in l:
            if "<-" in l:
                cell_info["functions"].append(l.split("<-")[0].strip())
                cell_info["num_functions"] += 1
            elif "=" in l:
                cell_info["functions"].append(l.split("=")[0].strip())
                cell_info["num_functions"] += 1
                
        elif "#" in l:
            cell_info["num_comments"] += 1
            cell_info["comments"].append(" ".join(l.split("#")[1:]).strip())

    return cell_info

def parse_ju_imports(list_of_code):
    """ Parse imports for Julia code cells """
    imports = []
    for code in list_of_code:
        if code.strip().startswith("using") and code.split("using")[1].strip() != "":
            im = code.split("using")[1].strip().split()[0]
            if im.endswith(":"):
                im = im[:-1]
                rest = code.split(":")[1].strip().split(",")
                for r in rest:
                    imports.append((im+"."+r.strip(), r.strip()))
            else:
                imports.append((im,""))
        elif code.strip().startswith("import") and code.split("import")[1].strip() != "":
            im = code.split("import")[1].strip().split()[0]
            imports.append((im,""))
    return imports

def parse_ju(cell_info):
    """ Parse imports, functions, and comments for Julia code cells. """

    cell_info["imports"] = parse_ju_imports(cell_info["code"])

    in_multiline = False
    for l in cell_info["code"]:
        l = l.lstrip()
        parts = l.split()
        if len(parts) >= 2:                
            if l.startswith("def"):
                cell_info["functions"].append(parts[1].split("(")[0])
                cell_info["num_functions"] += 1
            elif l.startswith("class"):
                cell_info["classes"].append(parts[1].split(":")[0])
                cell_info["num_classes"] += 1
        
        if in_multiline:
            cell_info["num_comments"] += 1
            cell_info["comments"].append(l)
        
        if l.startswith("#="):
            if not in_multiline:
                cell_info["num_comments"] += 1
                cell_info["comments"].append(l.replace('"',""))
                in_multiline = True
        elif l.startswith("=#"):
            if in_multiline:
                in_multiline = False
        elif "#" in l:
            cell_info["num_comments"] += 1
            cell_info["comments"].append(" ".join(l.split("#")[1:]).strip())

    return cell_info
    
if __name__ == "__main__":
    main()
