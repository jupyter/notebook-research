"""
Script to extract data from downloaded Jupyter notebooks.

Saves important information about notebooks, repos, and readmes
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

    global EXTENSION 
    EXTENSION = '_{0}'.format(worker) if worker != None else ''
    print('EXTENSION', EXTENSION)

    start = datetime.datetime.now()

    if local:
        current_csvs = set(os.listdir(PATH))
    else:
        current_csvs = list_s3_dir("csv/")

    # Extract basic data from json files (files created in query_git.py).
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
    if not set(["owners2{0}.csv".format(EXTENSION), "repos2{0}.csv".format(EXTENSION)]).issubset(current_csvs):
        owners2, repos2 = update_owners_repos(owners1, repos1, local)
        save = True
    else:
        # Get existing data.
        owners2_old = get_df("repos2{0}.csv".format(EXTENSION), local)
        repos2_old = get_df("repos2{0}.csv".format(EXTENSION), local)

        debug_print("Data found and opened for {0} owners and {1} repos".format(
            len(owners2_old), len(repos2_old)
        ))

        # Isolate rows that correspond to new notebooks.
        owners1_new = owners1[~owners1.owner_id.isin(owners2_old)]
        repos1_new = repos1[~repos1.repo_id.isin(repos2_old)]

        debug_print("Collecting for {0} owners and {1} repos".format(
            len(owners1_new), len(repos1_new)
        ))

        # If there are new notebooks, update owners and repos.
        if len(owners1_new) > 0:
            owners2_new, repos2_new = update_owners_repos(owners1_new, repos1_new, local)
            
            debug_print("Data collected for {0} owners and {1} repos".format(
                len(owners2_new), len(repos2_new)
            ))

            df_to_s3(owners2_new, "csv/owners2_temp.csv")
            df_to_s3(repos2_new, "csv/repos2_temp.csv")
            print("Saved temporary csvs.")

            owners2 = pd.concat([owners2_old, owners2_new], sort = True)
            repos2 = pd.concat([repos2_old, repos2_new], sort = True)
            save = True
        else:
            owners2 = owners2_old
            repos2 = repos2_old
            save = False
    
    # If new data, save.
    if save:
        debug_print("Saving combined data for {0} owners and {1} repos".format(
            len(owners2), len(repos2)
        ))
        if local:
            owners2.to_csv("{0}/owners2{1}.csv".format(PATH, EXTENSION), index = False)
            repos2.to_csv("{0}/repos2{1}.csv".format(PATH, EXTENSION), index = False)
        else:
            df_to_s3(owners2, "csv/owners2{0}.csv".format(EXTENSION))
            df_to_s3(repos2, "csv/repos2{0}.csv".format(EXTENSION))
        debug_print("Owners2 and Repos2 were created and saved.\n"+BREAK)
    
    debug_print("Repos2 was found and opened.\n"+BREAK)        
    
    ### Add information on readmes for each repo. #####################
    if not set(["readmes1{0}.csv".format(EXTENSION)]).issubset(current_csvs):
        readmes1 = clean_readmes(repos2, local)
        save = True
    else:
        # Get existing data.
        readmes1_old = get_df("readmes1{0}.csv".format(EXTENSION), local)

        debug_print("Found and opened data for {0} readmes.".format(
            len(readmes1_old)
        ))

        # Isolate rows of repos2 correspoding to new notebooks.
        repos2_new = repos2[~repos2.repo_id.isin(readmes1_old.repo_id)]

        debug_print("Collecting data for readmes on {0} repos.".format(
            len(repos2_new)
        ))

        # If there are new notebooks, clean readmes.
        if len(repos2_new) > 0:
            readmes1_new = clean_readmes(repos2_new, local)
            print("Collected data for {0} readmes.".format(
                len(readmes1_new)
            ))
            readmes1 = pd.concat([readmes1_old, readmes1_new], sort = True)
            save = True
        else:
            readmes1 = readmes1_old

    # If new data, save.
    if save:
        print("Saving combined data for {0} readmes.".format(
            len(readmes1)
        ))
        if local:
            readmes1.to_csv("{0}/readmes1{1}.csv".format(PATH, EXTENSION), index = False)
        else:
            df_to_s3(readmes1, "csv/readmes1{0}.csv".format(EXTENSION))
        
        debug_print("Readmes1 were created and saved.\n"+BREAK)

    ### Add data on cells within each notebook. #######################
    if not set(["notebooks2{0}.csv".format(EXTENSION),"cells1{0}.csv".format(EXTENSION)]).issubset(current_csvs):
        notebooks2, cells1 = add_nb_cells(notebooks1, local)
        save = True
    else:
        # Get existing data.
        notebooks2_old = get_df("notebooks2{0}.csv".format(EXTENSION), local)
        cells1_old = get_df("cells1{0}.csv".format(EXTENSION), local)

        debug_print("Found and opened data for {0} notebooks and {1} cells.".format(
            len(notebooks2_old), len(cells1_old)
        ))

        # Isolate rows of notebooks1 corresponding to new notebooks
        notebooks1_new = notebooks1[~notebooks1["file"].isin(notebooks2_old)]

        debug_print("Collecting data for {1} notebooks.".format(
            len(notebooks1_new)
        ))

        # If there are new notebooks, add cell data
        if len(notebooks1_new) > 0:
            notebooks2_new, cells1_new = add_nb_cells(notebooks1_new, local)
           
            debug_print("Collected data for {0} notebooks and {1} cells.".format(
                len(notebooks2_new), len(cells1_new)
            ))

            notebooks2 = pd.concat([notebooks2_old, notebooks2_new], sort = True)
            cells1 = pd.concat([cells1_old, cells1_new], sort = True)
            save = True
        else:
            notebooks2 = notebooks2_old
            cells1 = cells1_old
            save = False

    # If new data, save.
    if save:
        debug_print("Saving combined data for {0} notebooks and {1} cells".format(
            len(notebooks2), len(cells1)
        ))
        if local:
            notebooks2.to_csv("{0}/notebooks2{1}.csv".format(PATH, EXTENSION), index = False)
            cells1.to_csv("{0}/cells1{1}.csv".format(PATH, EXTENSION), index = False)
        else:
            df_to_s3(notebooks2, "csv/notebooks2{0}.csv".format(EXTENSION))
            df_to_s3(cells1, "csv/cells1{0}.csv".format(EXTENSION))

    
    ### Remove notebooks and associated cells, repos, readmes,  #######
    ### and owners with incomplete data.                        #######
    # if not set(["notebooks3.csv", "cells2.csv", "repos3.csv"]).issubset(current_csvs):
    remove_incomplete(notebooks2, cells1, repos2, local)
    
    # Check time and report status.
    end = datetime.datetime.now()
    debug_print("TOTAL TIME: {0}".format(end - start))

def get_df(file_name, local):
    if local:
        df = pd.read_csv("{0}/{1}".format(PATH, file_name))
    else:
        df = s3_to_df("csv/{0}".format(file_name))
    return df

# Equivalent to 5_repo_metadata_cleaning.ipynb
def update_owners_repos(owners, repos, local):
    """ Add information on Owners, repos, and Readmes"""
    
    new_repo_info = {}
    new_owner_info = {}
    repo_ids = list(repos.repo_id)
    missing = 0
    repo_json = None

    for i, repo_id in enumerate(repo_ids):
        # Keep track of progress.
        if i % COUNT_TRIGGER == 0:
            debug_print("{0} / {1} repo data files processed".format(i, len(repo_ids)))
        
        if local:
            try:
                with open("../data/repos/repo_{0}.json".format(repo_id), "r") as repo_json_file:
                    repo_json = json.load(repo_json_file)

            except Exception:    
                missing += 1   
                # Report missed files.
                msg = "Repo {0} metadata file did not open or could not process.".format(repo_id)
                write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)   
        else:
            try:
                obj = s3.Object("notebook-research","repos/repo_{0}.json".format(repo_id))
                repo_json = json.loads(obj.get()["Body"].read().decode("UTF-8"))
            except Exception:
                # Report missed files.
                msg = "Repo {0} metadata did not process.".format(repo_id)
                write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)

        if repo_json != None:
            if "message" in repo_json and (
                repo_json["message"] == "Not Found" or
                repo_json["message"] == "Bad credentials"
            ):
                # Report missed files.
                msg = "Repo {0} metadata file did not open.".format(repo_id)
                write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)
                continue

            if "owner" not in repo_json:
                print(repo_json)
            owner_id = repo_json["owner"]["id"]
                    
            if not repo_json["fork"]:   
                # Add repo info.
                repo_info = {
                    "repo_id": repo_id,
                    "fork": repo_json["fork"],
                    "language": repo_json["language"],
                    "forks_count": repo_json["forks_count"],
                    "stargazers_count": repo_json["stargazers_count"],
                    "watchers_count": repo_json["watchers_count"],
                    "subscribers_count": repo_json["subscribers_count"],
                    "network_count": repo_json["network_count"],
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

    # Display status.
    debug_print("We have {0} repos.".format(len(new_repo_info)))
    debug_print("Couldnt find {0} files.".format(missing))

    # Translate dictionaries to DataFrames and save to CSV.
    updated_owners = owners.merge(pd.DataFrame(new_owner_info).transpose().reset_index(drop=True), on = "owner_id")
    updated_repos = repos.merge(pd.DataFrame(new_repo_info).transpose().reset_index(drop=True), on = "repo_id")
    
    return updated_owners, updated_repos

def clean_readmes(repos, local):
    """ Add information on preferred readme for each repo. """
    
    readmes = {}
    if local:
        repo_ids = set([int(f.split(".")[0].split("_")[1]) for f in os.listdir("../data/readmes") if f[-5:] == ".json"])
    else:
        repo_ids = list_s3_dir('readmes/')
    
    unprocessed = 0
    for count, repo_id in enumerate(repo_ids):    
        # Keep track of progress.
        if count % COUNT_TRIGGER == 0:
            debug_print("{0} / {1} repo readme files processed".format(count, len(repo_ids)))
        
        if local:
            try:
                with open("../data/readmes/readme_{0}.json".format(repo_id), "r") as json_file:
                    try:
                        readme_json = json.load(json_file)
                    except Exception:
                        unprocessed+=1
                        msg = "Repo {0} readme did not process".format(repo_id)
                        write_to_log("../logs/repo_readme_cleaning_log.txt",msg)
                        
                        readme = {
                                "repo_id": repo_id,
                                "path": "",
                                "html_url": "",
                                "content": ""
                            }
                        if repo_id not in readmes:
                            readmes[repo_id] = readme
            except Exception:
                debug_print("Did not find readme_{0}.json".format(repo_id))
        else:
            try:
                obj = s3.Object("notebook-research","readmes/readme_{0}.json".format(repo_id))
                readme_json = json.loads(obj.get()["Body"].read().decode("UTF-8"))
            except Exception:
                # Report missed files.
                msg = "Repo {0} metadata did not process.".format(repo_id)
                write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)

        if readme_json:
            if "message" in readme_json and (
                readme_json["message"] == "Not Found" or 
                readme_json["message"] == "Bad credentials"
            ):
                unprocessed += 1
            else:
                readme = {
                    "repo_id": repo_id,
                    "path": readme_json["path"],
                    "html_url": readme_json["html_url"],
                    "content": readme_json["content"]
                }
                if repo_id not in readmes:
                    readmes[repo_id] = readme
                                        
                 
    # Display Status
    debug_print("We have {0} notebook readmes.".format(len(readmes)))
    debug_print("{0} repo readmes did not process or were not found.".format(unprocessed))

    # Translate dictionaries to DataFrames and save to CSV.
    readmes_df = pd.DataFrame(readmes).transpose().reset_index(drop=True)

    return readmes_df

# Equivalent to 6_compute_nb_data.ipynb
def add_nb_cells(notebooks, local):
    """ Add data on notebook and cell content. """
    new_nb_info, cells_info = get_all_nb_cells(notebooks, local)
    new_nb_info_df = pd.DataFrame(new_nb_info).transpose()

    debug_print("Updating existing notebooks dataframe with new info.")
    updated_notebooks = notebooks.merge(new_nb_info_df, on = "file", how = "left")

    debug_print("Creating dataframe with cell info.")
    cells = pd.DataFrame(cells_info).transpose().reset_index(drop=True)

    return updated_notebooks, cells
    
def get_all_nb_cells(notebooks, local):
    """ Get cell and notebook data for each notebook. """
    new_nb_info = {}
    all_cells_info = {}
    missing = []
    
    if local:
        current_notebooks = set(os.listdir("../data/notebooks"))
    else:
        current_notebooks = list_s3_dir("notebooks/")

    for count, row in notebooks.iterrows():
        
        # Track progress.
        file_name = row["file"]
        if count % 100 == 0:
            print("{0} / {1} notebooks processed for cell data".format(count, len(notebooks)))

        if file_name in current_notebooks:
            date_string = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
            if local:
                try:
                    f = "../data/notebooks/{0}".format(file_name)
                    with open(f) as data_file:
                        data = json.load(data_file)
                except Exception:
                    msg = "{0}: had trouble finding or loading {1}".format(date_string, row["file"])
                    write_to_log("../logs/nb_parse_log.txt", msg)
                    missing.append(file_name)

                    # Add row with missing values.
                    nb_info = {
                        "file": file_name,
                        "nbformat": "",
                        "nbformat_minor": "",
                        "num_cells": 0,
                        "kernel_lang": "",
                        "kernel_name": "",
                        "lang_name": "",
                        "lang_version": ""
                    }
                    if file_name not in new_nb_info:
                        new_nb_info[file_name]= nb_info
                    
                    continue
            else:
                try:
                    obj = s3.Object("notebook-research","notebooks/{0}".format(file_name))
                    data = json.loads(obj.get()["Body"].read().decode("UTF-8"))
                except Exception:
                    # Report missed files.
                    msg = "Notebook {0} did not open.".format(file_name)
                    write_to_log("../logs/repo_metadata_cleaning_log.txt", msg)
            if data:
                if isinstance(data, dict): 
                    keys = data.keys()
                else:
                    keys = []

                # Initialize row of data.
                nb_info = {
                    "file": file_name,
                    "nbformat": "",
                    "nbformat_minor": "",
                    "num_cells": 0,
                    "kernel_lang": "",
                    "kernel_name": "",
                    "lang_name": "",
                    "lang_version": ""
                }
                
                # Get nb top level format metadata.
                if "nbformat" in keys:
                    nb_info["nbformat"] = data["nbformat"]
                if "nbformat_minor" in keys:
                    nb_info["nbformat_minor"] = data["nbformat_minor"]
                
                # Get info from the metadata dictionary.
                if "metadata" in keys:
                    metadata_keys = data["metadata"].keys()

                    # Access language data.
                    if "kernelspec" in metadata_keys:
                        kernel_keys = data["metadata"]["kernelspec"].keys()
                        
                        # If google colab notebook, only Python 2.7 or 3.6 are possible.
                        if "colab" in metadata_keys:
                            if "name" in kernel_keys:
                                nb_info["kernel_lang"] = data["metadata"]["kernelspec"]["name"]
                                nb_info["kernel_name"] = data["metadata"]["kernelspec"]["display_name"]
                                if nb_info["kernel_lang"] == "python3":
                                    nb_info["lang_name"] = "python"
                                    nb_info["lang_version"] = "3.6"
                                elif nb_info["kernel_lang"] == "python2":
                                    nb_info["lang_name"] = "python"
                                    nb_info["lang_version"] = "2.7"

                        else:
                            if "language" in kernel_keys:
                                nb_info["kernel_lang"] = data["metadata"]["kernelspec"]["language"]
                            if "display_name" in kernel_keys:
                                nb_info["kernel_name"] = data["metadata"]["kernelspec"]["display_name"]

                    if "language_info" in metadata_keys and "colab" not in metadata_keys:
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
                
        else:
            # File missing, input row with missing values.
            missing.append(file_name)
            nb_info = {
                "file": file_name,
                "nbformat": "",
                "nbformat_minor": "",
                "num_cells": 0,
                "kernel_lang": "",
                "kernel_name": "",
                "lang_name": "",
                "lang_version": ""
            }
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
    
    # Remove magic lines because AST doesnt understand magic.
    code = "\n".join([c for c in list_of_code if not c.startswith("%")]).replace(";"," ")
    
    # Try to parse the entire cell.
    try:
        tree = ast.parse(code)
                    
    # Exception due to syntax error in code. Try line by line.
    # We could still miss a multi-line import, but better than nothing.                
    except Exception:
        for c in code.split("\n"):
            try:
                tree = ast.parse(c)
                        
            # If another syntax error, code cannot be read.
            except Exception:
                return imports, functions, classes
            

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
        
    return imports, functions, classes

def parse_py(cell_info):
    """ Parse imports, functions, classes, and comments for Python code cells. """
    
    imports, functions, classes = parse_py_ast(cell_info["code"])
    cell_info["imports"] = imports
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

def remove_incomplete(notebooks, cells, repos, local):
    """ Clean data to exclude incomplete results. """
    
    nbs_w_file = set(notebooks.file)
    
    df_nb_content_count = cells[["file", "lines_of_code", "num_words"]].groupby("file").sum().reset_index()
    nbs_w_cells = set(df_nb_content_count[np.logical_or(
            df_nb_content_count.lines_of_code != 0,
            df_nb_content_count.num_words != 0
        )].file.unique())
    debug_print("{0} out of the {1} downloaded notebooks have missing cell content.".format(len(notebooks) - len(nbs_w_cells), len(notebooks)))
    
    # repos_w_metadata = set(repos.repo_id.unique())
    # nbs_w_repo = set(notebooks[notebooks.repo_id.isin(repos_w_metadata)]["file"])
    # debug_print("{0} out of the {1} downloaded notebooks have missing repos.".format(len(notebooks) - len(nbs_w_repo), len(notebooks)))

    nbs_w_all = nbs_w_cells.intersection(nbs_w_file)
        
    updated_notebooks = notebooks[notebooks.file.isin(nbs_w_all)]
    updated_notebooks2 = updated_notebooks[["ipynb_checkpoints" not in n for n in updated_notebooks.file]]

    debug_print("{0} out of the {1} notebooks with full data are part of ipynb_checkpoints.".format(len(updated_notebooks) - len(updated_notebooks2), len(updated_notebooks)))

    updated_repos = repos[repos.repo_id.isin(updated_notebooks2.repo_id)]
    updated_cells = cells[cells.file.isin(nbs_w_all)]
    debug_print("\n{0} out of the {1} downloaded notebooks have all data. {2}% of notebooks had missing data and have been deleted.".format(len(updated_notebooks2), len(notebooks), round(100*(1 - len(updated_notebooks2)/len(notebooks)),3)))
    

    if local:
        updated_notebooks2.to_csv("{0}/notebooks3{1}.csv".format(PATH, EXTENSION), index = False)
        updated_cells.to_csv("{0}/cells2{1}.csv".format(PATH, EXTENSION), index = False)
        updated_repos.to_csv("{0}/repos3{1}.csv".format(PATH, EXTENSION), index = False)
    else:
        df_to_s3(updated_notebooks2, "csv/notebooks3{0}.csv".format(EXTENSION))
        df_to_s3(updated_cells, "csv/cells2{0}.csv".format(EXTENSION))
        df_to_s3(updated_repos, "csv/repos3{0}.csv".format(EXTENSION))
        
if __name__ == "__main__":
    main()