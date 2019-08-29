"""
Script to download Jupyter notebooks from GitHub.

Downloads all Jupyter notebooks listed in notebooks1.csv,
created in query_git.py.
"""

import datetime
import os
import pickle
import json
import time
import sys

import requests
import argparse
import pandas as pd
from io import BytesIO

from consts import (
    PATH,
    JSON_PATH,
    COUNT_TRIGGER,
    BREAK,
    HEADERS,
    s3
)

from funcs import (
    debug_print,
    write_to_log,
    s3_to_df,
    list_s3_dir
)


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-w","--worker", metavar="N", type=int, 
        help=(
            "GITHUB_TOKEN assigned to these files "
            + "(will use partition N stored in download_partitions.pickle)."
        )
    )
    parser.add_argument("-r","--repos", action="store_const", 
        dest="only_repos", const=True, default=False, 
        help="Download repos only."
    )
    parser.add_argument("-n","--notebooks", action="store_const", 
        dest="only_nbs", const=True, default=False, 
        help="Download notebooks only."
    )
    parser.add_argument("--local", action="store_const", 
        dest="local", const=True, default=False, 
        help="Save data locally instead of in S3."
    )
    args = parser.parse_args()
    worker = args.worker
    only_repos = args.only_repos
    only_nbs = args.only_nbs
    local = args.local

    # If both flags are specified, ignore.
    if only_repos and only_nbs:
        raise Exception("Cannot use both --repos and --notebooks flags. Use --help flag for more information.")

    # If a worker was specified, get partition data and correct header.
    if worker != None:
        print("Worker {0}".format(worker))
        
        try:
            if local:
                with open("download_partitions.pickle", "rb") as f:
                    partitions_download = pickle.load(f)
                partition = partitions_download[worker]
            else:
                obj = s3.Object("notebook-research", "download_partitions_{0}.pickle".format(worker))
                partition = pickle.load(BytesIO(obj.get()["Body"].read()))
        except Exception:
            print((
                "Download Partitions data were not found {0}. ".format(
                    "locally" if local else "in s3"
                )
                + "Please run parallelize_download.py and try again."
            ))
            sys.exit(0)
        
        notebooks1 = partition["notebooks"]

        obj = s3.Object("notebook-research", "csv/owners1.csv")
        owners = pd.read_csv(BytesIO(obj.get()["Body"].read()))

        obj = s3.Object("notebook-research", "csv/repos1.csv")
        repos = pd.read_csv(BytesIO(obj.get()["Body"].read()))


        owners1 = notebooks1[["owner_id"]].merge(owners[['owner_id','owner_login']], on = "owner_id", how = 'left').drop_duplicates()
        repos1 = notebooks1[["repo_id", 'owner_id']].merge(repos[['repo_id','repo_name']], on = 'repo_id', how = 'left').drop_duplicates()
        header = HEADERS[partition["id"]]

        debug_print(
            "Partition data for downloads found and opened. "
            + "Notebooks1, Owners1, and Repos1 were found and opened."+BREAK
        )

    # If a worker was not specified, get all data and use first header.
    else:
        try:
            if local:
                notebooks1 = pd.read_csv("{0}/notebooks1.csv".format(PATH))
                owners1 = pd.read_csv("{0}/owners1.csv".format(PATH))
                repos1 = pd.read_csv("{0}/repos1.csv".format(PATH))
            else:
                notebooks1 = s3_to_df("csv/notebooks1.csv")
                owners1 = s3_to_df("csv/owners1.csv")
                repos1 = s3_to_df("csv/repos1.csv")
        except Exception:
            print(
                "The files 'notebooks1.csv','repos1.csv', and "+
                "'owners1.csv' were not found. Please run query_git.py "+
                "and try again."
            )
            sys.exit(0)

        header = HEADERS[0]


    # Check time and display status.
    print("{0} notebooks, {1} repos, {2} owners".format(len(notebooks1), len(repos1), len(owners1)))
    check1 = datetime.datetime.now()
    write_to_log(
        "../logs/timing.txt", 
        "download CHECKPOINT 1: {0}".format(check1)
    )

    # Download full notebooks from github.
    if not only_repos:
        if local:
            current_files = set(os.listdir("../data/notebooks"))
        else:
            obj = s3.Object("notebook-research", "current_notebooks.pickle")
            current_files = pickle.load(BytesIO(obj.get()["Body"].read()))
        
        num_done = len(current_files)
        debug_print(
            "{0} notebooks have already been downloaded.".format(num_done)
        )
        
        download_nbs(notebooks1, local, current_files)
        check2 = datetime.datetime.now()
        write_to_log("../logs/timing.txt", "CHECKPOINT 2: {0}".format(check2))
        debug_print(
            "\nNotebooks have been downloaded. Time: {0}{1}".format(
            check2 - check1, BREAK
        ))
       
    # Download data from github.
    if not only_nbs:
        download_repo_data(repos1, owners1, header, local)
        check3 = datetime.datetime.now()
        write_to_log("../logs/timing.txt", "CHECKPOINT 3: {0}".format(check3))
        debug_print(
            "\nRepos have been downloaded. "+
            "Time: {0}{1}".format(check3 - check2, BREAK)
        )


def download_nbs(notebooks, local, current_files): 
    """ 
    Download notebooks from GitHub.
    Equivalent to Adam's 2_nb_download.ipynb.
    """
    debug_print("Downloading notebooks\n")
    already_done = 0
    checkpoints = 0
    new = 0
    count = 0
    
    for _, row in notebooks.sort_values(by="days_since").iterrows():
        date_string = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
        
        # Keep track of the download progress.
        if count % COUNT_TRIGGER == 0 or count == len(notebooks):
            debug_print("{0} / {1} notebooks downloaded.".format(
                count, len(notebooks)
            ))
        
        count += 1

        # Don't download files we already have. 
        # Don't download files in .ipynb_checkpoints.
        if row["file"] in current_files:
            already_done += 1
            continue
        if ".ipynb_checkpoints" in row["html_url"]:
            checkpoints += 1
    
        try:
            # Access the raw content webpage and download the file.
            raw_url = row["html_url"].replace(
                "github.com",
                "raw.githubusercontent.com"
            ).replace("/blob", "")
            r = requests.get(raw_url)

            # Save file.
            if local:
                filename = "../data/notebooks/{0}".format(row["file"])
                with open(filename, "w") as nb_file:
                    nb_file.write(r.text)
            else:
                obj = s3.Object(
                    "notebook-research","notebooks/{0}".format(row["file"])
                )
                obj.put(Body = bytes(r.text.encode("UTF-8")))
        
            new += 1
            msg = "{0}: downloaded {1}".format(date_string, row["file"])
            write_to_log("../logs/nb_log.txt", msg)

        except Exception:
            # Report missed files.
            msg = "{0}: had trouble downloading {1}".format(
                date_string, row["file"]
            )
            write_to_log("../logs/nb_log.txt", msg)
            debug_print(msg)

    debug_print(
        "{0} were already done. {1} were in ipynb checkpoints. {2} ".format(
            already_done, checkpoints, new
        )
        + "new notebooks were downloaded."
    )


def download_repo_data(repos, owners, header, local):
    """ Download repository metadata files from GitHub. """
    if len(repos) == 0 or len(owners) == 0:
        return
    
    data_frame = repos.merge(owners, on ="owner_id")

    # List files already downloaded.
    current_repos = os.listdir("../data/repos") if local else list_s3_dir("repos/")

    debug_print((
        "There are currently {0} repo metadata files saved."
    ).format(len(current_repos)))
    
    num_recorded_since = 0
    num_tried_since = 0
    hit_url = ''

    for i, row in data_frame.iterrows():
        num_tried_since += 1

        # Keep track of the download progress.
        if i % COUNT_TRIGGER == 0 or i == len(data_frame):
            debug_print("{0} / {1} repos downloaded.".format(
                i, len(data_frame)
            ))

        # Download repository metadata.
        repo_recorded = False
        if "repo_{0}.json".format(row["repo_id"]) not in current_repos:
            wait_time = 0        
            while not repo_recorded:
                time.sleep(wait_time)
                date_string = datetime.datetime.now().strftime(r"%Y-%m-%d %H:%M:%S")
                url = "https://api.github.com/repos/{0}/{1}".format(
                    row["owner_login"], row["repo_name"]
                )
                try:
                    # Query the api.
                    r = requests.get(url, headers = header)
                    j = r.json()
                    h = r.headers

                    # Handle rate limiting.
                    if h["Status"] == "403 Forbidden":
                        debug_print(
                            "{0}: Hit rate limit. Retry at {1}. {2} tried and {3} saved since last hit.".format(
                                h["Date"], time.ctime(int(h["X-RateLimit-Reset"])), num_tried_since, num_recorded_since
                            )
                        )
                        if hit_url == url:
                            print('Same one again, skipping')
                            repo_recorded = True
                            continue

                        wait_time = int(h["X-RateLimit-Reset"]) - time.time() + 1
                        num_tried_since = 0
                        num_recorded_since = 0
                        hit_url = url
                        continue
                   
                    if "message" in j and (
                        j["message"] == "Not Found" or
                        j["message"] == "Bad credentials"
                    ):
                        print(url,'Message:', j['message'])
                        raise Exception

                    # Save JSON File.
                    else:
                        if local:
                            filename = "../data/repos/repo_{0}.json".format(
                                row["repo_id"]
                            )
                            with open(filename, "w") as repo_file:
                                json.dump(j, repo_file)
                        else:
                            obj = s3.Object(
                                "notebook-research",
                                "repos/repo_{0}.json".format(row["repo_id"])
                            )
                            obj.put(Body = bytes(json.dumps(j).encode("UTF-8")))

                        # Report Status.
                        msg = "{0}: downloaded repo {1}".format(
                            date_string, row["repo_id"]
                        )
                        write_to_log("../logs/repo_metadata_query_log.txt", msg)
                        repo_recorded = True
                        wait_time = 0
                        num_recorded_since += 1

                except Exception as e:
                    # Report missed files.
                    msg = "{0}: had trouble downloading repo {1}".format(
                        date_string, row["repo_id"]
                    )
                    write_to_log("../logs/repo_metadata_query_log.txt", msg)
                    debug_print(msg)
                    debug_print(e)
                    repo_recorded = True

if __name__ == "__main__":
    main()