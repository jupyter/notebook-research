"""
Script to query GitHub for Jupyter notebooks in given size range. 

Queries GitHub for Jupyter Notebooks, downloads query metadata 
JSON files, and, if specified, processes basic data from all query 
files to notebooks1.csv. After notebooks1.csv is created, files 
can be downloaded with download.py.
"""

import time
import os
import datetime
import json

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
    s3
)

from funcs import (
    debug_print,
    write_to_log,
    df_to_s3,
    list_s3_dir
)

# Max number of results in one size range.
QUERY_CUTOFF = 950


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "min", type=int, 
        help="Minimum size to search."
    )
    parser.add_argument(
        "max", type=int, 
        help="Maximum size to search."
    )
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
        "--process", action="store_const", 
        dest="process", const=True, default=False, 
        help=(
            "Processes metadata at the end to update "
            + "notebooks.csv, otherwise, the program only "
            + "queries and downloads query files."
        )
    )
    parser.add_argument(
        "--worker", metavar="N", type=int, 
        help=(
            "GITHUB_TOKEN assigned to these sizes (workers "
            + "sorted in alphabetical order: {0}).".format(
                list(TOKENS.keys())
            )
        )
    )
    args = parser.parse_args()
    MIN = args.min
    MAX = args.max
    updating = args.updating
    worker = args.worker
    local = args.local
    process = args.process

    # If updating, look at saved_urls to determine a duplicate.
    # New versions of notebooks will overwrite earlier downloads.
    saved_urls = []
    if updating:
        # Double check update with user (this takes a long time).
        sure = input((
            "Would you like to update the corpus (i.e. "
            + "search for recently added notebooks)? "
            + "This may take a while. [y/n]: "
        ))
        if (
            sure.lower().strip() == "y" or 
            sure.lower().strip() == "yes"
        ):
            if "notebooks1.csv" in os.listdir(PATH):
                notebooks1 = pd.read_csv("{0}/notebooks1.csv".format(PATH))
                saved_urls = list(notebooks1.html_url)
        else:
            updating = False

    # Set worker.
    if worker != None:
        header = HEADERS[worker]
    else:
        header = HEADERS[0]

    # Log and display status.
    write_to_log(
        "../logs/timing.txt", 
        "Testing on the size range {0} to {1}".format(MIN, MAX)
    )
    start = datetime.datetime.now()
    write_to_log("../logs/timing.txt", "START: {0}".format(start))
    debug_print(
        BREAK 
        + "Downloading and formatting data for all Jupyter "
        + "Notebooks publicly stored on github." 
        + BREAK
    )        

    # List notebooks already downloaded.
    current_notebooks = (
        set(os.listdir("../data/notebooks")) if local 
        else list_s3_dir("notebooks/")
    ) if updating else []

    # Get json query files for given size range.
    num_needed = get_json(MIN, MAX, saved_urls, header, 
                          updating, local, current_notebooks)

    # Check time, log, and display status.
    check1 = datetime.datetime.now()
    write_to_log("../logs/timing.txt", "CHECKPOINT 1: {0}".format(check1))
    debug_print(
        "\nJson query files have been downloaded. "
        + "Time: {0}{1}".format(check1 - start, BREAK)
    )

    # Extract basic data from json files, save to CSV.
    if process:
        clean_metadata(num_needed, updating, local)

    # Check time, log, and display status.
    check2 = datetime.datetime.now()
    write_to_log("../logs/timing.txt","CHECKPOINT 2: {0}".format(check2))
    debug_print(
        "Notebooks1, Owners1, and Repos1 were created and saved. "
        + "Time: {0}{1}".format(check2 - check1, BREAK)
    )
    debug_print("All together, {0}".format(check2 - start))


def check_limit(limit_status):
    """ 
    Check limit status and sleeps when appropriate 
    in order to prevent rate-limiting by GitHub. 
    """

    if limit_status["limited"]:
        # Wait until we can query again.
        time.sleep(limit_status["wait_time"])
        limit_status["limited"] = False
    elif limit_status["remaining_queries"] == 0:
        # Sleep if about to hit limit.
        time.sleep(limit_status["reset_time"] - time.time() + 1)
    else:
        # Always sleep 4 seconds.
        time.sleep(4)
    return limit_status


def get_json(
    minimum, maximum, saved_urls, 
    header, updating, local, current_notebooks
):
    """ 
    Download json search results for all jupyter notebooks 
    on github within given size range (minimum - maximum). 
    
    Equivalent to Adam's 0_nb_metadata_download.
    """

    debug_print("Downloading query results by size.")
    current_files = (
        set(os.listdir(JSON_PATH)) if local 
        else list_s3_dir("json/")
    )
   
    num_needed = 0
    done = False
    stored_max = maximum
    
    while not done:
        # Change size range based on what has already been queried.
        if not updating:
            minimum, maximum = edit_size(current_files, minimum, maximum)
        
        if minimum > maximum:
            done = True
            continue
        
        size = str(minimum)+".."+str(maximum)
                
        # At this point, we have a range ready to be queried.
        debug_print("Querying {0} byte notebooks.".format(size))
        
        # Query size range.
        url = URL + size
        query_status = query(
            url, size, header, 
            saved_urls, updating, 
            local, current_notebooks
        )
        
        # Check number of search results.
        # We can only access 1000 results due to the query limit.
        if (
            query_status["num_results"] > QUERY_CUTOFF and   # Results over query limit.
            maximum != minimum and                           # Can decrease query range.
            query_status["page"] == 1                        # On the first page of query.
        ):
            # Cut query range in half.
            if maximum - (maximum - minimum)//2 != maximum:
                maximum = maximum - (maximum - minimum)//2
            else:
                maximum = minimum
            debug_print(
                "Too many results, trying a narrower range: " 
                + "{0} - {1}".format(minimum, maximum)
            )
            continue

        else:
            debug_print(
                "{0} / {1} results found".format(
                    len(query_status["all_items"]), 
                    query_status["num_results"]
                )
            )
            debug_print(
                "{0} are unique.".format(len(set(query_status["all_items"])))
            )
                
        # Move on to next search within original query range.
        minimum = maximum + 1
        maximum = stored_max
    
    return num_needed


def edit_size(current_files, minimum, maximum):
    """
    Update minimum and maximum to an unqueried range.

    Returns the minimum and maximum for the first
    range that has yet to be queried.

    Example: searching 0 - 100 when 23 - 45 have already
    been queried returns 0 - 22.
    """
    # Find sizes already queried.
    sizes_done = {}
    current_files = sorted(
        current_files, 
        key = lambda x: int(x.split("_")[2].split("..")[0]), 
    )
    for x in current_files:
        start = int(x.split("_")[2].split("..")[0])
        if ".." in x:
            end = int(x.split("_")[2].split("..")[1])
        else:
            end = start

        sizes_done[start] = end

    while True:
        minimum_done_start = min(sizes_done.keys())
        minimum_done_end = sizes_done[minimum_done_start]

        if (
            minimum >= minimum_done_start and # Minimum above min queried start.
            maximum > minimum_done_end        # Maximum above min queried end.
        ):
            # If minimum below min queried end, then range of
            # minimum..minimum_done_end has already been queried, 
            # so increase minimum.
            if minimum <= minimum_done_end:
                minimum = minimum_done_end + 1
                debug_print("Size {0}..{1} already queried".format(
                    minimum_done_start, minimum_done_end
                ))
            
            # Remove smallest query range, continue to next smallest.
            sizes_done.pop(minimum_done_start)
            if len(sizes_done) == 0:
                break
            else:
                continue
        break
        
    # Minimum is complete, decrease maximum if necessary.
    if len(sizes_done) > 0:
        minimum_done_start = min(sizes_done.keys())
        minimum_done_end = sizes_done[minimum_done_start]
        if maximum >= minimum_done_start:
            maximum = minimum_done_start - 1
            debug_print("Size {0}..{1} already queried".format(
                minimum_done_start, minimum_done_end
            ))
    
    return minimum, maximum


def query(url, size, header, 
        saved_urls, updating, 
        local, current_notebooks):
    """ 
    Query GitHub for notebooks of a given size and return query status.
    """

    # Set inital rate limiting management variables.
    limit_status = {
        "reset_time": time.time(),
        "limited": False,
        "wait_time": 0,
        "remaining_queries": 30
    }
    
    # Set initial query status variables.
    query_status = {
        "done": False,
        "page": 1,
        "another_page": False,
        "updating": updating,
        "local": local,
        "num_results": 0,
        "num_needed": 0,
        "all_items": []
    }

    while not query_status["done"]:
        # Handle rate limiting status.
        limit_status = check_limit(limit_status)

        # Save this page of results.
        r, limit_status, query_status = save_page(
            url, size, header, query_status,
            saved_urls, current_notebooks
        )
        if r == None:
            continue

        # If too many results, return. Handled in get_json.
        if (query_status["num_results"] > QUERY_CUTOFF and  # Too many results.
            size.split("..")[0] != size.split("..")[1]      # Can decrease query range (min!=max).
        ):
            query_status["done"] = True
            return query_status

        # Handle rate limiting status.
        if limit_status["limited"] and limit_status["wait_time"] != 0:
            continue
            
        # Move to the next page of results.
        if "next" in r.links:
            next_url = r.links["next"]["url"]
            query_status["another_page"] = True
            
            while (
                query_status["another_page"] and 
                len(query_status["all_items"]) != query_status["num_results"]
            ):
                query_status["page"] += 1
                debug_print("{0} to find, {1} found, {2} unique".format(
                    query_status["num_results"], 
                    len(query_status["all_items"]), 
                    len(set(query_status["all_items"]))
                ))
            
                # Handle rate limiting status.
                limit_status = check_limit(limit_status)

                # Save this page of results.
                r, limit_status, query_status = save_page(
                    next_url, size, header, query_status, 
                    saved_urls, current_notebooks
                )
                if r == None:
                    continue

                # Handle rate limiting status.
                if limit_status["limited"] and limit_status["wait_time"] != 0:
                    query_status["page"] -= 1
                    continue

                if "next" in r.links:
                    # Move on to next page of results.
                    next_url = r.links["next"]["url"]
                else:
                    # Completed last page of results.
                    query_status["another_page"] = False

        query_status["done"] = True

        # Report if too many results within a single size (e.g. 1200..1200).
        if (
            query_status["num_results"] > QUERY_CUTOFF and 
            size.split("..")[0] == size.split("..")[1]
        ):
            msg = "TOO MANY RESULTS: {0} bytes, {1} results".format(
                size.split("..")[0],
                query_status["num_results"]
            )
            write_to_log("../logs/nb_metadata_query_log.txt", msg)
            debug_print(msg)
    
    return query_status


def save_page(
    url, size, header, query_status, 
    saved_urls, current_notebooks
):
    """ Save results page to json file. """
    
    # Set inital rate limiting management variables.
    limit_status = {
        "reset_time": time.time(),
        "limited": False,
        "wait_time": 0,
        "remaining_queries": 30
    }
    
    # Query GitHub API.
    try:
        r = requests.get(url, headers = header)
        j = r.json()
        h = r.headers
    except requests.exceptions.Timeout:
        debug_print("Request timeout.")
        r = None
        limit_status["limited"] = True
        limit_status["wait_time"] = 60
        return r, limit_status, query_status

    # Handle 403 error if we have hit query rate.
    if "Status" not in h or h["Status"] == "403 Forbidden":
        try:
            debug_print(
                "{0}: Hit rate limit. Retry after {1} seconds".format(
                    h["Date"], 
                    h["Retry-After"]
                )
            )

            # Set to limited and update wait time.
            limit_status["limited"] = True
            limit_status["wait_time"] = int(h["Retry-After"])

        except Exception:
            # Default wait time to 1 minute.
            limit_status["limited"] = True
            limit_status["wait_time"] = 60

        return r, limit_status, query_status

    # Update rate limiting management variables.
    date = r.headers["Date"]
    query_status["num_results"] = int(j["total_count"])
    limit_status["remaining_queries"] = h["X-RateLimit-Remaining"]
    limit_status["reset_time"] = int(h["X-RateLimit-Reset"])

    # Write progress to log and display status.
    log_string = "{0}: {1} bytes {2} results".format(
        date, size, query_status["num_results"]
    )
    write_to_log("../logs/nb_metadata_query_log.txt", log_string)
    debug_print(log_string)

    # Check if query result is acceptable.
    if (
        query_status["num_results"] <= QUERY_CUTOFF or 
        query_status["page"] > 1 or
        size.split("..")[0] == size.split("..")[1]
    ):
        
        # Add days since.
        diff = datetime.datetime.now() - datetime.datetime(2019,1,1)
        j["days_since"] = (diff.days 
            + (diff.seconds + diff.microseconds/(10**6))/(60*60*24)
        )
        
        # Save this page.
        filename = "github_notebooks_{0}_p{1}.json".format(
            size, query_status["page"]
        )    
        if query_status["updating"]:
            filename = "github_notebooks_{0}_p{1}_{2}.json".format(
                size, query_status["page"], datetime.datetime.now()
            )

        if query_status["local"]:
            with open(JSON_PATH+filename, "w") as json_file:
                json.dump(j, json_file)
        else:
            obj = s3.Object("notebook-research","json/"+filename)
            obj.put(Body = bytes(json.dumps(j).encode("UTF-8")))
        
        # Display status.
        debug_print("Saved {0} bytes, p{1}".format(size, query_status["page"]))

        for item in j["items"]:
            # If updating, done if this html_url has already been downloaded.
            if query_status["updating"]:
                html_url = item["html_url"].replace("#","%23")
                file_name = item["file"]
                # If the same version of an existing notebook, done.
                if html_url in saved_urls:
                    debug_print(("This notebook has already been "
                        "downloaded! Stop looking here.")
                    )
                    query_status["another_page"] = False
                    query_status["done"] = True
                    break
                # If new version of an existing notebook, delete old.
                elif file_name in current_notebooks:
                    if query_status["local"]:
                        os.remove("../data/notebooks/{0}".format(file_name))
                    else:
                        s3.Object(
                            "notebook-research",
                            "notebooks/{0}".format(file_name)
                        ).delete()

            # If we"ve retrieved num_results notebooks, we"re done.
            path = item["repository"]["full_name"] + "/" + item["path"]
            query_status["all_items"].append(path)
            if len(query_status["all_items"]) == query_status["num_results"]:
                query_status["another_page"] = False
                query_status["done"] = True
                break

            query_status["num_needed"] += 1

        # Write progress to log adn display 
        log_string = "{0}: {1} bytes p{2} {3} items".format(
            date, size, query_status["page"], len(j["items"])
        )
        write_to_log("../logs/nb_metadata_query_log.txt", log_string)
        debug_print(log_string)

        # if less than 100 items on the page, it"s the last page
        # at most 10 pages
        if len(j["items"]) < 100 or query_status["page"] == 10:
            query_status["done"] = True

    return r, limit_status, query_status


def clean_metadata(num_needed, updating, local):
    """ 
    Extract information from metadata JSON files and save to CSVs. 
    Equivalent to Adam's 1_nb_metadata_cleaning.ipynb.
    """

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

                    if repo_id not in repos:
                        repo = {
                            "repo_name": item_repo["name"],
                            "owner_id": owner_id,
                            "repo_description": item_repo["description"],
                            "repo_fork": item_repo["fork"],
                            "repo_html_url": item_repo["html_url"],
                            "repo_private": item_repo["private"],
                        }
                        repos[repo_id] = repo

                    if owner_id not in owners:
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
                "we have {0} notebooks.").format(len(notebooks)))
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
        notebooks_df.to_csv("{0}/notebooks1.csv".format(PATH), index = False)
        owners_df.to_csv("{0}/owners1.csv".format(PATH), index = False)
        repos_df.to_csv("{0}/repos1.csv".format(PATH), index = False)
    else:
        df_to_s3(notebooks_df, "csv/notebooks1.csv")
        df_to_s3(owners_df, "csv/owners1.csv")
        df_to_s3(repos_df, "csv/repos1.csv")


if __name__ == "__main__":
    main()