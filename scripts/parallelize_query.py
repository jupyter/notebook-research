import pandas as pd
import sys
import os
import argparse
import time

from consts import NUM_WORKERS

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("min", type=int, help="Minimum size to search.")
    parser.add_argument("max", type=int, help="Maximum size to search.")
    parser.add_argument(
        "--update", action="store_const", 
        dest="updating", const=True, default=False, 
        help=(
            "Search notebooks that have been added "
            + "or updated since last search, along with new "
            + "notebooks"
        )
    )
    
    args = parser.parse_args()
    MIN = args.min
    MAX = args.max
    updating = args.updating

    # Load csv with approximated number of notebooks per size interval.
    data_df = pd.read_csv("./nb_counts.csv")

    # Calculate best size distribution among workers.
    workers = distribute_query(data_df, NUM_WORKERS, MIN, MAX)

    # Format commands.
    query_commands = []
    for _, worker in workers.iterrows():
        size = worker["size"]
        size_min = size.split("..")[0]
        size_max = size.split("..")[1]
        if size_min <= size_max:
            query_commands.append(("nohup python3 -u query_git.py {0} {1}"
                " --worker {2}{3}> query_{4}.log &").format(
                    size_min, size_max, 
                    worker["id"], 
                    " --update " if updating else " ",
                    worker["id"]
                )
            )


    i = 0
    for command in query_commands:
        if i in [5]:
            os.system(command)
            time.sleep(10)
        i += 1

def distribute_query(data_df, num_workers, MIN, MAX):
    """
    Find the best size partitions (from MIN to MAX) 
    to distribute among worker.
    """
    num_nbs = sum(data_df.number)
    expected_each = num_nbs // num_workers

    workers = []
    start_idx = 0
    end_idx = 0
    done_worker = False
    done_all = False

    for w in range(num_workers):
        if not done_all:
            done_worker = False
            while not done_worker:
                # If reached end of range, record partition.
                if data_df.iloc[end_idx]["size"] >= MAX:
                    total = data_df[start_idx:-1].number.sum()
                    size = "{0}..{1}".format(
                        (data_df.iloc[start_idx-1]["size"] if w > 0 else MIN), 
                        MAX
                    )
                    worker = {
                        "id": w,
                        "size": size,
                        "total": total
                    }
                    workers.append(worker)
                    done_worker = True
                    done_all = True
                    continue 

                # Notebooks in current partition vs. number expected.
                total = data_df[start_idx:end_idx+1].number.sum()
                diff = abs(total - expected_each)

                # Notebooks in next partition vs. number expected.
                total_next = data_df[start_idx:end_idx+2].number.sum()
                diff_next = abs(total_next - expected_each)

                # Record partition if closer than the next option.
                if ((total >= expected_each or diff < diff_next) 
                    and data_df.iloc[end_idx]["size"] > MIN):
                    size = "{0}..{1}".format(
                        (data_df.iloc[start_idx-1]["size"] if w > 0 else MIN), 
                        data_df.iloc[end_idx]["size"]
                    )
                    worker = {
                        "id": w,
                        "size": size,
                        "total": total
                    }
                    workers.append(worker)
                    done_worker = True
                    continue
                
                end_idx = end_idx+1

            # Set variables for the next worker.
            start_idx = end_idx+1
            end_idx = end_idx+1
            

    # If not all workers are used, split largest partitions in half.
    workers_df = pd.DataFrame(workers).sort_values(by="total")\
        .reset_index(drop=True)
    workers_df["id"] = list(range(len(workers_df)))

    while len(workers_df) < num_workers:
        minimum = int(workers_df.iloc[-1]["size"].split("..")[0])
        maximum = int(workers_df.iloc[-1]["size"].split("..")[1])
        half = minimum + (maximum - minimum) // 2
        new = {
            "id": len(workers_df),
            "size": ["{0}..{1}".format(minimum,half),
                    "{0}..{1}".format(half+1,maximum)],
            "total": [workers_df.iloc[-1]["total"]//2]*2
        }
        workers_df = workers_df[:-1].append(pd.DataFrame(new))\
            .sort_values(by="total")

    workers_df = workers_df.sort_values(by="size")
    workers_df["id"] = list(range(len(workers_df)))
    return workers_df

if __name__ == "__main__":
    main()