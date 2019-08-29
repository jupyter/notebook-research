## To Query on Your Own

### 1. Set up Directory Structure

- `query_git.py`
- `parallelize_query.py`
- `download.py`
- `parallelize_download.py`
- `extract_data.py`
- `parallelize_extract.py`
- analysis_notebooks
- data 
    - json
    - notebooks
    - repos
- logs
- csv

### 2. Querying Git

Querying git requires the use of at least one personal access token. These can be aquired in developer settings. For more information see the (GitHub API)[https://developer.github.com/v3/auth/].

Access tokens should be saved as environment variables with the prefix GITHUB_TOKEN. For instance: `export GITHUB_TOKEN3="..."`. 

```
# in parallel across multiple github tokens
# will run with nohup, progress saved in query_{#}.log
python3 parallelize_query.py min max [--update]

# -- or --

# all at once on one github token
python3 query_git.py min max [--update]
```

- `min` and `max` are limits for file sizes (in bytes) that will be queried. Files on GitHub range from 0 to 100,000,000 bytes (100MB).
- Adding the `--update` flag looks for new or updated notebooks in the given size range. Without this flag, the program will not search size ranges that have already been searched.

```
# if query was run in parallel, use process.py to combine
python3 process.py [--update]
```

### 3. Downloading Notebooks

```
# in parallel across multiple github tokens
# will run with nohup, progress saved in download_{#}.log
python3 parallelize_download.py [--local]

# -- or --

# all at once on one github token
python3 download.py
```

### 4. Processing Data

#### 4.1 Only if using data from Adam Rule's 2017 corpus
(https://library.ucsd.edu/dc/object/bb2733859v)

```
python3 convert.py
```

- Assumes the structure described above with `convert.py` in root directory, three downloaded CSVs in `csv`, and all downloaded notebooks (nb_0.ipynb, nb_1.ipynb, etc.) in `data/notebooks`.
            
#### 4.2

```
# in parallel across 10 workers (does not rely on GitHub tokens)
# will run with nohup, progress saved in extract_{#}.log
python3 parallelize_extract.py

# -- or --

# all at once
python3 extract_data.py
```

### 5. Running Analysis Notebooks

- Notebooks within the `analysis_notebooks` directory can be run only if `extract_data.py` has been run to completion such that final csv files are the the `csv` directory. Further, `aggregations.py` will need to be run to create aggregations for quicker analysis.
