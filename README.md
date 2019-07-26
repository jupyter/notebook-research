# notebook-research
# Public GitHub Notebook Corpus Research Collaboration

## Project Description
As an extension of work done by [Rule et al.](https://blog.jupyter.org/we-analyzed-1-million-jupyter-notebooks-now-you-can-too-guest-post-8116a964b536) in 2017, the goal of this project is to collect and analyze all public Jupyter Notebooks on GitHub ([~1 million in 2017](https://library.ucsd.edu/dc/collection/bb6931851t), [now nearly 5 million](https://github.com/parente/nbestimate/blob/master/estimate.ipynb)). This analysis will help designers, developers, and researchers in the Jupyter community quantitatively assess how people use notebooks, with an emphasis on applications in data science, machine learning, and information visualization. This GitHub repository holds the scripts used to search, download, and process notebooks from GitHub, the resulting CSV data files, and the set of notebooks used to analyze the corpus. Given the number and size of notebooks on GitHub, we expect these corpus notebooks to be stored in a separate repository optimized for data storage.

The results of this research can complement qualitative user studies and inform challenging UX questions to focus development on real user needs. This understanding of notebook applications is crucial to user-centered design. Because many of the notebooks hosted publicly on GitHub are created as part of educational endeavours such as online and in-person courses, these insights may be particularly valuable for the Jupyter education community.

## Collaboration

The initial collaboration will involve several individuals and institutions:
- Adam Rule: The original author of the work, and currently a postdoc at OHSU.
- Jenna Landy: An AWS summer data science intern and contributor to Project Jupyter.
- Markelle Kelly: A Project Jupyter software engineering and data science intern.
- Brian Granger: An AWS Principal TPM and co-founder of Project Jupyter.
- Tim George: The UX Designer/Research for Project Jupyter.
- The broader Jupyter open source community. There is an emerging community of HCI and researchers in Jupyter's open source community that we will engage with.

------------

## To Query on Your Own

### 1. Set up Directory Structure

- `query_git.py`
- `extract_data.py`
- analysis_notebooks
- data 
    - json
    - notebooks
    - readmes
    - repos
- logs
- csv

### 2. Querying Git

```
python3 query_git.py min max [--update]
```

- `min` and `max` are limits for file sizes that will be queried
- Adding the `--update` flag looks for new or updated notebooks in the given size range. 
  Without this flag, the program will not search size ranges that have already been searched.

### 3. Downloading Notebooks

```
python3 download.py
```

### 4. Processing Data

#### 4.1 Only if using data from Adam Rule's 2017 corpus
(https://library.ucsd.edu/dc/object/bb2733859v)

```
python3 convert.py
```

- Assumes the structure described above with `convert.py` in root directory, three downloaded
  CSVs in `csv`, and all downloaded notebooks (nb_0.ipynb, nb_1.ipynb, etc.) in `data/notebooks`.
            
#### 4.2

```
python3 extract_data.py
```

### 5. Running Analysis Notebooks

- Notebooks within the `analysis_notebooks` directory can be run only if `extract_data.py` has been run to completeion such that final csv files are the the `csv` directory.