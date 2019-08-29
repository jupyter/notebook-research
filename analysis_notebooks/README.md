# Analysis Notebooks

The analysis notebooks are best understood if looked at in the order indicated by `TableOfContents.ipynb`.

## Running Analysis Notebooks

Included in this repositry is a small sample dataset of 10 Jupyter Notebooks. This can be used to understand the structure of the data without downloading all 4.3 million notebooks. In the `analysis_data` directory are the aggregation results of `aggregate.py` on this data. They are all pickled objects to make it easier to keep data types consistent.

## If running with your own data

Notebooks within the `analysis_notebooks` directory can be run only if `extract_data.py` has been run to completion such that final csv files are the the `csv` directory. 

After this, the four final CSV files (`notebooks_final.csv`, `repos_final.csv`, `owners_final.csv`, and `cells_final.csv`) should be moved to the `data_final` directory here. 

Further, `aggregate.py` will need to be run to create aggregations for quicker analysis.