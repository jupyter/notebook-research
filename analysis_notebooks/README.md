# Analysis Notebooks

The analysis notebooks are best understood if looked at in the order indicated by [TableOfContents.ipynb](TableOfContents.ipynb).

## Running Analysis Notebooks

Included in this repositry is a small sample dataset of 10 Jupyter Notebooks. This can be used to understand the structure of the data without downloading all 4.3 million notebooks. In the `analysis_data` directory are the aggregation results of `aggregate.py` on this data. They are all pickled objects to make it easier to keep data types consistent.

## If running with your own data

1. Notebooks within the `analysis_notebooks` directory can be run only if `extract_data.py` has been run to completion. 
2. When data has been extracted, run through `PrepareData.ipynb` to get csv files from the S3 bucket, combine them, and remove missing values and notebooks that are a part of ipynb_checkpoints.

At this point, the four final CSV files (`notebooks_final.csv`, `repos_final.csv`, `owners_final.csv`, and `cells_final.csv`) are in the `data_final` directory here. 

3. Finally, run `aggregate.py` to create the aggregated DataFrame. This can take up to 20 hours to run, so I'd recommend running it as a background process and monitoring its log file: `nohup python3 -u aggregate.py > aggregate_status.log &`.
