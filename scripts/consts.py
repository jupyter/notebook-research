import subprocess
import boto3


# GitHub tokens and http request authentication.
ENV_VARS = subprocess.check_output("printenv", shell=True).decode().split("\n")

# Finds all GitHub tokens stored as environment variables.
# GitHub token variables must be prefixed with 'GITHUB_TOKEN'
# (e.g. export GITHUB_TOKEN1="...")
TOKENS = {
	token.split("=")[0]: token.split("=")[1] 
	for token in 
	[s for s in ENV_VARS if s.startswith("GITHUB_TOKEN")]
}

HEADERS = [{"Authorization": "token {0}".format(token)} 
           for token in TOKENS.values()]

NUM_WORKERS = len(HEADERS)


# Connecting to S3 data storage.
s3 = boto3.resource("s3")

bucket = s3.Bucket("notebook-research")


# Data collection constants.
URL = (
	"https://api.github.com/search/code?"
	+ "per_page=100&language=Jupyter+Notebook&"
	+ "sort=indexed&order=desc&q=ipynb+in:path+extension:ipynb+size:"
)

COUNT_TRIGGER = 10000

LOGGING = True

DEBUG_PRINTING = True

BREAK = "\n\n"+"-"*80+"\n\n"	# Line break for output.

PATH = "../csv"

JSON_PATH = "../data/json/"