from io import StringIO, BytesIO

import pandas as pd

from consts import (
    LOGGING,
    DEBUG_PRINTING,
    s3, 
    bucket
)

def write_to_log(f, msg):
	""" Write given message to given log file. """
	if LOGGING:
		log_file = open(f, "a")
		log_file.write(msg + "\n")
		log_file.close()

def debug_print(msg):
	""" Prints to console if DEBUG_PRINTING is True. """
	if DEBUG_PRINTING:
		print(msg)

def df_to_s3(data_frame, path):
	obj = s3.Object("notebook-research", path)
	csv_buffer = StringIO()
	data_frame.to_csv(csv_buffer, index = False)
	obj.put(Body=csv_buffer.getvalue())
	
def s3_to_df(path):
	df_obj = s3.Object("notebook-research", path)
	return pd.read_csv(BytesIO(df_obj.get()["Body"].read()), header = 0)

def list_s3_dir(path):
	list_dir = set([])
	for obj in bucket.objects.filter(Prefix = path):
		list_dir.add(obj.key.split("/")[1])
	return list_dir