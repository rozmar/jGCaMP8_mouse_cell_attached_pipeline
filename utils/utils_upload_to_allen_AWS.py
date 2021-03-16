import threading
import boto3
import os
import sys
from boto3.s3.transfer import TransferConfig
import random
import ray
import time
import json
# pip install boto3
bucket_name = "gcamp8deep"

# This is the variable to point at local folder, you can add multiple file paths in the list




def get_list_files_to_upload(local_root_tree):
    files = []
    for r, d, f in os.walk(local_root_tree):
        for file in f:
            files.append(os.path.join(r, file))

    return files


def file_path_replace(file_path, os_check=True):
    """Parses file path, possible based on OS.
    
    If Windows, replaces / with \\. Otherwise does nothing.

    Args:  
        file_path (str):  The file path
        os_check (boolean, optional):  If True, check the OS. False, perform operation regardless.  

    Returns: 
        A string representing the file path.
    """
    if os.name == "nt" and os_check:
        if "/" in file_path:
            return "\\{}".format(file_path.replace("/", "\\"))
    elif "\\" in file_path:
        return file_path.replace("\\\\", "/").replace("\\", "/")

    return file_path


def multi_part_upload_with_s3(conn, local_file, remote_file, bucket_name):
    # Multipart upload

    config = TransferConfig(
        multipart_threshold=1024 * 25,
        max_concurrency=5,
        multipart_chunksize=1024 * 25,
        use_threads=False,
    )
    file_path = local_file
    key_path = remote_file
    conn.upload_file(
        file_path,
        bucket_name,
        key_path,
        ExtraArgs={"ACL": "bucket-owner-full-control"},
        Config=config,
        #Callback=ProgressPercentage(file_path),
    )


# this is the callback function that updates progress
class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify we'll assume this is hooked up
        # to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)"
                % (self._filename, self._seen_so_far, self._size, percentage)
            )
            sys.stdout.flush()

@ray.remote   
def dotheupload(randomseed = None,creds=None):    
    list_folders = ["/home/rozmar/Mount/HDD_RAID_2_16TB/GCaMP8_deepinterpolation_export"]
    access_key = creds['access_key']
    secret_key = creds['secret_key']
    conn = boto3.client(
        service_name="s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )

    # We first clean path to be os independent
    os_fixed_paths = []
    for indiv_folder in list_folders:
        os_fixed_paths.append(file_path_replace(indiv_folder))

    # Shuffle for multi-threaded uploads.
    
    random.shuffle(list_folders)
    for index_folder, root_path in enumerate(list_folders):
        aws_root = os.path.basename(root_path)

        list_files = get_list_files_to_upload(root_path)
        if not randomseed == None:
            random.seed(randomseed)
        random.shuffle(list_files)
        for index_file, each_file in enumerate(list_files):
            # print('Copying '+each_file)
            local_file_aws = each_file[len(root_path) :]

            # For windows
            local_aws_base = aws_root + "\\" + local_file_aws
            local_aws_base = local_aws_base.replace("\\", "/")

            # For unix
            local_aws_base = local_aws_base.replace("//", "/")

            try:
                response = conn.head_object(Bucket=bucket_name, Key=local_aws_base,)
            except:
                print(each_file)
                multi_part_upload_with_s3(conn, each_file, local_aws_base, bucket_name)
    
with open('aws_creds.json') as f:
  creds = json.load(f)
cores = 10
ray.init(num_cpus = cores)
result_ids = list()
for coreidx in range(cores):
    result_ids.append(dotheupload.remote(coreidx+95,creds))        
ray.get(result_ids)    
ray.shutdown()