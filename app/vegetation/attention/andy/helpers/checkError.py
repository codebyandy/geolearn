import os
import glob

files = glob.glob(f"/scratch/users/avhuynh/jobs/*.err")
newest_err_file = max(files, key=os.path.getmtime)

with open(newest_err_file, 'r') as file:
    contents = file.read()
    print(contents)

