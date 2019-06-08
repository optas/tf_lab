import os

temp_files = ['./tf_nndistance_g.cu.o', './tf_nndistance_so.so', './tf_nndistance.pyc']

for f in temp_files:
    if os.path.exists(f):
        os.remove(f)
