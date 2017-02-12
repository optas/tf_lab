import os
import sys

category = '03001627'
model_path = os.path.join('/orions4-zfs/projects/lins2/Panos_Space/DATA/ShapeNetPointClouds/from_manifold_meshes/1024/',category)

train_file_list = open('trainlist.txt','w')
test_file_list = open('testlist.txt','w')

model_list = [line for line in os.listdir(model_path)]

for idx,line in enumerate(model_list):
    model_abs_path = os.path.join(model_path,line)
    if idx % 10 < 9:
        train_file_list.writelines(model_abs_path+'\n')
    else:
        test_file_list.writelines(model_abs_path+'\n')
