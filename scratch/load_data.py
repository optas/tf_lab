from global_variables import *
import numpy as np
import os

def readPointFile(filename):
    pointlist = np.zeros([Npoint,3],dtype=np.float32)
    numpoint = 0
    for l in open(filename):
        if not l.startswith('#'):
            x,y,z= map(float,l.strip().split())
            pointlist[numpoint][0] = x
            pointlist[numpoint][1] = y
            pointlist[numpoint][2] = z
            numpoint = numpoint + 1
    return pointlist

def load_data(filename):
    point_cloud_list = []
    model_list = []
    for model_path in open(filename):
        model_path = model_path.strip()
        pointcloud = readPointFile(model_path)
        model_id = model_path.split('/')[-1].split('_pts.txt')[0]
        point_cloud_list.append(pointcloud)
        model_list.append(model_id)
    return point_cloud_list,model_list
