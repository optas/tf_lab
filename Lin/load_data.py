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

def load_data():
    point_cloud_list = []
    model_list = [line.strip().split('_pts.txt')[0] for line in os.listdir(filelist)]
    for pc_file in model_list:
        filename = os.path.join(filelist,pc_file+'_pts.txt')
        pointcloud = readPointFile(filename)
        point_cloud_list.append(pointcloud)
    return point_cloud_list,model_list
