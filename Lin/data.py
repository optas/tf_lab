import numpy as np
import os
import sys
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print BASE_DIR
sys.path.append(BASE_DIR)

#config
Npoint = 3000
trX = []
trY = []

def readPointFile(fileName):
    pointList = np.zeros([Npoint,3])
    numPoint = 0
    for l in open(fileName):
            x,y,z = map(float,l.strip().split())
            pointList[numPoint][0] = x
            pointList[numPoint][1] = y
            pointList[numPoint][2] = z
            numPoint = numPoint + 1
    return pointList

def loadData():
    fileListPath = '/orions4-zfs/projects/lins2/Panos_Space/DATA/ShapeNetPointClouds/3000/no_segmentations/03001627'
    fileList = [line.strip().split('_pts.txt')[0] for line in os.listdir(fileListPath)]
    for line in fileList:
        filePath = os.path.join(fileListPath,line+'_pts.txt')
        if os.path.exists(filePath) and len(trX) < 1000 :
            pointcloud = readPointFile(filePath)
            trX.append(pointcloud)
            trY.append(pointcloud)
            print len(trX)

if __name__ == '__main__':
    loadData()
