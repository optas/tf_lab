from global_variables import *
import matplotlib.pyplot as plt
import os
import io
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D
def gen_plot(points,modelid,epoch):
    x = []
    y = []
    z = []
    for i in xrange(Npoint):
        x.append(points[i*3])
        y.append(points[i*3+1])
        z.append(points[i*3+2])
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.view_init(elev=10,azim=40)
    ax.scatter(x, y, z, c='r', marker='o')
    figname = str(epoch) + str(modelid)+'.png'
    plt.savefig(os.path.join('/orions4-zfs/projects/lins2/Lin_Space/DATA/Lin_Data/img/',figname))
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf
