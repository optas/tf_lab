def gen_plot(points,modelid,epoch)
    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    plt.save()
