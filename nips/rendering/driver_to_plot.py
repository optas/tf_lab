cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:,:3]

color = cmap[np.array([0] * 2500 + [2] * 2000), :]

point_np = points.transpose(2,1).data.numpy()

showpoints(point_np, color)