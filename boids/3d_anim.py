import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure(figsize=(50, 50))
ax = fig.add_subplot(111, projection='3d',
                    xlim=(0, 300), 
                    ylim=(0, 300),
                    zlim=(0, 300))

data = np.load('res_grid.npy')

scat = ax.scatter(data[0, :, 1, 0], data[0, :, 1, 1], data[0, :, 1, 2], s=30, )

def animate(i):
    pos = data[i, :, 1, :]
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    return scat 

ani = animation.FuncAnimation(fig, animate, range(1, len(data)),
                              interval=0.0001)
ani.save('boids_3d_grid.mp4', writer=writer)