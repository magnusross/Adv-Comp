import numpy as np
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d',
                    xlim=(-200, 200), 
                    ylim=(-200, 200),
                    zlim=(-200, 200))

data = np.load('results_mpi.npy')

scat = ax.scatter(data[0][0][:,0], data[0][0][:,1], data[0][0][:,2])

def animate(i):
    pos, vel = data[i]
    scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
    return scat 

ani = animation.FuncAnimation(fig, animate, range(1, len(data)),
                              interval=0.001)
ani.save('boids_3d.mp4', writer=writer)