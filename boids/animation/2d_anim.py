import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

data = np.load('./data_1000_500_better.npy')
print(data.shape)
fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, 
                    xlim=(0, 300), 
                    ylim=(0, 300))
ax.grid()

scat = ax.scatter(data[0][0][:,0], data[0][0][:,0],
                 s=0.5)

def animate(i):
    pos, vel = data[i]
    scat.set_offsets(pos)
    return scat 

ani = animation.FuncAnimation(fig, animate, range(1, len(data)),
                              interval=0.001)
ani.save('boids_mpi.mp4', writer=writer)