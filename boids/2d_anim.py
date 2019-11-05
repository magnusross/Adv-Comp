import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

data = np.load('results.npy')

fig = plt.figure()
ax = fig.add_subplot(111, autoscale_on=False, xlim=(-400, 400), ylim=(-200, 200))
ax.grid()

scat = ax.scatter(data[0][0][:,0], data[0][0][:,0],
                 s=0.5)

def animate(i):
    pos, vel = data[i]
    scat.set_offsets(pos)
    return scat 

ani = animation.FuncAnimation(fig, animate, range(1, len(data)),
                              interval=0.001)
ani.save('boids.mp4', writer=writer)