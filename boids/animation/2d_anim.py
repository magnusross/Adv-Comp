import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

#data to animate 
data = np.load('./../results/anim_data/2D_anim_data.npy')
print(data.shape)
fig = plt.figure()

#set domain size to correct one using xlim ylim 
ax = fig.add_subplot(111, autoscale_on=False, 
                    xlim=(0, 500), 
                    ylim=(0, 500))
ax.grid()
plt.axis('off')
scat = ax.scatter(data[0][0][:,0], data[0][0][:,0],
                 s=0.5)

def animate(i):
    pos, vel = data[i]
    scat.set_offsets(pos)
    return scat 

ani = animation.FuncAnimation(fig, animate, range(1, len(data)),
                              interval=0.01)

ani.save('boids_2D.mp4', writer=writer)