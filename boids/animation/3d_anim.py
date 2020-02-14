import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  

Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)

#set domain size to correct one using xlim ylim zlim
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d',
                    xlim=(0, 500), 
                    ylim=(0, 500),
                    zlim=(0, 500))

data = np.load('./../results/anim_data/3D_anim_data.npy')

scat = ax.scatter(data[0, 0, :, 0], data[0, 0, :, 1], data[0, 0, :, 2], s=50, )
plt.axis('off')

def animate(i):
    pos = data[i, 0, :, :]
    scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
    return scat 

ani = animation.FuncAnimation(fig, animate, range(1, len(data)),
                              interval=0.0001)
                        
ani.save('boids_3D.mp4', writer=writer)