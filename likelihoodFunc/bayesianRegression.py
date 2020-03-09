# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:53:07 2020

@author: wang0096
"""

import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

# Prior function
def pri(theta1,theta2):
    func = (1/np.sqrt(6.28))*np.exp((theta1**2 + theta2**2)**2/(-2))
    return func  #(1-(x**2+y**3))*exp(-(x**2+y**2)/2)

# Likelihood function given one observation
def likeli(theta1,theta2,obs_y,obs_x):
    func = (1/np.sqrt(6.28))*np.exp((obs_y-theta1-theta2*obs_x)**2/(-2))
    return func  #(1-(x**2+y**3))*exp(-(x**2+y**2)/2)

# define theta space
theta1 = np.arange(-5.0,5.1,0.1)
theta2 = np.arange(-5.0,5.1,0.1)
theta2 = theta2[::-1]  # minimum value from lowerleft corner
T1,T2 = meshgrid(theta1, theta2) # grid of point

# true linear function
obs_x = np.random.rand(3)[:,None]; obs_x = obs_x*3  # draw 3 observations in [0,3]
obs_y = 3 + 2*obs_x + np.random.randn(len(obs_x),1)*0.1
obs = np.hstack((obs_x, obs_y))

# loop over number of observations
posterior = 1
for k in range(3):  #(len(obs)):
    prior = pri(T1,T2) 
    likelihood = likeli(T1, T2, obs[k,1], obs[k,0]) # evaluation of the function on the grid
    posterior *= likelihood  #*prior
    #plt.imshow(prior,cmap='coolwarm',extent=[-5,5,-5,5])
plt.imshow(posterior,cmap='coolwarm',extent=[-5,5,-5,5])

# Plot 2D
im = imshow(posterior,cmap='coolwarm') # drawing the function
# adding the Contour lines with labels
#cset = contour(Z,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
#clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
# latex fashion title
#title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
show()

# Plot 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(T1, T2, posterior, rstride=1, cstride=1, 
                      cmap='coolwarm',linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

