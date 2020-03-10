#-------------------------------------------------------------------------------
# Name:        Toy Bayesian Inference for linear regression

##########
# Main
##########

# Purpose:     
#              1. for education and practice
#              2. ...
#              Main method is ...
#
# Modifications ~~:
#              1.
#
# Author:      Jiong Wang
#
# Created:     10/03/2020
# Copyright:   (c) JonWang 2020
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# Prior function for a simple linear model with only a interception and a slope
def pri(theta1,theta2):
    sigma = 3  # Standard deviation of the Gaussian prior
    func = (1/np.sqrt(6.28*sigma**2))*np.exp((theta1**2 + theta2**2)**2/(-2*sigma**2))
    return func

# Likelihood function for one observation
# For multiple observations only needs to multiply this function
def likeli(theta1,theta2,obs_y,obs_x):  # It is a function of theta with known observations
    sigma = 1  # Standard deviation of the Gaussian likelihood
    func = (1/np.sqrt(6.28*sigma**2))*np.exp((obs_y-theta1-theta2*obs_x)**2/(-2*sigma**2))
    return func

# True linear function
def linearF(x):
    func = 3 + 2*x
    return func

# Function for visualizing observations and likelihood
def plotLikelihood(obs_x, obs_y, k, likelihood):
    # Observations
    plt.subplot(1,2,1)
    plt.plot(obs_x,obs_y,'k.',obs_x[:k+1],obs_y[:k+1],'ro')
    plt.plot(true_x,true_y,'g--')
    plt.xlim([-5.,5.])
    plt.ylim([-7.5,12.5])
    plt.xlabel('x')
    plt.ylabel('y')
    # Likelihood
    plt.subplot(1,2,2)
    plt.imshow(likelihood,cmap='YlOrRd',extent=[-5,5,-5,5])
    plt.xlabel('θ1')
    plt.ylabel('θ2')

    

# Ground truth and noisy observations
obs_x = np.random.rand(3)[:,None]; obs_x = obs_x*8-4  # Draw 3 noisy observations in [-4,4]
obs_y = linearF(obs_x) + np.random.randn(len(obs_x),1)*1.0

true_x = np.arange(-5.0,5.1,0.1)  # Ground truth data showing the linear function
true_y = linearF(true_x)

# Visualize ground truth data and noisy observation
plt.plot(obs_x,obs_y,'ko',true_x,true_y,'g--')
plt.legend(['observed','true'], loc=0)
plt.xlim([-5.,5.])
plt.ylim([-7.5,12.5])
plt.xlabel('x')
plt.ylabel('y')


# Define theta space for visualizing likelihood and prior
theta1 = np.arange(-5.0,5.1,0.1)
theta2 = np.arange(-5.0,5.1,0.1)
theta2 = theta2[::-1]  # minimum value from lowerleft corner
T1,T2 = meshgrid(theta1, theta2) # grid of point

# Loop over number of observations
likelihood = 1
for k in range(3):  #(len(obs_y)):
    prior = pri(T1,T2) 
    likelihood *= likeli(T1, T2, obs_y[k], obs_x[k]) # evaluation of the function on the grid
    posterior = prior*likelihood


# Visualization in 2D
#im = imshow(posterior,cmap='YlOrRd') # drawing the function
#colorbar(im) # adding the colobar on the right
#show()
plotLikelihood(obs_x, obs_y, k, likelihood)
plt.imshow(prior,cmap='GnBu',extent=[-5,5,-5,5])
plt.imshow(likelihood,cmap='YlOrRd',extent=[-5,5,-5,5])


# Plot 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(T1, T2, posterior, rstride=1, cstride=1, 
                      cmap='coolwarm',linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

