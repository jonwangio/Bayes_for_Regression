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
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import stats


#####################################################
# 01 Functions for prior, likelihood, and visualization
#####################################################

# True linear model
def linearF(x, theta1, theta2):  # A very simple univariate linear model
    func = theta1 + theta2*x  # f(x)
    return func

# Plot sample Ordinary Least Square (OLS) fits
def plotSampleOLS(true_x, n):    
    obs_x = np.random.rand(n); obs_x = obs_x*10-5  # Draw n noisy observations in [-4,4]
    obs_y = linearF(obs_x, 3, 2) + np.random.randn(len(obs_x))*1.0  # f(x) = 3 + 2x + e
    true_y = linearF(true_x, 3, 2)
    # Linear fit using OLS
    r = stats.linregress(obs_x, obs_y)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(true_x,true_y,'g--')
    line, = ax.plot(true_x, linearF(true_x, r.intercept, r.slope), 'c-', linewidth=1)
    point, = ax.plot(obs_x, obs_y, 'ko')
    ax.legend(['true','fit','observed'], loc=0)
    ax.set_xlim([-5.,5.])
    ax.set_ylim([-7.5,12.5])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.tight_layout()
    
    def update(i):
        # Another set of observations
        obs_x = np.random.rand(n); obs_x = obs_x*10-5  # Draw n noisy observations in [-4,4]
        obs_y = linearF(obs_x, 3, 2) + np.random.randn(len(obs_x))*1.0  # f(x) = 3 + 2x + e
        r = stats.linregress(obs_x, obs_y)  # Another fit
        point.set_data(obs_x, obs_y)
        line.set_ydata(linearF(true_x, r.intercept, r.slope))
    
    ani = FuncAnimation(fig, update, frames=np.arange(0, 50), interval=1)
    ani.save('0_sampleOLS.gif', PillowWriter(fps=5))  # dpi=400, writer='imagemagick')
    
    return None

# Prior function for a simple linear model with only a interception and a slope
def pri(theta1,theta2):
    sigma = 5  # Standard deviation of the Gaussian prior
    func = (1/np.sqrt(6.28*sigma**2))*np.exp((theta1**2 + theta2**2)**2/(-2*sigma**2))
    return func

# Likelihood function for one observation
# For multiple observations only needs to multiply this function
def likeli(theta1,theta2,obs_y,obs_x):  # It is a function of theta with known observations
    sigma = 1  # Standard deviation of the Gaussian likelihood
    func = (1/np.sqrt(6.28*sigma**2))*np.exp((obs_y-theta1-theta2*obs_x)**2/(-2*sigma**2))
    return func

# Posterior estimation of the model parameters
def post(pri, likeli):
    return pri*likeli

# Predictive distribution of model based upon a SINGLE [theta1, theta2]
def pred(x_pred, y_pred, theta1, theta2, posterior):
     func = (y_pred - linearF(x_pred, theta1, theta2))*posterior
     return func

# Function for visualizing observations and likelihood
def plotLikelihood(obs_x, obs_y, k, likelihood):
    # Observations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
    ax1.plot(obs_x,obs_y,'k.',obs_x[:k+1],obs_y[:k+1],'ro')
    ax1.plot(true_x,true_y,'g--')
    ax1.set_xlim([-5.,5.])
    ax1.set_ylim([-7.5,12.5])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    # Likelihood
    ax2.imshow(likelihood,cmap='YlOrRd',extent=[-5,5,-5,5])
    ax2.set_xlabel('θ1')
    ax2.set_ylabel('θ2')
    plt.tight_layout()
    
# Function for visualizing Bayes and random draw from the bayes
def plotBayesDraw(T1, T2, bayes, true_x):
    T = np.hstack((T1.reshape(-1,1),T2.reshape(-1,1)))  # Columns of theta tuple
    p = bayes.reshape(-1,)  # Prior as the probability to draw 
    thetaInd = np.random.choice(np.arange(len(T)), None, p=p/np.sum(p))
    # Observations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
    ax1.imshow(bayes,cmap='YlGnBu',extent=[-5,5,-5,5])
    point, = ax1.plot(T[thetaInd][0], T[thetaInd][1], 'r.', markersize=15)
    ax1.set_xlabel('θ1')
    ax1.set_ylabel('θ2')
    
    # Distribution
    #'''
    # This is only for posterior model distribution
    trans = bayes/bayes.max()  # Visualizing posteriror distribution of model line transparency    
    # Iteration over all possible thetas
    for row in range(10, 60):  #T1.shape[0]):  # Theta indices
        for col in range(40,90):  #T1.shape[1]):
            # Each [theta1, theta2] along with possibility as transparency
            ax2.plot(true_x, T1[row,col]+T2[row,col]*true_x,'c-', alpha=trans[row,col]*0.6)
    #'''
    
    line, = ax2.plot(true_x, T[thetaInd][0]+T[thetaInd][1]*true_x,'k--')
    ax2.plot(obs_x,obs_y,'ko')
    ax2.set_xlim([-5.,5.])
    ax2.set_ylim([-7.5,12.5])
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    plt.tight_layout()
    
    def update(i):
        thetaInd = np.random.choice(np.arange(len(T)), None, p=p/np.sum(p))
        point.set_data(T[thetaInd][0], T[thetaInd][1])
        line.set_ydata(T[thetaInd][0]+T[thetaInd][1]*true_x)
    
    ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=100)
    ani.save('5_postDraw.gif', PillowWriter(fps=20))  # dpi=400, writer='imagemagick')
    
# Function for visualizing posterior with different number of observations
def plotObsPost(T1, T2, obs_x, obs_y, true_x, true_y):
    # Observations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
    ax1.plot(obs_x,obs_y,'ko',true_x,true_y,'g--')
    point, = ax1.plot(obs_x[0], obs_y[0], 'r.', markersize=15)
    ax1.set_xlim([-5.,5.])
    ax1.set_ylim([-7.5,12.5])
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    # Posterior
    prior = pri(T1,T2) 
    likelihood = likeli(T1, T2, obs_y[0], obs_x[0]) # evaluation of the function on the grid
    posterior = prior*likelihood
    im = ax2.imshow(posterior,cmap='YlGnBu',extent=[-5,5,-5,5])
    ax2.set_xlabel('θ1')
    ax2.set_ylabel('θ2')
    plt.tight_layout()
    
    def update(i):
        like=1 # evaluation of the function on the grid
        pr = pri(T1,T2)
        for k in range(i):  #(len(obs_y)):
            like *= likeli(T1, T2, obs_y[k], obs_x[k]) # evaluation of the function on the grid
        post = pr*like
        vmax = np.max(post)
        vmin = np.min(post)
        im.set_array(post)
        im.set_clim(vmin, vmax)
        point.set_data(obs_x[:i+1], obs_y[:i+1])
    
    ani = FuncAnimation(fig, update, frames=np.arange(0,len(obs_x)+1), interval=500)
    ani.save('4_postObs.gif', dpi=400, writer='imagemagick')
    
    
#####################################################
# 02 True model and noisy observations
#####################################################

# Ground truth and noisy observations
n = 3  # Number of observations
obs_x = np.random.rand(n); obs_x = obs_x*10-5  # Draw n noisy observations in [-4,4]
obs_y = linearF(obs_x, 3, 2) + np.random.randn(len(obs_x))*1.0  # f(x) = 3 + 2x + e

true_x = np.arange(-5.0,5.1,0.005)  # Ground truth data showing the linear function
true_y = linearF(true_x, 3, 2)

# Visualize ground truth data and noisy observation
plt.plot(obs_x,obs_y,'ko',true_x,true_y,'g--')
plt.legend(['observed','true'], loc=0)
plt.xlim([-5.,5.])
plt.ylim([-7.5,12.5])
plt.xlabel('x')
plt.ylabel('y')


#####################################################
# 03 Likelihood and prior for theta_1 and theta_2
#####################################################

# Define theta space for visualizing likelihood and prior
theta1 = np.arange(-5.0,5.1,0.1)
theta2 = np.arange(-5.0,5.1,0.1)
theta2 = theta2[::-1]  # minimum value from lowerleft corner
T1,T2 = meshgrid(theta1, theta2) # grid of point

# Loop over number of observations
likelihood = 1
for k in range(5):  #(len(obs_y)):
    prior = pri(T1,T2) 
    likelihood *= likeli(T1, T2, obs_y[k], obs_x[k]) # evaluation of the function on the grid    
    posterior = post(prior, likelihood)
    

#####################################################
# 04 Prediction
#####################################################

# Using a discrete simulation of integration
# Simulate each potential function by enumaerate all thetas
# Combine all potential functions by using the posterior distribution of thetas
# Define X, Y space for visualizing posteriori of Y=f(x)+e
xx = np.arange(-5.0,5.1,0.1)
yy = np.arange(-7.5,12.5,0.1)
yy = yy[::-1]  # minimum value from lowerleft corner
X,Y = meshgrid(xx, yy) # grid of point

'''
trans = posterior/posterior.max()  # Visualizing posteriror distribution of model line transparency

fig, ax = plt.subplots(figsize=(6,4))  # Set figure
ax.set_xlim([-5.,5.])
ax.set_ylim([-7.5,12.5])
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.tight_layout()

# Iteration over all possible thetas
for row in range(T1.shape[0]):  # Theta indices
    for col in range(T1.shape[1]):
        # Each [theta1, theta2] along with possibility as transparency
        ax.plot(true_x, T1[row,col]+T2[row,col]*true_x,'g--', alpha=trans[row,col])
'''


#####################################################
# 05 Visualization
#####################################################

# Visualization in 2D
# Likelihood
plotLikelihood(obs_x, obs_y, k, likelihood)
# Prior
plt.imshow(prior,cmap='YlGnBu',extent=[-5,5,-5,5])
plt.xlabel('θ1')
plt.ylabel('θ2')
# Animation of prior draw
plotBayesDraw(T1, T2, prior, true_x)
# Posterior
plt.imshow(posterior,cmap='YlGnBu',extent=[-5,5,-5,5])
plotObsPost(T1, T2, obs_x, obs_y, true_x, true_y)
# Animation of posterior draw
plotBayesDraw(T1, T2, posterior, true_x)
#im = imshow(posterior,cmap='YlOrRd') # drawing the function
#colorbar(im) # adding the colobar on the right
#show()


# Plot 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(T1, T2, posterior, rstride=1, cstride=1, 
                      cmap='coolwarm',linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

