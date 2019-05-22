"""
Created on wed Nov 2018

@author: Yibo Yang
"""

import sys
sys.path.insert(0, './Utilities/')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pyDOE import lhs

from models import Burgers_UQPINN

import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
    
if __name__ == "__main__":
    
    # Number of collocation points
    N_f = 10000

    # Number of training data on the boundary (Boundary condition)
    N_b = 100

    # Number of training data for the initial condition
    N_i = 50
    N_u = N_b + N_i

    # Dimension of input, output and latent variable
    X_dim = 1 
    Y_dim = 1
    T_dim = 1
    Z_dim = 1

    # Noise level (in this noise free case is zero)
    err_var = 0.1

    # x and t on the initial condition
    X_i = -1 + 2*np.random.random((N_i))[:,None]
    X_i =  np.sort(X_i, axis=0)
    T_i = np.zeros((N_i))[:,None]

    # x and t on the boundary condition
    X_b1 = np.ones((N_b // 2))[:,None]
    X_b2 = - np.ones((N_b // 2))[:,None]
    T_b1 = np.random.random(N_b // 2)[:,None]
    T_b2 = np.random.random(N_b // 2)[:,None]
    X_b = np.vstack((X_b1, X_b2))
    T_b = np.vstack((T_b1, T_b2))

    # x and t for training points (initial condition + boundary condition)
    X_u = np.vstack((X_i, X_b))
    T_u = np.vstack((T_i, T_b))

    # x and t for collocation points
    X_f = -1 + 2*np.random.random((N_f))[:,None]
    T_f = np.random.random((N_f))[:,None]

    # Noisy initial condition
    def f_initial(X):
        return - np.sin(np.pi * X)
    error = 1.0/np.exp(3.0*(abs(X_i)))*np.random.normal(0,err_var,X_i.size)[:,None]
    def ff(X, error):
        return -np.sin(np.pi*(X+2*error))+error
        
    X_ii = np.linspace(-1,1,500)
    Y_ii = f_initial(X_ii)
    Y_i = ff(X_i,error)

    # Plot the exact initial condition with the noisy data for the initial condition
    plt.figure(1, figsize=(6, 4))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.plot(X_ii, Y_ii, 'b-', label = "Exact", linewidth=2)
    plt.plot(X_i, Y_i, 'kx', label = "Noisy initial condition", alpha = 1.)
    plt.legend(loc='upper right', frameon=False, prop={'size': 11})
    ax = plt.gca()
    plt.xlim(-1.0, 1.0)
    plt.xlabel('$x$',fontsize=11)
    plt.ylabel('$u(0, x)$',fontsize=11)
    plt.savefig('./Initial.png', dpi = 600)

    # data for training 
    Y_b = np.zeros((N_b))[:,None]
    Y_u = np.vstack((Y_i, Y_b)) 

    # Loading the reference solution of Burgers equation
    x = np.linspace(-1., 1., 256)[:,None]
    t = np.linspace(0., 1., 100)[:,None]
    X, T = np.meshgrid(x,t)
    XT = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    X_star = XT[:,0][:,None]
    T_star = XT[:,1][:,None]

    data = scipy.io.loadmat('./burgers_shock.mat')
    Exact = np.real(data['usol']).T

    # Model creation
    layers_P = np.array([X_dim+T_dim+Z_dim,50,50,50,50,Y_dim])
    layers_Q = np.array([X_dim+T_dim+Y_dim,50,50,50,50,Z_dim])  
    layers_T = np.array([X_dim+T_dim+Y_dim,50,50,50,1])

    model = Burgers_UQPINN(X_f, T_f, X_u, T_u, Y_u, layers_P, layers_Q, layers_T, lam = 1.5, beta = 1.)
    
    model.train(nIter = 30000, N_u = N_u, N_f = N_f, batch_size_u = N_u, batch_size_f = N_f)
    
    # Prediction
    N_samples = 500
    samples_mean = np.zeros((X_star.shape[0], N_samples))
    for i in range(0, N_samples):
        samples_mean[:,i:i+1] = model.generate_sample(X_star, T_star)

    # Compare mean and variance of the predicted samples as prediction and uncertainty
    U_pred = np.mean(samples_mean, axis = 1)    
    U_pred = griddata(XT, U_pred.flatten(), (X, T), method='cubic')
    Sigma_pred = np.var(samples_mean, axis = 1)
    Sigma_pred = griddata(XT, Sigma_pred.flatten(), (X, T), method='cubic')

    # Compare the relative error between the prediciton and the reference solution 
    error_u = np.linalg.norm(Exact-U_pred,2)/np.linalg.norm(Exact,2)
    print('Error u: %e' % (error_u))  
    np.save('L2_error.npy', error_u)
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    

    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot(T_u, X_u, 'kx', label = 'Data (%d points)' % (Y_u.shape[0]), markersize = 4, clip_on = False)
    
    line = np.linspace(x.min(), x.max(), 2)[:,None]
    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('$u(t,x)$', fontsize = 10)


    ####### Row 1: u(t,x) slices ##################    
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = U_pred[25,:] - 2.0*np.sqrt(Sigma_pred[25,:])
    upper = U_pred[25,:] + 2.0*np.sqrt(Sigma_pred[25,:])
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    
    ax = plt.subplot(gs1[0, 1])
    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = U_pred[50,:] - 2.0*np.sqrt(Sigma_pred[50,:])
    upper = U_pred[50,:] + 2.0*np.sqrt(Sigma_pred[50,:])
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])
    ax.set_title('$t = 0.50$', fontsize = 10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    
    ax = plt.subplot(gs1[0, 2])
    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
    lower = U_pred[75,:] - 2.0*np.sqrt(Sigma_pred[75,:])
    upper = U_pred[75,:] + 2.0*np.sqrt(Sigma_pred[75,:])
    plt.fill_between(x.flatten(), lower.flatten(), upper.flatten(), 
                     facecolor='orange', alpha=0.5, label="Two std band")
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1,1.1])
    ax.set_ylim([-1.1,1.1])    
    ax.set_title('$t = 0.75$', fontsize = 10)
    savefig('./Prediction')
    

    fig, ax = newfig(1.0)
    ax.axis('off')
    
    #############       Uncertainty       ##################
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs2[:, :])
    
    h = ax.imshow(Sigma_pred.T, interpolation='nearest', cmap='rainbow', 
                  extent=[t.min(), t.max(), x.min(), x.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc = 'best')
    ax.set_title('Variance of $u(t,x)$', fontsize = 10)
    savefig('./Variance')

   
    
    