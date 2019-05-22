"""
Created on Wed Nov 2018

@author: Yibo Yang
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from pyDOE import lhs

from models import UQ_PINN

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

    # Load the data
    data = np.load('./nonlinear2d_data.npz')
    X = data['X']
    K = data['k']
    U = data['u']

    # Exact relation between k and u
    def k_vanGenuchten(u):
        alpha = 0.1
        n = 1.885
        m = 1.0 - 1.0/n
        s = (1.0 + (alpha*np.abs(u))**n)**(-m)
        k = np.sqrt(s)*(1.0 - (1.0 - s**(1.0/m))**m)**2
        return k


    N = 10000
    N_f = N
    N_u = 200
    N_b = 100 # for one boundary

    X_dim = 1 
    Y_dim = 1
    Z_dim = 2

    L1 = 10.
    L2 = 10.

    X_u = np.zeros((N_u,2))
    Y_u = np.zeros((N_u,1))
    X_f = np.zeros((N_f,2))

    # Boundary points
    x1_b1 = np.zeros(N_b)[:,None]
    x2_b1 = L2 * np.random.random(N_b)[:,None]
    X_b1 = np.hstack((x1_b1, x2_b1))
    x1_b2 = L1 * np.random.random(N_b)[:,None]
    x2_b2 = np.zeros(N_b)[:,None]
    X_b2 = np.hstack((x1_b2, x2_b2))
    x1_b3 = L1 * np.ones(N_b)[:,None]
    x2_b3 = L2 * np.random.random(N_b)[:,None]
    X_b3 = np.hstack((x1_b3, x2_b3))   
    x1_b4 = L1 * np.random.random(N_b)[:,None]
    x2_b4 = L2 * np.ones(N_b)[:,None]
    X_b4 = np.hstack((x1_b4, x2_b4))
    X_b = np.hstack((X_b1, X_b2))
    X_b = np.hstack((X_b, X_b3))
    X_b = np.hstack((X_b, X_b4))

    # Collocation points
    X1_f = L1 * np.random.random(N_f)[:,None]
    X2_f = L2 * np.random.random(N_f)[:,None]
    X_f = np.hstack((X1_f, X2_f))

    U_data = U
    X_data = X

    idx_u = np.random.choice(N, N_u, replace=False)
    for i in range(N_u):
        X_u[i,:] = X_data[idx_u[i],:]
        Y_u[i,:] = U_data[idx_u[i]]


    # Model creation
    layers_P_u = np.array([X_dim+X_dim+Z_dim,50,50,50,50,Y_dim])
    layers_Q = np.array([X_dim+X_dim+Y_dim,50,50,50,50,Z_dim])  
    layers_T = np.array([X_dim+X_dim+Y_dim,50,50,50,1])
    layers_P_k = np.array([Y_dim,50,50,50,50,Y_dim])

    model = UQ_PINN(X_u, X_b, Y_u, X_f, layers_P_u, layers_P_k, layers_Q, layers_T, lam = 1.5, beta = 1.0, q = 1., u_0 = - 10.)

    model.train(nIter = 30000)

    X_star = X
    k_star = K.T
    u_star = U.T

    # Domain bounds
    lb, ub = X.min(0), X.max(0)
    # Plot
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    XX, YY = np.meshgrid(x,y)

    K_plot = griddata(X_star, k_star.flatten(), (XX, YY), method='cubic')
    U_plot = griddata(X_star, u_star.flatten(), (XX, YY), method='cubic')

    N_samples = 500
    kkk = np.zeros((X_star.shape[0], N_samples))
    uuu = np.zeros((X_star.shape[0], N_samples))
    fff = np.zeros((X_star.shape[0], N_samples))
    for i in range(0, N_samples):
        kkk[:,i:i+1] = model.predict_k(X_star)
        uuu[:,i:i+1] = model.predict_u(X_star)
        fff[:,i:i+1] = model.predict_f(X_star)

    np.save('uuu0.npy', uuu)
    np.save('kkk0.npy', kkk)

    kkk_mu_pred = np.mean(kkk, axis = 1)
    kkk_Sigma_pred = np.var(kkk, axis = 1)
    uuu_mu_pred = np.mean(uuu, axis = 1)    
    uuu_Sigma_pred = np.var(uuu, axis = 1)
    fff_mu_pred = np.mean(fff, axis = 1)    
    fff_Sigma_pred = np.var(fff, axis = 1)


    K_mu_plot = griddata(X_star, kkk_mu_pred.flatten(), (XX, YY), method='cubic')
    U_mu_plot = griddata(X_star, uuu_mu_pred.flatten(), (XX, YY), method='cubic')
    F_mu_plot = griddata(X_star, fff_mu_pred.flatten(), (XX, YY), method='cubic')
    K_Sigma_plot = griddata(X_star, kkk_Sigma_pred.flatten(), (XX, YY), method='cubic')
    U_Sigma_plot = griddata(X_star, uuu_Sigma_pred.flatten(), (XX, YY), method='cubic')
    F_Sigma_plot = griddata(X_star, fff_Sigma_pred.flatten(), (XX, YY), method='cubic')

    fig = plt.figure(2,figsize=(12,12))
    plt.subplot(2,2,1)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.pcolor(XX, YY, K_plot, cmap='viridis')
    plt.colorbar().ax.tick_params(labelsize=15)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)  
    plt.title('Exact $k(x_1,x_2)$', fontsize=15)

    plt.subplot(2,2,2)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.pcolor(XX, YY, K_mu_plot, cmap='viridis')
    plt.colorbar().ax.tick_params(labelsize=15)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)  
    plt.title('Prediction $k(x_1,x_2)$', fontsize=15)

    plt.subplot(2,2,3)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.pcolor(XX, YY, np.abs(K_plot - K_mu_plot), cmap='viridis')
    plt.colorbar().ax.tick_params(labelsize=15)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)  
    plt.title('Error of $k(x_1,x_2)$', fontsize=15)
    
    plt.subplot(2,2,4)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.pcolor(XX, YY, np.abs(K_plot - K_mu_plot) / K_plot, cmap='viridis')
    plt.colorbar().ax.tick_params(labelsize=15)
    plt.xlabel('$x_1$', fontsize=15)
    plt.ylabel('$x_2$', fontsize=15)  
    plt.title('Relative error of $k(x_1,x_2)$', fontsize=15)
    plt.savefig('./reconstruction.png', dpi = 600)

    u = np.load('uuu0.npy')
    k = np.load('kkk0.npy')
    u_mu = np.mean(u, axis = 1)
    u = np.zeros((10000, 500))
    for i in range(500):
        u[:,i] = u_mu
        
    u = u.reshape(1,-1)
    k = k.reshape(1,-1)
    idx = np.random.choice(5000000, 1000, replace=False)
    u_p = u[:,idx]
    k_p = k[:,idx]


    u = np.linspace(-10.,-4., 1000)
    k = k_vanGenuchten(u)

    plt.figure(10, figsize=(6, 4))
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)   
    plt.plot(u_p,k_p, 'bo') 
    plt.plot(u,k, 'r-', label = "Exact", linewidth=2)
    ax = plt.gca()
    plt.xlabel('$u$',fontsize=11)
    plt.ylabel('$K(u)$',fontsize=11)
    plt.savefig('./UK.png', dpi = 600)


    
    