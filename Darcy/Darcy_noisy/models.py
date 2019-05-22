"""
Created on Wed Nov 2018

@author: Yibo Yang
"""

import tensorflow as tf
import numpy as np
import timeit

    
class UQ_PINN:
    # Initialize the class
    def __init__(self, X_u, X_b, Y_u, X_f, layers_P_u, layers_P_k, layers_Q, layers_T, lam = 1.5, beta = 1.0, q = 1, u_0 = - 10.):
                
        # Normalize data
        self.lb = np.array([0.0, 0.0])
        self.ub = np.array([10.0, 10.0])
        self.lbb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.ubb = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        X_u = (X_u - self.lb) - 0.5*(self.ub - self.lb)
        X_b = (X_b - self.lbb) - 0.5*(self.ubb - self.lbb)
        X_f = (X_f - self.lb) - 0.5*(self.ub - self.lb)


        self.q = q
        self.u_0 = u_0
        self.ksat = 10.
        
        self.x1_u = X_u[:,0:1] # dimension  N_u x 1
        self.x2_u = X_u[:,1:2] # dimension  N_u x 1
        self.y_u = Y_u           # dimension N_u

        self.x1_f = X_f[:,0:1]   # dimension N_f x 1
        self.x2_f = X_f[:,1:2]   # dimension N_f x 1

        # Position of the boundary 
        self.x1_b1 = X_b[:,0:1]
        self.x2_b1 = X_b[:,1:2]
        self.x1_b2 = X_b[:,2:3]
        self.x2_b2 = X_b[:,3:4]
        self.x1_b3 = X_b[:,4:5]
        self.x2_b3 = X_b[:,5:6]
        self.x1_b4 = X_b[:,6:7]
        self.x2_b4 = X_b[:,7:8]
        
        # Layers of the neural networks
        self.layers_P_u = layers_P_u
        self.layers_Q = layers_Q
        self.layers_T = layers_T
        self.layers_P_k = layers_P_k   

        # Dimensions of the inputs, outputs, latent variables 
        self.X_dim = self.x1_u.shape[1]
        self.Y_u_dim = self.y_u.shape[1]
        self.Y_k_dim = self.y_u.shape[1]
        self.Y_f_dim = self.y_u.shape[1]
        self.Z_dim = layers_Q[-1]

        # Regularization parameters
        self.lam = lam
        self.beta = beta

        # Ratio of training for generator and discriminator
        self.k1 = 1
        self.k2 = 5

        # Initialize network weights and biases        
        self.weights_P_u, self.biases_P_u = self.initialize_NN(layers_P_u)
        self.weights_Q, self.biases_Q = self.initialize_NN(layers_Q)
        self.weights_T, self.biases_T = self.initialize_NN(layers_T)
        self.weights_P_k, self.biases_P_k = self.initialize_NN(layers_P_k)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.x1_u_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x2_u_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x1_f_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x2_f_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.y_u_tf = tf.placeholder(tf.float32, shape=(None, self.Y_u_dim))
        self.y_k_tf = tf.placeholder(tf.float32, shape=(None, self.Y_k_dim))
        self.y_f_tf = tf.placeholder(tf.float32, shape=(None, self.Y_f_dim))

        self.x1_b1_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x2_b1_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x1_b2_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x2_b2_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x1_b3_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x2_b3_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x1_b4_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.x2_b4_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))

        self.z_b1_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))
        self.z_b2_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))
        self.z_b3_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))
        self.z_b4_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))
        self.z_u_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))
        self.z_f_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))

        self.y_u_pred = self.net_P_u(self.x1_u_tf, self.x2_u_tf, self.z_u_tf)
        self.y_b1_pred = self.get_b1(self.x1_b1_tf, self.x2_b1_tf, self.z_b1_tf)
        self.y_b2_pred = self.get_b2(self.x1_b2_tf, self.x2_b2_tf, self.z_b2_tf)
        self.y_b3_pred = self.get_b3(self.x1_b3_tf, self.x2_b3_tf, self.z_b3_tf)
        self.y_b4_pred = self.get_b4(self.x1_b4_tf, self.x2_b4_tf, self.z_b4_tf)
        self.y_k_pred = self.net_P_k(self.y_u_pred)
        self.y_f_pred = self.get_f(self.x1_f_tf, self.x2_f_tf, self.z_f_tf)

        # Generator loss (to be minimized)
        self.G_loss, self.KL_loss, self.recon_loss, self.PDE_loss = self.compute_generator_loss(self.x1_u_tf, self.x2_u_tf,
                                                                            self.y_u_pred, self.y_f_pred, self.y_b1_pred, self.y_b2_pred, 
                                                                            self.y_b3_pred, self.y_b4_pred, self.z_u_tf)

        # Discriminator loss (to be minimized)
        self.T_loss  = self.compute_discriminator_loss(self.x1_u_tf, self.x2_u_tf, self.y_u_tf, self.z_u_tf)

        # Define optimizer        
        self.optimizer_KL = tf.train.AdamOptimizer(1e-4)
        self.optimizer_T = tf.train.AdamOptimizer(1e-4)
        
        # Define train Ops
        self.train_op_KL = self.optimizer_KL.minimize(self.G_loss, 
                                                      var_list = [self.weights_P_u, self.biases_P_u, self.weights_P_k, self.biases_P_k,
                                                                  self.weights_Q, self.biases_Q])
                                                                    
        self.train_op_T = self.optimizer_T.minimize(self.T_loss,
                                                    var_list = [self.weights_T, self.biases_T])

        # Initialize Tensorflow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

    
    # Initialize network weights and biases using Xavier initialization
    def initialize_NN(self, layers):      
        # Xavier initialization
        def xavier_init(size):
            in_dim = size[0]
            out_dim = size[1]
            xavier_stddev = 1. / np.sqrt((in_dim + out_dim) / 2.)
            return tf.Variable(tf.random_normal([in_dim, out_dim], dtype=tf.float32) * xavier_stddev, dtype=tf.float32)   
        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
           
           
    # Evaluates the forward pass
    def forward_pass(self, H, layers, weights, biases):
        num_layers = len(layers)
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        H = tf.add(tf.matmul(H, W), b)
        return H
    
    def f(self, X_normalized):
        return tf.zeros_like(X_normalized) 

    # Decoder: p(y|x,z)
    def net_P_u(self, X1, X2, Z):
        Y = self.forward_pass(tf.concat([X1, X2, Z], 1),
                              self.layers_P_u,
                              self.weights_P_u,
                              self.biases_P_u)
        return Y
    
    # Encoder: q(z|x,y)
    def net_Q(self, X1, X2, Y):
        Z = self.forward_pass(tf.concat([X1, X2, Y], 1),
                              self.layers_Q,
                              self.weights_Q,
                              self.biases_Q)
        return Z
    
    # Discriminator
    def net_T(self, X1, X2, Y):
        T = self.forward_pass(tf.concat([X1, X2, Y], 1),
                              self.layers_T,
                              self.weights_T,
                              self.biases_T)        
        return T
    
    # Decoder: p(y|x,z)
    def net_P_k(self, U):
        Y = self.forward_pass(U,
                              self.layers_P_k,
                              self.weights_P_k,
                              self.biases_P_k)
        return self.ksat * tf.exp(Y)
    

    def get_u(self, X1, X2, Z):
        z_prior = Z       
        u = self.net_P_u(X1, X2, z_prior)
        return u

    def get_k(self, U):   
        u = self.net_P_k(U)
        return u    

    def get_b1(self, X1, X2, Z):   
        z_prior = Z       
        u = self.net_P_u(X1, X2, z_prior)
        u_x1 = tf.gradients(u, X1)[0]
        k = self.net_P_k(u)
        temp = self.q + k * u_x1
        return temp

    def get_b2(self, X1, X2, Z):   
        z_prior = Z       
        u = self.net_P_u(X1, X2, z_prior)
        u_x2 = tf.gradients(u, X2)[0]
        return u_x2

    def get_b3(self, X1, X2, Z):   
        z_prior = Z       
        u = self.net_P_u(X1, X2, z_prior)
        temp = u - self.u_0
        return temp

    def get_b4(self, X1, X2, Z):   
        z_prior = Z       
        u = self.net_P_u(X1, X2, z_prior)
        u_x2 = tf.gradients(u, X2)[0]
        return u_x2

    def get_f(self, X1, X2, Z_u):
        u = self.net_P_u(X1, X2, Z_u)
        k = self.net_P_k(u)
        u_x1 = tf.gradients(u, X1)[0]
        u_x2 = tf.gradients(u, X2)[0]
        f_1 = tf.gradients(k*u_x1, X1)[0]
        f_2 = tf.gradients(k*u_x2, X2)[0]
        f = f_1 + f_2
        return f
    
    def compute_generator_loss(self, x1_u, x2_u, y_u_pred, y_f_pred, y_b1_pred, y_b2_pred, y_b3_pred, y_b4_pred, z_u):    
        # Encoder: q(z|x,y)
        z_u_prior = z_u

        z_u_encoder = self.net_Q(x1_u, x2_u, y_u_pred)

        y_u_pred = self.net_P_u(x1_u, x2_u, z_u)
        T_pred = self.net_T(x1_u, x2_u, y_u_pred)

        # KL-divergence between the data and the generator samples
        KL = tf.reduce_mean(T_pred)
        
        # Entropic regularization
        log_q = - tf.reduce_mean(tf.square(z_u_prior-z_u_encoder))

        # Physics-informed loss
        loss_f = tf.reduce_mean(tf.square(y_f_pred)) + tf.reduce_mean(tf.square(y_b1_pred)) +\
                tf.reduce_mean(tf.square(y_b2_pred)) +  tf.reduce_mean(tf.square(y_b3_pred)) + tf.reduce_mean(tf.square(y_b4_pred))

        # Generator loss
        loss = KL + (1.0-self.lam)*log_q + self.beta * loss_f
        
        return loss, KL, (1.0-self.lam)*log_q, self.beta * loss_f
    
    
    def compute_discriminator_loss(self, X1, X2, Y, Z): 
        # Prior: p(z)
        z_prior = Z
        # Decoder: p(y|x,z)
        Y_pred = self.net_P_u(X1, X2, z_prior)                
        
        # Discriminator loss
        T_real = self.net_T(X1, X2, Y)
        T_fake = self.net_T(X1, X2, Y_pred)
        
        T_real = tf.sigmoid(T_real)
        T_fake = tf.sigmoid(T_fake)
        
        T_loss = -tf.reduce_mean(tf.log(1.0 - T_real + 1e-8) + \
                                 tf.log(T_fake + 1e-8)) 
        
        return T_loss

    # Trains the model
    def train(self, nIter = 20000): 

        start_time = timeit.default_timer()
        for it in range(nIter):     

            # Sampling from latent spaces
            z_u = np.random.randn(self.x1_u.shape[0], self.Z_dim)
            z_f = np.random.randn(self.x1_f.shape[0], self.Z_dim)
            z_b1 = np.random.randn(self.x1_b1.shape[0], self.Z_dim)
            z_b2 = np.random.randn(self.x1_b2.shape[0], self.Z_dim)
            z_b3 = np.random.randn(self.x1_b3.shape[0], self.Z_dim)
            z_b4 = np.random.randn(self.x1_b4.shape[0], self.Z_dim)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.x1_u_tf: self.x1_u, self.x2_u_tf: self.x2_u, self.x1_f_tf: self.x1_f, self.x2_f_tf: self.x2_f, 
                    self.y_u_tf: self.y_u, self.x1_b1_tf: self.x1_b1, self.x2_b1_tf: self.x2_b1, self.x1_b2_tf: self.x1_b2, self.x2_b2_tf: self.x2_b2,
                    self.x1_b3_tf: self.x1_b3, self.x2_b3_tf: self.x2_b3, self.x1_b4_tf: self.x1_b4, self.x2_b4_tf: self.x2_b4, 
                    self.z_u_tf: z_u, self.z_f_tf: z_f, self.z_b1_tf: z_b1, self.z_b2_tf: z_b2, self.z_b3_tf: z_b3, self.z_b4_tf: z_b4}
            
            # Run the Tensorflow session to minimize the loss
            for i in range(self.k1):
                self.sess.run(self.train_op_T, tf_dict)
            for j in range(self.k2):
                self.sess.run(self.train_op_KL, tf_dict)
            
            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_KL_value, reconv, loss_PDE = self.sess.run([self.KL_loss, self.recon_loss, self.PDE_loss], tf_dict)
                loss_T_value = self.sess.run(self.T_loss, tf_dict)
                print('It: %d, KL_loss: %.2e, Recon_loss: %.2e, PDE_loss: %.2e, T_loss: %.2e, Time: %.2f' % 
                      (it, loss_KL_value, reconv, loss_PDE, loss_T_value, elapsed))
                start_time = timeit.default_timer()
                

    # Evaluates predictions at test points           
    def predict_k(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict 
        z_u = np.random.randn(X_star.shape[0], self.Z_dim)    
        tf_dict = {self.x1_u_tf: X_star[:,0:1], self.x2_u_tf: X_star[:,1:2], self.z_u_tf: z_u}    
        k_star = self.sess.run(self.y_k_pred, tf_dict) 
        return k_star / self.ksat
    
    # Evaluates predictions at test points           
    def predict_u(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict   
        z_u = np.random.randn(X_star.shape[0], self.Z_dim)       
        tf_dict = {self.x1_u_tf: X_star[:,0:1], self.x2_u_tf: X_star[:,1:2], self.z_u_tf: z_u}      
        u_star = self.sess.run(self.y_u_pred, tf_dict) 
        return u_star
    
    # Evaluates predictions at test points           
    def predict_f(self, X_star): 
        # Center around the origin
        X_star = (X_star - self.lb) - 0.5*(self.ub - self.lb)
        # Predict      
        z_f = np.random.randn(X_star.shape[0], self.Z_dim) 
        tf_dict = {self.x1_f_tf: X_star[:,0:1], self.x2_f_tf: X_star[:,1:2], self.z_f_tf: z_f}     
        f_star = self.sess.run(self.y_f_pred, tf_dict) 
        return f_star

    # Predict the k as function of u
    def predict_k_from_u(self, u):
        tf_dict = {self.y_u_pred: u}
        k_star = self.sess.run(self.y_k_pred, tf_dict) 
        return k_star / self.ksat



    