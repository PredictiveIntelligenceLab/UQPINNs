"""
Created on Wed Nov 2018

@author: Yibo Yang
"""

import tensorflow as tf
import numpy as np
import timeit
    
class Burgers_UQPINN:
    # Initialize the class
    def __init__(self, X_f, T_f, X_u, T_u, Y_u, layers_P, layers_Q, layers_T, lam = 1.0, beta = 1.0):
                
        # Normalize data
        self.Xmean, self.Xstd = X_f.mean(0), X_f.std(0)
        self.Tmean, self.Tstd = T_f.mean(0), T_f.std(0)
        self.Ymean, self.Ystd = Y_u.mean(0), Y_u.std(0)
        X_f = (X_f - self.Xmean) / self.Xstd
        X_u = (X_u - self.Xmean) / self.Xstd
        T_f = (T_f - self.Tmean) / self.Tstd
        T_u = (T_u - self.Tmean) / self.Tstd

        # Jacobian of the PDE because of normalization
        self.Jacobian_X = 1 / self.Xstd
        self.Jacobian_T = 1 / self.Tstd

     
        self.X_f = X_f
        self.X_u = X_u
        self.T_f = T_f
        self.T_u = T_u
        self.Y_u = Y_u
        
        self.layers_P = layers_P
        self.layers_Q = layers_Q
        self.layers_T = layers_T
        
        self.X_dim = X_u.shape[1]
        self.T_dim = T_u.shape[1]
        self.Y_dim = Y_u.shape[1]
        self.Z_dim = layers_Q[-1]
        self.lam = lam
        self.beta = beta

        self.k1 = 1
        self.k2 = 5

        # Initialize network weights and biases        
        self.weights_P, self.biases_P = self.initialize_NN(layers_P)
        self.weights_Q, self.biases_Q = self.initialize_NN(layers_Q)
        self.weights_T, self.biases_T = self.initialize_NN(layers_T)
        
        # Define Tensorflow session
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        # Define placeholders and computational graph
        self.X_u_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.X_f_tf = tf.placeholder(tf.float32, shape=(None, self.X_dim))
        self.T_u_tf = tf.placeholder(tf.float32, shape=(None, self.T_dim))
        self.T_f_tf = tf.placeholder(tf.float32, shape=(None, self.T_dim))
        self.Y_u_tf = tf.placeholder(tf.float32, shape=(None, self.Y_dim))
        self.Z_u_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))
        self.Z_f_tf = tf.placeholder(tf.float32, shape=(None, self.Z_dim))

        self.Y_u_pred = self.net_P(self.X_u_tf, self.T_u_tf, self.Z_u_tf)
        self.Y_f_pred = self.get_r(self.X_f_tf, self.T_f_tf, self.Z_f_tf)

        # Generator loss (to be minimized)
        self.G_loss, self.KL_loss, self.recon_loss, self.PDE_loss  = self.compute_generator_loss(self.X_u_tf, self.T_u_tf, self.Y_u_tf, 
            self.Y_u_pred, self.X_f_tf, self.T_f_tf, self.Y_f_pred, self.Z_u_tf, self.Z_f_tf)
                                                
        # Discriminator loss (to be minimized)
        self.T_loss  = self.compute_discriminator_loss(self.X_u_tf, self.T_u_tf, self.Y_u_tf, self.Z_u_tf)
        
        # Generator samples of y given x and t
        self.sample = self.sample_generator(self.X_u_tf, self.T_u_tf, self.Z_u_tf)

        # Define optimizer        
        self.optimizer_KL = tf.train.AdamOptimizer(1e-4)
        self.optimizer_T = tf.train.AdamOptimizer(1e-4)
        
        # Define train Ops
        self.train_op_KL = self.optimizer_KL.minimize(self.G_loss, 
                                                      var_list = [self.weights_P, self.biases_P,
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
    
    # right hand side terms of the PDE (in this case are zero)
    def f(self, X_normalized): 
        return tf.zeros_like(X_normalized) 

    # Decoder: p(y|x,z)
    def net_P(self, X, T, Z):
        Y = self.forward_pass(tf.concat([X, T, Z], 1),
                              self.layers_P,
                              self.weights_P,
                              self.biases_P)
        return Y
    
    # Encoder: q(z|x,y)
    def net_Q(self, X, T, Y):
        Z = self.forward_pass(tf.concat([X, T, Y], 1),
                              self.layers_Q,
                              self.weights_Q,
                              self.biases_Q)
        return Z
    
    # Discriminator
    def net_T(self, X, T, Y):
        T = self.forward_pass(tf.concat([X, T, Y], 1),
                              self.layers_T,
                              self.weights_T,
                              self.biases_T)        
        return T
    
    # Physics-Informed neural network prediction
    def get_u(self, X, T, Z):
        z_prior = Z       
        u = self.net_P(X, T, z_prior)
        return u

    # Physics-Informed residual on the collocation points
    def get_r(self, X, T, Z):
        z_prior = Z    
        u = self.net_P(X, T, z_prior)
        u_t = tf.gradients(u, T)[0]
        u_x = tf.gradients(u, X)[0]
        u_xx = tf.gradients(u_x, X)[0]
        f = self.f(X)
        r = (self.Jacobian_T) * u_t + (self.Jacobian_X) * u * u_x - (0.01/np.pi) * (self.Jacobian_X ** 2) * u_xx - f
        return r    
    
    # Compute the generator loss
    def compute_generator_loss(self, X_u, T_u, Y_u, Y_u_pred, X_f, T_f, Y_f_pred, Z_u, Z_f):  
        # Prior:
        z_u_prior = Z_u
        z_f_prior = Z_f
        # Encoder: q(z|x,y)
        z_u_encoder = self.net_Q(X_u, T_u, Y_u_pred)
        z_f_encoder = self.net_Q(X_f, T_f, Y_f_pred)
        # Discriminator loss
        Y_pred = self.net_P(X_u, T_u, Z_u)
        T_pred = self.net_T(X_u, T_u, Y_pred)
        
        # KL-divergence between the data distribution and the model distribution
        KL = tf.reduce_mean(T_pred)

        # Entropic regularization
        log_q = - tf.reduce_mean(tf.square(z_u_prior-z_u_encoder))
        
        # Physics-informed loss
        loss_f = tf.reduce_mean(tf.square(Y_f_pred))

        # Generator loss
        loss = KL + (1.0-self.lam)*log_q + self.beta * loss_f
        
        return loss, KL, (1.0-self.lam)*log_q, self.beta * loss_f
    
    # Compute the discriminator loss
    def compute_discriminator_loss(self, X, T, Y, Z): 
        # Prior: p(z)
        z_prior = Z
        # Decoder: p(y|x,z)
        Y_pred = self.net_P(X, T, z_prior)                
        
        # Discriminator loss
        T_real = self.net_T(X, T, Y)
        T_fake = self.net_T(X, T, Y_pred)
        
        T_real = tf.sigmoid(T_real)
        T_fake = tf.sigmoid(T_fake)
        
        T_loss = -tf.reduce_mean(tf.log(1.0 - T_real + 1e-8) + \
                                 tf.log(T_fake + 1e-8)) 
        
        return T_loss
           
    
    # Fetches a mini-batch of data
    def fetch_minibatch(self, X_u, T_u, X_f, T_f, Y_u, N_batch_u, N_batch_f):
        N_u = X_u.shape[0]
        N_f = X_f.shape[0]
        idx_u = np.random.choice(N_u, N_batch_u, replace=False)
        idx_f = np.random.choice(N_f, N_batch_f, replace=False)
        X_u_batch = X_u[idx_u,:]
        T_u_batch = T_u[idx_u,:]
        X_f_batch = X_f[idx_f,:]
        T_f_batch = T_f[idx_f,:]
        Y_u_batch = Y_u[idx_u,:]
        return X_u_batch, T_u_batch, X_f_batch, T_f_batch, Y_u_batch
    
    
    # Trains the model
    def train(self, nIter = 20000, N_u = 300, N_f = 5000, batch_size_u = 50, batch_size_f = 500): 

        start_time = timeit.default_timer()

        # Store generator loss and discriminator loss as function of iteration
        iteration = []
        loss_D = []
        loss_G = []
        for it in range(nIter):     
            # Fetch a mini-batch of data
            X_u_batch, T_u_batch, X_f_batch, T_f_batch, Y_u_batch = self.fetch_minibatch(self.X_u, self.T_u, self.X_f, self.T_f,
                                                                     self.Y_u, batch_size_u, batch_size_f)

            Z_u = np.random.randn(batch_size_u, 1)
            Z_f = np.random.randn(batch_size_f, 1)

            # Define a dictionary for associating placeholders with data
            tf_dict = {self.X_u_tf: X_u_batch, self.T_u_tf: T_u_batch, self.Y_u_tf: Y_u_batch, self.X_f_tf: X_f_batch,
                     self.T_f_tf: T_f_batch, self.Z_u_tf: Z_u, self.Z_f_tf: Z_f}  
            
            # Run the Tensorflow session to minimize the loss
            for i in range(self.k1):
                self.sess.run(self.train_op_T, tf_dict)
            for j in range(self.k2):
                self.sess.run(self.train_op_KL, tf_dict)
        
            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_GG, loss_KL_value, reconv, loss_PDE = self.sess.run([self.G_loss, self.KL_loss, self.recon_loss, self.PDE_loss], tf_dict)
                loss_T_value = self.sess.run(self.T_loss, tf_dict)

                iteration.append(it)
                loss_D.append(loss_T_value)
                loss_G.append(loss_GG)
                print('It: %d, KL_loss: %.2e, Recon_loss: %.2e, PDE_loss: %.2e, T_loss: %.2e, Time: %.2f' % 
                      (it, loss_KL_value, reconv, loss_PDE, loss_T_value, elapsed))
                start_time = timeit.default_timer()

    # Generate samples of y given x by sampling from the latent space z
    def sample_generator(self, X, T, Z):        
        # Prior: 
        z_prior = Z       
        # Decoder: p(y|x,z)
        Y_pred = self.net_P(X, T, z_prior)      
        return Y_pred

    # Predict y given x
    def generate_sample(self, X_star, T_star):
        X_star = (X_star - self.Xmean) / self.Xstd
        T_star = (T_star - self.Tmean) / self.Tstd
        Z = np.random.randn(X_star.shape[0], 1)
        tf_dict = {self.X_u_tf: X_star, self.T_u_tf: T_star, self.Z_u_tf: Z}       
        Y_star = self.sess.run(self.sample, tf_dict) 
        Y_star = Y_star 
        return Y_star

    def predict_f(self, X_star, T_star): 
        # Center around the origin
        X_star = (X_star - self.Xmean) / self.Xstd
        T_star = (T_star - self.Tmean) / self.Tstd
        # Predict
        z_f = np.random.randn(X_star.shape[0], self.Z_dim) 
        tf_dict = {self.X_f_tf: X_star[:,0:1], self.T_f_tf: T_star, self.Z_f_tf: z_f}     
        f_star = self.sess.run(self.Y_f_pred, tf_dict) 
        return f_star

