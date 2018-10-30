__author__ = "Shreyas Seshadri, shreyas.seshadri@aalto.fi"
import numpy as np
import tensorflow as tf


tf.reset_default_graph() # debugging, clear all tf variables
#tf.enable_eager_execution() # placeholders are not compatible

import model_convNet 
import scipy.io


_FLOATX = tf.float32 



## LOAD DATA
loadFile = './rand_data.mat'
loaddata = scipy.io.loadmat(loadFile)
X = loaddata['X'] # style 1 training minibatches of size [frames,batchSize,dim]
Y = loaddata['Y'] # style 2 training minibatches of size [frames,batchSize,dim]
x = loaddata['feats_x'] # style 1 test data of size [frames,dim]
y = loaddata['feats_y'] # style 2 test data of size [frames,dim]


## PARAMETERS
residual_channels = 256
filter_width = 11
dilations = [1, 1, 1, 1, 1, 1]
input_channels = X[0][0].shape[2]
output_channels = X[0][0].shape[2]
cond_dim = None
postnet_channels= 256
do_postproc = True
do_gu = True


#residual_channels = 2
#filter_width = 3
#dilations = [1, 1, 1, 1]
#input_channels = X[0][0].shape[2]
#output_channels = X[0][0].shape[2]
#cond_dim = None
#postnet_channels= 5
#do_postproc = False

G = model_convNet.CNET(name='G', 
                       input_channels=input_channels,
                       output_channels= output_channels,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=cond_dim,
                       do_postproc=do_postproc,
                       do_GU=do_gu)

F = model_convNet.CNET(name='F', 
                       input_channels=input_channels,
                       output_channels= output_channels,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=cond_dim,
                       do_postproc=do_postproc,
                       do_GU=do_gu)

D_x = model_convNet.CNET(name='D_x', 
                       input_channels=output_channels,
                       output_channels= 1,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=cond_dim,
                       do_postproc=do_postproc,
                       do_GU=do_gu)

D_y = model_convNet.CNET(name='D_y', 
                       input_channels=output_channels,
                       output_channels= 1,
                       residual_channels=residual_channels,
                       filter_width=filter_width,
                       dilations=dilations,
                       postnet_channels=postnet_channels,
                       cond_dim=cond_dim,
                       do_postproc=do_postproc,
                       do_GU=do_gu)

# optimizer parameters
adam_lr = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.999

num_epochs = 1#3#200

# data placeholders of shape (batch_size, timesteps, feature_dim)
x_real = tf.placeholder(shape=(None, None, input_channels), dtype=_FLOATX)
y_real = tf.placeholder(shape=(None, None, input_channels), dtype=_FLOATX)

# X -> Y_hat -> X_hat_hat loop
y_hat = G.forward_pass(x_real)
x_hat_hat = F.forward_pass(y_hat)
x_id = F.forward_pass(x_real)
D_out_y_real = D_y.forward_pass(y_real)
D_out_y_fake = D_y.forward_pass(y_hat)

# Y -> X_hat -> Y_hat_hat loop
x_hat = F.forward_pass(y_real)
y_hat_hat = G.forward_pass(x_hat)
y_id = G.forward_pass(y_real)
D_out_x_real = D_x.forward_pass(x_real)
D_out_x_fake = D_x.forward_pass(x_hat)


# GAN loss
D_y_loss_gan = -tf.reduce_mean(D_out_y_real) + tf.reduce_mean(D_out_y_fake) 
D_x_loss_gan = -tf.reduce_mean(D_out_x_real) + tf.reduce_mean(D_out_x_fake) 
G_loss_gan = -tf.reduce_mean(D_out_y_fake)
F_loss_gan = -tf.reduce_mean(D_out_x_fake)

#recon loss
recon_loss_x = 10*tf.reduce_mean(tf.abs(x_real-x_hat_hat)) 
recon_loss_y = 10*tf.reduce_mean(tf.abs(y_real-y_hat_hat))

# identity loss
id_loss_x = 5*tf.reduce_mean(tf.abs(x_real-x_id))
id_loss_y = 5*tf.reduce_mean(tf.abs(y_real-y_id))

# gradient penalty
epsilon_shape = tf.stack([tf.shape(x_real)[0],tf.shape(x_real)[1],1]) # treat timestep similar to batch (TODO: or not?)
epsilon = tf.random_uniform(epsilon_shape, 0.001, 0.999)
y_grad = epsilon*y_real + (1.0-epsilon)*y_hat
d_hat_y = D_y.forward_pass(y_grad)
gradients_y = tf.gradients(d_hat_y, y_grad)[0]
gradnorm_y = tf.sqrt(tf.reduce_sum(tf.square(gradients_y), axis=[2])+1.0e-19)
gradient_penalty_y = 10*tf.reduce_mean(tf.square(gradnorm_y-1.0)) # magic weight factor 10 from the 'improved wgan' paper
D_loss_gradpen_y = gradient_penalty_y

epsilon_shape = tf.stack([tf.shape(y_real)[0],tf.shape(x_real)[1],1]) # treat timestep similar to batch (TODO: or not?)
epsilon = tf.random_uniform(epsilon_shape, 0.001, 0.999)
x_grad = epsilon*x_real + (1.0-epsilon)*x_hat
d_hat_x = D_x.forward_pass(x_grad)
gradients_x = tf.gradients(d_hat_x, x_grad)[0]
gradnorm_x = tf.sqrt(tf.reduce_sum(tf.square(gradients_x), axis=[2])+1.0e-19)
gradient_penalty_x = 10*tf.reduce_mean(tf.square(gradnorm_x-1.0)) # magic weight factor 10 from the 'improved wgan' paper
D_loss_gradpen_x = gradient_penalty_x

# additional penalty term to keep the scores from drifting too far from zero 
D_loss_zeropen_x = 1e-2 * tf.reduce_sum(tf.square(D_out_x_real))
D_loss_zeropen_y = 1e-2 * tf.reduce_sum(tf.square(D_out_y_real))

D_loss = D_y_loss_gan + D_x_loss_gan + D_loss_gradpen_y + D_loss_gradpen_x + D_loss_zeropen_x + D_loss_zeropen_y
Gen_loss = G_loss_gan + F_loss_gan + recon_loss_x + recon_loss_y + id_loss_x + id_loss_y
Gen_loss2 = G_loss_gan + F_loss_gan + recon_loss_x + recon_loss_y

theta_G = G.get_variable_list()
theta_F = F.get_variable_list()
theta_Dx = D_x.get_variable_list()
theta_Dy = D_y.get_variable_list()

Gen_solver = tf.train.AdamOptimizer(learning_rate=adam_lr,beta1=adam_beta1,beta2=adam_beta2).minimize(Gen_loss, var_list=[theta_G,theta_F])
Gen_solver2 = tf.train.AdamOptimizer(learning_rate=adam_lr,beta1=adam_beta1,beta2=adam_beta2).minimize(Gen_loss2, var_list=[theta_G,theta_F])

D_solver = tf.train.AdamOptimizer(learning_rate=adam_lr,beta1=adam_beta1,beta2=adam_beta2).minimize(D_loss, var_list=[theta_Dx,theta_Dy])


n_critic = 5

lossD_all = np.zeros((num_epochs*X.shape[0],7),dtype=float)
lossG_all = np.zeros((num_epochs*X.shape[0],7),dtype=float)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
tfconfig = tf.ConfigProto(gpu_options=gpu_options)

saveFile1 = './pred_res.mat'
saveFile2 = './errors.mat'
cont = 0;
with tf.Session(config=tfconfig) as sess:

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op) 

    saver = tf.train.Saver(max_to_keep=0)

    for epoch in range(num_epochs):
        
        # Train discriminator
        idx = np.random.permutation(X.shape[0])
        
        for batch_i in range(X.shape[0]):
            
            for critic_i in range(n_critic):

                # Train Discriminator                
                _, lossD, lossD_gan_x, lossD_gan_y, lossD_grad_y, lossD_grad_x, lossD_zero_x, lossD_zero_y = sess.run([D_solver, D_loss, D_x_loss_gan, D_y_loss_gan,D_loss_gradpen_y, D_loss_gradpen_x, D_loss_zeropen_x, D_loss_zeropen_y], feed_dict={x_real: X[idx[batch_i]][0],y_real: Y[idx[batch_i]][0]})
            lossD_all[cont][0] = lossD
            lossD_all[cont][1] = lossD_gan_x
            lossD_all[cont][2] = lossD_gan_y
            lossD_all[cont][3] = lossD_grad_y
            lossD_all[cont][4] = lossD_grad_x
            lossD_all[cont][5] = lossD_zero_x
            lossD_all[cont][6] = lossD_zero_y
            
            # Train generator
            if epoch<50:
                _, lossGen, lossG, lossF, loss_reconX, loss_reconY, loss_idX, loss_idY = sess.run([Gen_solver, Gen_loss, G_loss_gan, F_loss_gan, recon_loss_x, recon_loss_y, id_loss_x, id_loss_y], feed_dict={x_real: X[idx[batch_i]][0],y_real: Y[idx[batch_i]][0]})
            else:
                _, lossGen, lossG, lossF, loss_reconX, loss_reconY,  = sess.run([Gen_solver2, Gen_loss2, G_loss_gan, F_loss_gan, recon_loss_x, recon_loss_y], feed_dict={x_real: X[idx[batch_i]][0], y_real: Y[idx[batch_i]][0]})
            
            lossG_all[cont][0] = lossGen
            lossG_all[cont][1] = lossG
            lossG_all[cont][2] = lossF
            lossG_all[cont][3] = loss_reconX
            lossG_all[cont][4] = loss_reconY
            lossG_all[cont][5] = loss_idX
            lossG_all[cont][6] = loss_idX
            cont = cont+1
                
            #print("Errors for epoch %d and minibatch %d : Gen loss is %f, D loss is %f " % (epoch, batch_i, lossGen, lossD))
        print("Errors for epoch %d : Gen loss is %f, D loss is %f " % (epoch, lossGen, lossD))
        scipy.io.savemat(saveFile2,{"lossG_all":lossG_all,"lossD_all":lossD_all})
        

    
    #x
    no_utt = x.shape[0]
    y_pred = np.ndarray((no_utt,),dtype=object)
    x_recon = np.ndarray((no_utt,),dtype=object)
    x_pred_id = np.ndarray((no_utt,),dtype=object)
    for n_val in range(no_utt):  
        input_data = np.reshape(x[n_val][0], (1,x[n_val][0].shape[0],x[n_val][0].shape[1]))
        if  input_data.shape[0]!= 0:
            y_pred[n_val],x_recon[n_val],x_pred_id[n_val] = sess.run([y_hat,x_hat_hat,x_id], feed_dict={x_real: input_data})
        else:
                y_pred[n_val] = np.nan
                x_recon[n_val] = np.nan
                x_pred_id[n_val] = np.nan
#    
     # y_train
    no_utt = y.shape[0]
    x_pred = np.ndarray((no_utt,),dtype=object)
    y_recon = np.ndarray((no_utt,),dtype=object)
    y_pred_id = np.ndarray((no_utt,),dtype=object)
    for n_val in range(no_utt):
        input_data = np.reshape(y[n_val][0], (1,y[n_val][0].shape[0],y[n_val][0].shape[1]))        
        if  input_data.shape[0]!= 0:            
            x_pred[n_val],y_recon[n_val],y_pred_id[n_val] = sess.run([x_hat,y_hat_hat,y_id], feed_dict={y_real: input_data})
        else:
            x_pred[n_val] = np.nan
            y_recon[n_val] = np.nan
            y_pred_id[n_val] = np.nan
            
   

scipy.io.savemat(saveFile1,{"y_pred":y_pred,
                             "x_recon":x_recon,
                             "x_pred_id":x_pred_id,
                             "x_pred":x_pred,
                             "y_recon":y_recon,
                             "y_pred_id":y_pred_id})