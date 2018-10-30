__author__ = "Lauri Juvela, lauri.juvela@aalto.fi; Shreyas Seshadri, shreyas.seshadri@aalto.fi"

import os
import sys
import math
import numpy as np
import tensorflow as tf

_FLOATX = tf.float32 # todo: move to lib/precision.py

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d()):
    if shape is None:
        return tf.get_variable(name)
    else:  
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX)): 
    if shape is None:
        return tf.get_variable(name) 
    else:     
        return tf.get_variable(name, shape=shape, dtype=_FLOATX, initializer=initializer)
   

class CNET():

    def __init__(self,
                 name,
                 residual_channels=64,
                 filter_width=3,
                 dilations=[1, 2, 4, 8, 1, 2, 4, 8],
                 input_channels=123,
                 output_channels=48,
                 cond_dim = None,
                 cond_channels = 64,
                 postnet_channels=256,
                 do_postproc = True,
                 do_GU = True):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.filter_width = filter_width
        self.dilations = dilations
        self.residual_channels = residual_channels
        self.postnet_channels = postnet_channels
        self.do_postproc = do_postproc
        self.do_GU = do_GU

        if cond_dim is not None:
            self._use_cond = True
            self.cond_dim = cond_dim
            self.cond_channels = cond_channels
            
        else:
            self._use_cond = False

        self._name = name
        self._create_variables()

    def _create_variables(self):

        fw = self.filter_width
        r = self.residual_channels
        s = self.postnet_channels

        with tf.variable_scope(self._name):

            with tf.variable_scope('input_layer'):
                get_weight_variable('W', (fw, self.input_channels, 2*r))
                get_bias_variable('b', (2*r)) 

            if self._use_cond:
                with tf.variable_scope('embed_cond'):
                    get_weight_variable('W', (1, self.cond_dim, self.cond_channels))
                    get_bias_variable('b', (self.cond_channels))         

            for i, dilation in enumerate(self.dilations):
                with tf.variable_scope('conv_modules'):
                    with tf.variable_scope('module{}'.format(i)):
                        # (filter_width x input_channels x output_channels) 
                        get_weight_variable('filter_gate_W', (fw, r, 2*r)) 
                        get_bias_variable('filter_gate_b', (2*r))
                        
                        
                        if self.do_postproc:
                            get_weight_variable('skip_gate_W', (1, r, s)) 
                            get_bias_variable('skip_gate_b', (s))
                        
                        if self.do_GU:
                            get_weight_variable('post_filter_gate_W', (1, r, r)) 
                            get_bias_variable('post_filter_gate_b', (r))

                        if self._use_cond:
                            get_weight_variable('cond_filter_gate_W', (1, self.cond_channels, 2*r)) 
                            get_bias_variable('cond_filter_gate_b', (2*r)) 
            

            if self.do_postproc:                    
                with tf.variable_scope('postproc_module'):
                    # (filter_width x input_channels x output_channels)                     
                    get_weight_variable('W1', (fw, s, s)) 
                    get_bias_variable('b1', s)
                    if type(self.output_channels) is list:
                        get_weight_variable('W2', (fw, s, sum(self.output_channels))) 
                        get_bias_variable('b2', sum(self.output_channels))
                    else:
                        get_weight_variable('W2', (fw, s, self.output_channels)) 
                        get_bias_variable('b2', self.output_channels)

            
            with tf.variable_scope('last_layer'):
                # (filter_width x input_channels x output_channels) 

                if type(self.output_channels) is list:
                    get_weight_variable('W', (fw, r, sum(self.output_channels))) 
                    get_bias_variable('b', sum(self.output_channels))
                else:
                    get_weight_variable('W', (fw, r, self.output_channels)) 
                    get_bias_variable('b', self.output_channels)                    

    def get_variable_list(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self._name)          


    def _input_layer(self, main_input):
        with tf.variable_scope('input_layer'):

            W = get_weight_variable('W')
            b = get_bias_variable('b')
            r = self.residual_channels

            X = main_input
            Y = tf.nn.convolution(X, W, padding='SAME')
            Y += b
            #Y = tf.tanh(Y)
            Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])

        return Y

    def _embed_cond(self, cond_input):
        with tf.variable_scope('embed_cond'):
            W = get_weight_variable('W')
            b = get_bias_variable('b')

            Y = tf.nn.convolution(cond_input, W, padding='SAME') # 1x1 convolution
            Y += b

            return tf.tanh(Y)


    def _conv_module(self, main_input, residual_input, module_idx, dilation, cond_input=None):
        with tf.variable_scope('conv_modules'):
            with tf.variable_scope('module{}'.format(module_idx)):
                W = get_weight_variable('filter_gate_W') 
                b = get_bias_variable('filter_gate_b') 
                r = self.residual_channels
                
                if self.do_postproc:
                    W_s = get_weight_variable('skip_gate_W') 
                    b_s = get_weight_variable('skip_gate_b') 
                    
                if self.do_GU:
                    W_p = get_weight_variable('post_filter_gate_W') 
                    b_p = get_weight_variable('post_filter_gate_b')

                if self._use_cond:
                    V_cond = get_weight_variable('cond_filter_gate_W') 
                    b_cond = get_bias_variable('cond_filter_gate_b') 

                X = main_input
                # convolution
                Y = tf.nn.convolution(X, W, padding='SAME', dilation_rate=[dilation])
                # add bias
                Y += b

                if self._use_cond:
                    C = tf.nn.convolution(cond_input, V_cond, padding='SAME') # 1x1 convolution
                    C += b_cond
                    C = tf.tanh(C)
                    Y += C

                # filter and gate
                Y = tf.tanh(Y[:, :, :r])*tf.sigmoid(Y[:, :, r:])
                
                # add residual channel
                if self.do_postproc:    
                    skip_out = tf.nn.convolution(Y, W_s, padding='SAME')
                    skip_out += b_s
                else:
                    skip_out = []
                
                if self.do_GU:
                    Y = tf.nn.convolution(Y, W_p, padding='SAME')
                    Y += b_p
                    Y += X

        return Y, skip_out

    def _postproc_module(self, residual_module_outputs):
        with tf.variable_scope('postproc_module'):

            W1 = get_weight_variable('W1')
            b1 = get_bias_variable('b1')
            W2 = get_weight_variable('W2')
            b2 = get_bias_variable('b2')

            # sum of residual module outputs
            X = tf.zeros_like(residual_module_outputs[0])
            for R in residual_module_outputs:
                X += R

            Y = tf.nn.convolution(X, W1, padding='SAME')    
            Y += b1
            Y = tf.nn.relu(Y)

            Y = tf.nn.convolution(Y, W2, padding='SAME')    
            Y += b2

            if type(self.output_channels) is list:
                #import ipdb; ipdb.set_trace()
                output_list = []
                start = 0 
                for channels in self.output_channels:
                    output_list.append(Y[:,:,start:start+channels])
                    start += channels
                Y = output_list
            
        return Y
    
    def _last_layer(self, last_layer_ip):
        with tf.variable_scope('last_layer'):

            W = get_weight_variable('W')
            b = get_bias_variable('b')

            X = last_layer_ip            

            Y = tf.nn.convolution(X, W, padding='SAME')    
            Y += b

            if type(self.output_channels) is list:
                #import ipdb; ipdb.set_trace()
                output_list = []
                start = 0 
                for channels in self.output_channels:
                    output_list.append(Y[:,:,start:start+channels])
                    start += channels
                Y = output_list
            
        return Y

    def forward_pass(self, X_input, cond_input=None):
        
        skip_outputs = []
        with tf.variable_scope(self._name, reuse=True):

            if self._use_cond:
                C = self._embed_cond(cond_input)
            else:
                C = None    

            R = self._input_layer(X_input)
            X = R
            for i, dilation in enumerate(self.dilations):
                X, skip = self._conv_module(X, R, i, dilation, cond_input=C)
                skip_outputs.append(skip)

            if self.do_postproc:    
                Y = self._postproc_module(skip_outputs)    
            else:
                Y = self._last_layer(X)                

        return Y
                                                 