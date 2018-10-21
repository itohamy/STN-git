#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 13:32:35 2018

@author: fredman
"""

import tensorflow as tf
#import tf_funcs
import numpy as np
from tensorflow.python.framework import ops
from scipy.linalg import expm_frechet
import cv2
from random import randint,uniform
from tensorflow.contrib import layers
import tensorflow as tf
import tensorflow.contrib.layers as lays

'''
input - a list of desired transformations. 
 Can hold:
   The top 4 cover the affine transfromations broken down:
     'r' = rotation
     'sc' = scaling
     'sh' = sheer
     't' = translation
   Sine for the matrices of scale and rotations or for scale and sheer we
   know that e^a * e^b = e^(a+b), then we will first learn a+b and only then
   thake the matrix exponent in order to save time.
     'sc_r' : scale_and_rotate,
     'sc_sh' : scale_and_sheer,
   we will test what happens if we run an area preserving map and on top of
   it a uniform scaling map (these create a full affine map).
     "ap" : area_preserving,
     "us" : uniform_scaling,
   we will test what happens if we run a full affine map.
     "fa" : full_affine
   For for classification with the regular stn, we want a full affine without taking the expnent.
     "ane" : affine_no_exp

output - 
    1. The vectorized afine matrix (with shape (1,6)) for each image
    2. The vectorized r,s and t matrices (with shape (1,6)) for each image
        we return this in order to give different penalties based on our prior knowlege
        of the possible transformations. 
'''


def transfromation_parameters_regressor(transformations,images,keep_prob,width,batch_size,num_channels,
                                        activation_func="relu",weight_stddev=5e-4):
    activation = getattr(tf.nn,activation_func)

    transform_maps = []
    transform_maps_dict = {}

    methods = get_methods()

    correct_transformations_order(transformations,transform_maps_dict)
    valid_input = False

    #first we'll use the localization network to learn the different transfromation
    #parametes, and then we'll create each transformation by taking the matrix exponent
    #using its parameters.
    #we'll learn 7 parameters even though there's 6 degrees of freedom. So even though
    #if there is rotatation, we only need one param for sheer, we will learn both parameters,
    #since all the cases are covered this way, and its not so expensive to learn this one too.
    num_of_params = 7
    params = regress_parameters(images,width,keep_prob,activation,num_of_params,weight_stddev,num_channels)

    print("Going to...")
    for transformation in transformations:
        if transformation in methods:
            valid_input = True
            transformation_exponentiol,orig_transformation = methods[transformation](params)
            transform_maps.append(transformation_exponentiol)
            #transform_maps_dict[transformation] = tf.reshape(tf.slice(transform_maps[-1], [0, 0, 0], [-1, 2, 3]), [-1, 6])
            transform_maps_dict[transformation] = tf.reshape(orig_transformation,[-1,6])
    #rename the scaling key in the dictionary
    if 'sc' in transform_maps_dict:
        transform_maps_dict['sc'] = transform_maps_dict.pop('sc_r',transform_maps_dict['sc'])
        transform_maps_dict['sc'] = transform_maps_dict.pop('sc_sh',transform_maps_dict['sc'])

    if not valid_input:
        raise ValueError("Bad input parameters! Choose any combination of 'r','sc','t' in a list, and add the images.")
    if len(transform_maps) > 1:
        return tf.reshape(transformations_composition(transform_maps),[-1,6]),transform_maps_dict
    else:
        transformation = tf.slice(transform_maps[0],[0,0,0],[-1,2,3])
        return tf.reshape(transformation,[-1,6]),transform_maps_dict
    #        return tf.reshape(transformation, [-1, 6]), transform_maps_dict


def get_methods():
    return {"r":rotation,
            "sc":scaling,
            "sh":sheer,
            "t":translation,
            #           "sc_r" : scale_and_rotate,
            #           "sc_sh" : scale_and_sheer,
            #           "sh_r" : sheer_with_rotation,
            "ap":area_preserving,
            "us":uniform_scaling,
            "fa":full_affine,
            "ane":affine_no_exp
            }


def get_param_indexes():
    return {"r":0,
            "sc1":1,
            "sc2":2,
            "sh1":3,
            "sh2":4,
            "t1":5,
            "t2":6,
            }


# %%

#this is the stn's localization network
#here we construct it using a 2-layer FC network.
def regress_parameters(x,width,keep_prob,activation_func,num_of_params,weight_stddev,num_channels):

    net = tf.reshape(x,shape=[-1,128,128,num_channels])
    net = lays.conv2d(net,32,[5,5],stride=2,padding='SAME')
    net = lays.conv2d(net,16,[5,5],stride=2,padding='SAME')
    net = layers.flatten(net)
    net = lays.fully_connected(net, 7 ,activation_fn=tf.nn.relu,scope='fc-final')

    # #We'll setup the two-layer localisation network to figure out the
    # # parameters for an affine transformation of the input
    # # Create variables for fully connected layer
    # W_fc_loc1 = weight_variable([width * width * num_channels,20],weight_stddev)
    # #W_fc_loc1 = tf.Print(W_fc_loc1,[W_fc_loc1],message="W_fc_loc1: ",summarize=100)
    # b_fc_loc1 = bias_variable([20])
    # #b_fc_loc1 = tf.Print(b_fc_loc1,[b_fc_loc1],message="b_fc_loc1: ",summarize=100)
    #
    # W_fc_loc2 = weight_variable([20,num_of_params],weight_stddev)
    # # Use identity transformation as starting point
    # b_fc_loc2 = bias_variable([1,num_of_params])
    # # Define the two layer localisation network
    # h_fc_loc1 = activation_func(tf.matmul(x,W_fc_loc1) + b_fc_loc1)
    # #h_fc_loc1 = tf.Print(h_fc_loc1,[h_fc_loc1],message="h_fc_loc1: ",summarize=100)
    # # We can add dropout for regularizing and to reduce overfitting like so:
    # h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1,keep_prob)
    # #h_fc_loc1_drop = tf.Print(h_fc_loc1_drop,[h_fc_loc1_drop],message="h_fc_loc1_drop: ",summarize=100)
    # # Second layer
    # s_fc_loc2 = activation_func(tf.matmul(h_fc_loc1_drop,W_fc_loc2) + b_fc_loc2)
    # #s_fc_loc2 = tf.Print(s_fc_loc2,[s_fc_loc2],message="s_fc_loc2: ",summarize=100)

    return net #s_fc_loc2


def expm(params_matrix,batch_size,num_channels):
    # Take the matrix exponentioal of the affine map, inorder to get an affine-defiomorphism map.
    exp_params = tf.reshape(params_matrix,[-1,2,3])
    #need to append another row so that the matrix exponential will get a square matrix
    initial = tf.zeros_like(tf.slice(exp_params,[0,0,0],[-1,1,3]))
    initial = tf.cast(initial,tf.float32)
    exp_params = tf.concat([exp_params,initial],1)
    return matrix_expnential(exp_params,batch_size)


# %%

def matrix_expnential(matrices,batch_size):
    matrices = tf.cast(matrices,tf.float32)
    x_unpacked = tf.unstack(matrices,num=batch_size)  # defaults to axis 0, returns a list of tensors
    processed = []  # this will be the list of processed tensors
    for t in x_unpacked:
        t = tf.cast(t,tf.float64)
        result_tensor = tf.linalg.expm(t)
        processed.append(result_tensor)

    #output = tf.concat(tf.cast(processed, tf.float32), 0)
    return tf.reshape(processed,[-1,3,3])


#Define the matrix exponential gradient for back propagation
def register_gradient():
    if "MatrixExponential" not in ops._gradient_registry._registry:
        #Adding the gradient for the matrix exponential functionrixExponential" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("MatrixExponential")
        def _expm_grad(op,grad):
            # We want the backward-mode gradient (left multiplication).
            # Let X be the NxN input matrix.
            # Let J(X) be the the N^2xN^2 complete Jacobian matrix of expm at X.
            # Let Y be the NxN previous gradient in the backward AD (left multiplication)
            # We have
            # unvec( ( vec(Y)^T . J(X) )^T )
            #   = unvec( J(X)^T . vec(Y) )
            #   = unvec( J(X^T) . vec(Y) )
            # where the last part (if I am not mistaken) holds in the case of the
            # exponential and other matrix power series.
            # It can be seen that this is now the forward-mode derivative
            # (right multiplication) applied to the Jacobian of the transpose.
            grad_func = lambda x,y:expm_frechet(x,y,compute_expm=False)
            return tf.py_func(grad_func,[tf.transpose(op.inputs[0]),grad],tf.float64)


# %%
def transformations_composition(transform_maps):
    mat = transform_maps[0]
    for ind in range(1,len(transform_maps)):
        mat = tf.matmul(mat,transform_maps[ind])
    return tf.slice(mat,[0,0,0],[-1,2,3])


def weight_variable(shape,stddev):
    initial = tf.truncated_normal(shape,stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


# %%
# Runs
def rotation(params):
    print("Rotate !!!")
    param_indexes = get_param_indexes()
    params = tf.slice(params,[0,param_indexes['r']],[-1,1])

    zero_vec = tf.zeros_like(params)
    params_matrix = (tf.concat([zero_vec,params,zero_vec,-1 * params,zero_vec,zero_vec],1))

    one_vec = tf.ones_like(params)
    exp_params = (tf.concat(
        [tf.cos(params),-tf.sin(params),zero_vec,tf.sin(params),tf.cos(params),zero_vec,zero_vec,zero_vec,one_vec],1))
    exp_params = tf.reshape(exp_params,[-1,3,3])

    return exp_params,params_matrix


# %%
# Runs
def scaling(params):
    print("Scale !!!")

    #separate the 2 parameters
    param_indexes = get_param_indexes()
    vec1 = tf.slice(params,[0,param_indexes['sc1']],[-1,1])
    vec2 = tf.slice(params,[0,param_indexes['sc2']],[-1,1])

    # In the next line we add e^-6 to the zero vec, as a workaround to avoid nan values.
    # Without this, the zero vector causes nan values in the back propogation.
    zero_vec = tf.zeros_like(vec1)
    params_matrix = (tf.concat([vec1,zero_vec,zero_vec,zero_vec,vec2,zero_vec],1))

    one_vec = tf.ones_like(vec1)
    exp_params = (
        tf.concat([tf.exp(vec1),zero_vec,zero_vec,zero_vec,tf.exp(vec2),zero_vec,zero_vec,zero_vec,one_vec],1))
    exp_params = tf.reshape(exp_params,[-1,3,3])

    return exp_params,params_matrix


# %%
# Runs
#def scale_and_rotate(x, width, keep_prob, batch_size, activation_func, weight_stddev, num_channels):
#    num_of_params=3
#    print("Scale !!!")
#    print("Rotate !!!")
#
#    params = regress_parameters(x,width,keep_prob,activation_func,num_of_params,weight_stddev,num_channels)
#
#    #separate the 2 parameters
#    vec1 = (tf.slice(params, [0,0], [-1,1]))
#    vec2 = (tf.slice(params, [0,1], [-1,1]))
#    vec3 = (tf.slice(params, [0,2], [-1,1]))
#
#    # In the next line we add e^-6 to the zero vec, as a workaround to avoid nan values.
#    # Without this, the zero vector causes nan values in the back propogation.
#    zero_vec = tf.zeros_like(vec1)
#    params_matrix = (tf.concat([vec1, vec3, zero_vec, -1*vec3, vec2, zero_vec], 1))
#    exp_params = expm(params_matrix, batch_size, num_channels)
#
#    return exp_params, params_matrix

# %%
# Runs
#def scale_and_sheer(x, width, keep_prob, batch_size, activation_func, weight_stddev, num_channels):
#    num_of_params=4
#    print("Scale !!!")
#    print("Sheer !!!")
#
#    params = regress_parameters(x,width,keep_prob,activation_func,num_of_params,weight_stddev,num_channels)
#
#    #separate the 2 parameters
#    vec1 = (tf.slice(params, [0,0], [-1,1]))
#    vec2 = (tf.slice(params, [0,1], [-1,1]))
#    vec3 = (tf.slice(params, [0,2], [-1,1]))
#    vec4 = (tf.slice(params, [0,3], [-1,1]))
#
#    # In the next line we add e^-6 to the zero vec, as a workaround to avoid nan values.
#    # Without this, the zero vector causes nan values in the back propogation.
#    zero_vec = tf.zeros_like(vec1)
#    params_matrix = (tf.concat([vec1, vec3, zero_vec, vec4, vec2, zero_vec], 1))
#    exp_params = expm(params_matrix, batch_size, num_channels)
#
#    return exp_params, params_matrix


# %%
# Runs
#def sheer_with_rotation(x, width, keep_prob, batch_size, activation_func, weight_stddev, num_channels):
#    #this covers the case where also rotation was requested, in which case we only need
#    #to learn one parameter to get the sheer in both directions.
#
#    print("Sheer !!!")
#    num_of_params=1
#    params = regress_parameters(x,width,keep_prob,activation_func,num_of_params,weight_stddev,num_channels)
#
#    # In the next line we add e^-6 to the zero vec, as a workaround to avoid nan values.
#    # Without this, the zero vector causes nan values in the back propogation.
#    zero_vec = tf.zeros_like(params)
#    params_matrix = (tf.concat([zero_vec, params, zero_vec, zero_vec, zero_vec, zero_vec], 1))
#    exp_params = expm(params_matrix, batch_size, num_channels)
#
#    return exp_params, params_matrix

# Runs
def sheer(params):
    #this covers the case where sheer was requested without rotation, in which case we need
    #to learn 2 parameter2 to get the sheer in both directions.

    print("Sheer !!!")
    #separate the 2 parameters
    param_indexes = get_param_indexes()
    vec1 = tf.slice(params,[0,param_indexes['sh1']],[-1,1])
    vec2 = tf.slice(params,[0,param_indexes['sh2']],[-1,1])
    # In the next line we add e^-6 to the zero vec, as a workaround to avoid nan values.
    # Without this, the zero vector causes nan values in the back propogation.
    zero_vec = tf.zeros_like(vec1)
    # this is what we'll return to the regularization function which wants the params to be zero so that the exp will be the unit matrix
    params_matrix = (tf.concat([zero_vec,vec1,zero_vec,vec2,zero_vec,zero_vec],1))
    #    params_matrix1 = (tf.concat([zero_vec, vec1, zero_vec, zero_vec, zero_vec, zero_vec], 1))
    #    params_matrix2 = (tf.concat([zero_vec, zero_vec, zero_vec, vec2, zero_vec, zero_vec], 1))

    one_vec = tf.ones_like(vec1)
    exp_params1 = (tf.concat([one_vec,vec1,zero_vec,zero_vec,one_vec,zero_vec,zero_vec,zero_vec,one_vec],1))
    exp_params1 = tf.reshape(exp_params1,[-1,3,3])

    exp_params2 = (tf.concat([one_vec,zero_vec,zero_vec,vec2,one_vec,zero_vec,zero_vec,zero_vec,one_vec],1))
    exp_params2 = tf.reshape(exp_params2,[-1,3,3])

    exp_params = tf.matmul(exp_params1,exp_params2)

    return exp_params,params_matrix


# %%
# Runs
def translation(params):
    print("Translate !!!")
    #separate the 2 parameters
    param_indexes = get_param_indexes()
    vec1 = tf.slice(params,[0,param_indexes['t1']],[-1,1])
    vec2 = tf.slice(params,[0,param_indexes['t2']],[-1,1])

    # In the next line we add e^-6 to the zero vec, as a workaround to avoid nan values.
    # Without this, the zero vector causes nan values in the back propogation.
    zero_vec = tf.zeros_like(vec1)
    params_matrix = (tf.concat([zero_vec,zero_vec,vec1,zero_vec,zero_vec,vec2],1))
    #params_matrix = tf.Print(params_matrix,[params_matrix],message="params_matrix: ",summarize=100)

    one_vec = tf.ones_like(vec1)
    exp_params = (tf.concat([one_vec,zero_vec,vec1,zero_vec,one_vec,vec2,zero_vec,zero_vec,one_vec],1))
    exp_params = tf.reshape(exp_params,[-1,3,3])

    return exp_params,params_matrix


# %%
# Runs
def uniform_scaling(x,width,keep_prob,batch_size,activation_func,weight_stddev,num_channels):
    print("scale uniformaly !!!")
    num_of_params = 1
    params = regress_parameters(x,width,keep_prob,activation_func,num_of_params,weight_stddev,num_channels)

    # In the next line we add e^-6 to the zero vec, as a workaround to avoid nan values.
    # Without this, the zero vector causes nan values in the back propogation.
    zero_vec = tf.zeros_like(params)
    params_matrix = (tf.concat([params,zero_vec,zero_vec,zero_vec,params,zero_vec],1))
    exp_params = expm(params_matrix,batch_size,num_channels)

    return exp_params,params_matrix


# %%


def area_preserving(x,width,keep_prob,batch_size,activation_func,weight_stddev,num_channels):
    print("area_preserve !!!")

    params = learn_all_params(x,width,keep_prob,batch_size,activation_func,weight_stddev,num_channels)

    const_matrix = np.eye(6,dtype=np.float32)
    const_matrix[1,1] = 1.0
    const_matrix[0,1] = 1.0
    const_matrix = tf.constant(const_matrix)

    params_matrix = tf.matmul(params,const_matrix)

    exp_params = expm(params_matrix,batch_size,num_channels)

    return exp_params,params_matrix


# %%
# Runs
def full_affine(x,width,keep_prob,batch_size,activation_func,weight_stddev,num_channels):
    print("complete affine !!!")

    params_matrix = learn_all_params(x,width,keep_prob,batch_size,activation_func,weight_stddev,num_channels)

    exp_params = expm(params_matrix,batch_size,num_channels)

    return exp_params,params_matrix


# %%
# Runs
def affine_no_exp(x,width,keep_prob,batch_size,activation_func,weight_stddev,num_channels):
    print("Running STN without matrix exponential !!!")

    params = learn_all_params(x,width,keep_prob,batch_size,activation_func,weight_stddev,num_channels)

    #    #need to append another row sof [0,0,1]
    #    #separate the 2 parameters
    #    vec1 = (tf.slice(params, [0,1], [-1,1]))
    #    zero_vec = tf.zeros_like(vec1)
    #    one_vec = tf.ones_like(vec1)
    #    params_matrix = (tf.concat([params, zero_vec, zero_vec, one_vec], 1))
    #    params_matrix = tf.reshape(params_matrix, [-1, 3, 3])

    params_matrix = tf.reshape(params,[-1,2,3])

    #    exp_params = expm(params_matrix, batch_size, num_channels) #uncomment if you want to take the expnent without any regularization penalty.

    return params_matrix,params_matrix


# %%

#this is just another way to learn all 6 params without having to concat the vectors learned.
#can also just concat (like we did for scaling for example).
def learn_all_params(x,width,keep_prob,batch_size,activation_func,weight_stddev,num_channels):
    #We'll setup the two-layer localisation network to figure out the
    # parameters for an affine transformation of the input
    # Create variables for fully connected layer
    W_fc_loc1 = weight_variable([width * width * num_channels,20],weight_stddev)
    b_fc_loc1 = bias_variable([20])

    W_fc_loc2 = weight_variable([20,6],weight_stddev)
    # Use identity transformation as starting point
    initial = np.array([[1.,0,0],[0,1.,0]])
    initial = initial.astype('float32')
    initial = initial.flatten()
    b_fc_loc2 = tf.Variable(initial_value=initial,name='b_fc_loc2')

    # Define the two layer localisation network
    h_fc_loc1 = activation_func(tf.matmul(x,W_fc_loc1) + b_fc_loc1)
    # We can add dropout for regularizing and to reduce overfitting like so:
    h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1,keep_prob)
    # Second layer
    h_fc_loc2 = activation_func(tf.matmul(h_fc_loc1_drop,W_fc_loc2) + b_fc_loc2)

    return h_fc_loc2


def correct_transformations_order(transformations,transform_maps_dict):
    #Sine for the matrices of scale and rotations or for scale and sheer we
    #know that e^a * e^b = e^(a+b), then we will first learn a+b and only then
    #take the matrix exponent in order to save time.
    #If there are both sheer and rotation, then we only need to learn one more
    #param for the sheer.
    #    if ('sh' in transformations) and ('r' in transformations):
    #        transformations.remove('sh')
    #        transformations.append('sh_r')
    #    if ('sc' in transformations) and ('r' in transformations):
    #        transformations.remove('sc')
    #        transformations.remove('r')
    #        transformations.append('sc_r')
    #        transform_maps_dict['sc'] = None
    #    if ('sc' in transformations) and ('sh' in transformations):
    #        transformations.remove('sc')
    #        transformations.remove('sh')
    #        transformations.append('sc_sh')
    #        transform_maps_dict['sc'] = None
    #push the traslation operation to the end of the matrix.
    if 't' in transformations:
        transformations.remove('t')
        transformations.insert(0,'t')


# %%

def transform_input_images(images,rows,cols,batch_size):
    # this method takes one of the images in the training data
    # and rotates it in different angles to create a new training
    # data from that rotated digit
    transformed = None
    num_of_images = 10
    #We'll find the biggenst number which is >= mages.shape[0]
    #and that divides by batch_size
    #    batch_size_multiplier = np.floor(images.shape[0]/batch_size)
    #    num_of_training_data = batch_size*batch_size_multiplier
    val = int(np.floor(images.shape[0] / num_of_images))
    l = np.ones((num_of_images)) * val
    l[0] += int(images.shape[0] - np.sum(l))
    #    val = int(np.floor(num_of_training_data/num_of_images))
    #    l = np.ones((num_of_images))*val
    #    l[0] += int(num_of_training_data - np.sum(l))
    l = l.astype(int)
    for image_num in range(num_of_images):
        for ind in range(l[image_num]):
            #first we'll rotate the images with the following angle range
            rotation = randint(310,410) % 360
            im = np.reshape(images[image_num,:],(rows,cols))

            #can comment out the following lines if you dont want ratations
            M = cv2.getRotationMatrix2D((cols / 2,rows / 2),rotation,1)
            #            dst = cv2.warpAffine(im,M,(cols,rows)) #uncomment this if you only want rotations
            im = cv2.warpAffine(im,M,(cols,rows))  ##########  comment if you don't want rotations. ##########

            #now we will also translate and scale the images randomely
            #scale = (uniform(0.8, 1.4), uniform(0.8, 1.4))
            translation = (uniform(-0.2,0.2),uniform(-0.2,0.2))
            #tform = AffineTransform(scale=scale)
            #tform = AffineTransform(scale=scale, translation=translation)
            #dst = warp(im, tform, preserve_range=False, mode='edge')
            to_scale = uniform(0.65,1.30)
            dst = cv2_clipped_zoom(im,to_scale)
            #dst = im ########  Uncomment for rotations only ##########
            if transformed is not None:
                transformed = np.concatenate((transformed,np.reshape(dst,(1,rows * cols))))
            else:
                transformed = np.reshape(dst,(1,rows * cols))
    np.random.shuffle(transformed)
    return transformed


#give a diffrent penalty to each type of transformation magnituted
def transformation_regularization(affine_maps,regularizations):
    transformations = [*regularizations]
    stable_transformations = 0.
    unit_transformation = tf.constant([1.,0.,0.,0.,1.,0.])
    for transformation in transformations:
        if transformation in affine_maps:
            maps = affine_maps[transformation]
            if transformation == "ane":
                return tf.reduce_sum(regularizations[transformation] * tf.square(
                    tf.subtract(tf.reduce_mean(maps,0),unit_transformation)))
            elif transformation == "sc":
                l2_l1 = l2_l1_func(maps)
                reg = regularizations[transformation] * tf.reduce_sum(tf.reduce_sum(l2_l1,0))
            else:
                reg = regularizations[transformation] * tf.reduce_sum(tf.reduce_sum(tf.square(maps),1),0)
            stable_transformations += reg
    #    #scaling_lamda = 0.02
    #    scaling_lamda = 10
    #    translation_lamda = 30
    #    full_affine_lamda = 32
    #    stable_transformations = tf.constant([0.])
    #    if "t" in affine_maps: #if there was a translation map, then we want to minimize it's magnitude
    #        maps = affine_maps["t"] #get the transformation maps after the exponent
    #        stable_transformations = translation_lamda*tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(maps), 1)), 0)
    #    if ("sc" in affine_maps) or ("us" in affine_maps): #if there was a scaling map, then we want to minimize it's magnitude
    #        maps = get_maps(affine_maps,"sc","us") #get the transformation maps before the exponent
    ##        scaling_magnitude = tf.reduce_sum(scaling_lamda*tf.abs(tf.subtract(tf.reduce_mean(affine_maps["s"], 0), unit_scaling)))
    ##        stable_transformations = tf.add(stable_transformations, scaling_magnitude)
    #        scaling_magnitude = scaling_lamda*tf.square(tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(maps), 1)), 0))
    ##        scaling_magnitude = scaling_lamda*tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(maps), 1)), 0)
    #        stable_transformations = tf.add(stable_transformations, scaling_magnitude) # add to the translation magnitude
    #    if ("fa" in affine_maps) or ("ap" in affine_maps): #if there was a scaling map, then we want to minimize it's magnitude
    #        maps = get_maps(affine_maps,"fa","ap") #get the transformation maps before the exponent
    #        scaling_magnitude = full_affine_lamda*tf.reduce_sum(tf.sqrt(tf.reduce_sum(tf.square(maps), 1)), 0)
    #        stable_transformations = tf.add(stable_transformations, scaling_magnitude) #
    return stable_transformations


def l2_l1_func(
        maps):  #this function takes the square of all values greater than zero, and the absolute of all those smaller than zero
    #    m1 = tf.minimum(maps, 0)
    #    m1 = tf.square(tf.square(m1))
    #    m1 *= 0
    #    m2 = tf.maximum(maps, 0)
    #    m2 = tf.abs(m2)
    #    m2 *= 1e+16
    maps = tf.maximum(maps,0)
    maps = tf.square(maps)
    #    return tf.add(m1,m2)
    return maps


def get_maps(affine_maps,opt1,opt2):
    if opt1 in affine_maps:
        return affine_maps[opt1]
    return affine_maps[opt2]


# used during the transformation testing
# will perform uniform scaling on a image using the zoom_factor
def cv2_clipped_zoom(img,zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of
    the image without changing dimensions
    Args:
        img : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    height,width = img.shape[:2]  # It's also the final desired shape
    new_height,new_width = int(height * zoom_factor),int(width * zoom_factor)

    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1,x1 = max(0,new_height - height) // 2,max(0,new_width - width) // 2
    y2,x2 = y1 + height,x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1,x1,y2,x2 = bbox
    cropped_img = img[y1:y2,x1:x2]

    # Handle padding when downscaling
    resize_height,resize_width = min(new_height,height),min(new_width,width)
    pad_height1,pad_width1 = (height - resize_height) // 2,(width - resize_width) // 2
    pad_height2,pad_width2 = (height - resize_height) - pad_height1,(width - resize_width) - pad_width1
    pad_spec = [(pad_height1,pad_height2),(pad_width1,pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img,(resize_width,resize_height))
    result = np.pad(result,pad_spec,mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


# %%

#this is the stn's localization network
#here we construct it using a 4-layer CNN network.
def regress_parameters_conv(x,input_size,keep_prob,activation_func,num_of_params,weight_stddev,num_channels):
    # Convolutional Layer 1.
    filter_size1 = 5
    num_filters1 = 32

    # Convolutional Layer 2.
    filter_size2 = 5
    num_filters2 = 32

    # Fully-connected layer.
    fc_size = 32

    # The convolutional layers expect x to be encoded as a 4-dim tensor so we have to
    #reshape it so its shape is instead [num_images, img_height, img_width, num_channels]
    input_images = tf.reshape(x,shape=[-1,input_size,input_size,num_channels])

    layer_conv1,weights_conv1 = \
        new_conv_layer(input=input_images,
                       num_input_channels=num_channels,
                       filter_size=filter_size1,
                       num_filters=num_filters1,
                       weight_stddev=weight_stddev,
                       use_pooling=True)

    layer_conv2,weights_conv2 = \
        new_conv_layer(input=layer_conv1,
                       num_input_channels=num_filters1,
                       filter_size=filter_size2,
                       num_filters=num_filters2,
                       weight_stddev=weight_stddev,
                       use_pooling=True)

    #before sending the input to a non-convolutional layer, need to re-flatten it
    #(need to undo what we did in the reshape before the convolutional layer above)
    layer_flat,num_features = flatten_layer(layer_conv2)

    layer_fc1 = new_fc_layer(input=layer_flat,
                             num_inputs=num_features,
                             num_outputs=fc_size,
                             weight_stddev=weight_stddev,
                             use_relu=True)
    layer_fc1_drop = tf.nn.dropout(layer_fc1,keep_prob)

    layer_fc2 = new_fc_layer(input=layer_fc1_drop,
                             num_inputs=fc_size,
                             num_outputs=fc_size,
                             weight_stddev=weight_stddev,
                             use_relu=True)
    layer_fc2_drop = tf.nn.dropout(layer_fc2,keep_prob)

    layer_fc2 = new_fc_layer(input=layer_fc2_drop,
                             num_inputs=fc_size,
                             num_outputs=num_of_params,
                             weight_stddev=weight_stddev,
                             use_relu=False)

    return layer_fc2


def new_conv_layer(input,  # The previous layer.
                   num_input_channels,  # Num. channels in prev. layer.
                   filter_size,  # Width and height of each filter.
                   num_filters,  # Number of filters.
                   weight_stddev,
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size,filter_size,num_input_channels,num_filters]

    # Create new weights aka. filters with the given shape.
    weights = weight_variable(shape=shape,stddev=weight_stddev)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1,1,1,1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1,2,2,1],
                               strides=[1,2,2,1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer,weights


#A convolutional layer produces an output tensor with 4 dimensions. We will add fully-connected
# layers after the convolution layers, so we need to reduce the 4-dim tensor to 2-dim which can
# be used as input to the fully-connected layer.
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer,[-1,num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat,num_features


def new_fc_layer(input,  # The previous layer.
                 num_inputs,  # Num. inputs from prev. layer.
                 num_outputs,  # Num. outputs.
                 weight_stddev,
                 use_relu=True):  # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = weight_variable(shape=[num_inputs,num_outputs],stddev=weight_stddev)
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input,weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


def new_biases(length):
    return tf.Variable(tf.constant(0.05,shape=[length]))