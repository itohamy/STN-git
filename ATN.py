#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 18:16:52 2018

@author: fredman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:12:14 2018

@author: fredman
"""

########################################################################
###
###  This file creates takes in input images, and alignes them, rturning the
###  alined images, and the alignment data loss and regularization loss.
###  The alignment is achieved by using a pixel-location variance loss function.
###  We take the matrix exponent to form affine deffeomorphism transformations.
###
###  Implementations comment -
###  In order to implement the matrix exponential (and its gradient),
###  there was a need to use the batch_size in order to unstack the parametes before taking their exp.
###  For this I needed to use always the same batch size, so I possibly removed some images from
###  the training or test set, so that the number of images mod batch_size will be zero.
###
########################################################################

import os,sys

sys.path.insert(1,os.path.join(sys.path[0],'..'))
import tensorflow as tf
from atn_helpers.spatial_transformer import transformer
from atn_helpers.tranformations_helper import transfromation_parameters_regressor,transformation_regularization


class alignment_transformer_network:

    def __init__(self,input_data,requested_transforms,regularizations,batch_size,image_size,num_channels,num_classes,
                 weight_stddev,activation_func,only_stn,keep_prob,y=None):
        self.X = tf.reshape(input_data,shape=[-1,image_size * image_size * num_channels])  #reshaping to 2 dimensions
        self.y = y
        self.requested_transforms = requested_transforms
        self.regularizations = regularizations
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = tf.cast(num_classes,tf.int32)
        self.batch_size = batch_size
        self.only_stn = only_stn
        self.keep_prob = keep_prob
        self.weight_stddev = weight_stddev
        self.activation_func = activation_func
        self.sigma = 0.5  # for Geman-Mecclure robust function
        self.affine_maps = None
        self.logits = input_data
        self.transformations_regularizers = tf.constant([0.,0.,0.,0.,0.,0.])

    def stn_diffeo(self):

        with tf.variable_scope("atn"):
            x_tensor = tf.reshape(self.X,[-1,self.image_size,self.image_size,self.num_channels])
            h_fc_loc2,self.affine_maps = transfromation_parameters_regressor(self.requested_transforms,self.X,
                                                                             self.keep_prob,self.image_size,
                                                                             self.batch_size,self.num_channels,
                                                                             self.activation_func,self.weight_stddev)

            h_fc_loc2 = tf.Print(h_fc_loc2,[h_fc_loc2],message="h_fc_loc2: ",summarize=100)
            out_size = (self.image_size,self.image_size)
            logits = transformer(x_tensor,h_fc_loc2,out_size)
            #to avoid the sparse indexing warning, comment the next line, and uncomment the one after it.
            self.logits = tf.reshape(logits,shape=[-1,self.image_size,self.image_size,self.num_channels])
            #        self.logits = tf.reshape(logits, shape=[self.batch_size, self.image_size, self.image_size, self.num_channels])

            transformations = h_fc_loc2
            #transformations = tf.reshape(transformations, [-1,6])
            #slice only the first row to show corrent status
            self.transformations = (tf.slice(transformations,[0,0],[1,6]))

            return self.logits,self.affine_maps


    def compute_alignment_loss(self,lables_one_hot=None):

        with tf.variable_scope("atn"):

            self.lables_one_hot = lables_one_hot

            if self.only_stn == True:
                print("only STN is {}!! (so no alignment...)".format(self.only_stn))
                zero = tf.constant(0.)
                return zero

            if self.y is not None:  #we will align each class separetly
                self.alignment_loss = self.alignment_loss_per_class()
            else:  # we will align the entire data together
                self.alignment_loss, a, b = self.alignment_loss()

            return self.alignment_loss, a, b


    def compute_transformations_regularization(self,affine_maps=None):

        if self.only_stn == True:
            print("only STN is {}!! (so --currently(!)-- not calculating the reg loss...)".format(self.only_stn))
            zero = tf.constant(0.)
            return zero

        #if affine_maps is equal to None, then the user wants to compute congealng loss for a different layer than that for which
        # he ran the STN on. So he needs to give the parameter maps he got from the STN layer as an input.
        # otherwise we'll assume that on this layer he also ran the STN, and we'll already have the affine_maps in self.affine_maps.
        if affine_maps is None:
            affine_maps = self.affine_maps
        self.transformations_regularizers = transformation_regularization(affine_maps,
                                                                          self.regularizations)  #give a diffrent penalty to each type of transformation magnituted
        return self.transformations_regularizers

    #    old code. here I returned the alignment loss together with the transformed inputs.
    #    def atn_layer(self):
    #
    #        # Since x is currently [batch, height*width], we need to reshape to a
    #        # 4-D tensor to use it in a convolutional graph.  If one component of
    #        # `shape` is the special value -1, the size of that dimension is
    #        # computed so that the total size remains constant.  Since we haven't
    #        # defined the batch dimension's shape yet, we use -1 to denote this
    #        # dimension should not change size.
    #        x_tensor = tf.reshape(self.X, [-1, self.image_size, self.image_size, self.num_channels])
    ##        x_tensor = tf.reshape(self.X, [self.batch_size, self.image_size, self.image_size, self.num_channels])
    #        h_fc_loc2, affine_maps = transfromation_parameters_regressor(self.requested_transforms, self.X, self.keep_prob, self.image_size, self.batch_size, self.num_channels, self.activation_func, self.weight_stddev)
    #
    #
    #        #We'll create a spatial transformer module to identify discriminative
    #        # patches
    #        out_size = (self.image_size,self.image_size)
    #        logits = transformer(x_tensor, h_fc_loc2, out_size)
    #        #to avoid the sparse indexing warning, comment the next line, and uncomment the one after it.
    #        logits = tf.reshape(logits, shape=[-1, self.image_size, self.image_size, self.num_channels])
    ##        logits = tf.reshape(logits, shape=[self.batch_size, self.image_size, self.image_size, self.num_channels])
    #        self.logits = logits
    #
    #        #find the variance (sqaured difference) for each pixel stack in the batch
    #        #then we will sum the variances (might be more correct to take their average)
    #       # logits = tf.reshape(logits, shape=[-1,self.image_size**2])
    #
    #        transformations = h_fc_loc2
    #        #transformations = tf.reshape(transformations, [-1,6])
    #        #slice only the first row to show corrent status
    #        transformations = (tf.slice(transformations, [0,0], [1,6]))
    #
    #        if self.only_stn == True:
    #            zero = tf.reduce_sum(tf.constant([0.]))
    #            return logits, transformations, zero, zero
    #
    #        if self.y is not None: #we will align each class separetly
    #            alignment_losses, transformations_regularizers = self.alignment_loss_and_reg_per_class(affine_maps)
    #        else: # we will align the entire data together
    #            alignment_losses, transformations_regularizers = self.alignment_loss_and_reg(logits,affine_maps)
    #
    #        alignment_losses = tf.reduce_sum(tf.cast(alignment_losses, tf.float32))
    #
    #        return logits, transformations, alignment_losses, transformations_regularizers


    def get_transformations(self):
        return self.transformations


    def alignment_loss_per_class(self):
        self.y = tf.cast(self.y,tf.int32)
        class_inds = tf.range(self.num_classes)
        #        class_inds = tf.range(1,self.num_classes+1) # asher workaournd for the shvn case where the classes are indexed 1-5.
        class_inds = tf.cast(class_inds,tf.float32)
        alignment_losses = tf.map_fn(self.get_alignment_losses,class_inds)
        alignment_loss = tf.reduce_mean(alignment_losses)
        return alignment_loss


    def get_alignment_losses(self,class_ind,robust=False):
        if self.lables_one_hot is not None:
            class_data = tf.gather(self.logits,tf.where(tf.equal(tf.argmax(self.y,1),tf.cast(class_ind,tf.int64))))
        else:
            class_data = tf.gather(self.logits,tf.where(tf.equal(self.y,tf.cast(class_ind,tf.int32))))
        averages = tf.reduce_mean(class_data,0)
        diff = tf.subtract(class_data,averages)
        variance_loss = tf.reduce_mean(tf.square(diff),0)
        if robust == True:  # if we should calculate the robust loss, for example with the Geman-Mecclure function
            variance_loss = variance_loss / (variance_loss + self.sigma ** 2)
        loss = tf.reduce_mean(variance_loss)
        alignment_loss = tf.cond(tf.is_nan(loss),lambda:0.,
                                 lambda:loss)  # if the class didn't appear in the batch, it will give a nan value for the alignment of that class, so need to change it to zero
        return tf.cast(alignment_loss,dtype=tf.float32)


    def alignment_loss(self):

        # ------------------------ Our loss (with W) ------------------------------------------------------
        # #multiply W (binary mask) with the 3-dim image (tested):
        # img_slice = tf.slice(self.logits,[0,0,0,0],[-1,-1,-1,self.num_channels - 1])  # (64, 128, 128, 3)
        # #img_slice = tf.Print(img_slice,[img_slice],message="img_slice is a: ",summarize=100)
        # w_slice = tf.slice(self.logits,[0,0,0,self.num_channels - 1],[-1,-1,-1,-1])  # (64, 128, 128, 1)
        # #w_slice = tf.Print(w_slice,[w_slice],message="w_slice is a: ",summarize=100)
        # logits_new = tf.multiply(img_slice,w_slice)  # (64, 128, 128, 3)
        # #logits_new = tf.Print(logits_new,[logits_new],message="logits_new is a: ",summarize=100)
        # sum_weighted_imgs = tf.reduce_sum(logits_new,0)  # (128, 128, 3)
        # #sum_weighted_imgs = tf.Print(sum_weighted_imgs,[sum_weighted_imgs],message="sum_weighted_imgs is a: ",summarize=100)
        # sum_weights = tf.reduce_sum(w_slice,0)  # (128, 128, 1)
        # #sum_weights = tf.Print(sum_weights,[sum_weights],message="sum_weights is a: ",summarize=100)
        # sum_weights = tf.concat([sum_weights,sum_weights,sum_weights],2)  # (128, 128, 3)
        # #sum_weights = tf.Print(sum_weights,[sum_weights],message="sum_weights is a: ",summarize=100)
        #
        # # If "sum_weights" = 0 for pixel i, then "averages_new" should be 0
        # averages_new = tf.where(tf.less(sum_weights,1e-3),tf.zeros_like(sum_weighted_imgs),tf.divide(sum_weighted_imgs,sum_weights+1e-7)) # (128, 128, 3)
        # #averages_new = tf.Print(averages_new,[averages_new],message="averages_new is a: ",summarize=100)
        #
        # weighted_diff = tf.multiply(w_slice,tf.subtract(img_slice,averages_new))  # (64, 128, 128, 3)
        # #weighted_diff = tf.Print(weighted_diff,[weighted_diff],message="weighted_diff is a: ",summarize=100)
        # sum_weighted_diff = tf.reduce_sum(tf.square(weighted_diff),0)  # (128, 128, 3)
        #
        # # PRINTINGS
        # #tmp = tf.slice(sum_weighted_diff,[90,90,0],[-1,-1,-1])
        # # tmp = tf.Print(tmp,[tmp],message="sum_weighted_diff is a: ",summarize=100)
        # a = tf.reduce_max(w_slice)
        # #a = tf.Print(a,[a],message="max(img_slice): ")
        # b = tf.reduce_min(w_slice)
        # #b = tf.Print(b,[b],message="min(img_slice): ")
        #
        # # If "sum_weights" = 0 for pixel i, then "square_mean_new" should be 0
        # square_mean_new = tf.where(tf.less(sum_weights,1e-3),tf.zeros_like(sum_weighted_diff),tf.divide(sum_weighted_diff,sum_weights+1e-7))  # (128, 128, 3)
        # #square_mean_new = tf.Print(square_mean_new,[square_mean_new],message="square_mean_new is a: ",summarize=100)
        # #alignment_loss = tf.reduce_sum(square_mean_new) # tf.reduce_sum(square_mean_new) # return
        # alignment_loss = tf.reduce_sum(sum_weighted_diff)  # tf.reduce_sum(square_mean_new)
        # #alignment_loss = tf.Print(alignment_loss,[alignment_loss],message="alignment_loss is a: ",summarize=100)

        # ------------------------ Asher's loss (without W) ------------------------------------------------------
        averages = tf.reduce_mean(self.logits,0)
        diff = tf.subtract(self.logits,averages)
        sqaure_mean = tf.reduce_sum(tf.square(diff),0)  # reduce_mean
        robust_loss = sqaure_mean / (sqaure_mean + self.sigma ** 2)
        alignment_loss = tf.reduce_sum(robust_loss)  # reduce_mean
        a = tf.reduce_max(averages)
        b = tf.reduce_max(averages)

        return alignment_loss, a, b
