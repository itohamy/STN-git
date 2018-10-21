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

    def __init__(self,input_data,requested_transforms,regularizations,batch_size,image_size,num_channels,num_classes=10,
                 weight_stddev=5e-3,activation_func="tanh",keep_prob=1.,y=None):
        self.X = tf.reshape(input_data,shape=[-1,image_size * image_size * num_channels])  #reshaping to 2 dimensions
        self.y = y
        self.requested_transforms = requested_transforms
        self.regularizations = regularizations
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.keep_prob = keep_prob
        self.weight_stddev = weight_stddev
        self.activation_func = activation_func

    def atn_layer(self):
        # Since x is currently [batch, height*width], we need to reshape to a
        # 4-D tensor to use it in a convolutional graph.  If one component of
        # `shape` is the special value -1, the size of that dimension is
        # computed so that the total size remains constant.  Since we haven't
        # defined the batch dimension's shape yet, we use -1 to denote this
        # dimension should not change size.
        x_tensor = tf.reshape(self.X,[-1,self.image_size,self.image_size,self.num_channels])
        h_fc_loc2,affine_maps = transfromation_parameters_regressor(self.requested_transforms,self.X,self.keep_prob,
                                                                    self.image_size,self.batch_size,self.num_channels,
                                                                    self.activation_func,self.weight_stddev)

        #We'll create a spatial transformer module to identify discriminative
        # patches
        out_size = (self.image_size,self.image_size)
        logits = transformer(x_tensor,h_fc_loc2,out_size)
        #to avoid the sparse indexing warning, comment the next line, and uncomment the one after it.
        logits = tf.reshape(logits,shape=[-1,self.image_size,self.image_size,self.num_channels])
        #logits = tf.Print(logits,[logits],message="logits is a: ",summarize=100)

        #find the variance (sqaured difference) for each pixel stack in the batch
        #then we will sum the variances (might be more correct to take their average)
        # logits = tf.reshape(logits, shape=[-1,self.image_size**2])

        transformations = h_fc_loc2
        #transformations = tf.reshape(transformations, [-1,6])
        #slice only the first row to show corrent status
        transformations = (tf.slice(transformations,[0,0],[1,6]))

        alignment_losses,transformations_regularizers , a, b = self.alignment_loss_and_reg(logits,affine_maps)
        return logits,transformations,alignment_losses,transformations_regularizers, a, b

    def alignment_loss_and_reg(self,logits,affine_maps):
        # # multiply W (binary mask) with the 3-dim image (tested):
        # img_slice = tf.slice(logits,[0,0,0,0],[-1,-1,-1,self.num_channels - 1])  # (64, 128, 128, 3)
        # #img_slice = tf.Print(img_slice,[img_slice],message="img_slice is a: ",summarize=100)
        # w_slice = tf.slice(logits,[0,0,0,self.num_channels - 1],[-1,-1,-1,-1])  # (64, 128, 128, 1)
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
        # sum_weighted_diff = tf.reduce_mean(tf.square(weighted_diff),0)  # (128, 128, 3)
        #
        # # PRINTINGS
        # tmp = tf.slice(sum_weighted_diff,[90,90,0],[-1,-1,-1])
        # # tmp = tf.Print(tmp,[tmp],message="sum_weighted_diff is a: ",summarize=100)
        # a = tf.reduce_max(tmp)
        # # a = tf.Print(a,[a],message="max(sum_weighted_diff) is a: ")
        # b = tf.reduce_min(tmp)
        # # b = tf.Print(b,[b],message="min(sum_weighted_diff) is a: ")
        #
        # # If "sum_weights" = 0 for pixel i, then "square_mean_new" should be 0
        # square_mean_new = tf.where(tf.less(sum_weights,1e-3),tf.zeros_like(sum_weighted_diff),tf.divide(sum_weighted_diff,sum_weights+1e-7))  # (128, 128, 3)
        # #square_mean_new = tf.Print(square_mean_new,[square_mean_new],message="square_mean_new is a: ",summarize=100)
        # alignment_loss = tf.reduce_sum(square_mean_new) # tf.reduce_sum(square_mean_new)
        # #alignment_loss = tf.Print(alignment_loss,[alignment_loss],message="alignment_loss is a: ",summarize=100)
        #
        # transformations_regularizer = transformation_regularization(affine_maps,
        #                                                            self.regularizations)  #give a diffrent penalty to each type of transformation magnituted

        # # Asher's code:
        averages = tf.reduce_mean(logits,  0)
        #averages = tf.Print(averages,[averages],message="averages is a: ",summarize=100)
        diff = tf.subtract(logits,averages)
        #diff = tf.Print(diff,[diff],message="diff is a: ",summarize=100)
        sqaure_mean = tf.reduce_mean(tf.square(diff),0)
        #sqaure_mean = tf.Print(sqaure_mean,[sqaure_mean],message="sqaure_mean is a: ",summarize=100)
        alignment_loss = tf.reduce_sum(sqaure_mean)
        #alignment_loss = tf.Print(alignment_loss,[alignment_loss],message="alignment_loss is a: ",summarize=100)
        transformations_regularizer = transformation_regularization(affine_maps,self.regularizations)#give a diffrent penalty to each type of transformation magnituted
        a = tf.reduce_max(averages)
        b = tf.reduce_max(averages)
        return alignment_loss,transformations_regularizer , a , b


