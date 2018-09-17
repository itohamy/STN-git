########################################################################
###
###  This file creates a mnist-like dataset for a specific digit,
###  after rotating each of the specific digit's images, to
###  be alined to a fixed position.
###  This is achieved by using a pixel-location variance loss function.
###  Here we are restricting the transformations to only by rotations.
###  We also take the matrix exponent to form affine deffeomorphism
###  transformations.
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
import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_helper
from ATN import alignment_transformer_network
from atn_helpers.tranformations_helper import register_gradient
#from skimage.transform import warp, AffineTransform
from data_provider2 import DataProvider
from Plots import open_figure,PlotImages


# %% Load data
def main():
    # Here you can play with some parameters.
    digit_to_align = 7  #the digit which should we align
    n_epochs = 1
    iter_per_epoch = 500
    batch_size = 64

    num_channels = 4

    # possible trasromations = "r","sc","sh","t","ap","us","fa"
    # see explanations in transformations_helper.py
    requested_transforms = ["t"] #["r","t","sc","sh"]
    regularizations = {"r":0,"t":0.,"sc":150,"sh":0}
    #    requested_transforms = ["ane"]
    #    regularizations = {"ane":2500}
    #    requested_transforms = ["fa"]
    #    regularizations = {"fa":3.5}

    # param test_the_alignment_process
    # If true will transform a few images in the training data many times
    # and we'll test if the alignment works.
    test_the_alignment_process = False

    #param use_small_mnist
    #if is true, we will use a mnist with a much smaller amount of images for each digit.
    #if such a dataset doesnt exist, create it.
    #this is so that later we can run mnist classification on it, with the alignment cost, and see
    #if this allowes fater converging (and a more convex function)
    use_small_mnist = False
    minimal_imgs_per_digit = 10

    # param my_learning_rate
    # Gets good results with 1e-4. You can also set the weigts in the transformations_helper file
    # (good results also with 1e-4 initialization)
    my_learning_rate = 7e-5
    weight_stddev = 5e-4

    activation_func = "relu"

    prepare_figure()

    #measure the time
    start_time = time.time()

    # mnist_path = path_to_reg_mnist #will tell us which dataset to take the digit from later
    # if use_small_mnist:
    #     mnist_path = path_to_small_mnist
    #     mnist_helper.create_smaller_mnist(minimal_imgs_per_digit, path_to_small_mnist+"/", path_to_reg_mnist, height,width)
    #
    # # Load data and take only the desired digit images
    # params = (digit_to_align, batch_size, height,width, mnist_path, path_to_new_mnist, test_the_alignment_process)
    # X_train, y_train, X_test, y_test = mnist_helper.get_digit_data(*params)

    # !!! MINE !!!
    img_sz = 128
    video_name = "movies/BG.mp4"
    data = DataProvider(video_name,img_sz,num_channels)
    X_train = data.next_batch(batch_size,'train')
    X_test = data.next_batch(batch_size,'train')

    device = '/cpu:0'
    with tf.device(device):  #greate the graph
        loss,logits,transformations,b_s,x,keep_prob,optimizer , a, b= computational_graph(my_learning_rate,
                                                                                    requested_transforms,batch_size,
                                                                                    regularizations,activation_func,
                                                                                    weight_stddev,num_channels)

    #writer, summary_op = create_summaries(loss, x, logits, b_s)

    # We now create a new session to actually perform the initialization the variables:
    params = (data,iter_per_epoch,X_train,X_test,n_epochs,batch_size,loss,logits,transformations,b_s,x,keep_prob,optimizer,start_time,digit_to_align,img_sz,num_channels, a, b)
    run_session(*params)

    #measure the time
    # Set the precision.
    duration = time.time() - start_time
    print("Total runtime is " + "%02d" % (duration) + " seconds.")


# %%

def computational_graph(my_learning_rate,requested_transforms,batch_size,regularizations,activation_func,weight_stddev,
                        num_channels):
    x = tf.placeholder(tf.float32,[None,height * width * num_channels])  # input data placeholder for the atn layer
    # y = tf.placeholder(tf.float32, [None, 1])
    #batch size
    b_s = tf.placeholder(tf.float32,[1,])

    # Since x is currently [batch, height*width], we need to reshape to a
    # 4-D tensor to use it in a convolutional graph.  If one component of
    # `shape` is the special value -1, the size of that dimension is
    # computed so that the total size remains constant.  Since we haven't
    # defined the batch dimension's shape yet, we use -1 to denote this
    # dimension should not change size.
    keep_prob = tf.placeholder(tf.float32)
    atn = alignment_transformer_network(x,requested_transforms,regularizations,batch_size,width,num_channels,1,
                                        weight_stddev,activation_func,keep_prob)
    logits,transformations,alignment_loss,transformations_regularizer , a, b = atn.atn_layer()

    loss = compute_final_loss(alignment_loss,transformations_regularizer,num_channels)
    #    loss = mnist_loss + transformations_regularizer

    opt = tf.train.AdamOptimizer(learning_rate=my_learning_rate)
    optimizer = opt.minimize(loss)
    #grads = opt.compute_gradients(loss, [b_fc_loc2])

    return loss,logits,transformations,b_s,x,keep_prob,optimizer , a ,b


def compute_final_loss(alignment_loss,transformations_regularizer,num_channels):
    alignment_loss /= (width * width * num_channels)  # we need this, other wise the alignment loss is to big and we'll completely zoom out and and ruin the input image
    return alignment_loss + transformations_regularizer


# %%
def run_session(data,iter_per_epoch,X_train,X_test,n_epochs,batch_size,loss,logits,transformations,b_s,x,keep_prob,
                optimizer,start_time,digit_to_align,img_sz,num_channels, a, b):
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    #find the indexes needed for splitting the train and test sets into batchs with the desired batch size
    #iter_per_epoch,indices,test_iter_per_epoch,test_indices = prepare_splitting_data(X_train,X_test,batch_size)

    for epoch_i in range(n_epochs):
        for iter_i in range(iter_per_epoch):
            batch_xs = data.next_batch(batch_size,'train')  # X_train[indices[iter_i]:indices[iter_i+1]]
            loss_val,theta_val, a_val, b_val = sess.run([loss,transformations, a, b],
                                          feed_dict={
                                              b_s:[batch_size],
                                              x:batch_xs,
                                              keep_prob:1.0
                                          })
            if iter_i % 20 == 0:
                print('Iteration: ' + str(iter_i) + ' Loss: ' + str(loss_val))
                print("theta row 1 is: " + str(theta_val[0,:]))

            sess.run(optimizer,feed_dict={
                b_s:[batch_size],x:batch_xs,keep_prob:1.0})

        #Find accuracy on test data
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\nrunning test data...")
        accuracy = 0.
        for iter_i in range(batch_size):
            batch_xs = data.next_batch(batch_size,
                                       'train')  #X_test[test_indices[iter_i]:test_indices[iter_i+1]]
            loss_val,theta_val,test_imgs = sess.run([loss,transformations,logits],
                                                    feed_dict={
                                                        b_s:[batch_size],
                                                        x:batch_xs,
                                                        keep_prob:1.0
                                                    })
            accuracy += loss_val
        accuracy /= batch_size

        # plot the reconstructed images and their ground truths (inputs)
        imgs = []
        imgs_test = []
        titles = []
        for i in range(10):
            I = batch_xs[i,...]
            I = np.reshape(I,(img_sz,img_sz,num_channels))
            imgs.append(I[:,:,0:3])
            I = test_imgs[i,...]
            I = np.reshape(I,(img_sz,img_sz,num_channels))
            imgs_test.append(np.abs(I[:,:,0:3]))
            titles.append('')
        fig1 = open_figure(1,'Original Images',(7,3))
        PlotImages(1,2,5,1,imgs,titles,'gray',axis=True,colorbar=False)
        fig2 = open_figure(2,'Test Results',(7,3))
        PlotImages(2,2,5,1,imgs_test,titles,'gray',axis=True,colorbar=False)
        plt.show()
        fig1.savefig('f1.png')
        fig2.savefig('f2.png')

        #print ("theta row 1 is: "+str(theta_val[0,:]))
        #print ("theta row 10 is: "+str(theta_val[9,:]))
        print('Accuracy (%d): ' % (epoch_i + 1) + str(accuracy) + "\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        if np.isnan(accuracy):
            duration = time.time() - start_time
            print("Total runtime is " + "%02d" % (duration) + " seconds.")
            raise SystemExit

    # if update_transformed_mnist:
    #     #Run one forward pass again on the training data, in ordr to create a transformed mnist data
    #     all_training_imgs = prepare_new_mnist(sess,X_train,y_train,iter_per_epoch,indices,batch_ys,batch_xs,loss,logits,transformations,b_s,x,y,keep_prob)
    #
    # #show some of the test data before and after running the model which was learned
    # all_test_imgs = None#Find accuracy on test data
    # print("\n\nPreparing test images...")
    # for iter_i in range(test_iter_per_epoch):
    #     batch_xs = X_test[test_indices[iter_i]:test_indices[iter_i+1]]
    #     batch_ys = y_test[test_indices[iter_i]:test_indices[iter_i+1]]
    #     batch_size = batch_ys.size
    #
    #     loss_val,test_imgs = sess.run([loss,logits],
    #                             feed_dict={
    #                                          b_s: [batch_size],
    #                                          x: batch_xs,
    #                                          y: batch_ys,
    #                                          keep_prob: 1.0
    #                                        })
    #     if all_test_imgs is not None:
    #         all_test_imgs = np.concatenate((all_test_imgs,test_imgs))
    #     else:
    #         all_test_imgs = test_imgs
    # show_test_imgs(all_test_imgs,X_test,height,width)
    #
    # if update_transformed_mnist: #create a new mnist data set with the newly aligned data
    #     create_mnist_dataset(y_train,y_test,path_to_new_mnist,digit_to_align,all_training_imgs,all_test_imgs)

    sess.close()


# %%

#def layer_grid_summary(name, var, image_dims, BATCH_SIZE):
#    prod = np.prod(image_dims)
#    grid = form_image_grid(tf.reshape(var, [BATCH_SIZE, prod], [BATCH_SIZE*28, BATCH_SIZE*28], image_dims, 1))
#    return tf.summary.image(name, grid)

def create_summaries(loss,x,output,BATCH_SIZE):
    writer = tf.summary.FileWriter("../logs")
    tf.summary.scalar("Loss",loss)
    #    layer_grid_summary("Input", x, [28, 28], BATCH_SIZE)
    #    layer_grid_summary("Output", output, [28, 28], BATCH_SIZE)
    return writer,tf.summary.merge_all()


def prepare_splitting_data(X_train,X_test,batch_size):
    train_size = X_train.shape[0]
    iter_per_epoch = int(train_size / batch_size)
    indices = np.linspace(0,train_size,iter_per_epoch + 1)
    indices = indices.astype('int')

    test_size = X_test.shape[0]
    test_iter_per_epoch = int(test_size / batch_size)
    test_indices = np.linspace(0,test_size,test_iter_per_epoch + 1)
    test_indices = test_indices.astype('int')

    return iter_per_epoch,indices,test_iter_per_epoch,test_indices


def prepare_figure():
    plt.figure(17,figsize=(figure_size,figure_size))
    plt.clf()
    plt.subplots_adjust(left=0.125,bottom=0.1,right=0.3,top=0.9,wspace=0.2,hspace=0.4)


#show the first 10 figures of the test data after running the model which was learned
def show_test_imgs(all_test_imgs,X_test,height,width):
    #show the test images before and after the alingment
    images,messages = [[],[]]
    for ind in range(rows):
        original_test_img = np.reshape(X_test[ind,:],(height,width))
        alinged_test_img = np.reshape(all_test_imgs[ind,:],(height,width))
        images.append([original_test_img,alinged_test_img])
        messages.append(["Test image {} before".format(ind),"Test image {} after".format(ind)])
    for row in range(1,rows + 1):
        ind = row * 2
        for i in range(2):
            PlotImage(rows,cols,ind - 1 + i,messages[row - 1][i],images[row - 1][i],'jet')


def PlotImage(rows,cols,ind,title,image,color):
    ax = plt.subplot(rows,cols,ind)
    ax.set_title(title)
    plt.imshow(image,cmap=color,interpolation='None')


#Run one forward pass again on the training data, in ordr to create a transformed mnist data
def prepare_new_mnist(sess,X_train,y_train,iter_per_epoch,indices,batch_ys,batch_xs,loss,logits,transformations,b_s,x,y,
                      keep_prob):
    all_training_imgs = None
    for iter_i in range(iter_per_epoch):
        batch_xs = X_train[indices[iter_i]:indices[iter_i + 1]]
        batch_ys = y_train[indices[iter_i]:indices[iter_i + 1]]
        batch_size = batch_ys.size

        loss_val,training_imgs,theta_val = sess.run([loss,logits,transformations],
                                                    feed_dict={
                                                        b_s:[batch_size],
                                                        x:batch_xs,
                                                        y:batch_ys,
                                                        keep_prob:1.0
                                                    })

        if all_training_imgs is not None:
            all_training_imgs = np.concatenate((all_training_imgs,training_imgs))
        else:
            all_training_imgs = training_imgs
    return all_training_imgs


def create_mnist_dataset(y_train,y_test,path_to_new_mnist,digit_to_align,all_training_imgs,all_test_imgs):
    updated_training_imgs = []
    for im in all_training_imgs:
        #updated_training_imgs.append(np.reshape(im, (height,width)))
        updated_training_imgs.append(im)
    updated_test_imgs = []
    for im in all_test_imgs:
        #updated_test_imgs.append(np.reshape(im, (height,width)))
        updated_test_imgs.append(im)
    aligned_data = [updated_training_imgs,y_train,updated_test_imgs,y_test]

    for ind in range(len(aligned_data)):
        if ind % 2 == 1:
            label_list = [digit_to_align] * (len(aligned_data[ind]))
            aligned_data[ind] = label_list
        aligned_data[ind] = np.array(aligned_data[ind])
        print("***********")
        print(aligned_data[ind].shape[0])

    mnist_helper.create_new_mnist(aligned_data,height,width,path=path_to_new_mnist + str(digit_to_align) + "/")


# %%

if __name__ == '__main__':
    #register the gradient for matrix exponential
    register_gradient()

    #some global params which will probably not be changed
    path_to_reg_mnist = "../../data/mnist/regular_data"
    path_to_small_mnist = "../../data/mnist/minimal_data"
    path_to_new_mnist = "../../data/mnist/affine_deffeomorphism/"
    update_transformed_mnist = False
    rows = 10
    cols = 2
    figure_size = 5 * rows
    height,width = [128,128]

    main()