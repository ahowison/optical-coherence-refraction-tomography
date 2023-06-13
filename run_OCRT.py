from __future__ import print_function, division
from OCRT import OCRT2D
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
from time import time

sample_id = 'mouse_vas_deferens2'  # a string that denotes which data to run
data_directory = 'data/'  # where is the OCRT input data located?
save_directory = 'saved_models_and_variables/'  # where to save the tf graph after optimization

def main():
    a = OCRT2D(sample_id=sample_id,save_directory=save_directory)
    
    # adjust some parameters, depending on the sample:
    if sample_id in ['mouse_bladder', 'mouse_femoral_artery', 'mouse_trachea', 'mouse_vas_deferens1']:
        num_iter = 200
    elif sample_id in ['mouse_vas_deferens2']:
        num_iter = 300
    elif sample_id in ['insect_leg']:
        num_iter = 500
    elif sample_id in ['human_cornea']:
        num_iter = 500
        # this sample is more difficult to register, so use a multiresolution approach:
        a.use_multires = True
        a.size_factor_ = 1  # the final size_factor_ is 8
        a.switch_iter = 250
    else:
        raise Exception('invalid sample_id: ' + sample_id)
    
    # two tube sizes were used (same inner diameter, but different outer diameter):
    if sample_id in ['mouse_vas_deferens1', 'mouse_vas_deferens2', 'mouse_trachea']:
        a.tube_diameter = 1.066  # in mm
    else:
        a.tube_diameter = 1.108516
    
    print("checkpoint 1")

    a.load_data_and_resolve_constants(data_directory=data_directory)
    a.build_graph()

    print("checkpoint 2")

    # run through optimization loop:
    losses = list()
    feed_dict = a.get_feed_dict()
    for i in range(num_iter + 1):
        
        # if using multires, change the pixel resolution of the reconstruction
        if i == a.switch_iter and a.use_multires: 
            feed_dict[a.size_factor] = a.final_size_factor
        
        start = time()
        loss_i, _ = a.sess.run([a.loss_terms, a.train_op], feed_dict=feed_dict)
        losses.append(loss_i)
        print(i, loss_i, time()-start)
        # loss_i is a list of all the contributors to the scalar loss;
        # (see a.loss_terms or a.loss_term_names for names of the regularization terms)
        
        # monitor results periodically:
        if i % 10 == 0:
            recon_i = a.sess.run(a.recon, feed_dict=feed_dict)
            recon_i = recon_i.sum(2)  # only once slice along y contains nonzero values, because we are optimizing 2D
            plt.figure(figsize=(10, 10))
            plt.imshow(recon_i, cmap='gray_r')
            plt.title('OCRT reconstruction')
            plt.show()
            
            RI = a.sess.run(a.RI, feed_dict={a.xz_delta: np.zeros((60, 2))})  # remove xy_delta shifts
            
            plt.imshow(RI)
            plt.title('refractive index map')
            plt.colorbar()
            plt.show()
            
            plt.plot(losses)
            plt.legend(a.loss_term_names.eval())
            plt.title('loss terms')
            plt.show()

    print("checkpoint 3")

    a.save_graph()  # this graph must be saved in order to run filter optimization below

    print("checkpoint 4")

    #-------------Filter Optimization after Registration----------------#
    # remove previous tf graph and variables:
    a.sess.close()
    tf.reset_default_graph()
    del a

    print("checkpoint 5")

    # instantiate new object, this time for filter optimization
    a = OCRT2D(sample_id=sample_id,save_directory=save_directory)
    num_iter = 100
    a.infer_backprojection_filter = True
    a.use_spatial_shifts = False

    # as above, set the tube diameter depending on the sample:
    if sample_id in ['mouse_vas_deferens1', 'mouse_vas_deferens2', 'mouse_trachea']:
        a.tube_diameter = 1.066
    else:
        a.tube_diameter = 1.108516

    a.load_data_and_resolve_constants(data_directory=data_directory)
    a.build_graph()

    print("checkpoint 6")

    losses = list()
    feed_dict = a.get_feed_dict()
    for i in range(num_iter + 1):
        start = time()
        loss_i, _ = a.sess.run([a.loss_terms, a.train_op], feed_dict=feed_dict)
        losses.append(loss_i)
        print(i, loss_i, time()-start)
        
        # monitor results periodically:
        if i % 20 == 0:
            recon_i = a.sess.run(a.recon, feed_dict=feed_dict)
            recon_i = recon_i.sum(2)
            plt.figure(figsize=(10, 10))
            plt.imshow(recon_i, cmap='gray_r')
            plt.title('OCRT reconstruction')
            plt.show()
            
            plt.plot(losses)
            plt.legend(a.loss_term_names.eval())
            plt.title('loss terms')
            plt.show()

    # a.save_graph()  # this graph doesn't need to be saved


if __name__ == "__main__":
    main()