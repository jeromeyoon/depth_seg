import numpy as np
import os
import tensorflow as tf
import random,time,json,pdb,scipy.misc,glob,random
from model_queue import FCN
from test import EVAL
from utils import pp, save_images, to_json, make_gif, merge, imread, get_image,center_crop
from numpy import inf
from sorting import natsorted
import matplotlib as mpl
import scipy.io as sio
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("g_learning_rate", 0.00001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size",20, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 224, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "depth_seg", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "output", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("input_size", 64, "The size of image input size")
flags.DEFINE_float("gpu",0.5,"GPU fraction per process")
FLAGS = flags.FLAGS

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    
    if not os.path.exists(os.path.join('./logs',time.strftime('%d%m'))):
    	os.makedirs(os.path.join('./logs',time.strftime('%d%m')))

    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu)
    #with tf.Session() as sess:
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)) as sess:
        if FLAGS.is_train:
            fcn = FCN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,\
	    dataset_name=FLAGS.dataset,is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
	    fcn = EVAL(sess, batch_size=1,rgb_image_shape=[224,224,3],dataset_name=FLAGS.dataset,\
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)
        if FLAGS.is_train:
            fcn.train(FLAGS)
        else:
            train_rgb_data = open("db/oxford_data2_rgb_train.txt")
            train_rgblist = train_rgb_data.readlines()
            train_depth_data = open("db/oxford_data2_depth_train.txt")
            train_depthlist = train_depth_data.readlines()
	    shuf = range(0,len(train_rgblist))
            random.shuffle(shuf)
	    shuf = shuf[:10]		
	    save_files = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'FCN.model*'))
	    save_files  = natsorted(save_files)
	    savepath ='./Depth_seg'
	    if not os.path.exists(os.path.join(savepath)):
	        os.makedirs(os.path.join(savepath))

	    model = save_files[-2]
	    model = model.split('/')
	    model = model[-1]
	    fcn.load(FLAGS.checkpoint_dir,model)

            for m in range(len(shuf)):
            	rgbpath = train_rgblist[shuf[m]] 
            	rgb_img = scipy.misc.imread(rgbpath[:-1]).astype(np.float32)
	        rgb_img = center_crop(rgb_img,224) 
		rgb_img = np.reshape(rgb_img,(1,224,224,3))	
            	depthpath = train_depthlist[shuf[m]] 
	    	depth_img = sio.loadmat(depthpath[:-1])
	        depth_img = depth_img['depth']
	    	depth_img = np.reshape(depth_img,[depth_img.shape[0],depth_img.shape[1],1])
	    	depth_img = center_crop(depth_img,224) 	
	        start_time = time.time() 
	        predict = sess.run(fcn.pred_seg, feed_dict={fcn.rgb_images: rgb_img,fcn.keep_prob:1.0})
	        predict = np.squeeze(predict).astype(np.float32)
		
   	        print('time: %.8f' %(time.time()-start_time))     
	        if not os.path.exists(os.path.join(savepath,'%s' %(model))):
	             os.makedirs(os.path.join(savepath,'%s' %(model)))
	        savename = os.path.join(savepath,'%s/predict_%03d.jpg' % (model,shuf[m]))
	        scipy.misc.imsave(savename, predict.astype(np.uint8))
	        savename = os.path.join(savepath,'%s/gt_%03d.jpg' % (model,shuf[m]))
	        scipy.misc.imsave(savename, np.squeeze(depth_img).astype(np.uint8))



def color_image(image,num_class=16):
     norm = mpl.colors.Normalize(vmin=0,vmax=num_class)
     pdb.set_trace()
     mycm = plt.get_cmap('Sequential')
     return mycm(norm(image))


if __name__ == '__main__':
    tf.app.run()
