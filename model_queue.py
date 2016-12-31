import os,time,pdb,argparse,threading
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
from ops import *
from utils import *
from random import shuffle
from network import networks
import scipy.io as sio
import scipy.ndimage

class FCN(object):
    def __init__(self, sess, image_size=108, is_train=True,is_crop=True,\
                 batch_size=32,rgb_shape=[224, 224,3], depth_shape=[224, 224, 1],\
	         df_dim=64,dataset_name='default',checkpoint_dir=None):


        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.depth_shape = depth_shape
        self.rgb_shape = rgb_shape
        self.df_dim = df_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
	self.use_queue = True
	self.dropout =0.85
	self.vgg_path='imagenet-vgg-verydeep-19.mat'
	self.build_model()
	
    def build_model(self):
	
	if not self.use_queue:

        	self.rgb_images = tf.placeholder(tf.float32, [self.batch_size] + self.rgb_shape,
                                    name='rgb_images')
        	self.depth_images = tf.placeholder(tf.int32, [self.batch_size] + self.depth_shape,
                                    name='depth_images')
	else:
		print ' using queue loading'
		self.rgb_single = tf.placeholder(tf.float32,shape=self.rgb_shape)
		self.depth_single = tf.placeholder(tf.int32,shape=self.depth_shape)
		q = tf.FIFOQueue(4000,[tf.float32, tf.int32],[[self.rgb_shape[0],self.rgb_shape[1],3],[self.depth_shape[0],self.depth_shape[1],1]])
		self.enqueue_op = q.enqueue([self.rgb_single, self.depth_single])
		self.rgb_images,self.depth_images = q.dequeue_many(self.batch_size)

	self.keep_prob = tf.placeholder(tf.float32)
	net = networks(self.vgg_path)
	self.pred_seg, self.logits = net.inference(self.rgb_images,self.keep_prob)
	#net  = networks(self.batch_size,self.df_dim,self.dropout)
	#self.pred_seg, self.logits = net.FCN8(self.rgb_images)
	self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits,tf.squeeze(self.depth_images,squeeze_dims=[3]),name='entropy')))
	self.saver = tf.train.Saver(max_to_keep=10)
	#self.h_vars =[var for var in t_vars if 'h' in var.name]
	#self.fc_vars =[var for var in t_vars if 'fc' in var.name]
    def train(self, config):
        #####Train DCGAN####

        global_step1 = tf.Variable(0,name='global_step1',trainable=False)

	vars_list = tf.trainable_variables()
	#g_lr = tf.train.exponential_decay(config.g_learning_rate,global_step1,1000,0.5,staircase=True)
        optim = tf.train.AdamOptimizer(config.g_learning_rate,beta1=config.beta1)#.minimize(self.loss, global_step=global_step1,var_list=self.t_vars)
	grads = optim.compute_gradients(self.loss,var_list=vars_list)
	train_op = optim.apply_gradients(grads)
	    
	tf.initialize_all_variables().run()
	
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loda training and validation dataset path
        train_rgb_data = open("db/oxford_data2_rgb_train.txt")
        tmp1 = train_rgb_data.readlines()
        train_rgb_data = open("db/Oxford_data1_RGB_train.txt")
        tmp2 = train_rgb_data.readlines()
	train_rgblist = tmp1 + tmp2	
        train_depth_data = open("db/oxford_data2_depth_train.txt")
        tmp1 = train_depth_data.readlines()
        train_detph_data = open("db/Oxford_data1_depth_train.txt")
        tmp2 = train_depth_data.readlines()
	train_depthlist = tmp1 + tmp2

        val_rgb_data = open("db/Oxford_data1_RGB_val.txt")
        val_rgblist = val_rgb_data.readlines()
        val_depth_data = open("db/Oxford_data1_depth_val.txt")
        val_depthlist = val_depth_data.readlines()
	
	VGG_mean =[103.939,116.779,123.68]
	batch_idxs = min(len(train_rgblist), config.train_size)/config.batch_size
	val_batch_idxs = min(len(val_rgblist), config.train_size)/config.batch_size
	shuf = range(len(train_rgblist))
	random.shuffle(shuf)
	val_range = range(len(val_rgblist))
	if self.use_queue:
	    # creat thread
	    coord = tf.train.Coordinator()
            num_thread =8
            for i in range(num_thread):
 	        t = threading.Thread(target=self.load_and_enqueue,args=(coord, train_rgblist, train_depthlist, shuf,i,num_thread))
	 	t.start()

	if self.use_queue:
	    for epoch in xrange(config.epoch):
	        #shuffle = np.random.permutation(range(len(data)))
		sum_loss =0.0
		sum_valloss =0.0

		if epoch ==0:
		    train_log = open(os.path.join("logs",'train_%s.log' %config.dataset),'w')
		    val_log = open(os.path.join("logs",'val_%s.log' %config.dataset),'w')
		else:
	    	    train_log = open(os.path.join("logs",'train_%s.log' %config.dataset),'aw')
	    	    val_log = open(os.path.join("logs",'val_%s.log' %config.dataset),'aw')

		#for idx in xrange(0,batch_idxs):
		for idx in xrange(0,1):
        	     start_time = time.time()
		     _,loss,pred,logits =self.sess.run([train_op,self.loss,self.pred_seg,self.logits],feed_dict={self.keep_prob:self.dropout})
		   
		     sum_loss += loss
		     print("Epoch: [%2d] [%4d/%4d] time: %4.4f loss:%.6f" \
		     % (epoch, idx, batch_idxs,time.time() - start_time,loss))

		train_log.write('epoch %06d mean_loss:%.6f\n' %(epoch,sum_loss/batch_idxs))
		train_log.close()
	        self.save(config.checkpoint_dir,global_step1)

		#### Validation ####	
		for idx in xrange(0,val_batch_idxs):
		     batch_files = val_range[idx*config.batch_size:(idx+1)*config.batch_size]
    		     batches = get_image(val_rgblist[batch_file],val_depthlist[batch_file],self.image_size,is_crop=self.is_crop,VGG_mean)		     
		     batches = np.array(batches).astype(np.float32)
		     batch_images = np.reshape(batches[:,:,:,:3],[config.batch_size,self.image_size,self.image_size,3])
		     batch_depths = np.reshape(batches[:,:,:,-1],[config.batch_size,self.image_size,self.image_size,1])

        	     start_time = time.time()
		     _,loss =self.sess.run([self.loss],feed_dict={self.rgb_images:batch_images,self.depth_images:batch_depths,self.keep_prob:1.0})
		     sum_valloss += loss
		     if idx % 50 ==0:
		         print("Epoch: [%2d] Val[%4d/%4d] time: %4.4f loss:%.6f" \
		         % (epoch, idx, batch_idxs,time.time() - start_time,loss))
		val_log.write('epoch %06d mean_loss:%.6f\n' %(epoch,sum_valloss/val_batch_idxs))
		val_log.close()

	else:
	    print('You should use multi thread \n')
	    """
	    for epoch in xrange(config.epoch):
	         # loda training and validation dataset path
	         shuffle_ = range(len(train_rgblist))
	         #shuffle_ = np.random.permutation(range(len(data)))
	         batch_idxs = min(len(train_rgblist), config.train_size)/config.batch_size
		    
	         for idx in xrange(0, batch_idxs):
        	     start_time = time.time()
		     batch_files = shuffle_[idx*config.batch_size:(idx+1)*config.batch_size]
    		     batches = [get_image(train_rgblist[batch_file],train_depthlist[batch_file],self.image_size, is_crop=self.is_crop) for batch_file in batch_files]

		     batches = np.array(batches).astype(np.float32)
		     batch_images = np.reshape(batches[:,:,:,:-1],[config.batch_size,self.image_size,self.image_size,3])
		     batchlabel_images = np.reshape(batches[:,:,:,-1],[config.batch_size,self.image_size,self.image_size,1])
		     
		     # Update Normal D network
		     _,loss= self.sess.run([train_op,self.loss], feed_dict={self.rgb_images: batch_images,self.depth_images:batchlabel_images })
		     if np.isnan(loss):
			 for i in range(2):
			     pdb.set_trace()
			     tmp = np.squeeze(pred[i,:,:,:])
			     scipy.misc.imsave('tmp%d.jpeg' %i,tmp)

		     print("Epoch: [%2d] [%4d/%4d] time: %4.4f loss:%.6f" \
		     % (epoch, idx, batch_idxs,time.time() - start_time,loss))
	         self.save(config.checkpoint_dir,epoch)
	    """

    def save(self, checkpoint_dir, step):
        model_name = "FCN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name,self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

	    
    def load_and_enqueue(self,coord,file_list, depth_list, shuf,idx=0,num_thread=1):
	count =0;
	length = len(file_list)
	while not coord.should_stop():
	    i = (count*num_thread + idx) % length;
	    #r = random.randint(0,3)
            rgbpath = file_list[shuf[i]] 
            rgb_img = scipy.misc.imread(rgbpath[:-1])
	    rgb_img = center_crop(rgb_img,self.image_size) 	
	    depth_path = depth_list[shuf[i]]
            #depth_img = scipy.misc.imread(depth_path[:-1])
	    depth_img = sio.loadmat(depth_path[:-1])
	    depth_img = depth_img['depth']
	    depth_img = np.reshape(depth_img,[depth_img.shape[0],depth_img.shape[1],1])
	    depth_img = center_crop(depth_img,self.image_size)
            self.sess.run(self.enqueue_op,feed_dict={self.rgb_single:rgb_img, self.depth_single:depth_img})
	    #print('thread count %d \n' %count)
	    count +=1
		
