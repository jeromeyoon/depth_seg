import pdb
from ops import *
import tensorflow as tf

class networks(object):
    """
    def __init__(self,batch_size,df_dim,keep_prob):
	self.batch_size = batch_size
	self.df_dim = df_dim
 	self.num_class =7 
	self.keep_prob = keep_prob
    def FCN8(self,image):
	
        h1_1 =tf.nn.relu(conv2d(image,self.df_dim,k_h=3,k_w=3,padding='SAME',name='h1_1'))
        h1_2 =tf.nn.relu(conv2d(h1_1,self.df_dim,k_h=3,k_w=3,padding='SAME',name='h1_2'))
	h1_3 = maxpool(h1_2) #224 ->112

        h2_1 =tf.nn.relu(conv2d(h1_3,self.df_dim*2,k_h=3,k_w=3,padding='SAME',name='h2_1'))
        h2_2 =tf.nn.relu(conv2d(h2_1,self.df_dim*2,k_h=3,k_w=3,padding='SAME',name='h2_2'))
	h2_3 = maxpool(h2_2) #112->56

        h3_1 =tf.nn.relu(conv2d(h2_3,self.df_dim*4,k_h=3,k_w=3,padding='SAME',name='h3_1'))
        h3_2 =tf.nn.relu(conv2d(h3_1,self.df_dim*4,k_h=3,k_w=3,padding='SAME',name='h3_2'))
        h3_3 =tf.nn.relu(conv2d(h3_2,self.df_dim*4,k_h=3,k_w=3,padding='SAME',name='h3_3'))
	h3_4 = maxpool(h3_3) #56->28

	h4_1 =tf.nn.relu(conv2d(h3_4,self.df_dim*8,k_h=3,k_w=3,padding='SAME',name='h4_1'))
        h4_2 =tf.nn.relu(conv2d(h4_1,self.df_dim*8,k_h=3,k_w=3,padding='SAME',name='h4_2'))
        h4_3 =tf.nn.relu(conv2d(h4_2,self.df_dim*8,k_h=3,k_w=3,padding='SAME',name='h4_3'))
	h4_4 = maxpool(h4_3) #28 ->14

	h5_1 =tf.nn.relu(conv2d(h4_4,self.df_dim*8,k_h=3,k_w=3,padding='SAME',name='h5_1'))
        h5_2 =tf.nn.relu(conv2d(h5_1,self.df_dim*8,k_h=3,k_w=3,padding='SAME',name='h5_2'))
        h5_3 =tf.nn.relu(conv2d(h5_2,self.df_dim*8,k_h=3,k_w=3,padding='SAME',name='h5_3'))
	h5_4 = maxpool(h5_3) #14->7
	
	fc6_1  = conv2d(h5_4,4096,k_h=7,k_w=7,padding='SAME',name='fc6')
	fc6_2 = tf.nn.dropout(fc6_1,self.keep_prob)
	
	fc7_1  = conv2d(fc6_2,4096,k_h=1,k_w=1,padding='SAME',name='fc7')
	fc7_2 = tf.nn.dropout(fc7_1,self.keep_prob)

	fc8_1 = conv2d(fc7_2,self.num_class,k_h=1,k_w=1,name='fc8') # classs 10 segmentation

	fc9_1 = deconv2d(fc8_1,[self.batch_size,14,14,self.df_dim*8], name = 'fc9_1')
	#fc9_2 = conv2d(h4_4,self.num_class,k_h=1,k_w=1,name ='fc9_2')
	fc9_2 = tf.add(fc9_1,h4_4)
	fc10_1 = deconv2d(fc9_2,[self.batch_size,28,28,self.df_dim*4], name = 'fc10_1')
	#fc10_2 = conv2d(h3_4,self.num_class,k_h=1,k_w=1,name ='fc10_2')
	fc10_2 = tf.add(fc10_1,h3_4)
	#fc11_1 = deconv2d(fc10_2,[self.batch_size,56,56,self.num_class], name = 'fc11_1')
	fc11_1 = deconv2d(fc10_2,[self.batch_size,image.get_shape().as_list()[1],image.get_shape().as_list()[1],self.num_class],k_h=16,k_w=16,d_h=8,d_w=8,padding='SAME', name = 'fc11_1')
	fc11_2 = tf.argmax(fc11_1,dimension=3)
	return fc11_2,fc11_1
    """
    def __init__(self,weights):
        self.data  = scipy.io.loadmat(weights)
	#self.weights = np.squeeze(data['layers'])

    def vgg_net(self,weights,image,reuse=False):
        layers = (
         'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

         'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
         'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

         'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
         'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

         'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
         'relu5_3', 'conv5_4', 'relu5_4'
        )
        net = {}
        current = image
 	with tf.variable_scope(tf.get_variable_scope(),reuse=reuse):	
            for i, name in enumerate(layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = weights[i][0][0][0][0]
                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = self.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                    bias =  self.get_variable(bias.reshape(-1), name=name + "_b")
                    current = self.conv2d_basic(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current, name=name)
                elif kind == 'pool':
                    current = self.avg_pool_2x2(current)
                net[name] = current
	    #current = self.avg_pool_2x2(net['conv5_3'])
	    #net['pool5'] = current	
 
            return net

    def conv2d_basic(self,x, W, bias):
        conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
        return tf.nn.bias_add(conv, bias)

    def avg_pool_2x2(self,x):
        return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    def get_variable(self,weights,name):
        init = tf.constant_initializer(weights, dtype=tf.float32)
        var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
        return var
   
    def weight_variable(self,shape, stddev=0.02, name=None):
        # print(shape)
        initial = tf.truncated_normal(shape, stddev=stddev)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, initializer=initial)

    def bias_variable(self,shape, name=None):
        initial = tf.constant(0.0, shape=shape)
        if name is None:
            return tf.Variable(initial)
        else:
            return tf.get_variable(name, initializer=initial)

    def conv2d_transpose_strided(self,x, W, b, output_shape=None, stride = 2):
        # print x.get_shape()
        # print W.get_shape()
        if output_shape is None:
            output_shape = x.get_shape().as_list()
            output_shape[1] *= 2
            output_shape[2] *= 2
            output_shape[3] = W.get_shape().as_list()[2]
        # print output_shape
        conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding="SAME")
        return tf.nn.bias_add(conv, b)

    def process_image(self,image, mean_pixel):
        return image - mean_pixel
    def inference(self,image, keep_prob):
        """
        Semantic segmentation network definition
        :param image: input image. Should have values in range 0-255
        :param keep_prob:
        :return:
        """
	NUM_OF_CLASSESS = 51
	IMAGE_SIZE = 224
        print("setting up vgg initialized conv layers ...")
        model_data = self.data
        mean = model_data['normalization'][0][0][0]
        mean_pixel = np.mean(mean, axis=(0, 1))

        weights = np.squeeze(model_data['layers'])

        processed_image = self.process_image(image, mean_pixel)

        with tf.variable_scope("inference"):
            image_net = self.vgg_net(weights, processed_image)
            conv_final_layer = image_net["conv5_3"]

            pool5 = self.max_pool_2x2(conv_final_layer)

            W6 = self.weight_variable([7, 7, 512, 4096], name="W6")
            b6 = self.bias_variable([4096], name="b6")
            conv6 = self.conv2d_basic(pool5, W6, b6)
            relu6 = tf.nn.relu(conv6, name="relu6")
            relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

            W7 = self.weight_variable([1, 1, 4096, 4096], name="W7")
            b7 = self.bias_variable([4096], name="b7")
            conv7 = self.conv2d_basic(relu_dropout6, W7, b7)
            relu7 = tf.nn.relu(conv7, name="relu7")
            relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

            W8 = self.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
            b8 = self.bias_variable([NUM_OF_CLASSESS], name="b8")
            conv8 = self.conv2d_basic(relu_dropout7, W8, b8)
            # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

            # now to upscale to actual image size
            deconv_shape1 = image_net["pool4"].get_shape()
            W_t1 = self.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
            b_t1 = self.bias_variable([deconv_shape1[3].value], name="b_t1")
            conv_t1 = self.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
            fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

            deconv_shape2 = image_net["pool3"].get_shape()
            W_t2 = self.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
            b_t2 = self.bias_variable([deconv_shape2[3].value], name="b_t2")
            conv_t2 = self.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
            fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

            shape = tf.shape(image)
            deconv_shape3 = tf.pack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
            W_t3 = self.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
            b_t3 = self.bias_variable([NUM_OF_CLASSESS], name="b_t3")
            conv_t3 = self.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

            annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

        return tf.expand_dims(annotation_pred, dim=3), conv_t3

