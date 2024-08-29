import numpy as np
import sys
import os

import tensorflow as tf
import tensorflow.keras.layers
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import optimizers,callbacks
from tensorflow.keras.layers import Layer, Lambda, Conv2D, Dropout,Dense,Activation,Input,GlobalAveragePooling1D
from tensorflow.keras.layers import Reshape,Flatten,BatchNormalization,MaxPooling1D,AveragePooling2D,Reshape,Attention
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping,History,ModelCheckpoint

from Common_Model import Common_Model
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix,f1_score
from datetime import datetime
from Config import Config

# Random seed setting
from numpy.random import seed
seed(1024)
tf.random.set_seed(2048)


def margin_loss(y_true, y_pred):
    """
        Margin Loss
        :param y_true: [None, n_classes]
        :param y_pred: [None, num_capsule]
        :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))



def softmax(x, axis=-1):
    """
        softmax in Dynamic Routings
    """
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex/K.sum(ex, axis=axis, keepdims=True)

# def caps_dropout(X, drop_probability):
#     keep_probability = 1 - drop_probability
#     assert 0 <= keep_probability <= 1
#     # 这种情况下把全部元素都丢弃。
#     # all the uints are dropouted in the case.
#     if keep_probability == 0:
#         return X.zeros_like()
#     (num_samples, num_capsules, capsule_dim) = K.shape(X)
#     # 随机选择一部分该层的输出作为丢弃元素。
#     # some of layers are dropouted randomly.
#     mask = K.random_uniform( shape = (num_samples, num_capsules, 1), minval = 0.0, maxval = 1.0 )
#         # 0, 1.0, X.shape[0]*X.shape[2], ctx=X.context).reshape((X.shape[0],1,X.shape[2],1,1)) < keep_probability
#     mask = mask[mask < keep_probability].assign(1.0)
#     mask = mask[mask >= keep_probability].assign(0.0)
#     # 保证 E[dropout(X)] == X
#     scale =  1 / keep_probability
#     return mask * X *scale

def PrimaryCapssquash(vectors, axis=-1):
    """
        The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
        :param vectors: some vectors to be squashed, N-dim tensor
        :param axis: the axis to squash
        :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def DigitCapssquash(Value, axis = -1):
    """
        Squash activation in PrimaryCaps
        :return: a Tensor with same shape as input vectors
    """
    Square_Vector = K.sum(K.square(Value), axis, keepdims=True)
    Proportion = Square_Vector / (1 + Square_Vector) / K.sqrt(Square_Vector + K.epsilon())
    Output = Proportion * Value
    return Output

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    """
        Apply Conv2D `n_channels` times and concatenate all capsules
        :return: output tensor, shape=[None, num_capsule, dim_capsule]
    """
    output = tf.keras.layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = tf.keras.layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return tf.keras.layers.Lambda(PrimaryCapssquash, name='primarycap_squash')(outputs)

# Smooth label operation
def smooth_labels(labels, factor=0.1):
    """
        smooth the labels
        returned the smoothed labels
    """
    labels *= (1 - factor)
    labels += (factor / labels.shape[1])
    return labels


class Mask(Layer):
    """
    Mask a Tensor with shape=[None, num_capsule, dim_vector] either by the capsule with max length or by an additional 
    input mask. Except the max-length capsule (or specified capsule), all vectors are masked to zeros. Then flatten the
    masked Tensor.
    For example:
        ```
        x = keras.layers.Input(shape=[8, 3, 2])  # batch_size=8, each sample contains 3 capsules with dim_vector=2
        y = keras.layers.Input(shape=[8, 3])  # True labels. 8 samples, 3 classes, one-hot coding.
        out = Mask()(x)  # out.shape=[8, 6]
        # or
        out2 = Mask()([x, y])  # out2.shape=[8,6]. Masked with true labels y. Of course y can also be manipulated.
        ```
    """
    def call(self, inputs, **kwargs):
        if type(inputs) is list:  # true label is provided with shape = [None, n_classes], i.e. one-hot code.
            assert len(inputs) == 2
            inputs, mask = inputs
        else:  # if no true label, mask by the max length of capsules. Mainly used for prediction
            # compute lengths of capsules
            x = K.sqrt(K.sum(K.square(inputs), -1))
            # generate the mask which is a one-hot code.
            # mask.shape=[None, n_classes]=[None, num_capsule]
            mask = K.one_hot(indices=K.argmax(x, 1), num_classes=x.get_shape().as_list()[1])

        # inputs.shape=[None, num_capsule, dim_capsule]
        # mask.shape=[None, num_capsule]
        # masked.shape=[None, num_capsule * dim_capsule]
        masked = K.batch_flatten(inputs * K.expand_dims(mask, -1))
        return masked

    def compute_output_shape(self, input_shape):
        if type(input_shape[0]) is tuple:  # true label provided
            return tuple([None, input_shape[0][1] * input_shape[0][2]])
        else:  # no true label provided
            return tuple([None, input_shape[1] * input_shape[2]])

    def get_config(self):
        config = super(Mask, self).get_config()
        return config

class Length(Layer):
    """
    Compute the length of vectors. This is used to compute a Tensor that has the same shape with y_true in margin_loss.
    Using this layer as model's output can directly predict labels by using `y_pred = np.argmax(model.predict(x), 1)`
    inputs: shape=[None, num_vectors, dim_vector]
    output: shape=[None, num_vectors]
    """
    def call(self, inputs, **kwargs):
        return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape[:-1]

    def get_config(self):
        config = super(Length, self).get_config()
        return config

class Capsule(tensorflow.keras.layers.Layer):
    """
        DigitCaps layer
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = DigitCapssquash
        else:
            self.activation = activations.get(activation)

    def get_config(self):
       config = {"num_capsule":self.num_capsule,
                 "dim_capsule":self.dim_capsule,
                 "routings":self.routings,
                 "share_weights":self.share_weights,
                 "activation":self.activation
                }
       base_config = super(Capsule, self).get_config()
       return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        #input_dim_capsule = 8
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,#input_dim_capsule = 64
                                            self.num_capsule * self.dim_capsule), # 6*64
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])


        # r1 = tf.shape(u_vecs)
        # tf.print('Size of u_vecs: ',r1)
        # r1 = tf.shape(u_hat_vecs)
        # tf.print('Size of U_hat_vecs: ' , r1)
        # r1 = tf.shape(self.W)
        # print('Size of weights: ', r1)

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))

        # b = tf.shape(u_hat_vecs)
        # print('New U_HAT_VECS Shape: ', b)
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # b = tf.shape(u_hat_vecs)
        # print('New 1 u-hat_vecs: ', b)

        b = K.zeros_like(u_hat_vecs[:,:,:,0])
        # r1 = tf.shape(b)
        # print('b shape: ', r1)

        for i in range(self.routings): #Routings
            c = softmax(b, 1)

            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

class CPAC_Model(Common_Model):
    def __init__(self, input_shape,num_classes,**params):
        super(CPAC_Model,self).__init__(**params)
        self.data_shape = input_shape
        self.num_classes = num_classes
        self.matrix = []
        self.eva_matrix = []
        self.acc = 0

    def save_model(self, model_name):
        """
            Store the model weights as model_name.h5 and model_name.json in the /Models directory
        """
        now_time = datetime.now().strftime('%m-%d~%H:%M:%S')
        
        h5_save_path = 'Models/' + model_name + now_time+ '.h5'
        self.train_model.save_weights(h5_save_path)

        save_json_path = 'Models/' + model_name + now_time + '.json'
        with open(save_json_path, "w") as json_file:
            json_file.write(self.model.to_json())
    
    def create_model(self, args):
        self.inputs=Input(shape = (self.data_shape[0],self.data_shape[1],1))
        self.conv_1 = Conv2D(filters=64, kernel_size=3, name=None)(self.inputs)
        self.conv_1 = BatchNormalization(axis=-1)(self.conv_1, training = False)
        self.conv_1 = Activation('elu')(self.conv_1)
        self.conv_1 = AveragePooling2D()(self.conv_1)
        self.conv_1 = Dropout(0.5)(self.conv_1)
        
        self.conv_2 = Conv2D(filters=64, kernel_size=3, name=None)(self.conv_1)
        self.conv_2 = BatchNormalization(axis=-1)(self.conv_2, training = False)
        self.conv_2 = Activation('elu')(self.conv_2)
        self.conv_2 = AveragePooling2D()(self.conv_2)
        self.conv_2 = Dropout(0.25)(self.conv_2)

        # self.conv_3 = Conv2D(filters=64, kernel_size=3, name=None)(self.conv_2)
        # self.conv_3 = BatchNormalization(axis=-1)(self.conv_3, training = False)
        # self.conv_3 = Activation('elu')(self.conv_3)
        # self.conv_3 = AveragePooling2D()(self.conv_3)
        # self.conv_3 = Dropout(0.25)(self.conv_3)
        self.cap = self.conv_2
        self.primarycaps = PrimaryCap( self.cap, dim_capsule=16, n_channels=3, kernel_size=3, strides=1, padding='valid' )


        self.cap = self.primarycaps
        self.sa = Attention(use_scale =True )([self.primarycaps,self.primarycaps,self.primarycaps])
        self.cap = Lambda(lambda x: tf.multiply(x[0], x[1]))([self.cap, self.sa])
        # (num_samples, num_cap, cap_dim) = K.shape(self.cap)
        # Capsule Drop Out
        if args.capsule_drop_prob:
            self.cap = Dropout( rate = args.capsule_drop_prob, noise_shape = ( None, None, 1 ) ) ( self.cap )

        self.capsule = Capsule(args.digit_cap_num_capsule, args.digit_cap_capsule_dim, 3, True)(self.cap)
        # if args.if_reconstruction:
        self.out_caps = Length(name='capsnet')(self.capsule)
        if not args.if_reconstruction:
            self.out_caps = softmax(self.out_caps)

        # self.out_caps = tf.keras.activations.softmax(self.out_caps)
        # print('Shape of out_caps: ', tf.shape(self.out_caps))
        # Decoder network.
        self.y = Input(shape=(self.num_classes,))
        self.masked_by_y = Mask()([self.capsule, self.y])  # The true label is used to mask the output of capsule layer. For training
        # self.masked = Mask()(self.capsule)  # Mask using the capsule with maximal length. For prediction

        # Shared Decoder model in training and prediction
        decoder = Sequential(name='decoder')
        decoder.add(Dense(256, activation='relu', input_dim=args.digit_cap_capsule_dim*self.num_classes))
        decoder.add(Dense(512, activation='relu'))
        decoder.add(Dense(1024, activation='relu'))
        decoder.add(Dense(np.prod(self.data_shape), activation='sigmoid'))
        decoder.add(Reshape(target_shape=self.data_shape, name='out_recon'))

        
        if args.if_reconstruction:

            # Models for training and evaluation (prediction)
            self.train_model = Model([self.inputs, self.y], [ self.out_caps, decoder(self.masked_by_y) ] )
            self.eval_model = Model(self.inputs, self.out_caps)
            # self.GA = GlobalAveragePooling1D()(self.capsule)
            # self.drop = Dropout(0.2)(self.GA)
            # self.predictions = Dense(self.num_classes, activation='softmax')(self.drop)
            # self.train_model = Model(inputs = [self.inputs,self.y],  = self.predictions)
            self.train_model.compile(loss = [margin_loss, 'mse'],
                               # loss_weights=[1., 0.392],
                               loss_weights = [args.margin_loss_weight, args.mse_weight],
                               optimizer = Adam(learning_rate=0.001,beta_1=0.975, beta_2=0.932,epsilon=1e-8), metrics = {'capsnet':'accuracy', 'decoder': 'mse'})
        else:
            self.GA = GlobalAveragePooling1D()(self.capsule)
            self.out_caps = Dense(self.num_classes, activation='softmax')(self.GA)
            # Models for training and evaluation (prediction)
            self.train_model = Model(self.inputs, self.out_caps )
            self.eval_model =self.train_model # Eval model and Train Model Same, in case of no reconstruction
            # self.GA = GlobalAveragePooling1D()(self.capsule)
            # self.drop = Dropout(0.2)(self.GA)
            # self.predictions = Dense(self.num_classes, activation='softmax')(self.drop)
            # self.train_model = Model(inputs = [self.inputs,self.y],  = self.predictions)
            self.train_model.compile(loss = [margin_loss],
                               # loss_weights=[1., 0.392],
                            #    loss_weights = [args.margin_loss_weight],
                               optimizer = Adam(learning_rate=0.1,beta_1=0.975, beta_2=0.932,epsilon=1e-8), metrics = 'accuracy')
        print("Model create succes!")
        print(self.train_model.summary())
    
    def train(self, args, x, y, x_test = None, y_test = None, data_name = None, fold = None , random = None):
        """
            train(): train the model on the given training set
            input:
                x (numpy.ndarray): the training set samples
                y (numpy.ndarray): training set labels
                x_val (numpy.ndarray): test set samples
                y_val (numpy.ndarray): test set labels
                n_epochs (int): number of epochs
        """
        avg_accuracy = 0
        avg_loss = 0
        n_split = fold
        filepath='./Models/'
        kfold = KFold(n_splits=n_split, shuffle=True, random_state= random)
        i=1
        flag = 0
        for train, test in kfold.split(x, y):
            flag = 1
            self.create_model(args)
            y[train] = smooth_labels( y[train], 0.1 )
            # y[test] = smooth_labels( y[test], 0.1 )
            folder_address = filepath+data_name
            if not os.path.exists(folder_address):
                os.mkdir(folder_address)
            weight_path = folder_address+'/weights.best_'+str(i)+".hdf5"
            max_acc = 0
            max_f1 = 0
            best_eva_list = []
            for epoch in range(args.num_epochs):
                print("epoch/max_epochs:",str(epoch+1)+'/'+str(args.num_epochs)+" best_acc:"+str(round(max_acc*10000)/100)+" best_F1:"+str(round(max_f1*10000)/100))
                if args.if_reconstruction:
                    self.train_model.fit([x[train], y[train]],[y[train], x[train]], batch_size = 64, epochs = 1, verbose=2)
                    evaluate_list = self.train_model.evaluate([x[test],  y[test]], [y[test], x[test]])
                    # print(self.train_model.metrics_names, evaluate_list)
                    # print('Hello Stupid, Shubha')
                    y_pred, _ = self.train_model.predict([x[test],  y[test]])
                else:
                    self.train_model.fit(x[train],y[train], batch_size = 64, epochs = 1, verbose=1)
                    evaluate_list = self.train_model.evaluate(x[test],  y[test])
                    # print(self.train_model.metrics_names, evaluate_list)
                    # print('Hello Stupid, Shubha')
                    y_pred = self.train_model.predict(x[test])
                f1 = f1_score(np.argmax(y[test],axis=1), np.argmax(y_pred,axis=1), average='weighted')
                if args.if_reconstruction:
                    if evaluate_list[3]>max_acc:
                        max_acc = evaluate_list[3]
                        best_eva_list = evaluate_list
                        y_pred_best = y_pred
                else:
                    if evaluate_list[1]>max_acc:
                        max_acc = evaluate_list[1]
                        best_eva_list = evaluate_list
                        y_pred_best = y_pred
                if f1>max_f1:
                    max_f1 = f1
            # break

            avg_loss += best_eva_list[0]
            avg_accuracy += best_eva_list[3]
            print(str(i)+'_Model evaluation: ', evaluate_list,"   Now ACC:",str(round(avg_accuracy*10000)/100/i))
            i+=1
            self.matrix.append(confusion_matrix(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1)))

            em = classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=Config.CLASS_LABELS,output_dict=True)
            self.eva_matrix.append(em)
            print(classification_report(np.argmax(y[test],axis=1),np.argmax(y_pred_best,axis=1), target_names=Config.CLASS_LABELS))

            if flag == 1:
                break

        # print("Average ACC:",avg_accuracy/n_split)
        # self.acc = avg_accuracy/n_split
        print("Average ACC:",avg_accuracy)
        self.acc = avg_accuracy
        self.trained = True

    
    def predict(self, sample):
        """
            predict(): identify the emotion of the audio
            input:
                samples: the audio features to be recognized
            Output:
                list: the results
        """
        if not self.trained:
            sys.stderr.write("No Model.")
            sys.exit(-1)
        return np.argmax(self.eval_model.predict(sample,verbose=2), axis=1)