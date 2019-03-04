
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras import datasets
from tensorflow.python.keras import applications
from tensorflow.python.keras import models
import os
from tensorflow.python.keras import backend as K

from cifar10vgg import cifar10vgg


def normalize(X_train,X_test):
    #this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0,1,2,3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test

class Attack :

    def __init__(self, model, tol, num_steps, step_size, random_start):
        self.model =model
        self.tol=tol # 어떤 역할을 하는지
        self.num_steps = num_steps
        self.step_size = step_size # learning_rate
        self.rand = random_start # True or False

        # variable
        self.xs=tf.Variable(np.zeros((1,32,32,3), dtype=np.float32),
                            name="modifier")
        self.orig_xs=tf.placeholder(tf.float32, [None, 32, 32, 3])
        self.ys=tf.placeholder(tf.int32, [None]) # shape : N

        self.epsilon =0.1

        # difference between orig_xs and xs
        delta=tf.clip_by_value(self.xs, -2 ,2)-self.orig_xs
        delta=tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs=tf.assign(self.xs, self.orig_xs+delta)

        self.logits=self.model(self.xs) # sess
        # self.logits=self.xs
        print(self.logits) # Tensor("sequential/activation_14/Softmax:0", shape=(1, 10), dtype=float32)
        logits=self.logits

        # obfuscated-gradient
        label_mask=tf.one_hot(self.ys,10) # ont-hot
        correct_logit=tf.reduce_sum(label_mask*logits, axis=1) # shape : (N.?)
        wrong_logit=tf.reduce_max((1-label_mask)*logits- label_mask*1e-4, axis=1) # shape : (N,?)

        self.loss=(correct_logit-wrong_logit)

        start_vars=set(x.name for x in tf.global_variables()) # variable names
        optimizer=tf.train.AdamOptimizer(step_size*1)

        grad, var=optimizer.compute_gradients(self.loss, [self.xs])[0]

        self.train=optimizer.apply_gradients([(tf.sign(grad), var)])

        end_vars=tf.global_variables()
        self.new_vars=[x for x in end_vars if x.name not in start_vars]


    def perturb(self, x,y, sess):

        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)

        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):
            sess.run(self.train, feed_dict={self.ys:y, self.orig_xs:x}) # error
            sess.run(self.do_clip_xs,{self.orig_xs : x})

        return sess.run(self.xs)




if '__name__'=="__main__":
    (x_train, y_train), (x_test, y_test) =datasets.cifar10.load_data()
    x_train,x_test= normalize(x_train,x_test)


    sess=tf.Session()
    K.set_session(sess) # keras session setting
    model=cifar10vgg(train=False).model
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.0005),
                         loss=tf.losses.softmax_cross_entropy,
                         metrics=['accuracy'])

    attack=Attack(model, 1,1,1, False)

    xs=tf.placeholder(tf.float32,(1,32,32,3))

    # original
    idx=0
    image=x_train[idx]
    plt.imshow(image)

    #plt.show()

    plt.savefig("./original/{}_original.png".format(idx))

    image=np.expand_dims(image,0)
    label=y_train[idx]
    print("Image label : {}".format(label))

    print("Clean Model Prediction",np.argmax(model.predict(image)))

    print("\tLogits : {}".format(model.predict(image, batch_size=1)))


    # adversarial
    adversarial=attack.perturb(image, label, sess)
    plt.imshow(adversarial[0])
    #plt.show()
    plt.savefig("./adversarial/{}_adversarial.png".format(idx))

    print("Max distortion", np.max(np.abs(adversarial[0]-image[0])))

    print('Adversarial Model Prediction', np.argmax(model.predict(adversarial)))
    print('\tLogits', model.predict(adversarial))
