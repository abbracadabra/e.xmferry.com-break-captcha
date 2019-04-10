import numpy as np
import PIL.Image as Image
from PIL import ImageDraw
import random
import os
import tensorflow as tf
import time
from  numpy.random import randint
base = os.getcwd()

input = tf.placeholder(dtype='float32', shape=(None,20,20,1), name='input')
input2 = input/255.
label = tf.placeholder(dtype=tf.int64, shape=(None), name='label')
label2 = tf.one_hot(label,36)
temp = tf.layers.conv2d(inputs=input2,filters=64,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
temp = tf.layers.conv2d(inputs=temp,filters=64,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
temp = tf.layers.dropout(temp)
temp = tf.layers.conv2d(inputs=temp,filters=64,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
temp = tf.layers.conv2d(inputs=temp,filters=64,kernel_size=(3,3),padding="SAME",activation=tf.nn.leaky_relu,kernel_initializer=tf.keras.initializers.he_normal())
temp = tf.layers.dropout(temp)
temp = tf.layers.flatten(temp)
temp = tf.layers.dense(temp,36)
prob = tf.nn.softmax(temp)
ixx = tf.argmax(prob,name='pred',axis=-1)
acc = tf.reduce_sum(tf.cast(tf.math.equal(tf.squeeze(tf.argmax(prob,axis=-1)),label),dtype=tf.int32))/tf.shape(label)[0]
loss = tf.losses.softmax_cross_entropy(label2,temp)
op = tf.train.AdamOptimizer().minimize(loss)

def addnoise(img):
    _a = np.random.choice([0,1],p=[0.4,0.6])
    if _a:
        draw = ImageDraw.Draw(img)
        draw.line((randint(-30, 20), randint(-10, 20),randint(0, 50), randint(0, 30)), fill=255,width=randint(1, 3))
        _a = np.random.choice([0, 1], p=[0.5, 0.5])
        if _a:
            draw.line((randint(-30, 20), randint(-10, 20), randint(0, 50), randint(0, 30)), fill=255,
                      width=randint(1, 3))
    return img


def get():
    x = []
    lb=[]
    vb = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    while True:
        c = random.choice(vb)
        cf = os.path.join(base, 'dd', c + '.jpg')
        aa = Image.open(cf)
        bb = Image.new('L', (20, 20))
        bb.paste(aa, (np.random.randint(0,5), 0))
        bb = addnoise(bb)
        bb = np.array(bb)
        bb = np.expand_dims(bb, axis=-1)
        x.append(bb)
        lb.append(vb.index(c))
        if len(x) == 5:
            yield x,lb
            x=[]
            lb=[]

if __name__=='__main__':
    saver = tf.train.Saver()
    sess = tf.Session()
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess,os.path.join(base,'mdl','mdl'))
    for x,lb in get():
        ls,_,ac = sess.run([loss,op,acc], feed_dict={input:np.float32(x),label:lb})
        print(ls,ac)
        if int(time.time()%10)==0:
            saver.save(sess,os.path.join(base,'mdl','mdl'))












