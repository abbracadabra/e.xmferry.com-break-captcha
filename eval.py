import argparse
parser = argparse.ArgumentParser()
parser.add_argument("imagepath")
args = parser.parse_args()
impath = args.imagepath

import numpy as np
import PIL.Image as Image
import os
import tensorflow as tf
base = os.getcwd()

im = Image.open(impath).convert('L')
im = im.point(lambda p: p > 185 and 255)
im = im.point(lambda p: 255-p)
width, height = im.size
ims=[]
_ = Image.new('L', (20, 20))
_.paste(im.crop((0,0,19,height)),(0,0))
ims.append(np.array(_))
_ = Image.new('L', (20, 20))
_.paste(im.crop((18, 0, 39, height)),(0,0))
ims.append(np.array(_))
_ = Image.new('L', (20, 20))
_.paste(im.crop((39, 0, 59, height)),(0,0))
ims.append(np.array(_))
_ = Image.new('L', (20, 20))
_.paste(im.crop((59, 0, 78, height)),(0,0))
ims.append(np.array(_))

saver = tf.train.import_meta_graph('mdl/mdl.meta')
sess = tf.Session()
saver.restore(sess,'mdl/mdl')
ip = tf.get_default_graph().get_tensor_by_name('input:0')
pd = tf.get_default_graph().get_tensor_by_name('pred:0')
fg = sess.run(pd,feed_dict={ip:np.expand_dims(np.array(ims),axis=-1)})
vb = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
print([vb[i] for i in fg])