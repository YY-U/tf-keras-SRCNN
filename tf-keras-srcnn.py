import os, sys, math, time, datetime
import glob
import math
import random
from PIL import Image

import numpy as np 
import matplotlib.pyplot as plt 

from tensorflow.python import keras 
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.models import Model,Sequential 
from tensorflow.python.keras.layers import Conv2D,Dense,Input,MaxPooling2D, UpSampling2D,Lambda 
from tensorflow.python.keras.preprocessing.image import load_img,img_to_array,array_to_img,ImageDataGenerator

import cv2 as cv

runcase = int(sys.argv[1])

#flag = 'train'
#flag = 'inference'

#低解像画像生成
def drop_resolution(x,scale = 3.0): 
    size = (x.shape[0], x.shape[1]) 
    small_size = (int(size[0]/scale),int( size[1]/ scale)) 
    img = array_to_img(x) 
    small_img = img.resize(small_size,3) 
    
    return img_to_array(small_img.resize( img.size, 3))



#入力データの生成
def data_generator(data_dir,mode,scale=2.0,target_size=(200,200),batch_size=32,shuffle=True):
    for imgs in ImageDataGenerator().flow_from_directory(
      directory=data_dir,
      classes=[mode],
      class_mode=None,
      color_mode='rgb',
      target_size=target_size,
      batch_size=batch_size,
      shuffle=shuffle
    ):

      x=np.array([
          drop_resolution(img,scale)for img in imgs
      ])
      yield x/255.,imgs/255.

#ピーク信号対雑音比の定義
def psnr(y_true,y_pred):
  return -10*K.log(K.mean(K.flatten((y_true - y_pred))**2) )/np.log(10)

DATA_DIR='dataset/'
N_TRAIN_DATA=1000
N_TEST_DATA=100
BATCH_SIZE=32

train_data_generator=data_generator(DATA_DIR,'train',batch_size=BATCH_SIZE)
test_x,test_y=next(data_generator(DATA_DIR,'test',batch_size=N_TEST_DATA,shuffle=False))

if runcase == 1:
  #model
  model=Sequential()
  model.add(Conv2D(filters=64,kernel_size=9,padding='same',activation='relu',input_shape=(None,None,3)))
  model.add(Conv2D(filters=32,kernel_size=1,padding='same',activation='relu'))
  model.add(Conv2D(filters=3,kernel_size=5,padding='same'))
  model.summary()

  #metric=PSNRにて学習を実行
  model.compile(loss='mean_squared_error',optimizer='adam',metrics=[psnr])
  model.fit_generator(train_data_generator,validation_data=(test_x,test_y),steps_per_epoch=N_TRAIN_DATA/BATCH_SIZE,
  epochs=60#50
  )
  #テストデータに対して適応
  pred=model.predict(test_x)
  array_to_img(test_x[0]).show()#input
  array_to_img(test_y[0]).show()#GT
  array_to_img(pred[0]).show()#output
  dt_now=datetime.datetime.now()
  model_path=os.path.join("./srcnn_model/","model"+dt_now.strftime('%Y-%m-%d-%H'))
  if not os.path.exists(model_path):
        os.makedirs(model_path)
  model.save(os.path.join(model_path,'srcnn.h5'))

if runcase == 2:
  #model = keras.models.load_model('model/srcnn/srcnn.h5',compile=False)
  model = keras.models.load_model('model/srcnn/srcnn.h5',compile=False)
  model.summary()
  pred=model.predict(test_x)
  print(pred.shape)

  cur_dir_path=os.getcwd()
  output_dir=os.path.join(cur_dir_path,"output_image")
  if not os.path.exists(output_dir):
        os.makedirs(output_dir)

  dt_now=datetime.datetime.now()
  output_img_path=os.path.join(output_dir,"image"+dt_now.strftime('%Y-%m-%d-%H'))
  if not os.path.exists(output_img_path):
        os.makedirs(output_img_path)
  
  def save_images(out_path, img):
    #img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    #cv.imwrite(out_path, img[:,:,::-1])
    #cv.imwrite(out_path, img)
    im=array_to_img(pred[0])
    im.save(output_img_path)
  
  
  #save_images(output_img_path, array_to_img(pred[0]))
  for i in range(0,10):
   im1=array_to_img(test_x[i])
   im2=array_to_img(test_y[i])
   im3=array_to_img(pred[i])

   im1.save(os.path.join(output_img_path,"{}_image_input.png".format("%04d"%i)),quality=95)
   im2.save(os.path.join(output_img_path,"{}_image_GT.png".format("%04d"%i)),quality=95)
   im3.save(os.path.join(output_img_path,"{}_image_output.png".format("%04d"%i)),quality=95)

  #array_to_img(pred[0]).show()#output
  """
  array_to_img(test_x[0]).show()#input
  array_to_img(test_y[0]).show()#GT
  array_to_img(pred[0]).show()#output

  array_to_img(test_x[0]).save('input.png')#input
  array_to_img(test_y[0]).save('GT.png')#GT
  array_to_img(pred[0]).save('output.png')#output
  """
  

  


"""
saver = tf.train.Saver() 
with tf.Session() as sess:  
    #変数の値をmodel/model.ckptに保存する
    saver.save(sess,"model/model")
"""

"""
#Saverを利用すると
saver = tf.train.Saver() 
with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    # model/ model. ckpt から 変数 の 値 を リストア する
    saver.restore(sess,save_path="model/model.ckpt") 
    print(sess.run(b)) 
    print(sess.run(b))
"""



