# Import the Libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
import glob
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Activation,LeakyReLU,ReLU,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.applications.vgg16 import VGG16

from glob import glob
glob('F:/computer vision/code/er/Disease_detection/disease/train/*')

from glob import glob
glob('F:/computer vision/code/er/Disease_detection/disease/test/*')

class VGG16_Disease:
  def __init__(self):
    self.train_path='F:/computer vision/code/er/Disease_detection/disease/train'
    self.test_path='F:/computer vision/code/er/Disease_detection/disease/test'
    self.height,self.width=224,224
    self.labels=['COVID19','NORMAL','PNEUMONIA']

  def Disease_Architecture(self,Train_data_preprocess,Test_data_preprocess):
    try:
      """          VGG16 Architecturre         """
      vgg16 = VGG16(input_shape=(self.height,self.width,3),weights='imagenet',include_top=False) # give our image to vgg16
      for layers in vgg16.layers:
        layers.Trainable=False
      x= Flatten()(vgg16.output) # 1d array
      predict=Dense(3,activation='softmax')(x) # two or more classes
      model=Model(inputs=vgg16.inputs,outputs=predict) # taking vgg16 inputs
      print(model.summary())
      Train_data=Train_data_preprocess.flow_from_directory(self.train_path,target_size=(self.height,self.width),class_mode='categorical',classes=self.labels,batch_size=32)
      Test_data=Test_data_preprocess.flow_from_directory(self.test_path,target_size=(self.height,self.width),class_mode='categorical',classes=self.labels,batch_size=32)
      model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
      model.fit(Train_data,validation_data=Test_data,epochs=2,steps_per_epoch=len(Train_data))
      return model

    except:
      print(f'error in main:{sys.exc_info()}')
  def Disease_prediction(self,model):
    try:
      image=cv2.imread('F:/computer vision/code/er/Disease_detection/disease/train/PNEUMONIA/PNEUMONIA(322).jpg')
      cv2.imshow('image', image)
      cv2.waitKey()
      cv2.destroyAllWindows()
      print(f'Original size of image:{image.shape}')
      image=cv2.resize(image,(self.height,self.width))# set the same size when image given to vgg16
      print(f'After resize of image:{image.shape}')
      image=np.array(image)/255.0
      image=np.expand_dims(image,axis=0)
      result=model.predict(image)
      print(result)
      sol=np.argmax(result,axis=1)
      print(sol[0])
      print(self.labels[sol[0]])

    except:
      print(f'error in main:{sys.exc_info()}')

  def Disease_preprocess(self):
    try:
      Train_data_preprocess = ImageDataGenerator(rescale=1./255.0,shear_range=0.2,horizontal_flip=True,zoom_range=0.2)
      Test_data_preprocess = ImageDataGenerator(rescale=1./255.0)
      model=self.Disease_Architecture(Train_data_preprocess,Test_data_preprocess)
      self.Disease_prediction(model)

    except:
      print(f'error in main:{sys.exc_info()}')


if __name__=='__main__':
  try:
    obj = VGG16_Disease()
    obj.Disease_preprocess()
  except:
    print(f'error in main:{sys.exc_info()}')