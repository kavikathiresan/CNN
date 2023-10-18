# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import sys
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Flatten,Activation,LeakyReLU,ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu,sigmoid
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
import logging
logging.basicConfig(filename='sales.log',level=logging.DEBUG,format='%(filename)s:%(message)s')

from glob import glob
glob('F:/computer vision/code/er/Dog_cat_DatasetsCNN/Dog_cat_Dataset/training_set/*')

from glob import glob
glob('F:/computer vision/code/er/Dog_cat_DatasetsCNN/Dog_cat_Dataset/test_set/*')



class VGG19_Binary:
  def __init__(self):
    self.Train_data_path='F:/computer vision/code/er/Dog_cat_DatasetsCNN/Dog_cat_Dataset/training_set' # take the path for train and test
    self.Test_data_path='F:/computer vision/code/er/Dog_cat_DatasetsCNN/Dog_cat_Dataset/test_set'
    self.height,self.width = 250,250 # set the size for model
    self.labels=['cats','dogs']


  def Binary_architecture(self,Train_data_preprocess,Test_data_preprocess):
    try:
      """                   VGG19 Architecture                     """
      vgg19=VGG19(input_shape=(self.height,self.width,3),weights='imagenet',include_top=False) # give our image to vgg16
      for layers in vgg19.layers:
        layers.trainable=False
      x = Flatten()(vgg19.output) # 1d arry
      predict=Dense(1,activation='sigmoid')(x)  # binary classes
      model=Model(inputs=vgg19.inputs,outputs=predict)
      print(model.summary())
      Train_data=Train_data_preprocess.flow_from_directory(self.Train_data_path,target_size= (self.height,self.width),classes=self.labels,class_mode='binary',batch_size=32)
      Test_data=Test_data_preprocess.flow_from_directory(self.Test_data_path,target_size=(self.height,self.width),classes=self.labels,class_mode='binary',batch_size=32)
      model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
      model.fit(Train_data,validation_data=Test_data,epochs=2,steps_per_epoch=len(Train_data))
      return model

    except:
      print(f'error in main:',sys.exc_info())

  def Binary_prediction(self,model):
    try:
      image_1=plt.imread('F:/computer vision/code/er/Dog_cat_DatasetsCNN/Dog_cat_Dataset/training_set/dogs/dog.183.jpg')
      plt.imshow(image_1)
      print(f'original image:{image_1.shape}')
      # Take the size with same training data
      image_1=cv2.resize(image_1, (self.height,self.width))
      print("After assign size for image:{}".format(image_1.shape))
      image_1=np.array(image_1)/255.0
      image_1=np.expand_dims(image_1,axis=0)
      result=model.predict(image_1)
      if result[0][0]>0.5:
        print("Its is a Dog")
      else:
        print("Its is a Cat")

    except:
      print(f'error in main:',sys.exc_info())


  def Binary_EDAvalidation(self,model):
    try:
      train_accuracy = model.history.history['accuracy']
      train_loss = model.history.history['loss']
      test_accuracy = model.history.history['val_accuracy']
      test_loss = model.history.history['val_loss']
      plt.figure(figsize=(7,6))
      plt.subplot(1,2,1)
      plt.plot(np.arange(1,3),train_loss,color='r',label='train_loss')
      plt.plot(np.arange(1,3),train_accuracy,color='y',label='train_accuracy')
      plt.legend(loc='best')
      plt.xlabel("epochs")
      plt.ylabel('Loss and accuracy checking')
      plt.subplot(1,2,2)
      plt.plot(np.arange(1,3),test_loss,color='g',label='test_loss')
      plt.plot(np.arange(1,3),test_accuracy,color='m',label='test_accuracy')
      plt.xlabel("epochs")
      plt.ylabel('Loss and accuracy checking')
      plt.legend(loc='best')
      plt.show()
      return model

    except:
      print(f'error in main:',sys.exc_info())

  def Binary_preprocess(self):
    try:
      Train_data_preprocess = ImageDataGenerator(rescale=1./255.0,shear_range=0.2,horizontal_flip=True,zoom_range=0.2)
      Test_data_preprocess = ImageDataGenerator(rescale=1./255.0)
      model=self.Binary_architecture(Train_data_preprocess,Test_data_preprocess)
      self.Binary_EDAvalidation(model)
      self.Binary_prediction(model)

    except:
      print(f'error in main:',sys.exc_info())

if __name__=='__main__':
  try:
    obj=VGG19_Binary()
    obj.Binary_preprocess()

  except:
    print(f'error in main:',sys.exc_info())
