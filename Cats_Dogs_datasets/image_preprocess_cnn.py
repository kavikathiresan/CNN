# Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPool2D,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu,sigmoid

"""**Import the path file**"""

from glob import glob
glob('/content/drive/MyDrive/Colab Notebooks/Dog_cat_DatasetsCNN/Dog_cat_Dataset/training_set/*')

from glob import glob
glob('/content/drive/MyDrive/Colab Notebooks/Dog_cat_DatasetsCNN/Dog_cat_Dataset/test_set*')

class CNN:
  def __init__(self,path):
    self.path=path
    self.Train_data_path='/content/drive/MyDrive/Colab Notebooks/Dog_cat_DatasetsCNN/Dog_cat_Dataset/training_set'
    self.Test_data_path='/content/drive/MyDrive/Colab Notebooks/Dog_cat_DatasetsCNN/Dog_cat_Dataset/test_set'
    self.image_height,self.image_width=100,100#Set the fixed size so while modeling images its easy to understand


  def EDA_validation(self,model):
    try:
      Training_accuracy=model.history.history['Accuracy']
      Training_loss=model.history.history['loss']
      plt.figure(figsize = (5,5))
      plt.plot(np.arange(1,11),Training_loss,color='b',label='accuracy')
      plt.plot(np.arange(1,11),Training_accuracy,color='r',label='loss')
      plt.title('Cats and Dogs Training Data Report')
      plt.show()
      # Lets take the cat or dog image and check it
      image=plt.imread('/content/drive/MyDrive/Colab Notebooks/Dog_cat_DatasetsCNN/Dog_cat_Dataset/training_set/dogs/dog.77.jpg')
      print(image)
      print(image.shape)
      plt.imshow(image)
      def prediction(path):
        image_1=cv2.imread(path)
        print('original image:',image_1.shape)
        # Take the size with same training data
        image_1=cv2.resize(image_1,(100,100))
        print(image_1.shape)
        # since the training was done with pixel values after scaling down to 0 - 1 so for test also we need to scale down into 0 - 1
        image_1=np.array(image_1)/255.0
        image_1=np.expand_dims(image_1,axis=0)
        result=model.predict(image_1)
        print(result)
        if result[0][0]>0.5:
          print("this image is Dog")
        else:
          print("this image is Cat")
        images=plt.imread(path)
        plt.imshow(images)
      prediction('/content/drive/MyDrive/Colab Notebooks/Dog_cat_DatasetsCNN/Dog_cat_Dataset/training_set/cats/cat.73.jpg')
    except Exception as e:
      print(f'error in main:{e.__str__()}')


  def image_buliding(self,train_data_preprocess,test_data_preprocess):
    try:
      labeles=['cats','dogs']#take the lables
      """ NOW assign the Train data and Test data for architecture"""
      Train_data=train_data_preprocess.flow_from_directory(self.Train_data_path,target_size=(self.image_height,self.image_width),classes=labeles,class_mode='binary',batch_size=32)
      Test_data=test_data_preprocess.flow_from_directory(self.Test_data_path,target_size=(self.image_height,self.image_width),classes=labeles,class_mode='binary',batch_size=32)
      model=Sequential()# Inital steps for buliding the architecture
      model.add(Conv2D(128,kernel_size=(3,3),input_shape=(self.image_height,self.image_width,3),padding='same',activation='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Conv2D(32,kernel_size=(3,3),padding='same',activation='relu'))
      model.add(MaxPool2D(pool_size=(2,2)))
      model.add(Flatten()) # one dimensional array # 1D given to ANN
      model.add(Dense(32,activation='relu'))# hidden layer 1
      model.add(Dense(16,activation='relu'))# hidden layer 2
      model.add(Dense(1,activation='sigmoid'))# Output layer
      model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['Accuracy'])#data has given to architecture
        #Training Begins
      model.fit(Train_data,epochs=10)
      return model

    except Exception as e:
      print(f'error in main:{e.__str__()}')


  def image_process(self):
    try:
      train_data_preprocess=ImageDataGenerator(rescale=1./255,shear_range=0.2,horizontal_flip=True,zoom_range=0.2)
      test_data_preprocess=ImageDataGenerator(rescale=1./255)
      model=self.image_buliding(train_data_preprocess,test_data_preprocess)
      self.EDA_validation(model)

    except Exception as e:
      print(f'error in main:{e.__str__()}')
if __name__=='__main__':
  obj=CNN('/content/drive/MyDrive/Colab Notebooks/Dog_cat_DatasetsCNN/Dog_cat_Dataset')
  obj.image_process()






