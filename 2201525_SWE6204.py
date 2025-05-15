import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import os
#^IMPORTS

base_dir = 'J:/OneDrive/UNI/S Shukla/SWE6204/A1/MURA-v1.1'#path to access dataset
#setting up base directory to access the dataset
train_labelledCSV = os.path.join(base_dir, 'train_labeled_studies.csv')
valid_labelledCSV = os.path.join(base_dir, 'valid_labeled_studies.csv')
train_imagePathCSV = os.path.join(base_dir, 'train_image_paths.csv')
valid_imagePthCSV = os.path.join(base_dir, 'valid_image_paths.csv')
#^rCreating paths for the CSV files

try:
  train_labelled = pd.read_csv(train_labelledCSV)
  valid_labelled = pd.read_csv(valid_labelledCSV)
  train_imagePath = pd.read_csv(train_imagePathCSV)
  valid_imagePath = pd.read_csv(valid_imagePthCSV)
  #^Reading CSV files
except:
  print('Error reading CSV files')
  exit()


trainImagePaths = train_labelled.iloc[:, 0].tolist()
trainLabelsRaw = train_labelled.iloc[:, 1].tolist()
#converting training data to lists to allow easier reading in python
trainLabels = [1 if label == 'positive' else 0 for label in trainLabelsRaw]
#seperating the training data into good and bad scans -- how it is displayed in CSV
validImagePaths = valid_labelled.iloc[:, 0].tolist()
validLabelsRaw = valid_labelled.iloc[:, 1].tolist()
#converting validation data to lists
validLabels = [1 if label == 'positive' else 0 for label in validLabelsRaw]
#seperating validation data

class DataGen(tf.keras.utils.Sequence):
  def __init__(self, imagePaths, labels, batchSize, imageHeight, imageWidth, augment = False):
    self.imagePaths = imagePaths
    self.labels = labels
    self.batchSize = batchSize
    self.imageHeight = imageHeight
    self.imageWidth = imageWidth
    self.augment = augment
    self.base_dir = 'J:/OneDrive/UNI/S Shukla/SWE6204/A1'
    #declaring main variables

  def __len__(self):
    return int(np.ceil(len(self.imagePaths)/float(self.batchSize)))
  #function to decide how many steps in an epoch
  
  def __getitem__(self, idx):
    batchStart = idx * self.batchSize
    batchEnd = (idx + 1) * self.batchSize
    batchPaths = self.imagePaths[batchStart:batchEnd]
    batchLabels = self.labels[batchStart:batchEnd]
    #declare start and endpoints and paths

    batchImagesArr = []
    batchLabelsArr = []
    #create arrays

    for i, path in enumerate(batchPaths):
      fullPath = self.base_dir + '/' + path
      img_files = [f for f in os.listdir(fullPath) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('._')]
      #ensuring the only files read are the ones with the end attatchments of ^^^^^^^^^^^^^^^^6 and will not start with certain characters to avoid crashing
      for img_file in img_files:
        full_img_path = os.path.join(fullPath, img_file)
        #print(full_img_path) --> Debug command to see what file is messing up
        img = image.load_img(full_img_path, target_size = (self.imageHeight, self.imageWidth))
        #load image to correct size
        imgArr = image.img_to_array(img)
        #formating the image to an array
        batchImagesArr.append(imgArr)
        #appending image to array
        batchLabelsArr.append(batchLabels[i])
        #appending index of array to labels array

        if len(batchImagesArr) == self.batchSize:
          break
      if len(batchImagesArr) == self.batchSize:
        break
    if len(batchImagesArr) != self.batchSize:
      print(f"warning not loaded full batch. Loaded {len(batchImagesArr)} images")
      return np.array([]), np.array([])
    #clauses to avoid batch sizes being misread or too extreme -- error faced where some data was not correct size so need these clauses
    

    images = np.array(batchImagesArr)
    labels = np.array(batchLabelsArr).reshape(self.batchSize, 1)
    #reshaping array to allow for minimal errors to present when running code

    print("image shape", images.shape)
    print("label shape", labels.shape)

    return images, labels

  
imageHeight = 256
imageWidth = 256
batchSize = 20
#declaring dimensions


trainGen = DataGen(
    trainImagePaths, trainLabels, batchSize, imageHeight, imageWidth, augment = True
)
#creating training data


validGen = DataGen(
    validImagePaths, validLabels, batchSize, imageHeight, imageWidth, augment = False
)
#creating validation data

print("Train data gen created")
print("valid data gen created")
#acknowledge data is created

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imageHeight, imageWidth, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
#setting up CNN --> Keras sequential

model.summary()
#produce summary of model so far

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
#compile current model to process through trainign and validations


epochs = 10

history =model.fit(
  trainGen,
  steps_per_epoch = len(trainGen),
  epochs = epochs,
  validation_data = validGen,
  validation_steps = len(validGen)
)
#function to run the training and validating algorithms 



evaluation = model.evaluate(validGen, steps=len(validGen))
print("Validation Loss:", evaluation[0])
print("Validation Accuracy: ", evaluation[1])
#evaluation parameters to allow the validation to work fully

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc ='upper right')
plt.title('Training and Validation Loss')
plt.show()
#plot graphs based on training and validation to allow visual monito9ring of outputs


def predImg(model, path, imageHeight, imageWidth):
  img = image.load_img(path, target_size=(imageHeight, imageWidth))
  imgArr = image.img_to_array(img)
  imgArr = np.expand_dims(imgArr, axis=0)
  imgArr /= 255.

  prediction = model.predict(imgArr)
  return prediction[0][0]

newImagePath = 'J:/OneDrive/UNI/S Shukla/SWE6204/A1/Testdata/healthy2.jpeg'
probOfAbnormality = predImg(model, newImagePath, imageHeight, imageWidth)

if probOfAbnormality > 0.5:
  print("Broken bone detected")
else:
  print("The bone looks healthy")
  #iteration to check if the bone image provided is normal or abnormal --> done by using the 

print("Probability:", probOfAbnormality)
#outputs probability of abnormality
