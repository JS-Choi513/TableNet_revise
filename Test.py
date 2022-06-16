#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model as modelLoader


myHeight, myWidth = 256, 256

def imageNormalization(theInputImage):
    anInputImage = tf.cast(theInputImage, tf.float32) / 255.0
    
    return anInputImage


def bitmapDecoding(theImage):
    anImageDecoded = tf.image.decode_jpeg(theImage, channels=3)
    anResizedImage = tf.image.resize(anImageDecoded, [myHeight, myWidth])
    
    return anResizedImage


def imageProcessing(theFilePath):
    anNormalizedImage = imageNormalization(bitmapDecoding(tf.io.read_file(theFilePath)))

    return anNormalizedImage


def imageMasking(theTableMask, theColumnMask):
    aMaxTableMask = tf.argmax(theTableMask, axis = -1)
    aPredictedTableMask = aMaxTableMask[..., tf.newaxis]

    aMaxColumnMask = tf.argmax(theColumnMask, axis = -1)
    aPredictedColumnMask = aMaxColumnMask[..., tf.newaxis]
    
    return aPredictedTableMask[0], aPredictedColumnMask[0]


def imageDisplaying(theList):
    plt.figure(figsize = (30, 30))

    aTitle = ['Input Image', 'Table Mask', 'Column Mask']

    for anIndex in range(len(theList)):
        plt.subplot(1, len(theList), anIndex + 1)
        plt.title(aTitle[anIndex])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(theList[anIndex]))
        plt.axis('off')
        
    plt.show()
    

# main code area
aModel = modelLoader("/home/js/Downloads/tablenet_model/model/tablenet_ow")

anOriginalImage = tf.data.Dataset.list_files("/home/js/download_pdf/test*.jpeg")
aMappedImage = anOriginalImage.map(imageProcessing, num_parallel_calls = tf.data.experimental.AUTOTUNE)
aBatch = aMappedImage.batch(1)

for anImage in aBatch:
    aPredictedTableMask, aPredictedColumnMask = aModel.predict(anImage, verbose = 1)
    aTableMask, aColumnMask = imageMasking(aPredictedTableMask, aPredictedColumnMask)
      
    imageDisplaying([anImage[0], aTableMask, aColumnMask])
    


# %%
