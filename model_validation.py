from lib2to3.pgen2.token import AT
from re import A
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

from pdf2image import convert_from_path, convert_from_bytes

myHeight, myWidth = 512, 512

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


def imageMasking(theTableMask):
    aMaxTableMask = tf.argmax(theTableMask, axis = -1)
    aPredictedTableMask = aMaxTableMask[..., tf.newaxis]
    return aPredictedTableMask[0]


def imageDisplaying(theList):
    plt.figure(figsize = (30, 30))

    aTitle = ['Input Image', 'Table Mask', 'Column Mask']

    for anIndex in range(len(theList)):
        plt.subplot(1, len(theList), anIndex + 1)
        plt.title(aTitle[anIndex])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(theList[anIndex]))
        plt.axis('off')
        
    plt.show()



def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ['Input Image', 'Table Mask']
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        image = display_list[i]
        plt.imshow(tf.keras.preprocessing.image.array_to_img(image))
        plt.axis('off')
    plt.show()

class F1_Score(tf.keras.metrics.Metric):

    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        p = self.precision_fn(y_true, tf.argmax(y_pred, axis=-1))
        r = self.recall_fn(y_true, tf.argmax(y_pred, axis=-1))
        # since f1 is a variable, we use assign
        self.f1.assign(2 * ((p * r) / (p + r + 1e-6)))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)
        
def validation(data_path, model_path):

    model = keras.models.load_model(model_path,custom_objects={'F1_Score':F1_Score})
    model.summary()
    anOriginalImage = tf.data.Dataset.list_files(data_path)
    aMappedImage = anOriginalImage.map(imageProcessing, num_parallel_calls = tf.data.experimental.AUTOTUNE)
    aBatch = aMappedImage.batch(1)
    for anImage in aBatch:
        aPredictedTableMask, aPredictedColumnMask = model.predict(anImage, verbose = 1)
        #print(aPredictedTableMask.shape)
        #aPredictedTableMask = np.array(aPredictedTableMask,dtype=int)
        #aPredictedTableMask = aPredictedTableMask.reshape(512,1024)
        #np.savetxt("/home/js/modeloutput.txt",aPredictedTableMask)
        #np.save('/home/js/modeloutput.txt',aPredictedTableMask)
        
        aTableMask = imageMasking(aPredictedTableMask)
        aColumnMask = imageMasking(aPredictedColumnMask)
        aTable = np.array(aTableMask[0],dtype=int)
        print(type(aTable))
        print(aTable.shape)
        
        np.savetxt("/home/js/modeloutput.txt",aTable)
        np.save('/home/js/modeloutput.txt',aTable)
        imageDisplaying([anImage[0], aTableMask, aColumnMask])  
    



#test_imgs = convert_from_path("/home/js/download_pdf/paper3.pdf")

#for i, page in enumerate(test_imgs):
#    page.save("/home/js/download_pdf/"+"test"+str(i)+".jpeg","JPEG")


validation("/home/js/download_pdf/test*.jpeg","/home/js/Downloads/tablenet_model/model_binary/tablenet_Resnet_only")

