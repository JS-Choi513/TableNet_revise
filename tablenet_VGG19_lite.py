import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import pytesseract
from sklearn.model_selection import train_test_split
gpus = tf.config.experimental.list_physical_devices('GPU')

"""# paths"""

originalImage = "/home/js/Downloads/Marmot_data/10.1.1.1.2006_3.bmp"
imageMask = "/home/js/Downloads/Marmot_data/10.1.1.1.2006_3.xml"
fileSavepath = "/home/js/Downloads/Tablenet_test/"
table_mask_path = "/home/js/Downloads/Tablenet_test/"
org_image_path = "/home/js/Downloads/Tablenet_test/"
dataPath = "/home/js/Downloads/Marmot_data/"


"""# reading dataframe"""

final_dataframe = pd.read_csv("/home/js/Downloads/Tablenet_test/final_dataframe.csv")
final_dataframe.head(2)

"""# data generator"""

X_train, X_test = train_test_split(final_dataframe, test_size=0.2)
print("train",X_train.head())
print("\n")
print("test",X_test.head())


training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(X_train['image'].values, tf.string),
            tf.cast(X_train['table_mask'].values, tf.string),
        )
    )
)



testing_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(X_test['image'].values, tf.string),
            tf.cast(X_test['table_mask'].values, tf.string),
        )
    )
)

# https://www.tensorflow.org/tutorials/load_data/images

@tf.function
def load_image(image, table_mask):

    image = tf.io.read_file(image)
    table_mask=tf.io.read_file(table_mask)

    image=tf.io.decode_bmp(image, channels=3)
    image=tf.image.resize(image, [512, 512])
    image = tf.cast(image, tf.float32) / 255.0

    table_mask=tf.io.decode_jpeg(table_mask, channels=1)
    table_mask=tf.image.resize(table_mask, [512, 512])
    table_mask = table_mask / 255.0
    



    return image, {"table_mask":table_mask}



# creating dataset object
train = training_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test = testing_dataset.map(load_image)

BATCH_SIZE = 2
BUFFER_SIZE = 10
train_steps = len(X_train) // BATCH_SIZE

# for feeding to training
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE)


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


for image, mask in train.take(1):

    sample_image = image
    sample_table_mask = mask['table_mask']


    print(image.shape)
    print(mask['table_mask'].shape)
    display([image, mask['table_mask']])



for image, mask in test_dataset.take(1):

    sample_image = image
    print(mask)
    print("space")
    print(mask['table_mask'])    
    sample_table_mask = mask['table_mask'][0]
    print("space")
    print(sample_table_mask.shape)
    display([image[0], sample_table_mask])



"""# model"""
from keras.layers import Input
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import UpSampling2D
from keras.layers import Concatenate
from keras.layers import Layer
from keras.layers import Activation
from keras.layers import Conv2DTranspose
#from tensorflow.keras.utils import plot_model
from keras import backend as K
import tensorflow as tf
import keras



tf.keras.backend.clear_session()


class table_mask(Layer):

    def __init__(self):
        super().__init__()
        self.conv_7 = Conv2D(kernel_size=(1,1), filters=128, kernel_regularizer=tf.keras.regularizers.l2(0.002))
        self.upsample_pool4 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_pool3 = UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.upsample_final = Conv2DTranspose(filters=2, kernel_size=3, strides=2, padding='same', activation='softmax')

    def call(self, input):
        
        x = self.conv_7(input)
        x = self.upsample_pool4(x)
       
        x = self.upsample_pool3(x)
        
        x = UpSampling2D((2,2))(x)
        x = UpSampling2D((2,2))(x)

        x = self.upsample_final(x)

        return x





input_shape = (512, 512, 3)
input_ = Input(shape=input_shape)
vgg19_ = keras.applications.vgg19.VGG19(
    include_top=False,
    weights="imagenet",
    input_tensor=input_,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)



for layer in vgg19_.layers:
    print("fixed!")
    
    layer.trainable = False


conv_1_1_1 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', name="block6_conv1", kernel_regularizer=tf.keras.regularizers.l2(0.004))(vgg19_.output)
conv_1_1_1_drop = Dropout(0.8)(conv_1_1_1)

conv_1_1_2 = Conv2D(filters=128, kernel_size=(1, 1), activation='relu', name="block6_conv2", kernel_regularizer=tf.keras.regularizers.l2(0.004))(conv_1_1_1_drop)
conv_1_1_2_drop = Dropout(0.8)(conv_1_1_2)

table_mask = table_mask()(conv_1_1_2_drop)

model = Model(input_, table_mask)

model.summary()
tf.keras.utils.plot_model(model,show_shapes=True,show_layer_names=True)

"""# model trainning"""



losses = {
    "table_mask": 'sparse_categorical_crossentropy',
}

tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)



filepath = "/home/js/Downloads/Tablenet_test/table_net.h5"
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor = "val_table_mask_loss", save_best_only=True, verbose = 0, mode="min")

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5,)


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


metrics = [F1_Score()]

global init_lr
init_lr = 0.0001

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=init_lr, epsilon=1e-8,),
              loss=losses,
              metrics=metrics )

def show_predictions(dataset=None, num=1):

    if dataset:
        for image, mask in dataset.take(1):
            table_mask_pred = model.predict(image)
            mask_1 = mask['table_mask'][0]            
            print("mask_2 shape",mask_1.shape)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(mask_1))
            plt.show()
            
            table_mask_pred = tf.argmax(table_mask_pred, axis=-1)
            table_mask_pred = table_mask_pred[..., tf.newaxis][0]


            im=tf.keras.preprocessing.image.array_to_img(image[0])
            im.save('image.png')

            im=tf.keras.preprocessing.image.array_to_img(table_mask_pred)
            im.save('table_mask_pred.png')


            display([image[0], table_mask_pred])




class DisplayCallback(tf.keras.callbacks.Callback):

    def __init__(self):
        self.history = {'val_table_mask_loss':[]}
        self.init_lr = init_lr

    #def on_epoch_end(self, epoch, logs=None):
    #    if epoch % 2 == 0:
    #        show_predictions(test_dataset, 1)
            


EPOCHS = 100
VAL_SUBSPLITS = 30
VALIDATION_STEPS = len(X_test)//BATCH_SIZE//VAL_SUBSPLITS

history = model.fit(train_dataset,
                              epochs=EPOCHS,
                              steps_per_epoch=train_steps,
                              validation_data=test_dataset,
                              validation_steps=VALIDATION_STEPS,
                              callbacks=[model_checkpoint, es, DisplayCallback()])
'''

'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("tablenet_VGG19_lite_100epoch_loss.png")
#plt.plot(history.history['table_mask_f1_score'])
#plt.plot(history.history['val_table_mask_f1_score'])
plt.plot(history.history['f1_score'])
plt.plot(history.history['val_f1_score'])
plt.title('model score')
plt.ylabel('table f1 score')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("tablenet_VGG19_lite_100epoch_F1.png")





model.save('model/tablenet_VGG19_lite_100')





