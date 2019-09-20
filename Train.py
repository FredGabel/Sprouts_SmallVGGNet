"""
20sept2019, FG : Initial commit
Train and predict scripts for custom VGGNet NN
Count number of sprouts in the image
------
Training initially done in keras tensorflow backend
"""

# Ignored the future warning
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from smallvggnet import SmallVGGNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
import gc

# Building the train_imgs sequences, 100 imgs for each label 0,1,2
train_dir = "dataset/train"
test_dir = "dataset/test"

train_0 = ["dataset/train/0/{}".format(i) for i in os.listdir(train_dir + "/0")]
train_1 = ["dataset/train/1/{}".format(i) for i in os.listdir(train_dir + "/1")]
train_2 = ["dataset/train/2/{}".format(i) for i in os.listdir(train_dir + "/2")]

train_imgs = train_0[:100] + train_1[:100] + train_2[:100]
random.shuffle(train_imgs)

del train_0
del train_1
del train_2
gc.collect()


def pre_process_image(list_of_images):
    """ resizes X images and creates y label list """
    X = []  # images
    y = []  # labels
    for image in list_of_images:
        imread = cv2.imread(image, cv2.IMREAD_COLOR)
        X.append(cv2.resize(imread, (64, 64), interpolation=cv2.INTER_CUBIC))
        if '0.' in image:
            y.append(0)
        elif '1.' in image:
            y.append(1)
        elif '2' in image:
            y.append(2)
    return X, y

# X image data y labels
X, y = pre_process_image(train_imgs)

# scale the raw pixel intensities to the range [0, 1]
X = np.array(X, dtype="float")/255.0
y = np.array(y)

# split the training data / testing data
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.20, random_state=2)

# convert the labels from integers to vectors
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_val = lb.transform(y_val)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2)

# initialize the smallvggnet
model = SmallVGGNet.build(width=64, height=64, depth=3, classes=len(lb.classes_))

# initialize parameters for learning
INIT_LR = 0.01
EPOCHS = 75
BS = 32

# initialize model and optimize - use categorical_crossentropy as more than 2 classes
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
H = model.fit_generator(aug.flow(X_train, y_train, batch_size=BS),
                        validation_data=(X_val, y_val),
                        steps_per_epoch=len(X_train) // BS,
                        epochs=EPOCHS)

## evaluate the network - Does not work for now
#print("[INFO] evaluating network...")
#predictions = model.predict(X_val, batch_size=32)
#print(classification_report(y_val.argmax(axis=1),
#                            predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("output/smallvggnet_plot.png")

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save("output/Sprouts_smallvggnet.h5")
f = open("output/Sprouts_smallvggnet", "wb")
f.write(pickle.dumps(lb))
f.close()
