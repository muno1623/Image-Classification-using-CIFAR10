import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.datasets import mnist, cifar10
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow.keras.utils as np_utils
import tensorflow.keras.utils as np_utils
from tensorflow.keras.models import load_model
import pydot

opt = tf.keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(y_test)

#img_r = x_train[0].shape[0]
#img_c = x_train[0].shape[1]
#reshaping array for keras with depth icluded
#x_train = x_train.reshape(x_train.shape[0], img_r, img_c, 1)
#x_test = x_test.reshape(x_test.shape[0], img_r, img_c, 1)
#store the shape of single image
#input_shape = (img_r, img_c, 1)
#change image type to float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#nomralize
x_train = x_train / 255
x_test = x_test / 255

#one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_of_classes = y_test.shape[1]

#Training
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_of_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

loss_and_metrics = model.evaluate(x_test, y_test, verbose=0)
print("Loss Score", loss_and_metrics[0])
print("Accuracy Score", loss_and_metrics[1])
print(history)

#Visualizing Model
np_utils.plot_model(model, "E:/Computer Vision and Machine Learning Projects/BuildingCNN/model_plot.png",show_shapes=True, show_layer_names=True )

#saving model
model.save("E:/Computer Vision and Machine Learning Projects/BuildingCNN/cifar10_cn_10ep.h5")

#plotting model
#Loss Plot
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
ep = range(1, len(loss_values)+1)
line1 = plt.plot(ep, val_loss_values, label = "Validation Loss")
line2 = plt.plot(ep, loss_values, label = "Training Loss")
plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()
#Accuracy Plot
loss_values = history_dict["accuracy"]
val_loss_values = history_dict["val_accuracy"]
ep = range(1, len(loss_values)+1)
line1 = plt.plot(ep, val_loss_values, label = "Validation Accuracy")
line2 = plt.plot(ep, loss_values, label = "Training Accuracy")
plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

#Prdeiction
#loading model
classifier = load_model("E:/Computer Vision and Machine Learning Projects/BuildingCNN/cifar10_cn_10ep.h5")
for i in range (0, 10):
    input_image = x_test[np.random.randint(0, len(x_test))]
    imageL = cv2.resize(input_image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    input_image = input_image.reshape(1, 32, 32, 3)
    black = [0, 0, 0]
    expand_image = cv2.copyMakeBorder(imageL, 0, 0, 0, imageL.shape[0], cv2.BORDER_CONSTANT, value=black)
    #expand_image = cv2.cvtColor(expand_image, cv2.COLOR_GRAY2BGR)
    print(input_image.shape)
    classes = str(classifier.predict_classes(input_image, 1, verbose=0)[0])
    cv2.putText(expand_image, classes, (152, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    cv2.imshow("Name", expand_image)
    cv2.waitKey(0)
