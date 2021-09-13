## Code is originally work on Google Colab

## mount google drive
from google.colab import drive  #Open files from Google Drive
drive.mount('/gdrive')
%cd /gdrive

##install zip file 
pip install rarfile

## extract zip file
import rarfile
rar_ref = rarfile.RarFile("/gdrive/MyDrive/CS7267/3c_balanced.rar", 'r')
rar_ref.extractall("/tmp")
rar_ref.close()


import os
base_dir = '/tmp/3c_balanced'
train_dir = os.path.join(base_dir, 'Train')
validation_dir = os.path.join(base_dir, 'Validation')
test_dir = os.path.join(base_dir, 'Test')

# Directory with our training fake pictures
train_fake_dir = os.path.join(train_dir, 'fake')

# Directory with our training real pictures
train_real_dir = os.path.join(train_dir, 'real')

# Directory with our training realMask pictures
train_realMask_dir = os.path.join(train_dir, 'realMask')

# Directory with our training fake pictures
validation_fake_dir = os.path.join(train_dir, 'fake')

# Directory with our validation real pictures
validation_real_dir = os.path.join(validation_dir, 'real')

# Directory with our validation realMask pictures
validation_realMask_dir = os.path.join(validation_dir, 'realMask')

##model
from keras.layers import *
from keras.models import Sequential
from keras.applications import VGG16
from keras.optimizers import Adam
opt = Adam(lr=0.001)

VGG =  VGG16(weights = 'imagenet', include_top = False, input_shape = (160, 160, 3))

VGG.trainable = False

model = Sequential()
model.add(VGG)

model.add(Flatten())

model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
#model.add(Dense(64, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))
print(model.summary())

model.compile(optimizer=opt,loss = 'categorical_crossentropy', metrics = ['accuracy'])


##call back
from keras.callbacks import ReduceLROnPlateau
LR_function=ReduceLROnPlateau(monitor='val_accuracy', patience=3, # 3 epochs modify LR 
                             verbose=1,
                             factor=0.5,# LR reduce 0.5
                             min_lr=0.00001 #mininum learning rate 0.00001
              )


from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt 
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

#batch_size= 20 to 32
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=32, target_size=(160,160), class_mode='categorical')
val_generator = val_datagen.flow_from_directory(validation_dir, batch_size=32, target_size=(160,160), class_mode='categorical')


##training
import time
begin_time = time.time()

history = model.fit(train_generator, steps_per_epoch=100, epochs=50, verbose=1, validation_data=val_generator, callbacks=[LR_function])

end_time = time.time()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()

print("Training time: ", end_time - begin_time)

#===================#
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(160, 160), batch_size=32, class_mode='categorical',shuffle=False)

#Confution Matrix and Classification Report
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
cm = confusion_matrix(test_generator.classes, y_pred)
print(cm)
print('Classification Report')
target_names = ['Fake','Real','Withmask']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))

plt.matshow(cm)
plt.colorbar()
plt.xlabel('Pred')
plt.ylabel('Test')
plt.xticks(np.arange(cm.shape[1]),target_names)
plt.yticks(np.arange(cm.shape[1]),target_names)
plt.show()
