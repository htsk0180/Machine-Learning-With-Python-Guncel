# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 20:57:50 2021

@author: 104863
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# ilkleme
classifier = Sequential()

# convolution ve pooling başlangıç #
# Adım 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu')) # 3 rgb ' yu temsil eder. 64,64 ise 64x64 lük matris.

# Adım 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# 2. convolution katmanı
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# convolution ve pooling bitiş #

# Adım 3 - Flattening
classifier.add(Flatten())

# Adım 4 - YSA
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# CNN ve resimler
from keras.preprocessing.image import ImageDataGenerator # bu özellik resimleri tek tek almaya yarar.

train_datagen = ImageDataGenerator(rescale = 1./255, # resim işleme ile ilgili filtreler.
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255, # resim işleme ile ilgili filtreler.
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

training_set = train_datagen.flow_from_directory('veriler/training_set',
                                                 target_size = (64, 64), #64x64 lük matris.
                                                 batch_size = 1,
                                                 class_mode = 'binary') #64x64 lük matris.

test_set = test_datagen.flow_from_directory('veriler/test_set',
                                            target_size = (64, 64), #64x64 lük matris.
                                            batch_size = 1,
                                            class_mode = 'binary')
# ysa eğitimi.
classifier.fit_generator(training_set,
                         epochs=1,# kaç epoch
                         validation_data = test_set
                         )

import numpy as np
import pandas as pd

test_set.reset()
pred=classifier.predict_generator(test_set,verbose=1)

# yuvarlama
pred[pred > .5] = 1
pred[pred <= .5] = 0
print('prediction gecti')

test_labels = []
for i in range(0,int(203)):
    test_labels.extend(np.array(test_set[i][1]))
    
print('test_labels')
print(test_labels)

dosyaisimleri = test_set.filenames

sonuc = pd.DataFrame()
sonuc['dosyaisimleri']= dosyaisimleri
sonuc['tahminler'] = pred
sonuc['test'] = test_labels   

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, pred)
print (cm)



