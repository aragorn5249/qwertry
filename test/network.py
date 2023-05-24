import sys
# !conda install --yes --prefix {sys.prefix} -c anaconda pydotimport matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import reset_default_graph
import cv2
import random
import urllib
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Activation
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import plot_model
import os
number_of_examples = len(all_data)
let_know = int(number_of_examples / 10)

for idx, example in enumerate(all_data):
    if (idx+1)%let_know == 0:
        print(f'processing {idx + 1}')
    resized_down = cv2.resize(example['X'], (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_LINEAR)
    
    all_data_processed.append({'X': np.array(resized_down), 'Y': label_dict[example['Y']]})
    classNames = {value:key for key, value in label_dict.items()}
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(18,12))
axes = axes.flatten()
number_of_examples = len(all_data_processed)
for idx, axis in enumerate(axes):
    idx = random.randint(0, number_of_examples)
    example = all_data_processed[idx]
    axis.axis('off')
    axis.set_title(f"{classNames[example['Y']]}")
    axis.imshow(example['X'])
    random.seed(42)
    random.seed(42)
    X = np.array([example['X'] for example in all_data_processed])
Y = np.array([example['Y'] for example in all_data_processed])
print(f"{bcolors.BOLD}Rozmiar cech (X): {X.shape}, rozmiar flagi/indykatora klasy (Y): {Y.shape}{bcolors.ENDC}")
split_ratio = 0.6
split_idx = int(len(Y)*split_ratio)
ohe = OneHotEncoder()
Y = ohe.fit_transform(Y.reshape(-1,1))
# proszę zadeklarować poniższe zmienne korzystając z macierzy X i Y (podpowiedź: slice):
X_train = None
X_test = None
Y_train = None
Y_test = None
X_train = X[:split_idx,:,:,:]
X_test = X[split_idx:,:,:,:]
Y_train = Y[:split_idx]
Y_test = Y[split_idx:]
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
Y_train = Y_train.todense()
Y_test = Y_test.todense()

print ("X_train_flat shape: " + str(X_train_flat.shape))
print ("X_test_flat shape: " + str(X_test_flat.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("Y_test shape: " + str(Y_test.shape))
print ("X_train shape: " + str(X_train.shape))
X_train_flat = X_train_flat / 255.from utils import decrypt_pickle, SelectFilesButton, bcolors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import reset_default_graph
import cv2
import random
import urllib
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Activation
from tensorflow.keras.optimizers import SGD, Adam
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import plot_model
import os

from utils import decrypt_pickle, SelectFilesButton, bcolors
all_data = decrypt_pickle('catvsnotcat.pkl.aes', password="WSB_ML")
IMG_SIZE = 64
label_dict = {'cat': 1, 'not-cat': 0}
all_data_processed = []
number_of_examples = len(all_data)
let_know = int(number_of_examples / 10)

for idx, example in enumerate(all_data):
    if (idx+1)%let_know == 0:
        print(f'processing {idx + 1}')
    resized_down = cv2.resize(example['X'], (IMG_SIZE, IMG_SIZE), interpolation= cv2.INTER_LINEAR)
    
    all_data_processed.append({'X': np.array(resized_down), 'Y': label_dict[example['Y']]})
    classNames = {value:key for key, value in label_dict.items()}
fig, axes = plt.subplots(nrows=2,ncols=3, figsize=(18,12))
axes = axes.flatten()
number_of_examples = len(all_data_processed)
for idx, axis in enumerate(axes):
    idx = random.randint(0, number_of_examples)
    example = all_data_processed[idx]
    axis.axis('off')
    axis.set_title(f"{classNames[example['Y']]}")
    axis.imshow(example['X'])
    random.seed(42)
    random.shuffle(all_data_processed)
    X = np.array([example['X'] for example in all_data_processed])
Y = np.array([example['Y'] for example in all_data_processed])
print(f"{bcolors.BOLD}Rozmiar cech (X): {X.shape}, rozmiar flagi/indykatora klasy (Y): {Y.shape}{bcolors.ENDC}")
split_ratio = 0.6
split_idx = int(len(Y)*split_ratio)
ohe = OneHotEncoder()
Y = ohe.fit_transform(Y.reshape(-1,1))
# proszę zadeklarować poniższe zmienne korzystając z macierzy X i Y (podpowiedź: slice):
X_train = None
X_test = None
Y_train = None
Y_test = None
X_train = X[:split_idx,:,:,:]
X_test = X[split_idx:,:,:,:]
Y_train = Y[:split_idx]
Y_test = Y[split_idx:]
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
Y_train = Y_train.todense()
Y_test = Y_test.todense()

print ("X_train_flat shape: " + str(X_train_flat.shape))
print ("X_test_flat shape: " + str(X_test_flat.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("Y_test shape: " + str(Y_test.shape))
print ("X_test shape: " + str(X_test.shape))
X_train_flat = X_train_flat / 255.
X_test_flat = X_test_flat / 255.
reset_default_graph()
try:
    #poniżej wywołanie Sequential z keras
    model = Sequential(name='Simple_model')

    #proszę uzupełnić liczbę neuronów w warstwie wartością 4096, wymiarowością wejścia równej długości rozwiniętego wektora ze 
    #zdjęcia, uniform (rozkład równomierny) jako funkcję inicjalizacji kerneli [string] i relu jako funkcję aktywacji [string]
    model.add(Dense(None, input_dim=None, kernel_initializer="please_fill", activation="please_fill"))

    #uzupełnić liczbę neuronów w warstwie na 512, funkcję aktywacji na relu i inicjalizację kerneli (wag, nie biasów) na uniform
    model.add(Dense(None, activation="please_fill", kernel_initializer="please_fill"))

    #uzupełnić liczbę neuronów na liczbę wyjść (czyli 2 klasy)
    model.add(Dense(None))

    #uzupełnić aktywację typem softmax
    model.add(Activation("please_fill"))
    
except:
    print(f'{bcolors.BOLD}{bcolors.FAIL}Proszę poprawnie uzupełnić powyższe miejsca gdzie występuje None lub "please_fill"{bcolors.ENDC}')
    # rozwiązanie
reset_default_graph()
model = Sequential(name='Simple_model')
model.add(Dense(4096, input_dim=12288, kernel_initializer="uniform", activation="relu"))
model.add(Dense(512, activation="relu", kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))
model.summary()
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
log_dir = os.path.join('logs','model', datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
try:    
    #uzupełnić funkcję kosztu na binary_crossentropy, jako metodę optymalizację - Adam(), jako metrykę accuracy
    model.compile(loss="please_fill", optimizer=Adam(), metrics=["please_fill"])
    
    #proszę wpierw podać treningowe dane X (jeden wektor), potem treningowe dane Y, ustawić liczbę epok na 50, wielkość batcha
    #na 64, zaś jako dane walidacyjne trzeba podać tupla (w nawiasach okrągłych) z tesotwymi danymi X i testowymi danymi Y
    model.fit(None, None, epochs=None, batch_size=None, validation_data=(X_test_flat, Y_test), verbose=1, callbacks=[tensorboard_callback])
except:
    print(f'{bcolors.BOLD}{bcolors.FAIL}Proszę poprawnie uzupełnić powyższe miejsca gdzie występuje None lub "please_fill"{bcolors.ENDC}')
    #rozwiązanie
log_dir = os.path.join('logs','model', datetime.now().strftime('%Y%m%d-%H%M%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
history_model = model.fit(X_train_flat, Y_train, epochs=50, batch_size=64, verbose=1, validation_data=(X_test_flat, Y_test), callbacks=[tensorboard_callback])
plt.plot(history_model.history['accuracy'])
plt.plot(history_model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.ylim(ymin=0, ymax=1)
plt.show()
plt.plot(history_model.history['loss'])
plt.plot(history_model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
reset_default_graph()
try:
    
    model_reg = Sequential(name='Model_with_Dropout')
    model_reg.add(Dense(4096, input_dim=12288, init="uniform", activation="relu"))
    
    #proszę podać prawdopodobieństwo ucięcia połączenia - 0.5 czyli 50%
    model_reg.add(Dropout(rate=None))
    model_reg.add(Dense(512, activation="relu", kernel_initializer="uniform"))
    
    #proszę podać prawdopodobieństwo ucięcia połączenia - 0.6 czyli 60%
    model_reg.add(Dropout(rate=None))

    model_reg.add(Dense(2))
    model_reg.add(Activation("softmax"))
except:
    print(f'{bcolors.BOLD}{bcolors.FAIL}Proszę poprawnie uzupełnić powyższe miejsca gdzie występuje None lub "please_fill"{bcolors.ENDC}')
    #rozwiązanie
reset_default_graph()
model_reg = Sequential(name='Model_with_Dropout')
model_reg.add(Dense(4096, input_dim=12288, kernel_initializer="uniform", activation="relu"))
model_reg.add(Dropout(rate=0.5))
model_reg.add(Dense(512, activation="relu", kernel_initializer="uniform"))
model_reg.add(Dropout(rate=0.6))
model_reg.add(Dense(2))
model_reg.add(Activation("softmax"))
model_reg.summary()
plot_model(model_reg, show_shapes=True, show_layer_names=True, to_file='model_reg.png')
log_dir = os.path.join('logs','regularized_model', datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#proszę zwrócić uwagę, że używamy tu innego optimizera - sgd i podajemy krok uczenia
sgd = SGD(lr=0.02)
model_reg.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
history_model_reg = model_reg.fit(X_train_flat, Y_train, epochs=50, batch_size=64, verbose=2, validation_data=[X_test_flat, Y_test], callbacks=[tensorboard_callback])
fig, axes = plt.subplots(nrows=1 ,ncols=2, figsize=(20,7))

axes[0].plot(history_model_reg.history['accuracy'])
axes[0].plot(history_model_reg.history['val_accuracy'])
axes[0].set_title('regularized model accuracy')
axes[0].set_ylabel('accuracy')
axes[0].set_xlabel('epoch')
axes[0].legend(['train', 'test'], loc='upper left')
axes[0].set_ylim(ymin=0, ymax=1)

axes[1].plot(history_model.history['accuracy'])
axes[1].plot(history_model.history['val_accuracy'])
axes[1].set_title('simple model accuracy')
axes[1].set_ylabel('accuracy')
axes[1].set_xlabel('epoch')
axes[1].legend(['train', 'test'], loc='upper left')
axes[1].set_ylim(ymin=0, ymax=1)
plt.show()
fig, axes = plt.subplots(nrows=1 ,ncols=2, figsize=(20,7))

y_max = max(max(history_model_reg.history['loss']), max(history_model_reg.history['val_loss']), max(history_model.history['loss']), max(history_model.history['val_loss']))

axes[0].plot(history_model_reg.history['loss'])
axes[0].plot(history_model_reg.history['val_loss'])
axes[0].set_title('regularized model loss')
axes[0].set_ylabel('loss')
axes[0].set_xlabel('epoch')
axes[0].legend(['train', 'test'], loc='upper left')
axes[0].set_ylim(ymin=0, ymax=y_max)

axes[1].plot(history_model.history['loss'])
axes[1].plot(history_model.history['val_loss'])
axes[1].set_title('simple model loss')
axes[1].set_ylabel('loss')
axes[1].set_xlabel('epoch')
axes[1].legend(['train', 'test'], loc='upper left')
axes[1].set_ylim(ymin=0, ymax=y_max)

plt.show()
print ("X_train shape: " + str(X_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("Y_test shape: " + str(Y_test.shape))
X_train = X_train / 255.
X_test = X_test / 255.
reset_default_graph()
try:
    model_cnn = Sequential(name="CNN_model")

    #proszę zadeklarować liczbę filtrów, pierwszy argument, na 16, rozmiar filtra 3 na 3 czyli tuple (3, 3), funkcję aktywacji
    #na relu, inicjalizację kernela (wagi) na he_uniform, padding na same ("ten sam") i odpowiedni rozmiar wejścia (X_train)
    model_cnn.add(Conv2D(None, (None, None), activation='please_fill', kernel_initializer='please_fill', padding='please_fill', input_shape=(None, None, None)))
    #w MaxPooling proszę ustawić rozmiar filtra na 2 na 2 czyli tuple (2, 2)
    model_cnn.add(MaxPooling2D((None, None)))
    
    #proszę zadeklarować liczbę filtrów, pierwszy argument, na 32, rozmiar filtra 3 na 3 czyli tuple (3, 3), funkcję aktywacji
    #na relu, inicjalizację kernela (wagi) na he_uniform, padding na same ("ten sam")
    model_cnn.add(Conv2D(None, (None, None), activation='please_fill', kernel_initializer='please_fill', padding='please_fill'))
    #w MaxPooling proszę ustawić rozmiar filtra na 2 na 2 czyli tuple (2, 2)
    model_cnn.add(MaxPooling2D((2, 2)))
    
    #proszę zadeklarować liczbę filtrów, pierwszy argument, na 64, rozmiar filtra 3 na 3 czyli tuple (3, 3), funkcję aktywacji
    #na relu, inicjalizację kernela (wagi) na he_uniform, padding na same ("ten sam")
    model_cnn.add(Conv2D(None, (None, None), activation='please_fill', kernel_initializer='please_fill', padding='please_fill'))
    #w MaxPooling proszę ustawić rozmiar filtra na 2 na 2 czyli tuple (2, 2)
    model_cnn.add(MaxPooling2D((None, None)))
    
    #Poniżej rzutujemy wszystko na jeden długi wektor i postępujemy jak ze zwykłą siecią pełnych połączeń
    model_cnn.add(Flatten())
    model_cnn.add(Dropout(rate=0.3))
    model_cnn.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
    model_cnn.add(Dropout(rate=0.3))
    model_cnn.add(Dense(2))
    model_cnn.add(Activation('softmax'))
except:
    #TypeError ValueErro
    print(f'{bcolors.BOLD}{bcolors.FAIL}Proszę poprawnie uzupełnić powyższe miejsca gdzie występuje None lub "please_fill"{bcolors.ENDC}')
    #rozwiązanie
reset_default_graph()
model_cnn = Sequential(name="CNN_model")

model_cnn.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(64, 64, 3)))
model_cnn.add(MaxPooling2D((2, 2)))
model_cnn.add(Dropout(0.3))

model_cnn.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_cnn.add(MaxPooling2D((2, 2)))
model_cnn.add(Dropout(0.3))


model_cnn.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model_cnn.add(MaxPooling2D((2, 2)))
model_cnn.add(Dropout(0.3))


model_cnn.add(Flatten())
model_cnn.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
model_cnn.add(Dropout(rate=0.3))
model_cnn.add(Dense(2))
model_cnn.add(Activation('softmax'))

model_cnn.summary()
log_dir = os.path.join('logs','cnn_model', datetime.now().strftime('%d_%m_%Y-%H_%M_%S'))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
sgd = SGD(0.02)
model_cnn.compile(loss="binary_crossentropy", optimizer=Adam(), metrics=["accuracy"])
history_model_cnn = model_cnn.fit(X_train, Y_train, epochs=50, batch_size=128, verbose=2, validation_data=[X_test, Y_test], callbacks=[tensorboard_callback])
plt.plot(history_model_cnn.history['accuracy'])
plt.plot(history_model_cnn.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.ylim(ymin=0, ymax=1)
plt.show()
plt.plot(history_model_cnn.history['loss'])
plt.plot(history_model_cnn.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
