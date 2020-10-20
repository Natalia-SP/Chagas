#!/usr/bin/env python
# coding: utf-8

# # Segmentación de amastigotes T.Cruzi

# ### Se importan las librerías necesarias

# In[1]:


import tensorflow as tf
import os
import numpy as np
import random
import cv2
import glob
import tifffile as tiff
import imagecodecs

from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import staintools
import re


# ### Ajustes iniciales:
# #### Tamaño de las imágenes
# #### Directorios
# #### Matrices donde guardaremos las imágenes
# #### Organizamos las imágenes en orden natural

# In[2]:


# Semilla random para verificar los resultados
seed = 42
np.random.seed = seed

# Dimensiones de las imágenes
IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3

# Funciones para organizar las listas de manera natural
def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)',text) ]

# Directorios raíz
XTRAIN_PATH = sorted(glob.glob('/almac/alfonso_tv/X_train/*.tif'), key=natural_keys)
YTRAIN_PATH = sorted(glob.glob('/almac/alfonso_tv/Y_train/*.tif'), key=natural_keys)
#TEST_PATH = sorted(glob.glob('/almac/alfonso_tv/X_train/*.tif'), key=natural_keys)

# Número de imágenes con las que se trabajarán (Opcional)
#XTRAIN_PATH = XTRAIN_PATH[500:1000]
#YTRAIN_PATH = YTRAIN_PATH[500:1000]
#TEST_PATH = TEST_PATH[:3500]
 
# Matrices donde se guardarán las imágenes (Opcional)
#X_train = np.zeros((len(XTRAIN_PATH), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
#Y_train = np.zeros((len(YTRAIN_PATH), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
#X_test = np.zeros((len(TEST_PATH), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)


# ## Cargamos las imágenes

# In[3]:


x_train = np.zeros((len(XTRAIN_PATH), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype = np.uint8)
for n,item in tqdm(enumerate(XTRAIN_PATH), total=len(XTRAIN_PATH)):
    img = tiff.imread(item)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    x_train[n]=img
print(len(x_train))


# In[30]:


tiff.imshow(x_train[2999])


# ## Cargamos las máscaras

# In[4]:


y_train = np.zeros((len(YTRAIN_PATH), IMG_HEIGHT, IMG_WIDTH, 1), dtype = np.bool)
for n,item in tqdm(enumerate(YTRAIN_PATH), total=len(YTRAIN_PATH)):
    img = cv2.imread(item)[:,:,:1]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    y_train[n]=(img)
print(len(y_train))


# In[50]:


tiff.imshow(y_train[0])


# ## Detección de casos positivos

# In[5]:


# Creamos un diccionario con los nombres e imágenes cargadas
nombre_e_imagen_y = dict(zip(YTRAIN_PATH, y_train))
nombre_e_imagen_x = dict(zip(XTRAIN_PATH, x_train)) 


# In[6]:


print(nombre_e_imagen_y[YTRAIN_PATH[0]])


# In[7]:


P_num = 0 # Numero de casos positivos
Y_Positives = [] # Lista de casos positivos
N_num = 0 # Numero de casos negativos
Y_Negatives = [] # Lista de casos negativos

for name in tqdm(nombre_e_imagen_y.keys()):
    if True in nombre_e_imagen_y[name]: # Verifica si el promedio de los valores de la máscara == 0 para verificar su clase
        Y_Positives.append(name)
        P_num += 1
    else:
        Y_Negatives.append(name)
        N_num += 1

print("El número de casos positivos es: ", P_num)
print("El número de casos negativos es:  ", N_num)


# In[8]:


print(Y_Negatives[0:4])


# ## Cambiar ruta de Y_train a X_train

# In[9]:


X_Negatives = []
for item in Y_Negatives:
    item = item.split('/')
    #print(item)
    item[3] = 'X_train'
    #print(item)
    item = '/'.join(item)
    #print(item)
    X_Negatives.append(item)
print(X_Negatives[0:4])


# In[10]:


# Se eliminan los casos negativos presentes en el diccionario de xtrain
for negative in X_Negatives:
    if negative in nombre_e_imagen_x:
        nombre_e_imagen_x.pop(negative)
        
for negative in Y_Negatives:
    if negative in nombre_e_imagen_y:
        nombre_e_imagen_y.pop(negative)


# In[11]:


# Se verifica que no haya quedado alguno
for negative in X_Negatives:
    if negative in nombre_e_imagen_x:
        print(True)
    if negative == X_Negatives[-1]:
        print('Ya no hay casos negativos en xtrain')

for negative in Y_Negatives:
    if negative in nombre_e_imagen_y:
        print(True)
    if negative == Y_Negatives[-1]:
        print('Ya no hay casos negativos en ytrain')


# In[12]:


# Variables limpias de casos negativos, deben coincidir X y Y
X = list(nombre_e_imagen_x.values())
print(len(X))
Y = list(nombre_e_imagen_y.values())
print(len(Y))


# In[27]:


print(X[0].shape)


# In[13]:


tiff.imshow(X[0])


# In[27]:


tiff.imshow(Y[0])


# ## U-Net

# In[14]:


# Design our model architecture here
def keras_model(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS):
    # U-NET Model
    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs) # Normalización de los pixeles

    #Contraction path
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #Expansive path 
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    unet = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    unet.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.MeanIoU(num_classes=2)]) # MeanIoU
    unet.summary() # Table to verify model components
    
    return unet


# In[15]:


unet = keras_model(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)


# In[16]:


print(unet)


# ## Cross-validation

# In[17]:


from sklearn.model_selection import KFold


# In[18]:


callbacks = [
        tf.keras.callbacks.EarlyStopping(patience = 10, monitor ='val_loss'),
        tf.keras.callbacks.TensorBoard(log_dir='logs_final'),
        tf.keras.callbacks.ModelCheckpoint('Model_for_Chagas.h5', verbose=1, save_best_only='True')]


# In[19]:


def get_score(model, X_train, X_test, Y_train, Y_test):
    model.fit(X_train, Y_train, batch_size=1, epochs=50,validation_split=0.1, callbacks=callbacks)
    return model.evaluate(X_test, Y_test)


# ### Entrenamiento

# In[20]:


scores_unet = []

kf = KFold(5)
kf.split(X,Y)
#print(len(X[0]))
for train, test in kf.split(X, Y):
    X_train, X_test, Y_train, Y_test = np.array(X)[train], np.array(X)[test], np.array(Y)[train], np.array(Y)[test]
    scores_unet.append(get_score(unet,X_train,X_test,Y_train,Y_test))


# ## Resultados y predicciones

# In[22]:


for scores in scores_unet:
    print('Val loss: {}'.format(scores[0]))
    print('Binary Accuracy: {}'.format(scores[1]))
    print('MeanIoU: {}'.format(scores[2]))


# In[23]:


Val_loss = 0
Binary_Acc = 0
MeanIoU = 0
for scores in scores_unet:
    if scores[0] > Val_loss:
        Val_loss = scores[0]
    if scores[1] > Binary_Acc:
        Binary_Acc = scores[1]
    if scores[2] > MeanIoU:
        MeanIoU = scores[2]
print('Los mejores resultados fueron:\nVal loss de ',Val_loss,'\nBinary Accuracy de ',Binary_Acc,'\nMeanIoU de ',MeanIoU)


# In[24]:


pVal = 0
pBin = 0
pMean = 0
for scores in scores_unet:
    pVal += scores[0]
    pBin += scores[1]
    pMean += scores[2]
pVal = pVal/5
pBin = pBin/5
pMean = pMean/5
print('El promedio de cada uno fue:\nVal loss de ',pVal,'\nBinary Accuracy de ',pBin,'\nMeanIoU de ',pMean)


# In[37]:


preds_train = unet.predict(X_train[:int(X_train.shape[0]*0.9)], verbose=1)
preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val = unet.predict(X_train[int(X_train.shape[0]*0.9):], verbose=1)
preds_val_t = (preds_val > 0.5).astype(np.uint8)


# In[46]:


# Perform a sanity check on some random training samples
fig=plt.figure(figsize=(25,15))
ix = 2
fig.add_subplot(1,3,1)
plt.imshow(X_train[ix])
fig.add_subplot(1,3,2)
plt.imshow(np.squeeze(Y_train[ix]))
fig.add_subplot(1,3,3)
plt.imshow(np.squeeze(preds_train_t[ix]))
plt.show()

# Perform a sanity check on some random validation samples
fig=plt.figure(figsize=(25,15))
ix2 = 3
fig.add_subplot(1,3,1)
plt.imshow(X_train[int(X_train.shape[0]*0.9):][ix2])
fig.add_subplot(1,3,2)
plt.imshow(np.squeeze(Y_train[int(Y_train.shape[0]*0.9):][ix2]))
fig.add_subplot(1,3,3)
plt.imshow(np.squeeze(preds_val_t[ix2]))
plt.show()


# ## Métricas.py

# In[47]:


import Metricas as m
n = 3
validation_mask = m.get_confusion_matrix_overlaid_mask(X_train[n], Y_train[n], preds_train_t[n])

plt.figure(figsize=(10,10))
plt.imshow(validation_mask)
plt.axis('off')
plt.title("Confusion Matrix Mask's Overlay")
plt.show()


# In[60]:


score_train = unet.evaluate(X_train,Y_train, steps=10)
print('Acctrain =', score_train[1])


# In[59]:


print(Y_train.shape)
print(preds_train.shape)


# In[62]:


print('Metricas entrenamiento completo')
m.metricas(Y_train, preds_train_t)


# In[64]:


objects = ['Nido','Fondo']
matrix = m.matriz_confusion(Y_train,preds_train_t)
m.plot_confusion_matrix(matrix, classes=objects, normalize=False)


# In[65]:


m.plot_confusion_matrix(matrix, classes=objects, normalize=True)


# In[67]:


df = m.conteo(Y_train, preds_train_t)
df.to_csv('Res_final.csv')


# In[68]:


df


# ### Tensorboard

# In[2]:


#%load_ext tensorboard


# In[3]:


#tensorboard --logdir=/ --host localhost --port 8501


# In[ ]:


import matplotlib.pyplot as plt

epoch_max = np.argmax(results.history['val_loss'])
plt.figure(num=None, figsize=(4, 3))
plt.plot(results.history['loss'], label='training')
plt.plot(results.history['val_loss'], label='validation')
plt.legend(loc='lower right')
plt.plot(epoch_max, results.history['val_loss'][epoch_max],'*')
plt.title('Loss')
plt.show()


# In[ ]:




