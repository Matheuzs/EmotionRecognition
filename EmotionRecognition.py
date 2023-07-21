import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import cv2 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, model_from_json
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


print("OpenCV Version:", cv2.__version__)
print("TensorFlow Version:", tf.__version__)



#Criando a função para normalizar 

def normalizar(x):
    x = x.astype('float32')
    x = x/255.0
    return x 

faces = normalizar(faces)

#Precisamos fazer o ohe das emoções para termos um vetor com 0 e 1 (1 para a emoção)
emocoes = pd.get_dummies(data['emotion']).values

x_treino, x_teste,y_treino, y_teste = train_test_split(faces, emocoes, test_size = 0.1, random_state = 42)
x_treino, x_val, y_treino, y_val = train_test_split(x_treino, y_treino, test_size = 0.1, random_state = 41)


#Montando a arquitetura rede neural
n_features = 64
n_labels = 7
batch_size =64
epochs = 100
width, height = 48,48

model = Sequential()

model.add(Conv2D(n_features, kernel_size=(3,3), activation='relu',
                 input_shape=(width, height, 1), data_format = 'channels_last',
                 kernel_regularizer = l2(0.01)))
model.add(Conv2D(n_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*n_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*n_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*n_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*n_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(2*2*2*n_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(2*2*2*n_features, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(2*2*2*n_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*2*n_features, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(2*n_features, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(n_labels, activation = 'softmax')) #numero de saida precisa ser igual ao numero de classes

model.summary()



#Vamos fazer a compilação do modelo 
model.compile(loss='categorical_crossentropy', 
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy']
              )

arquivo_modelo = 'modelo_01_expressoes.h5'
arquivo_modelo_json = 'modelo_01_expressoes.json'

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)


early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')
checkpointer = ModelCheckpoint(arquivo_modelo, monitor = 'val_loss', verbose=1, save_best_only=True)




model_json = model.to_json()
with open(arquivo_modelo_json, 'w') as json_file:
    json_file.write(model_json)





#Treinanmento da nossa rede neural

history = model.fit(np.array(x_treino),
                    np.array(y_treino),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data = (np.array(x_val), np.array(y_val)),
                    shuffle=True,
                    callbacks=[lr_reducer, early_stopper, checkpointer])
