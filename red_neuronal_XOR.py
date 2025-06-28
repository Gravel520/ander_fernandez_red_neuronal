'''
Script en Python.
Crearemos una red neuronal artificial muy sencilla en Python con Keras
y Tensorflow para comprender su uso. Implementaremos la compuerta XOR.

Las compuertas XOR.
Tenemos dos entradas binarias (1 ó 0) y la salida será 1 sólo si una de
las entradas es verdadera (1) y la otra falsa (0).
Es decir que de cuatro combinaciones posibles, sólo dos tienen salida 1
y las otras dos serán 0, como vemos aquí:
    * XOR(0,0) = 0
    * XOR(0,1) = 1
    * XOR(1,0) = 1
    * XOR(1,1) = 0

Crearemos una red neuronal con datos de entrada las 4 combinaciones XOR
y sus 4 salidas ordenadas.
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Cargamos las 4 combinaciones de las compuertas XOR.
training_data = np.array([[0,0], [0,1], [1,0], [1,1]], 'float32')

# Cargamos las salidas en el mismo orden que se obtienen.
target_data = np.array([[0], [1], [1], [0]])

# Creamos el modelo.
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=1000, verbose=False)

# Evaluamos el modelo.
scores = model.evaluate(training_data, target_data)

print('\n%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
print(model.predict(training_data).round())
