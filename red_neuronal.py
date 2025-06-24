'''
Script en Python.
Las redes neuronales están compuestas de neuronas, que a su vez
se agrupan en capas: cada neurona de cada capa está conectada
con todas las neuronas de la capa anterior. En ada neurona,
se realiarán una serie de operaciones las cuales, al optimizar,
conseguiremos que nuestra red aprenda.
Lo primero es programar las capas de neuronas.

CREANDO LAS CAPAS DE NEURONAS.
    Como funciona una capa de neuronas:
    1 Una capa recibe valores, llamados inputs. En la primera
    capa, esos valores vendrán definidos por los datos de 
    entrada, mientras que el resto de capas recibirán el 
    resultado de la capa anterior.

    2 Se realiza una suma ponderada de todos los valores de
    entrada. Para hacer esa ponderación necesitamos una matriz
    de pesos, conocida como W. Tendrá tantas filas como neuronas
    la capa anterior y tantas columnas como neuronas tiene esa
    capa.

    3 Al resultado de la suma ponderada anterior se le sumará
    otro parámetro, conocido como bias, o b. En este caso cada
    neurona tiene su propio bias, por lo que las dimensiones
    del vector bias será una columna y tantas filas como
    neuronas tiene esa capa.

    4 La función de activación. Para evitar que toda la red
    neuronal se pueda reducir a una simple regresión lineal, al
    resultado de la suma del bias a la suma ponderada se le
    aplica una función de activación. Este resultado será el de
    la neurona.
'''

'''
Para poder montar una capa de una red neuronal solo necesitamos
saber el número de neuronas en la capa y el número de neuronas
de la capa anterior. Con eso podremos crear tanto W como b.
Para crear esta estructura vamos a crear una clase, que llamaremos
capa. Además, vamos a inicializar los parámetros (b y W) con
datos aleatorios.
'''

import numpy as np
from scipy import stats

class capa():
    def __init__(self, n_neuronas_capa_anterior, n_neuronas, funcion_act):
        self.funcion_act = funcion_act
        self.b = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=n_neuronas).reshape(1, n_neuronas), 3)
        self.W = np.round(stats.truncnorm.rvs(-1, 1, loc=0, scale=1, size=n_neuronas * n_neuronas_capa_anterior).reshape(n_neuronas_capa_anterior, n_neuronas), 3)

'''
Script en Python.
Función de activación: Función Sigmoide
Esta función básicamente recibe un valor x y devuelve un valor
entre 0 y 1. Esto hace que sea una función muy interesante,
ya que indica la probabilidad de un estado.
'''

import math
import matplotlib.pyplot as plt

sigmoid = (
    lambda x:1 / (1 + np.exp(-x)),
    lambda x:x * (1 - x)
)

rango = np.linspace(-10, 10).reshape([50, 1])
datos_sigmoide = sigmoid[0](rango)
datos_sigmoide_derivada = sigmoid[1](rango)

# Creamos los gráficos
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
axes[0].plot(rango, datos_sigmoide)
axes[1].plot(rango, datos_sigmoide_derivada)
#fig.tight_layout()
plt.show()

'''
Script en Python.
Función de activación: Función ReLu.
Es muy simple: para valores negativos, la función devuelve 
cero. Para valores positivos, la función devuelve el mismo
valor.
'''

def derivada_relu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

relu = (
    lambda x: x * (x > 0),
    lambda x:derivada_relu(x)
)

datos_relu = relu[0](rango)
datos_relu_derivada = relu[1](rango)

# Volvemos a definir rango que ha sido cambiado
rango = np.linspace(-10, 10).reshape([50, 1])

# Creamos los gráficos
plt.cla()
fig, axes = plt.subplots(nrows=1, ncols=2, figsize= (15, 5))
axes[0].plot(rango, datos_relu[:,0])
axes[1].plot(rango, datos_relu_derivada[:,0])
#plt.close('all')
plt.show()

'''
Programando una red neuronal en Python.
Simplemente tendremos que indicar tres cosas: el número de capas
que tiene la red, el número de neuronas en cada capa y la función
de activación que se usará en cada una de las capas.

En este caso, usaremos la red neuronal para solucionar un 
problema de clasificación de dos clases, para lo cual usaremos
una red pequeña, de 4 capas que se compondrá de:
    * Una capa de entrada con dos neuronas, ya que usaremos dos
        variables.
    * Dos capas ocultas, una de 4 neuronas y otra de 8.
    * Una capa de salida, con una única neurona que predecirá
        la clase.

Asimismo, tenemos que definir qué función de activación se 
usará en cada capa. En nuestro caso, usaremos la función ReLu en
todas las capas menos en la última, en la cual usaremos la 
función sigmoide.

Por otro lado, Python no permite crear una lista de funciones. Por
eso, hemos definido las funciones relu y sigmoid como funciones
ocultas usando lambda.
'''

# Número de neuronas en cada capa.
# El primer valor es el número de columnas de la capa de entrada.
neuronas = [2, 4, 8, 1]

# Funciones de activación usadas en cada capa.
funciones_activacion = [relu, relu, sigmoid]

'''
Con todo esto, ya podemos crear la estructura de nuestra red
neuronal programada en Python. Lo haremos de forma iterativa e
iremos guardando esta estructura en un nuevo objeto, llamado
'red_neuronal'.
'''

red_neuronal = []

for paso in range(len(neuronas) -1):
    x = capa(neuronas[paso], neuronas[paso+1], funciones_activacion[paso])
    red_neuronal.append(x)

print(f'Estructura de la red neuronal:\n{red_neuronal}')

'''
Con esto ya tenemos la estructura de nuestra red neuronal. Ahora
solo quedarían dos pasos más: por un lado, conectar la red para
que nos de una predicción y un error y, por el otro lado, ir
propagando ese error hacia atrás para ir entrenando a nuestra 
red neuronal.
'''

'''
Haciendo que nuestra red neuronal prediga.
Lo único que tenemos que hacer es definir los cálculos que tiene
que seguir, que son 3: multiplicar los valores de entrada por la
matriz de pesos W, sumar el parámetro bias b, y por último, 
aplicar la función de activación.
'''

# Veamos el ejemplo de la primera capa. Para multiplicar los 
#   valores de entrada por la matriz de pesos tenemos que hacer
#   una multiplicación matricial.
X = np.round(np.random.randn(20, 2), 3) # Ejemplo de vector de entrada
z = X @ red_neuronal[0].W
print(z[:10,:], X.shape, z.shape)

# Ahora, hay que sumar el parámetro bias (b) al resultado
#   anterior de z.
z = z + red_neuronal[0].b
print(z[:5,:])

# Finalmente habría que aplicar la función de activación de
#   esa capa.
a = red_neuronal[0].funcion_act[0](z)
print(f'{a[:5,:]}\n\n')

'''
Con esto, tendríamos el resultado de la primera capa, que a su
vez es la entrada para la segunda capa y así hasta la última.
Por tanto, queda bastante claro que todo esto lo podemos definir
de forma iterativa dentro de un bucle.
'''

output = [X]

for nun_capa in range(len(red_neuronal)):
    z = output[-1] @ red_neuronal[nun_capa].W + red_neuronal[nun_capa].b
    a = red_neuronal[nun_capa].funcion_act[0](z)
    output.append(a)

print(output[-1])

'''
Así, tendríamos la estimación para cada una de las clases de este
ejercicio de prueba. Como es la primera ronda, la red no ha 
entrenado nada, por lo que el resultado es aleatorio. Por tanto,
solo quedaría una cosa: entrenar a nuestra red neuronal programada
en Python.
'''

'''
Entrenar tu red neuronal
Creando la función de coste.
Para poder entrenar la red neuronal lo primero que debemos hacer es
calcular cuánto ha fallado. Para ello usaremos uno de los estimadores más
típicos en el mundo del machine learning: el error cuadrático medio (MSE).
'''

def mse(Ypredich, Yreal):
    # Calculamos el error
    x = (np.array(Ypredich) - np.array(Yreal)) ** 2
    x = np.mean(x)

    # Calculamos la derivada de la función
    y = np.array(Ypredich) - np.array(Yreal)
    return (x, y)

'''
Con esto, vamos a 'inventarnos' nas clases (0 o 1) para los valores que
nuestra red neuronal ha predicho antes. Así, calcularemos el error 
cuadrático medio.
'''

from random import shuffle

Y = [0] * 10 + [1] * 10
shuffle(Y)
Y = np.array(Y).reshape(len(Y), 1)
print(mse(output[-1], Y)[0])

'''
Ahora que ya tenemos el error calculado, tenemos que irlo propagando hacia
atrás para ir ajustando los parámetros. Haciendo esto de forma iterativa,
nuestra red neuronal irá mejorando sus predicciones, es decir, disminuirá
su error. Así es como se entrena a una red neuronal.

Backpropagation y gadient descent: entrenando a nuestra red neuronal.
Gradient descent: optimizando los parámetros.
Los valores están lejos del valor óptimo, por lo que deberíamos hacer
que nuestro parámetro llegue a allí. ¿Cómo lo hacemos?
Usaremos gradient descent. Este algoritmo utiliza el error en el punto en 
el que nos encontramos y calcula las derivadas parciales en dicho punto.
Esto nos devuelve el vector gradiente, es decir, un vector de direcciones
hacia donde el error se incremente. Por tanto, si usamos el inverso de
ese valor, iremos hacia abajo. En definitiva, gradient descent calcula
la inversa del gradiente para saber qué valores deben tomas los
hiperparámetros.
Cuánto nos movamos hacia abajo dependerá de otro hiperparámetro: el 
learning rate.
Con gradient descent a cada iteración nuestros parámetros se irán 
acercando a un valor óptimo, hasta que lleguen a un punto óptimo, a partir
del cual nuestra red dejará de aprender.

Backpropagation: calculando el error en cada capa.
La única manera de calcular el error de cada neurona en cada capa es
haciendo el proceso inverso: primero calculamos el error de la última
capa, con lo que podremos calcular el error de la capa anterior y así
hasta completar todo el proceso.
'''

print(red_neuronal[-1].b)
print(red_neuronal[-1].W)

'''
El error lo calculamos como la derivada de la función de coste sobre el
resultado de la capa siguiente por la derivada de la función de activación.
En nuestro caso, el resultado del último valor está en la capa -1, mientras
que la capa que vamos a optimizar es la anteúltima (posición -2). Además,
como hemos definido las funciones como un par de funciones, simplemente
tendremos que indicar el resultado de la función de la posición [1] en
ambos casos.
'''

# Backprop en la última capa
a = output[-1]
x = mse(a, Y)[1] * red_neuronal[-2].funcion_act[1](a)
print(x)
print('-----------------------------------------------')

# Definimos el learning rate
lr = 0.05

# Creamos el índice inverso para ir de derecha a izquierda
back = list(range(len(output) -1))
back.reverse()

# Creamos el vector delta donde meteremos los errores en cada capa
delta = []

for capa in back:
    # Backprop #

    # Guardamos los resultados de la última capa antes de usar backprop
    #   para poder usarlas en gradient descent
    a = output[capa +1][1]

    # Backprop en la última capa
    if capa == back[0]:
        x = mse(a, Y)[1] * red_neuronal[capa].funcion_act[1](a)
        delta.append(x)

    # Backprop en el resto de capas
    else:
        x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
        delta.append(x)

    # Guardamos los valores de W para poder usarlos en la iteración siguiente
    W_temp = red_neuronal[capa].W.transpose()

    # Gradient Descent #

    # Ajustamos los valores de los parametros de la capa
    red_neuronal[capa].b = red_neuronal[capa].b - delta[-1].mean() * lr
    red_neuronal[capa].W = red_neuronal[capa].W - (output[capa].T @ delta[-1]) * lr

print(f'MSE: {mse(output[-1], Y)[0]}')
print(f'Estimación: {output[-1]}')
print('********************************************')

'''
CASO PRÁCTICO:
Definición del problema: clasificación de puntos.
Vamos a clasificar puntos de dos nubes de puntos. Para ello, lo primero
que vamos a hacer es crear una función que nos devuelva puntos aleatorios
alrededor de un círculo imaginario de radio R.
'''

def circulo(num_datos = 100, R = 1, minimo = 0, maximo = 1):
    pi = math.pi
    r = R * np.sqrt(stats.truncnorm.rvs(minimo, maximo, size= num_datos)) * 10
    theta = stats.truncnorm.rvs(minimo, maximo, size= num_datos) * 2 * pi * 10

    x = np.cos(theta) * r
    y = np.sin(theta) * r

    y = y.reshape((num_datos, 1))
    x = x.reshape((num_datos, 1))

    # Vamos a reducir el número de elementos para que no cause un Overflow
    x = np.round(x, 3)
    y = np.round(y, 3)

    df = np.column_stack([x, y])
    return(df)

'''
Ahora, crearemos dos sets de datos aleatorios, cada uno de 150 puntos y
con radios diferentes. La idea de hacer que los datos se creen de forma
aleatorio es que puedan solaparse, de tal manera que a la red neuronal le
cueste un poco y el resultado no sea perfecto.
'''

datos_1 = circulo(num_datos=150, R=2)
datos_2 = circulo(num_datos=150, R=0.5)
X = np.concatenate([datos_1, datos_2])
X = np.round(X, 3)

Y = [0] * 150 + [1] * 150
Y = np.array(Y).reshape(len(Y), 1)

'''
Con esto ya tendríamos nuestros datos de entrada (X) y sus correspondientes
etiquetas (Y). Teniendo esto en cuenta, visualicemos cómo es el problema que
debe resolver nuestra red neuronal.
'''

plt.cla()
plt.scatter(X[0:150,0], X[0:150,1], c="b")
plt.scatter(X[150:300,0], X[150:300,1], c="r")
plt.show()

'''
Entrenamiento de nuestra red neuronal
Lo primero de todo, vamos a crear funciones a partir del código que 
hemos generado anteriormente.
'''

def entrenamiento(X, Y, red_neuronal, lr = 0.01):
    # Output guardara el resultado de cada capa
    # En la capa 1, el resultado es el valor de entrada
    output = [X]

    for num_capa in range(len(red_neuronal)):
        z = output[-1] @ red_neuronal[num_capa].W + red_neuronal[num_capa].b

        a = red_neuronal[num_capa].funcion_act[0](z)

        # Incluimos el resultado de la capa a output
        output.append(a)

    # Backpropagation
    back = list(range(len(output)-1))
    back.reverse()

    # Guardaremos el error de la capa en delta
    delta = []

    for capa in back:
        # Backprop # delta

        a = output[capa+1]

        if capa == back[0]:
            x = mse(a, Y)[1] * red_neuronal[capa].funcion_act[1](a)
            delta.append(x)

        else:
            x = delta[-1] @ W_temp * red_neuronal[capa].funcion_act[1](a)
            delta.append(x)

        W_temp = red_neuronal[capa].W.transpose()

        # Gradient Descent #
        red_neuronal[capa].b = red_neuronal[capa].b - np.mean(delta[-1], axis=0, keepdims=True) * lr
        red_neuronal[capa].W = red_neuronal[capa].W - output[capa].transpose() @ delta[-1] * lr

    return output[-1]

'''
Ya tenemos nuestra función de red neuronal funcionando. Ahora, simplemente
tenemos que indicar los parámetros y el número de rondas, y esperar para
ver cómo va aprendiendo nuestra red neuronal y cómo de bien se le da con
el problema que hemos planteado.
Vamos a usar la función de entrenamiento. Además, vamos a ir guardando
tanto las predicciones que hace como el error que está cometiendo. De
esta manera podremos visualizar cómo ha entrenado nuestra red.
'''

error = []
predicciones = []

for epoch in range(0, 500):
    ronda = entrenamiento(X=X, Y=Y, red_neuronal=red_neuronal, lr=0.001)
    predicciones.append(ronda)
    temp = mse(np.round(predicciones[-1]), Y)[0]
    error.append(temp)

epoch = list(range(0, 500))
plt.plot(epoch, error)
plt.show()
