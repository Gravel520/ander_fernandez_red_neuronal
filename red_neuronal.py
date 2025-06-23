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
fig.tight_layout()
#plt.show()

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
#plt.show()

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

'''