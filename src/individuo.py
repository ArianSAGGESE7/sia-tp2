"""
Este archivo contiene:
- Estructura para guardar info de cada individuo de la poblacion.
- Funciones de evaluacion de similitud con imagen objetivo
"""

from dataclasses import dataclass
import numpy as np
import cv2

@dataclass
class Individuo:
    cromosoma: np.array     # encoding [x1, y1, ... xn, yn, R, G, B, A]
    costo: float            # comparativa con imagen original
    img_dims: tuple         # dimensiones imagen de referencia
    n_poligonos: int        # numero de poligonos
    n_lados: int            # numero de lados


def poligono_aleatorio(img, lados=3):
    "genera vector de la forma [x1, y1, x2, y2, ... xi, yi, cR, cG, cB, cA]"
    l, w, num_color = img.shape
    puntos = []
    color = np.random.randint(0, 256, num_color)
    x = np.random.randint(0, w+1, lados)
    y = np.random.randint(0, l+1, lados)
    puntos = np.vstack((x, y)).T.reshape(-1)
    return np.hstack((puntos, color))


def crear_individuo(img, num_poligonos, num_lados):
    "genera un individuo de la poblacion"
    ind = Individuo(
        cromosoma=np.vstack([poligono_aleatorio(img, num_lados) for i in range(num_poligonos)]),
        costo=0,
        img_dims=img.shape,
        n_poligonos=num_poligonos,
        n_lados=num_lados
    )
    return ind


def generar_poblacion(img, num_individuos, num_poligonos):
    "Genera una población de M individuos con N triángulos cada uno"
    return [crear_individuo(img, num_poligonos, 3) for _ in range(num_individuos)]    


def crear_imagen(individuo: Individuo):
    "Crea una imagen con el cromosoma de un individuo"
    colores = individuo.cromosoma[:,-4:]                    # Las 4 ultimas columnas son canales RGBA
    shape = (individuo.n_poligonos, individuo.n_lados, 2)   # De un array flatten a [[p1, p2, p3], [p1, p2, p3] ...]
    poligonos = individuo.cromosoma[:,:-4].reshape(shape)
    
    img = np.zeros(individuo.img_dims)                      # Lienzo vacio
    for i in range(poligonos.shape[0]):                     # Armo matriz con triangulos dentro
        poligono = poligonos[i]
        color = colores[i].tolist()
        img = cv2.fillPoly(img, [poligono], color)
    return img


def calcular_costo(img1, img2):
    """Calcula el Error Cuadrático Medio (MSE) entre dos imágenes"""
    """Lo usamos como función de fitness"""    
    return (np.abs(img1 - img2)).mean(axis=None)
    
