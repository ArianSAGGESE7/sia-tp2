"""
funciones de cruza
"""

import numpy as np
from .individuo import Individuo

def _cruce_un_punto(c1, c2, **kwargs):
    i = np.random.randint(1, len(c1))
    c3 = np.hstack((c1[:i], c2[i:]))
    return c3


def _cruce_n_puntos(c1, c2, n=2, **kwargs):
    i = 1
    for _ in range(n):
        i = np.random.randint(i, len(c1))
        c1, c2 = np.hstack((c1[:i], c2[i:])), np.hstack((c2[:i], c1[i:]))
    return c1


def _cruce_uniforme(c1, c2, **kwargs):
    mask = np.random.randint(0, 2, len(c1)).astype(bool)
    c3 = np.where(mask, c1, c2)
    return c3


def _cruce_anular(c1, c2, **kwargs):
    p = np.random.randint(0, len(c1))
    l = np.random.randint(1, len(c1)//2)
    swap_index = [(p+i) % len(c1) for i in range(l)]
    c3 = c1.copy()
    for index in swap_index:
        c3[index] = c2[index]
    return c3


funcion_cruza = {
    "cruce_un_punto": _cruce_un_punto,
    "cruce_n_puntos": _cruce_n_puntos,
    "cruce_uniforme": _cruce_uniforme,
    "cruce_anular": _cruce_anular,
}

def cruzar(ind1, ind2, metodo="cruce_un_punto", **kwargs):
    orig_shape = ind1.cromosoma.shape
    cromosoma1 = ind1.cromosoma.reshape(-1)
    cromosoma2 = ind2.cromosoma.reshape(-1)
    cromosoma3 = funcion_cruza[metodo](cromosoma1, cromosoma2, **kwargs).reshape(orig_shape)
    
    hijo = Individuo(
        cromosoma=cromosoma3,
        costo=0,
        img_dims=ind1.img_dims,
        n_poligonos=ind1.n_poligonos,
        n_lados=ind1.n_lados
    )
    return hijo

