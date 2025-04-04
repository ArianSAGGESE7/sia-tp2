"""
funciones de cruza
"""

import numpy as np
from .individuo import Individuo

def _cruce_un_punto(c1, c2):
    #import pdb; pdb.set_trace()
    i = np.random.randint(1, len(c1))
    c3 = np.hstack((c1[:i], c2[i:]))
    return c3


def _cruce_uniforme(c1, c2):
    mask = np.random.randint(0, 2, len(c1)).astype(bool)
    c3 = np.where(mask, c1, c2)
    return c3


funcion_cruza = {
    "cruce_un_punto": _cruce_un_punto,
    "cruce_uniforme": _cruce_uniforme
}

def cruzar(ind1, ind2, metodo="cruce_un_punto"):
    orig_shape = ind1.cromosoma.shape
    cromosoma1 = ind1.cromosoma.reshape(-1)
    cromosoma2 = ind2.cromosoma.reshape(-1)
    cromosoma3 = funcion_cruza[metodo](cromosoma1, cromosoma2).reshape(orig_shape)
    
    hijo = Individuo(
        cromosoma=cromosoma3,
        costo=0,
        img_dims=ind1.img_dims,
        n_poligonos=ind1.n_poligonos,
        n_lados=ind1.n_lados
    )
    return hijo

