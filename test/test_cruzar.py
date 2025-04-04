import cv2
import numpy as np

from src.individuo import crear_individuo
from src.cruzar import cruzar

img = cv2.imread("test/image_00.png")
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)


def test_cruce():
    # al cruzar los individuos el cromosoma debe tener la misma forma
    # asi siempre representamos la misma info
    ind1 = crear_individuo(img, 1, 3)
    ind2 = crear_individuo(img, 1, 3)

    ind3 = cruzar(ind1, ind2, "cruce_uniforme")
    assert ind3.cromosoma.shape == ind1.cromosoma.shape
    
    ind3 = cruzar(ind1, ind2, "cruce_un_punto")
    assert ind3.cromosoma.shape == ind1.cromosoma.shape
    
    ind3 = cruzar(ind1, ind2, "cruce_n_puntos", n=2)
    assert ind3.cromosoma.shape == ind1.cromosoma.shape

    ind3 = cruzar(ind1, ind2, "cruce_anular")
    assert ind3.cromosoma.shape == ind1.cromosoma.shape
