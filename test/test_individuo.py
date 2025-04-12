import cv2
from copy import copy
from src.individuo import crear_individuo, crear_imagen, calcular_costo


img = cv2.imread("Images/image_00.png")
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

def test_individuo():
    individuo = crear_individuo(img, num_poligonos=10, num_lados=3)
    out = crear_imagen(individuo)

    # La matriz de imagen viene como columnas (coordenada x) y filas (coordenada y)
    # Acá se verifica que el triangulo creado quede dentro de la matriz
    for p in range(individuo.n_poligonos):
        valores_x = individuo.cromosoma[p][:-4:2]
        assert all(valores_x <= individuo.img_dims[1])

        valores_y = individuo.cromosoma[p][1:-4:2]
        assert all(valores_y <= individuo.img_dims[0])


def test_fitness():
    # Dos imagenes iguales deberian dar error 0
    individuo1 = crear_individuo(img, num_poligonos=1, num_lados=3)
    individuo2 = copy(individuo1)
    imgs = [crear_imagen(ind) for ind in [individuo1, individuo2]]
    assert calcular_costo(*imgs) == 0
    
    # Dos imagenes diferentes deberian dar error != 0
    individuo1 = crear_individuo(img, num_poligonos=1, num_lados=3)
    individuo2 = crear_individuo(img, num_poligonos=1, num_lados=3)
    imgs = [crear_imagen(ind) for ind in [individuo1, individuo2]]
    assert calcular_costo(*imgs) != 0
