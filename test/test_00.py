

import cv2
from copy import copy
from src.mainp2 import crear_individuo, draw, fitness



# Tamaño del lienzo
# def test_00():
#     WIDTH, HEIGHT = 256, 256  
#     M = 9   # Número de individuos
#     N = 40  # Triángulos por individuo
#     img_size = (256, 256)  # Tamaño de la imagens
#     img_ref = Image.open("test/image_00.png").convert("RGBA")
#     img_ref = img_ref.resize(img_size)
#     Generations = 1000;
#     individual = src.mainp2.genetic_algorithm(M,N,img_size,img_ref,Generations)
#     img1 = src.mainp2.draw(individual,0)
#     img1.save("out.png")


img = cv2.imread("test/image_00.png")
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

def test_individuo():
    individuo = crear_individuo(img, num_poligonos=10, num_lados=3)
    out = draw(individuo)

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
    imgs = [draw(ind) for ind in [individuo1, individuo2]]
    assert fitness(*imgs) == 0
    
    # Dos imagenes diferentes deberian dar error != 0
    individuo1 = crear_individuo(img, num_poligonos=1, num_lados=3)
    individuo2 = crear_individuo(img, num_poligonos=1, num_lados=3)
    imgs = [draw(ind) for ind in [individuo1, individuo2]]
    assert fitness(*imgs) != 0


