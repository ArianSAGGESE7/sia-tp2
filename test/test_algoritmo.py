import cv2
from src.mainp2 import genetic_algorithm
from src.individuo import crear_imagen, crear_individuo

img = cv2.imread("test/image_00.png")
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
img = cv2.resize(img, (100, 100)) 

opciones = {
    "num_poligonos": 1,
    "num_lados": 3,
    "num_individuos": 100,
    "num_generaciones": 500,
    "num_seleccion_elite": 1,
    "prob_mutacion": 0.1,
    "cant_mutacion": 0.1,
    "metodo_de_cruza": "cruce_uniforme",
    "kwargs_cruza": {},
    "metodo_de_seleccion": "ranking",
    "kwargs_seleccion": {"K": 3},
}


mejor = genetic_algorithm(img, opciones)
mejor_img = crear_imagen(mejor)
cv2.imwrite("out.jpg", mejor_img)
cv2.imwrite("target.jpg", img)


