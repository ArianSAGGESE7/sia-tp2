import cv2
from src.mainp2 import genetic_algorithm
from src.individuo import crear_imagen, crear_individuo

img = cv2.imread("test/image_00.png")
img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)

opciones = {
    "num_poligonos": 5,
    "num_lados": 3,
    "num_individuos": 100,
    "num_generaciones": 30,
    "num_seleccion_elite": 5,
    "prob_mutacion": 0.5,
    "cant_mutacion": 0.3,
    "metodo_de_cruza": "cruce_un_punto"
}


mejor = genetic_algorithm(img, opciones)
mejor_img = crear_imagen(mejor)
cv2.imwrite("out.png", mejor_img)

