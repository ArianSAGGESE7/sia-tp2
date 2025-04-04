import numpy as np
from PIL import Image, ImageDraw
import random

import src.mainp2

# Tamaño del lienzo
def test_00():
    WIDTH, HEIGHT = 256, 256  



    M = 9   # Número de individuos
    N = 40  # Triángulos por individuo
    img_size = (256, 256)  # Tamaño de la imagens
    img_ref = Image.open("test/image_00.png").convert("RGBA")
    img_ref = img_ref.resize(img_size)
    Generations = 100;
    individual = src.mainp2.genetic_algorithm(M,N,img_size,img_ref,Generations)
    img1 = src.mainp2.draw_population(individual,0)
    img1.save("out.png")
