"""
Este archivo contiene:
- funciones para seleccionar los diferentes individuos de la poblacion
- ponder los otros esquemas
"""

import numpy as np
import random



def seleccionar(poblacion):
    len_poblacion = len(poblacion)
    return random.choices(poblacion[:len_poblacion//2], k=2)

