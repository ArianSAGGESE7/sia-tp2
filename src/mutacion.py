"""
Este archivo contiene:
- Función para mutar al individuo
"""
import numpy as np

def mutar(ind, prob_mutacion, cant_mutacion):
    "muta al individuo"
    if np.random.random() < prob_mutacion:        
        
        m, n = ind.cromosoma.shape
        genes_a_mutar = np.floor(cant_mutacion * ind.cromosoma.size).astype(int)
        
        for _ in range(genes_a_mutar):
            i = np.random.randint(m) # fila
            j = np.random.randint(n) # columna

            # primero checkeamos que no estemos mutando fuera del rango
            #   0   1   2   3   4   5   6   7
            # x11 y11 x12 y12   R   G   B   A
            # x21 y21 x22 y22   R   G   B   A
            
            if j < (n - 4): # cambia coordenada
                if i % 2 == 0: # numero par - coordenada x
                    mut = np.random.randint(ind.img_dims[1])
                else: # numero impar - coordenada y
                    mut = np.random.randint(ind.img_dims[0])
            else: # cambia color
                mut = np.random.randint(256)
            ind.cromosoma[i, j] = mut
    
    return ind
