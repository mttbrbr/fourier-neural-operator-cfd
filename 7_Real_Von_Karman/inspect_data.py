import scipy.io
import numpy as np

# Carichiamo il dataset
data = scipy.io.loadmat('cylinder_nektar_wake.mat')

print("=== Esplorazione Dataset Reale di Navier-Stokes ===")
print(f"Chiavi nel file: {data.keys()}")

# Estraiamo i tensori per capire la forma
U_star = data['U_star'] # Velocità (u, v)
p_star = data['p_star'] # Pressione
t_star = data['t']      # Tempo
X_star = data['X_star'] # Coordinate (x, y)

print(f"Coordinate spaziali (X_star): {X_star.shape}")
print(f"Velocità (U_star): {U_star.shape}")
print(f"Pressione (p_star): {p_star.shape}")
print(f"Istantanee temporali (t): {t_star.shape}")
