import scipy.io
import numpy as np
import torch
from scipy.interpolate import griddata
import os

print("=== Pre-elaborazione Dati di Navier-Stokes ===")
data = scipy.io.loadmat('cylinder_nektar_wake.mat')

X_star = data['X_star'] # (5000, 2)
U_star = data['U_star'] # (5000, 2, 200) -> u, v over 200 timesteps

# Calcoliamo i limiti del dominio fisico
x_min, x_max = X_star[:, 0].min(), X_star[:, 0].max()
y_min, y_max = X_star[:, 1].min(), X_star[:, 1].max()
print(f"Limiti X: [{x_min:.2f}, {x_max:.2f}]")
print(f"Limiti Y: [{y_min:.2f}, {y_max:.2f}]")

# Creiamo una griglia regolare ad alta risoluzione (es. 128x64)
# La scia è molto allungata sull'asse X
nx, ny = 128, 64
grid_x, grid_y = np.mgrid[x_min:x_max:nx*1j, y_min:y_max:ny*1j]

# Il cilindro in questo dataset si trova in (0,0) con raggio 0.5. 
# Creiamo una maschera per il cilindro
cylinder_mask = (grid_x**2 + grid_y**2) <= (0.5**2)

num_time_steps = U_star.shape[2]
video_tensor = np.zeros((num_time_steps, nx, ny, 2), dtype=np.float32)

print(f"Interpolazione di {num_time_steps} fotogrammi sulla griglia regolare...")
for t in range(num_time_steps):
    # Estraiamo le velocità U e V al tempo t
    u = U_star[:, 0, t]
    v = U_star[:, 1, t]
    
    # Interpoliamo i punti sparsi sulla nostra nuova griglia regolare
    u_grid = griddata(X_star, u, (grid_x, grid_y), method='cubic', fill_value=0)
    v_grid = griddata(X_star, v, (grid_x, grid_y), method='cubic', fill_value=0)
    
    # Azzeriamo la velocità dentro il cilindro (condizione di non slittamento)
    u_grid[cylinder_mask] = 0.0
    v_grid[cylinder_mask] = 0.0
    
    video_tensor[t, :, :, 0] = u_grid
    video_tensor[t, :, :, 1] = v_grid
    
    if (t+1) % 50 == 0:
        print(f"Completato fotogramma {t+1}/{num_time_steps}")

# Salviamo la griglia e il video come tensori PyTorch per l'addestramento
torch.save({
    'video': torch.tensor(video_tensor),
    'grid_x': torch.tensor(grid_x),
    'grid_y': torch.tensor(grid_y),
    'mask': torch.tensor(cylinder_mask)
}, 'gridded_wake_data.pt')

print("\nDati convertiti con successo in 'gridded_wake_data.pt'!")
print(f"Shape finale del tensore video: {video_tensor.shape} (Time, X, Y, Channels)")
