import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
from geo_fno import GeoFNO2d
from train_cylinder import generate_cylinder_data

def main():
    modes1, modes2, width = 12, 12, 32
    r_res, theta_res = 64, 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generazione Video Dinamico Geo-FNO su: {device}")

    # 1. Carichiamo Modello e Griglie
    model = GeoFNO2d(modes1, modes2, width).to(device)
    try:
        model.load_state_dict(torch.load('geofno_model.pth', weights_only=True))
        grids = torch.load('cylinder_grid.pth', weights_only=True)
        X_grid = grids['X'].numpy()
        Y_grid = grids['Y'].numpy()
    except FileNotFoundError:
        print("Devi prima eseguire 'python train_cylinder.py'!")
        return
    model.eval()

    # Vogliamo variare la velocità del vento in ingresso nel tempo (un'onda sinusoidale)
    num_frames = 60
    base_wind = 2.5
    wind_amplitude = 1.5
    
    frames_true = []
    frames_pred = []
    frames_error = []
    wind_values = []

    print("Calcolo inferenze per ogni frame del video...")
    for t in range(num_frames):
        # Il vento oscillerà dolcemente
        U_in_val = base_wind + wind_amplitude * np.sin(t * 2 * np.pi / num_frames)
        wind_values.append(U_in_val)
        
        # 2. Ricalcoliamo il ground truth usando le formule fisiche con il NUOVO vento
        R_cylinder, R_max = 1.0, 5.0
        r = torch.linspace(R_cylinder, R_max, r_res)
        theta = torch.linspace(0, 2*math.pi, theta_res)
        R_grid, Theta_grid = torch.meshgrid(r, theta, indexing='ij')
        
        V_r = U_in_val * (1.0 - (R_cylinder**2 / R_grid**2)) * torch.cos(Theta_grid)
        V_theta = -U_in_val * (1.0 + (R_cylinder**2 / R_grid**2)) * torch.sin(Theta_grid)
        
        true_Vx = (V_r * torch.cos(Theta_grid) - V_theta * torch.sin(Theta_grid)).numpy()
        true_Vy = (V_r * torch.sin(Theta_grid) + V_theta * torch.cos(Theta_grid)).numpy()
        true_mag = np.sqrt(true_Vx**2 + true_Vy**2)
        frames_true.append(true_mag)
        
        # 3. Chiediamo al Geo-FNO di predirlo
        U_in_tensor = torch.full((r_res, theta_res, 1), U_in_val, dtype=torch.float32)
        X_tensor = torch.tensor(X_grid, dtype=torch.float32).unsqueeze(-1)
        Y_tensor = torch.tensor(Y_grid, dtype=torch.float32).unsqueeze(-1)
        
        inp = torch.cat([U_in_tensor, X_tensor, Y_tensor], dim=-1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction = model(inp)
            
        pred_Vx = prediction[0, :, :, 0].cpu().numpy()
        pred_Vy = prediction[0, :, :, 1].cpu().numpy()
        pred_mag = np.sqrt(pred_Vx**2 + pred_Vy**2)
        frames_pred.append(pred_mag)
        
        # 4. Calcoliamo l'errore
        frames_error.append(np.abs(true_mag - pred_mag))

    # --- Creazione dell'animazione ---
    print("Creazione del file GIF (questo potrebbe richiedere un minuto)...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Valori massimi per bloccare la scala dei colori ed evitare sfarfallii
    vmax = base_wind + wind_amplitude + 0.5 
    
    circle1 = plt.Circle((0, 0), 1.0, color='black')
    axes[0].add_patch(circle1)
    pcm1 = axes[0].pcolormesh(X_grid, Y_grid, frames_true[0], cmap='jet', shading='gouraud', vmin=0, vmax=vmax*1.5)
    axes[0].set_aspect('equal')
    
    circle2 = plt.Circle((0, 0), 1.0, color='black')
    axes[1].add_patch(circle2)
    pcm2 = axes[1].pcolormesh(X_grid, Y_grid, frames_pred[0], cmap='jet', shading='gouraud', vmin=0, vmax=vmax*1.5)
    axes[1].set_aspect('equal')
    
    circle3 = plt.Circle((0, 0), 1.0, color='black')
    axes[2].add_patch(circle3)
    pcm3 = axes[2].pcolormesh(X_grid, Y_grid, frames_error[0], cmap='Reds', shading='gouraud', vmin=0, vmax=0.5)
    axes[2].set_aspect('equal')

    def update(frame):
        # Rimuoviamo la mesh vecchia e la ridisegniamo 
        for ax in axes:
            for coll in list(ax.collections):
                coll.remove()
        
        curr_u = wind_values[frame]
        fig.suptitle(f"Generalizzazione Geo-FNO | Vento Ingresso: {curr_u:.2f} m/s", fontsize=16)
        
        axes[0].pcolormesh(X_grid, Y_grid, frames_true[frame], cmap='jet', shading='gouraud', vmin=0, vmax=vmax*1.5)
        axes[1].pcolormesh(X_grid, Y_grid, frames_pred[frame], cmap='jet', shading='gouraud', vmin=0, vmax=vmax*1.5)
        axes[2].pcolormesh(X_grid, Y_grid, frames_error[frame], cmap='Reds', shading='gouraud', vmin=0, vmax=0.5)
        
        axes[0].set_title("Soluzione Analitica")
        axes[1].set_title("Predizione Geo-FNO")
        axes[2].set_title("Errore Assoluto")
        
        return axes

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=100)
    ani.save('geo_fno_dynamic_wind.gif', writer='pillow', fps=10)
    print("Video salvato in 'geo_fno_dynamic_wind.gif'.")

if __name__ == "__main__":
    main()
