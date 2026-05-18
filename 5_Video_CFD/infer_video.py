import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from fno_video import FNO2d_Video
from train_video import generate_vortex_video_data

def main():
    modes1, modes2, width, resolution = 12, 12, 32, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eseguo Rollout Autoregressivo su: {device}")

    model = FNO2d_Video(modes1, modes2, width).to(device)
    try:
        model.load_state_dict(torch.load('fno_video_model.pth', weights_only=True))
    except FileNotFoundError:
        print("Devi prima eseguire 'python train_video.py'!")
        return
    model.eval()

    # Vogliamo simulare un video lungo 40 fotogrammi
    rollout_steps = 40
    
    # Generiamo una "vera" sequenza fisica da usare come paragone (Ground Truth)
    # Ne generiamo una sola, ma le chiediamo di calcolare tutti i 40 steps con la vera fisica
    print("Generazione simulazione fisica reale di riferimento...")
    x_data, y_data = generate_vortex_video_data(1, resolution, timesteps=rollout_steps)
    
    # L'input iniziale per la nostra rete neurale sarà SOLO IL FOTOGRAMMA T=0.
    # Da qui in poi, la rete dovrà usare la propria predizione come input per il passo successivo.
    initial_input = x_data[0].unsqueeze(0).to(device)
    
    # Griglie spaziali (costanti) da appendere a ogni nuovo fotogramma predetto
    # x_data ha shape: [H, W, 3(field, grid_x, grid_y)]
    grid_x = initial_input[0, :, :, 1].unsqueeze(-1)
    grid_y = initial_input[0, :, :, 2].unsqueeze(-1)

    predicted_frames = [initial_input[0, :, :, 0].cpu().numpy()]
    true_frames = [initial_input[0, :, :, 0].cpu().numpy()]
    
    # Estraiamo i veri frame futuri dal dataset generato per confrontarli
    for i in range(rollout_steps):
        true_frames.append(y_data[i, :, :, 0].numpy())

    print("Inizio Autoregressive Rollout (La rete predice il futuro in loop)...")
    
    current_input = initial_input
    
    with torch.no_grad():
        for step in range(rollout_steps):
            # Predici il frame t+1
            pred_t_plus_1 = model(current_input)
            
            # Salva la predizione per il video
            pred_numpy = pred_t_plus_1[0, :, :, 0].cpu().numpy()
            predicted_frames.append(pred_numpy)
            
            # --- IL PASSAGGIO CHIAVE DELL'AUTOREGRESSIONE ---
            # Il nuovo input diventa la predizione corrente + le griglie fisse
            new_field = pred_t_plus_1[0] # shape [H, W, 1]
            current_input = torch.cat([new_field, grid_x, grid_y], dim=-1).unsqueeze(0)

    # --- CREAZIONE DELL'ANIMAZIONE (GIF) ---
    print("Rollout completato! Generazione del Video...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("FNO Fluidodinamica: Rollout Autoregressivo")
    
    im1 = ax1.imshow(true_frames[0], cmap='magma', origin='lower', animated=True, vmin=0, vmax=2)
    ax1.set_title("Fisica Reale (Ground Truth)")
    
    im2 = ax2.imshow(predicted_frames[0], cmap='magma', origin='lower', animated=True, vmin=0, vmax=2)
    ax2.set_title("Simulazione FNO (Immaginata dalla rete)")
    
    def update(frame):
        im1.set_array(true_frames[frame])
        im2.set_array(predicted_frames[frame])
        fig.suptitle(f"FNO Fluidodinamica | Fotogramma: {frame}/{rollout_steps}")
        return im1, im2,

    ani = animation.FuncAnimation(fig, update, frames=rollout_steps+1, interval=100, blit=True)
    
    # Salva come GIF (non richiede FFMPEG)
    video_path = 'fno_vortex_rollout.gif'
    ani.save(video_path, writer='pillow', fps=10)
    print(f"Fatto! Apri il file '{video_path}' per vedere il tuo primo vero video CFD generato dall'AI!")

if __name__ == "__main__":
    main()
