import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from fno_masked import FNO2d_Masked

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generazione Animazione Vortici di Von Kármán su: {device}")

    # 1. Caricamento Dati e Modello
    try:
        data = torch.load('gridded_wake_data.pt', weights_only=True)
        model = FNO2d_Masked(12, 12, 32).to(device)
        model.load_state_dict(torch.load('fno_real_wake_model.pth', weights_only=True))
    except FileNotFoundError:
        print("Errore: Assicurati di aver fatto girare sia preprocess_data.py che train_real_wake.py")
        return
        
    model.eval()

    video = data['video']
    grid_x = data['grid_x'].unsqueeze(-1).float()
    grid_y = data['grid_y'].unsqueeze(-1).float()
    mask = data['mask'].unsqueeze(-1).float()

    # Vogliamo fare un video di 50 fotogrammi per testare l'Autoregressione pura
    num_rollout_steps = 50
    start_frame = 100 # Partiamo da metà dataset
    
    # Questo è l'UNICO fotogramma che la rete vedrà
    initial_frame = video[start_frame]
    
    true_frames = []
    pred_frames = []
    
    # 2. Raccogliamo i veri fotogrammi per il confronto (usiamo la magnitudine della velocità)
    for t in range(num_frames_to_show := num_rollout_steps + 1):
        frame = video[start_frame + t]
        mag = np.sqrt(frame[:,:,0]**2 + frame[:,:,1]**2).numpy()
        true_frames.append(mag)
        
    # 3. Rollout Autoregressivo col FNO
    print("Avvio Rollout Autoregressivo col modello AI...")
    current_frame = initial_frame
    pred_frames.append(np.sqrt(current_frame[:,:,0]**2 + current_frame[:,:,1]**2).numpy())
    
    with torch.no_grad():
        for step in range(num_rollout_steps):
            inp = torch.cat([current_frame, mask, grid_x, grid_y], dim=-1).unsqueeze(0).to(device)
            next_frame_pred = model(inp)[0].cpu() # Shape: (128, 64, 2)
            
            # Salviamo per il video
            pred_mag = np.sqrt(next_frame_pred[:,:,0]**2 + next_frame_pred[:,:,1]**2).numpy()
            pred_frames.append(pred_mag)
            
            # Il ciclo autoregressivo: l'output diventa il nuovo input
            current_frame = next_frame_pred

    # 4. Creazione GIF
    print("Creazione GIF (Vortices_AI_vs_Physics.gif)...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Trasponiamo le immagini per far scorrere il fluido in orizzontale
    im1 = ax1.imshow(true_frames[0].T, cmap='turbo', origin='lower', animated=True, vmin=0, vmax=1.2)
    ax1.set_title("Soluzione Navier-Stokes (Reale)")
    
    im2 = ax2.imshow(pred_frames[0].T, cmap='turbo', origin='lower', animated=True, vmin=0, vmax=1.2)
    ax2.set_title("Simulazione FNO Autoregressiva (AI)")

    def update(frame):
        im1.set_array(true_frames[frame].T)
        im2.set_array(pred_frames[frame].T)
        fig.suptitle(f"Von Kármán Vortex Shedding | Fotogramma: {frame}/{num_rollout_steps}")
        return im1, im2,

    ani = animation.FuncAnimation(fig, update, frames=num_rollout_steps+1, interval=80, blit=True)
    ani.save('Vortices_AI_vs_Physics.gif', writer='pillow', fps=12)
    print("SUCCESSO! Apri 'Vortices_AI_vs_Physics.gif' per vedere l'AI che simula Navier-Stokes!")

if __name__ == "__main__":
    main()
