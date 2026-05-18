import torch
import matplotlib.pyplot as plt
import numpy as np

from geo_fno import GeoFNO2d
from train_cylinder import generate_cylinder_data

def main():
    modes1, modes2, width = 12, 12, 32
    r_res, theta_res = 64, 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Inferenza Flusso Cilindro su: {device}")

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

    # 2. Generiamo UN campione casuale
    input_tensor, true_output, _, _ = generate_cylinder_data(1, r_res, theta_res)
    U_in_val = input_tensor[0, 0, 0, 0].item() # La velocità del vento in ingresso casuale
    input_tensor = input_tensor.to(device)

    # 3. Predizione Geo-FNO
    with torch.no_grad():
        prediction = model(input_tensor)
        
    # Calcoliamo la magnitudine della velocità: sqrt(V_x^2 + V_y^2)
    pred_Vx = prediction[0, :, :, 0].cpu().numpy()
    pred_Vy = prediction[0, :, :, 1].cpu().numpy()
    pred_magnitude = np.sqrt(pred_Vx**2 + pred_Vy**2)
    
    true_Vx = true_output[0, :, :, 0].cpu().numpy()
    true_Vy = true_output[0, :, :, 1].cpu().numpy()
    true_magnitude = np.sqrt(true_Vx**2 + true_Vy**2)

    # 4. Creazione del Grafico (Attenzione: pcolormesh gestisce griglie curve!)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Geo-FNO: Flusso attorno a un cilindro (Vento Ingresso: {U_in_val:.2f})")
    
    # Riquadro 1: Fisica Reale
    # Disegniamo il cilindro (il buco nero al centro)
    circle1 = plt.Circle((0, 0), 1.0, color='black')
    axes[0].add_patch(circle1)
    # pcolormesh "piega" l'immagine rettangolare secondo le coordinate X e Y che le diamo
    pcm1 = axes[0].pcolormesh(X_grid, Y_grid, true_magnitude, cmap='jet', shading='gouraud')
    axes[0].set_title("Soluzione Analitica Reale")
    axes[0].set_aspect('equal')
    fig.colorbar(pcm1, ax=axes[0], label='Velocità')

    # Riquadro 2: Geo-FNO
    circle2 = plt.Circle((0, 0), 1.0, color='black')
    axes[1].add_patch(circle2)
    pcm2 = axes[1].pcolormesh(X_grid, Y_grid, pred_magnitude, cmap='jet', shading='gouraud')
    axes[1].set_title("Predizione Geo-FNO")
    axes[1].set_aspect('equal')
    fig.colorbar(pcm2, ax=axes[1], label='Velocità')

    plt.tight_layout()
    plt.savefig('cylinder_flow_result.png', dpi=200)
    print("Fatto! L'immagine 'cylinder_flow_result.png' è stata generata.")

if __name__ == "__main__":
    main()
