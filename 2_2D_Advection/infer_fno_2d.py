import torch
import matplotlib.pyplot as plt
from fno_2d import FNO2d
from train_fno_2d import generate_synthetic_data_2d

def main():
    modes1, modes2, width, resolution = 12, 12, 32, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eseguo l'inferenza 2D su: {device}")

    # 1. Carichiamo la rete
    model = FNO2d(modes1, modes2, width).to(device)
    try:
        model.load_state_dict(torch.load('fno2d_model.pth', weights_only=True))
    except FileNotFoundError:
        print("Errore: devi prima eseguire 'python train_fno_2d.py'!")
        return
    model.eval()

    # 2. Generiamo UN SOLO campione di test
    input_tensor, true_output = generate_synthetic_data_2d(1, resolution)
    input_tensor = input_tensor.to(device)

    # 3. Facciamo la predizione
    with torch.no_grad():
        prediction = model(input_tensor)
        
    # Spostiamo tutto su CPU e rimuoviamo le dimensioni extra per il plot (da [1, 64, 64, 1] a [64, 64])
    input_image = input_tensor[0, :, :, 0].cpu().numpy()  # Prendiamo solo il campo 'a', non le griglie x e y
    true_image = true_output[0, :, :, 0].cpu().numpy()
    pred_image = prediction[0, :, :, 0].cpu().numpy()
    
    # Calcoliamo la differenza assoluta tra la realtà e la nostra rete
    error_image = abs(true_image - pred_image)

    # 4. Creiamo un bel grafico esplicativo!
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # A. Input (Il passato)
    im0 = axes[0].imshow(input_image, cmap='hot', origin='lower')
    axes[0].set_title("1. Input (Tempo T=0)\n(La macchia di fluido iniziale)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # B. Ground Truth (Il futuro reale)
    im1 = axes[1].imshow(true_image, cmap='hot', origin='lower')
    axes[1].set_title("2. Realtà (Tempo T=1)\n(Dove si è spostata fisicamente)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # C. Predizione (Il futuro immaginato dalla rete)
    im2 = axes[2].imshow(pred_image, cmap='hot', origin='lower')
    axes[2].set_title("3. Predizione Rete (Tempo T=1)\n(Cosa ha imparato la rete)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # D. Errore (La differenza)
    im3 = axes[3].imshow(error_image, cmap='coolwarm', origin='lower')
    axes[3].set_title("4. Errore Assoluto\n(Differenza tra 2 e 3)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('fno_2d_explanation.png', dpi=150)
    print("Grafico 2D salvato in 'fno_2d_explanation.png'. Apri questo file, spiegherà tutto visivamente!")

if __name__ == "__main__":
    main()
