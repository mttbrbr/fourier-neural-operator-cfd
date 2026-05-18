import torch
import matplotlib.pyplot as plt
from fno_2d import FNO2d
from train_poisson_2d import generate_poisson_data

def main():
    modes1, modes2, width, resolution = 12, 12, 32, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eseguo l'inferenza Poisson su: {device}")

    # 1. Carichiamo la rete addestrata per Poisson
    model = FNO2d(modes1, modes2, width).to(device)
    try:
        model.load_state_dict(torch.load('fno2d_poisson_model.pth', weights_only=True))
    except FileNotFoundError:
        print("Errore: devi prima eseguire 'python train_poisson_2d.py'!")
        return
    model.eval()

    # 2. Generiamo UN campione di test mai visto dalla rete
    input_tensor, true_output = generate_poisson_data(1, resolution)
    input_tensor = input_tensor.to(device)

    # 3. Facciamo la predizione
    with torch.no_grad():
        prediction = model(input_tensor)
        
    # Estraiamo le immagini
    # input_tensor ha shape [1, 64, 64, 3]. Il canale 0 è il campo di forza f(x,y)
    f_image = input_tensor[0, :, :, 0].cpu().numpy()
    true_u_image = true_output[0, :, :, 0].cpu().numpy()
    pred_u_image = prediction[0, :, :, 0].cpu().numpy()
    
    error_image = abs(true_u_image - pred_u_image)

    # 4. Creiamo il grafico
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # A. Input f(x,y)
    im0 = axes[0].imshow(f_image, cmap='magma', origin='lower')
    axes[0].set_title("1. INPUT: Campo di forze $f(x,y)$ \n(Generato casualmente)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    # B. Ground Truth u(x,y)
    im1 = axes[1].imshow(true_u_image, cmap='viridis', origin='lower')
    axes[1].set_title("2. REALTÀ: Campo Pressione $u(x,y)$\n(Soluzione matematica di -Δu = f)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # C. Predizione FNO
    im2 = axes[2].imshow(pred_u_image, cmap='viridis', origin='lower')
    axes[2].set_title("3. PREDIZIONE RETE $u(x,y)$\n(Senza usare nessun risolutore matematico)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # D. Errore
    im3 = axes[3].imshow(error_image, cmap='coolwarm', origin='lower')
    axes[3].set_title("4. ERRORE ASSOLUTO\n(Differenza tra 2 e 3)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig('fno_2d_poisson_result.png', dpi=150)
    print("\nFatto! Il grafico è stato salvato in 'fno_2d_poisson_result.png'.")
    print("Guarda quanto i campi di input (magma) siano complessi rispetto a prima!")

if __name__ == "__main__":
    main()
