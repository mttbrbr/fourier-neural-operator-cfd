import torch
import matplotlib.pyplot as plt
import math

# Importiamo l'architettura della nostra rete
from fno_1d import FNO1d

def generate_single_sample(spatial_resolution):
    """Genera un singolo campione casuale per testare la rete."""
    x = torch.linspace(0, 2*math.pi, spatial_resolution)
    
    # Parametri casuali per un'onda
    k1, k2 = torch.randint(1, 5, (2,))
    phi1, phi2 = torch.rand(2) * 2 * math.pi
    
    # Input a(x) al tempo t=0
    a_x = torch.sin(k1 * x + phi1) + 0.5 * torch.cos(k2 * x + phi2)
    
    # Soluzione reale (Ground Truth) u(x) al tempo t=1 (traslata di shift=1.0)
    shift = 1.0
    u_x_true = torch.sin(k1 * (x - shift) + phi1) + 0.5 * torch.cos(k2 * (x - shift) + phi2)
    
    # Prepariamo l'input nel formato che la rete si aspetta: (batch=1, punti, 2_features)
    grid = x.view(1, -1, 1)
    a_x_input = a_x.view(1, -1, 1)
    
    input_data = torch.cat([a_x_input, grid], dim=-1)
    
    return input_data, u_x_true, a_x, x

def main():
    # 1. Configurazione
    modes = 16
    width = 64
    resolution = 1024
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Eseguo l'inferenza su: {device}")

    # 2. Inizializza il modello vuoto e carica i pesi allenati
    model = FNO1d(modes, width).to(device)
    
    try:
        model.load_state_dict(torch.load('fno1d_model.pth', weights_only=True))
        print("Pesi del modello caricati con successo!")
    except FileNotFoundError:
        print("Errore: file 'fno1d_model.pth' non trovato. Hai eseguito il training?")
        return

    # 3. Mettiamo il modello in modalità "Evaluation" (disabilita dropout, batchnorm, ecc.)
    model.eval()

    # 4. Generiamo un nuovo dato mai visto dalla rete
    input_tensor, true_output, original_input, x_grid = generate_single_sample(resolution)
    
    # Spostiamo l'input sulla GPU
    input_tensor = input_tensor.to(device)

    # 5. Inferenza (senza calcolare i gradienti per risparmiare memoria)
    with torch.no_grad():
        prediction = model(input_tensor)
        
    # Spostiamo la predizione di nuovo sulla CPU per il plotting e togliamo le dimensioni extra
    prediction = prediction.cpu().squeeze()
    
    # 6. Visualizzazione dei risultati
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_grid.numpy(), original_input.numpy(), label='Input $a(x)$ (Tempo $t=0$)', color='gray', linestyle='--')
    plt.plot(x_grid.numpy(), true_output.numpy(), label='Ground Truth $u(x)$ (Tempo $t=1$)', color='blue', linewidth=2)
    plt.plot(x_grid.numpy(), prediction.numpy(), label='Predizione FNO $\hat{u}(x)$', color='red', linestyle=':', linewidth=3)
    
    plt.title('Test del Fourier Neural Operator 1D')
    plt.xlabel('Spazio $x$')
    plt.ylabel('Ampiezza')
    plt.legend()
    plt.grid(True)
    
    # Salva il grafico
    plt.savefig('fno_prediction_result.png')
    print("Grafico salvato in 'fno_prediction_result.png'. Aprilo per vedere il risultato!")

if __name__ == "__main__":
    main()
