import torch
import matplotlib.pyplot as plt
import numpy as np

# Importiamo la rete 2D che hai appena addestrato
from fno_2d import FNO2d

def main():
    modes1, modes2, width, resolution = 12, 12, 32, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Avvio ambiente interattivo su: {device}")

    # 1. Carichiamo la rete addestrata
    model = FNO2d(modes1, modes2, width).to(device)
    try:
        model.load_state_dict(torch.load('fno2d_poisson_model.pth', weights_only=True))
    except FileNotFoundError:
        print("Errore: devi prima eseguire 'python train_poisson_2d.py'!")
        return
    model.eval()

    # 2. Prepariamo i dati costanti (Le griglie X e Y che la rete richiede)
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X_grid, Y_grid = torch.meshgrid(x, y, indexing='ij')
    X_tensor = X_grid.unsqueeze(-1)
    Y_tensor = Y_grid.unsqueeze(-1)

    # Il nostro campo di forza 'f' iniziale (tutto a zero)
    f_data = np.zeros((resolution, resolution), dtype=np.float32)

    # 3. Setup della Finestra Interattiva con Matplotlib
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.canvas.manager.set_window_title("FNO CFD Interattivo - Equazione di Poisson")

    # Inizializziamo le immagini
    im1 = ax1.imshow(f_data, cmap='magma', origin='lower', vmin=-3, vmax=3)
    ax1.set_title("INPUT: Campo di Forze f(x,y)\n[Tieni premuto e Disegna col Mouse!]")
    
    # Facciamo una prima predizione (su campo vuoto) per inizializzare l'output
    u_data = np.zeros((resolution, resolution), dtype=np.float32)
    im2 = ax2.imshow(u_data, cmap='viridis', origin='lower', vmin=-1, vmax=1)
    ax2.set_title("OUTPUT LIVE: Pressione u(x,y)\n(Calcolato in tempo reale dal FNO)")

    # Testo di aiuto
    plt.figtext(0.5, 0.02, "Tasto Sinistro: Aggiungi Forza (+) | Tasto Destro: Forza Negativa (-) | Premi 'C' per pulire", 
                ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})

    # 4. Funzione per aggiornare la predizione Live
    def update_prediction():
        # Convertiamo il numpy array che stiamo disegnando in Tensor
        f_tensor = torch.tensor(f_data, dtype=torch.float32).unsqueeze(-1)
        
        # Uniamo f, X, Y (Shape: [1, 64, 64, 3])
        inputs = torch.cat([f_tensor, X_tensor, Y_tensor], dim=-1).unsqueeze(0).to(device)
        
        # Inferenza ultra-veloce (frazione di millisecondo su ROCm)
        with torch.no_grad():
            prediction = model(inputs)
            
        u_pred = prediction[0, :, :, 0].cpu().numpy()
        
        # Aggiorniamo i grafici
        im1.set_data(f_data)
        # Adattiamo i colori dell'output dinamicamente
        im2.set_data(u_pred)
        im2.set_clim(vmin=u_pred.min() - 0.1, vmax=u_pred.max() + 0.1)
        
        fig.canvas.draw_idle()

    # 5. Gestione del Mouse (Disegno)
    is_drawing = False
    mouse_button = 1 # 1=Left, 3=Right

    def add_force_blob(event):
        nonlocal f_data
        if event.inaxes != ax1: return
        
        # Coordinate del mouse sulla griglia 64x64
        x_idx, y_idx = int(event.xdata), int(event.ydata)
        
        # Creiamo un "pennello" a forma di campana Gaussiana
        y_g, x_g = np.ogrid[0:resolution, 0:resolution]
        dist = np.sqrt((x_g - x_idx)**2 + (y_g - y_idx)**2)
        blob = np.exp(-(dist**2) / (2 * 2**2)) # Raggio del pennello = 2
        
        # Aggiungiamo o togliamo forza
        if mouse_button == 1:
            f_data += blob
        elif mouse_button == 3:
            f_data -= blob
            
        # Limitiamo i valori massimi per evitare instabilità nei colori
        f_data = np.clip(f_data, -5, 5)
        update_prediction()

    def on_press(event):
        nonlocal is_drawing, mouse_button
        is_drawing = True
        mouse_button = event.button
        add_force_blob(event)

    def on_release(event):
        nonlocal is_drawing
        is_drawing = False

    def on_motion(event):
        if is_drawing:
            add_force_blob(event)

    def on_key(event):
        nonlocal f_data
        if event.key.lower() == 'c':
            f_data.fill(0) # Pulisce lo schermo
            update_prediction()

    # Colleghiamo gli eventi di Matplotlib
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print("\n=== FINESTRA INTERATTIVA APERTA! ===")
    print("- Usa il mouse nel riquadro di SINISTRA per 'dipingere' il campo di forze.")
    print("- Guarda il riquadro di DESTRA aggiornarsi in tempo reale!")
    
    # Mostriamo la finestra (blocca l'esecuzione dello script finché non la chiudi)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()

if __name__ == "__main__":
    main()
