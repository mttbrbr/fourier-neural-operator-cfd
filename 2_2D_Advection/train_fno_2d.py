import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math

from fno_2d import FNO2d

# La stessa loss che abbiamo usato in 1D
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average: return torch.mean(diff_norms/y_norms)
            else: return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def generate_synthetic_data_2d(num_samples, spatial_resolution):
    """
    Genera un problema CFD giocattolo 2D (es. Avvezione/Traslazione di una macchia di fluido).
    Immagina una macchia di calore (una Gaussiana) che si sposta diagonalmente nel tempo.
    """
    print(f"Generazione {num_samples} campioni 2D in corso...")
    grid_size = spatial_resolution
    
    # Creiamo la griglia spaziale (x, y)
    x = torch.linspace(0, 1, grid_size)
    y = torch.linspace(0, 1, grid_size)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    
    # Inizializziamo i tensori di input (tempo 0) e output (tempo 1)
    a_xy = torch.zeros(num_samples, grid_size, grid_size)
    u_xy = torch.zeros(num_samples, grid_size, grid_size)
    
    # La macchia si sposterà di questa quantità in basso a destra
    shift_x = 0.2
    shift_y = 0.2
    
    for i in range(num_samples):
        # Posizione casuale iniziale della macchia di fluido
        x0 = torch.rand(1) * 0.4 + 0.1
        y0 = torch.rand(1) * 0.4 + 0.1
        sigma = 0.05 + torch.rand(1) * 0.05 # Dimensione casuale della macchia
        
        # Campo iniziale a(x, y) al tempo t=0
        a_xy[i] = torch.exp(-((grid_x - x0)**2 + (grid_y - y0)**2) / (2 * sigma**2))
        
        # Campo finale u(x, y) al tempo t=1 (traslato)
        u_xy[i] = torch.exp(-((grid_x - (x0 + shift_x))**2 + (grid_y - (y0 + shift_y))**2) / (2 * sigma**2))
    
    # Aggiungiamo le coordinate spaziali come features per aiutare la rete
    # Shape diventerà: (batch, dim_x, dim_y, 3_features)
    grid_x_t = grid_x.unsqueeze(0).unsqueeze(-1).repeat(num_samples, 1, 1, 1)
    grid_y_t = grid_y.unsqueeze(0).unsqueeze(-1).repeat(num_samples, 1, 1, 1)
    a_xy_t = a_xy.unsqueeze(-1)
    
    input_data = torch.cat([a_xy_t, grid_x_t, grid_y_t], dim=-1)
    target_data = u_xy.unsqueeze(-1)
    
    return input_data, target_data

def main():
    ntrain = 500
    ntest = 100
    batch_size = 10
    learning_rate = 0.001
    epochs = 40
    
    modes1 = 12 # Modalità da tenere su asse X
    modes2 = 12 # Modalità da tenere su asse Y
    width = 32  # Canali latenti (teniamo basso per fare in fretta)
    resolution = 64 # Immagini 64x64 pixel

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    x_train, y_train = generate_synthetic_data_2d(ntrain, resolution)
    x_test, y_test = generate_synthetic_data_2d(ntest, resolution)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = FNO2d(modes1, modes2, width).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = LpLoss()

    print("Inizio Addestramento 2D...")
    for ep in range(epochs):
        model.train()
        t1 = time.time()
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()
        
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                test_l2 += criterion(out, y).item()

        train_l2 /= len(train_loader)
        test_l2 /= len(test_loader)
        t2 = time.time()
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1:03d} | Time: {t2-t1:.2f}s | Train Rel L2: {train_l2:.5f} | Test Rel L2: {test_l2:.5f} | LR: {scheduler.get_last_lr()[0]:.5f}")

    torch.save(model.state_dict(), 'fno2d_model.pth')
    print("Modello salvato in 'fno2d_model.pth'")

if __name__ == "__main__":
    main()
