import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math

from fno_1d import FNO1d

# --- 1. Funzione di Loss per FNO (Relative L2 Error) ---
# Nel campo degli operatori neurali, l'errore L2 relativo è lo standard
# al posto del classico MSE (Mean Squared Error).
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# --- 2. Generazione di Dati Sintetici (Equazione di Avvezione Lineare) ---
# Per iniziare subito senza scaricare file pesanti, creiamo un dataset sintetico.
# Simuliamo un'onda che si sposta nel tempo: u(x, t) = a(x - c*t)
def generate_synthetic_data(num_samples, spatial_resolution):
    print("Generazione dei dati sintetici in corso...")
    x = torch.linspace(0, 2*math.pi, spatial_resolution)
    
    # Input a(x): combinazione di seni e coseni con frequenze e fasi casuali
    a_x = torch.zeros(num_samples, spatial_resolution)
    # Output u(x): l'input traslato di una fase fissa (simulando t=1 e velocità c)
    u_x = torch.zeros(num_samples, spatial_resolution)
    
    shift = 1.0 # traslazione
    
    for i in range(num_samples):
        k1, k2 = torch.randint(1, 5, (2,))
        phi1, phi2 = torch.rand(2) * 2 * math.pi
        
        # a(x)
        a_x[i] = torch.sin(k1 * x + phi1) + 0.5 * torch.cos(k2 * x + phi2)
        # u(x) al tempo t=1
        u_x[i] = torch.sin(k1 * (x - shift) + phi1) + 0.5 * torch.cos(k2 * (x - shift) + phi2)
        
    # Il FNO ha bisogno anche della griglia spaziale come input (x)
    # Forma finale dell'input: (batch, spaziale, 2 features [a(x), x])
    grid = x.view(1, -1, 1).repeat(num_samples, 1, 1)
    a_x = a_x.view(num_samples, -1, 1)
    
    input_data = torch.cat([a_x, grid], dim=-1)
    target_data = u_x.view(num_samples, -1, 1)
    
    return input_data, target_data

# --- 3. Loop di Addestramento ---
def main():
    # Iperparametri
    ntrain = 1000
    ntest = 200
    batch_size = 20
    learning_rate = 0.001
    epochs = 50
    modes = 16  # Frequenze da mantenere
    width = 64  # Dimensione dello spazio latente
    resolution = 1024 # Punti della griglia spaziale

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sto usando il device: {device}")

    # Creazione dei dati
    x_train, y_train = generate_synthetic_data(ntrain, resolution)
    x_test, y_test = generate_synthetic_data(ntest, resolution)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    # Inizializzazione Modello
    model = FNO1d(modes, width).to(device)
    print(f"Parametri del modello: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Ottimizzatore, Scheduler e Loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = LpLoss()

    print("Inizio Addestramento...")
    for ep in range(epochs):
        model.train()
        t1 = time.time()
        train_mse = 0
        train_l2 = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            
            # Loss relativa (L2) per aggiornare i pesi
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_l2 += loss.item()

        scheduler.step()
        
        # Validazione
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

    # Salva i pesi del modello
    torch.save(model.state_dict(), 'fno1d_model.pth')
    print("Modello salvato in 'fno1d_model.pth'")

if __name__ == "__main__":
    main()
