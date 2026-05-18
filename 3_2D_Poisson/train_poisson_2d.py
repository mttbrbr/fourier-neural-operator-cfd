import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math

# Usiamo la stessa architettura 2D!
from fno_2d import FNO2d

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

def generate_poisson_data(num_samples, resolution):
    """
    Genera dati per l'Equazione di Poisson: -Δu = f
    Questa è l'equazione più importante (e costosa da calcolare) nei solutori CFD per 
    trovare il campo delle pressioni nei fluidi incomprimibili (Navier-Stokes).
    
    Input (f): Un campo di forze generato casualmente (Gaussian Random Field).
    Output (u): Il campo di potenziale (es. la pressione) che risolve l'equazione.
    """
    print(f"Generazione di {num_samples} campioni per l'Equazione di Poisson...")
    
    # Prepariamo le frequenze per calcolare le derivate spaziali con la FFT
    kx = torch.fft.fftfreq(resolution, d=1.0/resolution) * 2 * math.pi
    ky = torch.fft.fftfreq(resolution, d=1.0/resolution) * 2 * math.pi
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    
    # Operatore di Laplace nel dominio di Fourier: -(kx^2 + ky^2)
    K_sq = Kx**2 + Ky**2
    K_sq[0, 0] = 1.0 # Evitiamo la divisione per zero sulla frequenza media
    
    f_all = torch.zeros(num_samples, resolution, resolution)
    u_all = torch.zeros(num_samples, resolution, resolution)
    
    for i in range(num_samples):
        # 1. Generiamo rumore bianco casuale
        noise = torch.randn(resolution, resolution, dtype=torch.cfloat)
        
        # 2. Lo rendiamo un "Gaussian Random Field" smussandolo (attenuiamo le alte frequenze)
        f_hat = noise / (K_sq + 1.0)**1.5 
        f_hat[0, 0] = 0.0 # Assicuriamoci che la media sia zero per la periodicità
        
        # Torniamo nel dominio spaziale per ottenere il campo di forze f(x,y)
        f = torch.fft.ifft2(f_hat).real
        
        # 3. Risolviamo l'Equazione di Poisson in modo esatto!
        # Matematicamente, se -Δu = f, nel dominio di Fourier u_hat = f_hat / (kx^2 + ky^2)
        u_hat = f_hat / K_sq
        u_hat[0, 0] = 0.0
        
        # Torniamo nel dominio spaziale per ottenere la soluzione esatta u(x,y)
        u = torch.fft.ifft2(u_hat).real
        
        # Normalizziamo i campi per aiutare l'addestramento della rete
        f = f / f.std()
        u = u / u.std()
        
        f_all[i] = f
        u_all[i] = u
        
    # Aggiungiamo le coordinate spaziali
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    X = X.unsqueeze(0).repeat(num_samples, 1, 1).unsqueeze(-1)
    Y = Y.unsqueeze(0).repeat(num_samples, 1, 1).unsqueeze(-1)
    f_tensor = f_all.unsqueeze(-1)
    u_tensor = u_all.unsqueeze(-1)
    
    # Input: [forza_f, griglia_x, griglia_y]
    inputs = torch.cat([f_tensor, X, Y], dim=-1)
    
    return inputs, u_tensor

def main():
    ntrain = 800
    ntest = 200
    batch_size = 10
    learning_rate = 0.001
    epochs = 50
    
    modes1 = 12
    modes2 = 12
    width = 32
    resolution = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Risoluzione PDE complessa su: {device}")

    # Generazione dei dati complessi
    x_train, y_train = generate_poisson_data(ntrain, resolution)
    x_test, y_test = generate_poisson_data(ntest, resolution)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = FNO2d(modes1, modes2, width).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = LpLoss()

    print("Inizio Addestramento (Equazione di Poisson)...")
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

    torch.save(model.state_dict(), 'fno2d_poisson_model.pth')
    print("Modello Poisson salvato in 'fno2d_poisson_model.pth'")

if __name__ == "__main__":
    main()
