import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
import math
from fno_ard_2d import FNO2d_ARD

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

def generate_ard_data(num_samples, resolution):
    """
    Genera dati per l'Equazione Advezione-Reazione-Diffusione (ARD):
    -D * Δu + (vx*du/dx + vy*du/dy) + k*u = f
    D è costante (=0.01). vx, vy e k variano casualmente per ogni campione.
    """
    print(f"Generazione {num_samples} campioni ARD...")
    D = 0.01 # Diffusione fissa
    
    kx = torch.fft.fftfreq(resolution, d=1.0/resolution) * 2 * math.pi
    ky = torch.fft.fftfreq(resolution, d=1.0/resolution) * 2 * math.pi
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    K_sq = Kx**2 + Ky**2
    
    inputs_list = []
    targets_list = []
    
    for i in range(num_samples):
        # Parametri casuali
        vx = (torch.rand(1).item() * 10) - 5 # Vento X tra -5 e 5
        vy = (torch.rand(1).item() * 10) - 5 # Vento Y tra -5 e 5
        k_react = (torch.rand(1).item() * 5) + 0.1 # Reazione tra 0.1 e 5.1
        
        # Generiamo il campo di forze f (Input)
        noise = torch.randn(resolution, resolution, dtype=torch.cfloat)
        f_hat = noise / (K_sq + 1.0)**1.5
        f_hat[0, 0] = 0.0
        f = torch.fft.ifft2(f_hat).real
        f = f / f.std()
        
        # Risolviamo nel dominio di Fourier
        # Spettro operatore: D*(kx^2 + ky^2) + i*(vx*kx + vy*ky) + k_react
        L_hat = D * K_sq + k_react + 1j * (vx * Kx + vy * Ky)
        u_hat = f_hat / L_hat
        u_hat[0,0] = 0.0
        u = torch.fft.ifft2(u_hat).real
        u = u / u.std() if u.std() > 0 else u
        
        # Creiamo i tensori di input spaziali
        f_tensor = f.unsqueeze(-1)
        vx_tensor = torch.full((resolution, resolution, 1), vx)
        vy_tensor = torch.full((resolution, resolution, 1), vy)
        k_tensor = torch.full((resolution, resolution, 1), k_react)
        
        # Griglie
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        X_tensor = X.unsqueeze(-1)
        Y_tensor = Y.unsqueeze(-1)
        
        # 6 Canali!
        inp = torch.cat([f_tensor, vx_tensor, vy_tensor, k_tensor, X_tensor, Y_tensor], dim=-1)
        inputs_list.append(inp.unsqueeze(0))
        targets_list.append(u.unsqueeze(-1).unsqueeze(0))
        
    return torch.cat(inputs_list, dim=0), torch.cat(targets_list, dim=0)

def main():
    ntrain, ntest = 800, 200
    batch_size, epochs = 10, 40
    modes1, modes2, width, resolution = 12, 12, 32, 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Addestramento ARD su: {device}")

    x_train, y_train = generate_ard_data(ntrain, resolution)
    x_test, y_test = generate_ard_data(ntest, resolution)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = FNO2d_ARD(modes1, modes2, width).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = LpLoss()

    for ep in range(epochs):
        model.train()
        train_l2 = 0
        t1 = time.time()
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
                test_l2 += criterion(model(x), y).item()

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1:03d} | Time: {time.time()-t1:.2f}s | Train L2: {train_l2/len(train_loader):.4f} | Test L2: {test_l2/len(test_loader):.4f}")

    torch.save(model.state_dict(), 'fno2d_ard_model.pth')
    print("Modello ARD salvato in 'fno2d_ard_model.pth'")

if __name__ == "__main__":
    main()
