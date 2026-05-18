import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
import time

from geo_fno import GeoFNO2d

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

def generate_cylinder_data(num_samples, r_res=64, theta_res=128):
    """
    Genera il Flusso Potenziale (senza viscosità) attorno a un cilindro.
    Utilizziamo una "O-Grid": la griglia è topologicamente un rettangolo 
    (Raggio x Angolo), ma fisicamente mappa uno spazio con un buco al centro!
    Questo è l'essenza di come funziona un Geo-FNO.
    """
    print(f"Generazione flussi CFD attorno al cilindro ({num_samples} campioni)...")
    
    R_cylinder = 1.0 # Raggio del cilindro
    R_max = 5.0      # Distanza del bordo esterno
    
    # 1. Creiamo la griglia strutturata (computazionale)
    r = torch.linspace(R_cylinder, R_max, r_res)
    theta = torch.linspace(0, 2*math.pi, theta_res)
    R_grid, Theta_grid = torch.meshgrid(r, theta, indexing='ij')
    
    # 2. Trasformiamola nella griglia fisica (deformata)
    X_grid = R_grid * torch.cos(Theta_grid)
    Y_grid = R_grid * torch.sin(Theta_grid)
    
    inputs_list = []
    targets_list = []
    
    for i in range(num_samples):
        # Generiamo una velocità del vento in ingresso casuale (U_in)
        U_in = (torch.rand(1).item() * 4.0) + 1.0 # Vento tra 1.0 e 5.0
        
        # 3. Soluzione analitica del flusso potenziale attorno al cilindro
        # Velocità radiale (V_r) e tangenziale (V_theta)
        V_r = U_in * (1.0 - (R_cylinder**2 / R_grid**2)) * torch.cos(Theta_grid)
        V_theta = -U_in * (1.0 + (R_cylinder**2 / R_grid**2)) * torch.sin(Theta_grid)
        
        # Convertiamo in velocità cartesiane (V_x, V_y)
        V_x = V_r * torch.cos(Theta_grid) - V_theta * torch.sin(Theta_grid)
        V_y = V_r * torch.sin(Theta_grid) + V_theta * torch.cos(Theta_grid)
        
        # 4. Creiamo i tensori Input / Output
        # Input: Velocità costante di base + Coordinate Fisiche Deformate
        U_in_tensor = torch.full((r_res, theta_res, 1), U_in)
        X_tensor = X_grid.unsqueeze(-1)
        Y_tensor = Y_grid.unsqueeze(-1)
        
        inp = torch.cat([U_in_tensor, X_tensor, Y_tensor], dim=-1)
        
        # Target: Campo vettoriale (V_x, V_y) calcolato dalla fisica
        tgt = torch.cat([V_x.unsqueeze(-1), V_y.unsqueeze(-1)], dim=-1)
        
        inputs_list.append(inp.unsqueeze(0))
        targets_list.append(tgt.unsqueeze(0))
        
    return torch.cat(inputs_list, dim=0), torch.cat(targets_list, dim=0), X_grid, Y_grid

def main():
    ntrain = 400
    ntest = 100
    batch_size = 20
    epochs = 60
    modes1, modes2, width = 12, 12, 32
    r_res, theta_res = 64, 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Addestramento Geo-FNO su {device}...")

    # Generazione dati
    x_train, y_train, X_grid, Y_grid = generate_cylinder_data(ntrain, r_res, theta_res)
    x_test, y_test, _, _ = generate_cylinder_data(ntest, r_res, theta_res)
    
    # Salviamo la griglia fisica deformata per poterla disegnare dopo nell'inferenza!
    torch.save({'X': X_grid, 'Y': Y_grid}, 'cylinder_grid.pth')

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = GeoFNO2d(modes1, modes2, width).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
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

    torch.save(model.state_dict(), 'geofno_model.pth')
    print("Modello Geo-FNO salvato in 'geofno_model.pth'")

if __name__ == "__main__":
    main()
