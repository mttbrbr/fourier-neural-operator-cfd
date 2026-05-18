import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import time
import math

from fno_video import FNO2d_Video

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

def generate_vortex_video_data(num_samples, resolution, timesteps=10):
    """
    Genera una sequenza video di un fluido che vortica (simulazione Semi-Lagrangiana).
    Ogni campione è una "storia" di N fotogrammi.
    Vogliamo insegnare alla rete: Frame[t] -> Frame[t+1]
    """
    print(f"Generazione dati video (Vortici e miscelazione) - {num_samples} campioni...")
    
    # Griglia da -1 a 1 per il grid_sample (che calcola lo spostamento del fluido)
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    
    # Creiamo un campo di velocità a vortice (ruota attorno al centro)
    # V_x = -Y, V_y = X
    V_x = -Y
    V_y = X
    
    dt = 0.1 # Delta time per frame
    
    # Coordinate arretrate nel tempo (da dove arriva il fluido?)
    # Metodo Semi-Lagrangiano
    back_x = X - V_x * dt
    back_y = Y - V_y * dt
    
    # Formattiamo per grid_sample: [batch, H, W, 2(x,y)]
    grid_flow = torch.stack([back_x, back_y], dim=-1).unsqueeze(0)
    
    inputs_list = []
    targets_list = []
    
    for i in range(num_samples):
        # Campo iniziale T=0 (2 o 3 macchie di colore/fluido casuali)
        field = torch.zeros((1, 1, resolution, resolution))
        for _ in range(3):
            cx, cy = (torch.rand(2) * 1.6) - 0.8
            r = (torch.rand(1) * 0.2) + 0.1
            blob = torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * r**2))
            field[0, 0] += blob
            
        # Advezione nel tempo (creiamo la sequenza video di N fotogrammi)
        frames = [field]
        for t in range(timesteps):
            # Il fluido al tempo t+1 è il fluido al tempo t, tirato indietro lungo le linee di velocità
            next_field = F.grid_sample(frames[-1], grid_flow, mode='bilinear', padding_mode='border', align_corners=True)
            frames.append(next_field)
            
        # Ora creiamo le coppie Input/Target per l'addestramento:
        # Vogliamo che la rete impari a mappare (Frame 0 -> Frame 1), (Frame 1 -> Frame 2), ecc.
        for t in range(timesteps):
            inp_frame = frames[t][0, 0]
            tgt_frame = frames[t+1][0, 0]
            
            # Aggiungiamo le griglie come features
            # Nota: le griglie per l'addestramento le normalizziamo da 0 a 1 per aiutare la rete
            x_norm = torch.linspace(0, 1, resolution)
            y_norm = torch.linspace(0, 1, resolution)
            X_n, Y_n = torch.meshgrid(x_norm, y_norm, indexing='ij')
            
            inp_tensor = torch.cat([inp_frame.unsqueeze(-1), X_n.unsqueeze(-1), Y_n.unsqueeze(-1)], dim=-1)
            
            inputs_list.append(inp_tensor.unsqueeze(0))
            targets_list.append(tgt_frame.unsqueeze(-1).unsqueeze(0))
            
    # Combiniamo tutto. Avremo (num_samples * timesteps) esempi totali
    return torch.cat(inputs_list, dim=0), torch.cat(targets_list, dim=0)

def main():
    ntrain_seq = 100 # 100 sequenze * 10 frames = 1000 esempi
    ntest_seq = 20
    batch_size = 20
    epochs = 40
    modes1, modes2, width, resolution = 12, 12, 32, 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Addestramento Autoregressivo su: {device}")

    # Generiamo 10 fotogrammi per ogni sequenza
    x_train, y_train = generate_vortex_video_data(ntrain_seq, resolution, timesteps=10)
    x_test, y_test = generate_vortex_video_data(ntest_seq, resolution, timesteps=10)
    
    print(f"Totale esempi singoli (fotogrammi): Train={len(x_train)}, Test={len(x_test)}")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

    model = FNO2d_Video(modes1, modes2, width).to(device)
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

    torch.save(model.state_dict(), 'fno_video_model.pth')
    print("Modello salvato in 'fno_video_model.pth'")

if __name__ == "__main__":
    main()
