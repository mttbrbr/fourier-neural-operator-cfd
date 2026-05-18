import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

from fno_masked import FNO2d_Masked

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Addestramento Autoregressivo su Dati Reali CFD ({device})...")

    # 1. Carichiamo i dati pre-processati
    try:
        data = torch.load('gridded_wake_data.pt', weights_only=True)
    except FileNotFoundError:
        print("Errore: Esegui prima 'python preprocess_data.py'")
        return

    video = data['video'] # Shape: (200, 128, 64, 2)
    grid_x = data['grid_x']
    grid_y = data['grid_y']
    mask = data['mask'].float() # 1 nel cilindro, 0 fuori
    
    num_frames = video.shape[0]
    
    # 2. Prepariamo le coppie (T) -> (T+1)
    inputs = []
    targets = []
    
    for t in range(num_frames - 1):
        # Input: Velocità u(t), v(t), Mask, GridX, GridY
        frame_t = video[t]
        grid_X_tensor = grid_x.unsqueeze(-1).float()
        grid_Y_tensor = grid_y.unsqueeze(-1).float()
        mask_tensor = mask.unsqueeze(-1).float()
        
        inp = torch.cat([frame_t, mask_tensor, grid_X_tensor, grid_Y_tensor], dim=-1)
        
        # Target: Velocità u(t+1), v(t+1)
        tgt = video[t+1]
        
        inputs.append(inp.unsqueeze(0))
        targets.append(tgt.unsqueeze(0))
        
    X_all = torch.cat(inputs, dim=0)
    Y_all = torch.cat(targets, dim=0)
    
    # Split train/test (I primi 150 frame per imparare la fisica, gli ultimi 49 per testare la generalizzazione)
    train_split = 150
    X_train, Y_train = X_all[:train_split], Y_all[:train_split]
    X_test, Y_test = X_all[train_split:], Y_all[train_split:]

    batch_size = 10
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test), batch_size=batch_size, shuffle=False)

    # 3. Addestramento Modello FNO
    modes1, modes2, width = 12, 12, 32
    model = FNO2d_Masked(modes1, modes2, width).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = LpLoss()
    
    epochs = 40
    print(f"Inizio Addestramento su {train_split} coppie di fotogrammi...")
    
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

    torch.save(model.state_dict(), 'fno_real_wake_model.pth')
    print("Modello salvato in 'fno_real_wake_model.pth'")

if __name__ == "__main__":
    main()
