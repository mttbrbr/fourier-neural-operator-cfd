import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d_Masked(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d_Masked, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Input: 5 canali -> u(t), v(t), maschera_cilindro, grid_x, grid_y
        self.p = nn.Linear(5, self.width)
        
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.mlp0 = nn.Conv2d(self.width, self.width, 1)
        self.mlp1 = nn.Conv2d(self.width, self.width, 1)
        self.mlp2 = nn.Conv2d(self.width, self.width, 1)
        self.mlp3 = nn.Conv2d(self.width, self.width, 1)

        self.q = nn.Linear(self.width, 128)
        # Output: 2 canali -> u(t+1), v(t+1)
        self.q_out = nn.Linear(128, 2)

    def forward(self, x):
        # Salviamo la maschera per forzare le velocità a zero dentro il cilindro alla fine!
        mask = x[..., 2:3] # Il terzo canale è la maschera (1 se cilindro, 0 altrimenti)

        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.conv0(x)
        x2 = self.mlp0(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv1(x)
        x2 = self.mlp1(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv2(x)
        x2 = self.mlp2(x)
        x = F.gelu(x1 + x2)

        x1 = self.conv3(x)
        x2 = self.mlp3(x)
        x = x1 + x2

        x = x.permute(0, 2, 3, 1)
        x = self.q(x)
        x = F.gelu(x)
        x = self.q_out(x)
        
        # Riapplichiamo la maschera per garantire la condizione "No-Slip"
        # dove c'è il cilindro (mask == 1), la velocità deve essere 0.
        # dove c'è fluido (mask == 0), manteniamo la predizione.
        x = x * (1.0 - mask) 
        
        return x
