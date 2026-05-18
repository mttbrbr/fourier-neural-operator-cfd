import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        """
        2D Fourier layer. Fa la FFT 2D, filtra le frequenze, e fa la iFFT 2D.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Frequenze sull'asse X
        self.modes2 = modes2 # Frequenze sull'asse Y

        scale = (1 / (in_channels * out_channels))
        # Nei segnali 2D dobbiamo considerare due set di pesi per gli angoli della matrice di Fourier
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        # Moltiplicazione complessa: batch, in_channel, x, y * in_channel, out_channel, x, y
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Trasformata di Fourier 2D
        x_ft = torch.fft.rfft2(x)

        # Tensore per i risultati (attenzione alle dimensioni di rfft2 che dimezza l'ultima dimensione)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # Moltiplichiamo le basse frequenze. In 2D si prendono gli "angoli" della matrice delle frequenze
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Trasformata inversa 2D
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Input ha 3 features: il campo iniziale a(x,y) e le griglie spaziali x e y
        self.p = nn.Linear(3, self.width)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        
        self.mlp0 = nn.Conv2d(self.width, self.width, 1)
        self.mlp1 = nn.Conv2d(self.width, self.width, 1)
        self.mlp2 = nn.Conv2d(self.width, self.width, 1)
        self.mlp3 = nn.Conv2d(self.width, self.width, 1)

        self.q = nn.Linear(self.width, 128)
        self.q_out = nn.Linear(128, 1) # L'output è 1 feature: la soluzione u(x,y)

    def forward(self, x):
        # x shape: (batch, dim_x, dim_y, features)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2) # (batch, channels, x, y) per le Conv2d

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

        x = x.permute(0, 2, 3, 1) # Torniamo a (batch, x, y, channels)
        x = self.q(x)
        x = F.gelu(x)
        x = self.q_out(x)
        
        return x
