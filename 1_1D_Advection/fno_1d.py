import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. Fa la FFT, moltiplica per i pesi nel dominio delle frequenze, e fa la iFFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Numero di modalità di Fourier da tenere (le basse frequenze)

        # Pesi apprendibili nel dominio di Fourier (numeri complessi)
        scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    def compl_mul1d(self, input, weights):
        # Moltiplicazione complessa: (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Trasformata di Fourier Veloce per input reali (rfft)
        x_ft = torch.fft.rfft(x)

        # Inizializziamo il tensore per i risultati nel dominio delle frequenze
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        # Moltiplichiamo solo le basse frequenze (da 0 a modes1)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Trasformata inversa (irfft) per tornare al dominio spaziale
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        """
        Architettura completa del Fourier Neural Operator 1D.
        """
        self.modes1 = modes
        self.width = width

        # A) Lifting: da 2 feature di input (es. valore della funzione a(x) e coordinata x) alla 'width'
        self.p = nn.Linear(2, self.width)

        # B) I layer di Fourier (con bypass lineare in parallelo)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        
        self.mlp0 = nn.Conv1d(self.width, self.width, 1)
        self.mlp1 = nn.Conv1d(self.width, self.width, 1)
        self.mlp2 = nn.Conv1d(self.width, self.width, 1)
        self.mlp3 = nn.Conv1d(self.width, self.width, 1)

        # C) Projection: proietta dalla 'width' all'output desiderato (es. 1 valore, la soluzione u(x, t))
        self.q = nn.Linear(self.width, 128)
        self.q_out = nn.Linear(128, 1)

    def forward(self, x):
        # x.shape = (batch, punti_spaziali, features_input)
        
        # 1. Lifting
        x = self.p(x) 
        x = x.permute(0, 2, 1) # Riordiniamo le dimensioni per le Conv1d: (batch, width, punti_spaziali)

        # 2. Fourier Layers (4 strati in sequenza con attivazione GELU)
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
        x = x1 + x2 # Niente attivazione all'ultimo livello di feature

        # 3. Projection
        x = x.permute(0, 2, 1) # Riordiniamo: (batch, punti_spaziali, width)
        x = self.q(x)
        x = F.gelu(x)
        x = self.q_out(x)
        
        return x
