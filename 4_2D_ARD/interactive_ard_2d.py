import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np

from fno_ard_2d import FNO2d_ARD

def main():
    modes1, modes2, width = 12, 12, 32
    resolution = 256  # <-- Aumentata la risoluzione per la zero-shot super-resolution!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FNO2d_ARD(modes1, modes2, width).to(device)
    try:
        model.load_state_dict(torch.load('fno2d_ard_model.pth', weights_only=True))
    except FileNotFoundError:
        print("Devi prima eseguire 'python train_ard_2d.py'!")
        return
    model.eval()

    # Griglie costanti
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X_grid, Y_grid = torch.meshgrid(x, y, indexing='ij')
    X_tensor = X_grid.unsqueeze(-1)
    Y_tensor = Y_grid.unsqueeze(-1)

    f_data = np.zeros((resolution, resolution), dtype=np.float32)
    
    # Parametri iniziali
    curr_vx = 0.0
    curr_vy = 0.0
    curr_k = 0.1

    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title("FNO ARD Interattivo: Vento e Reazione")
    
    ax1 = fig.add_axes([0.05, 0.3, 0.4, 0.6])
    ax2 = fig.add_axes([0.55, 0.3, 0.4, 0.6])

    im1 = ax1.imshow(f_data, cmap='magma', origin='lower', vmin=-3, vmax=3)
    ax1.set_title("1. Disegna la Sorgente f(x,y)\n(Tieni premuto sx/dx)")
    
    u_data = np.zeros((resolution, resolution), dtype=np.float32)
    im2 = ax2.imshow(u_data, cmap='viridis', origin='lower', vmin=-1, vmax=1)
    ax2.set_title("2. Soluzione Live u(x,y)")

    # --- SLIDERS ---
    axcolor = 'lightgoldenrodyellow'
    ax_vx = fig.add_axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_vy = fig.add_axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_k = fig.add_axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

    s_vx = Slider(ax_vx, 'Vento X (Advezione)', -5.0, 5.0, valinit=curr_vx)
    s_vy = Slider(ax_vy, 'Vento Y (Advezione)', -5.0, 5.0, valinit=curr_vy)
    s_k = Slider(ax_k, 'Assorbimento (Reazione)', 0.1, 5.0, valinit=curr_k)

    def update_prediction(val=None):
        vx_val = s_vx.val
        vy_val = s_vy.val
        k_val = s_k.val
        
        f_tensor = torch.tensor(f_data, dtype=torch.float32).unsqueeze(-1)
        vx_tensor = torch.full((resolution, resolution, 1), vx_val, dtype=torch.float32)
        vy_tensor = torch.full((resolution, resolution, 1), vy_val, dtype=torch.float32)
        k_tensor = torch.full((resolution, resolution, 1), k_val, dtype=torch.float32)
        
        inputs = torch.cat([f_tensor, vx_tensor, vy_tensor, k_tensor, X_tensor, Y_tensor], dim=-1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            u_pred = model(inputs)[0, :, :, 0].cpu().numpy()
            
        im1.set_data(f_data)
        im2.set_data(u_pred)
        if u_pred.max() - u_pred.min() > 0.01:
            im2.set_clim(vmin=u_pred.min(), vmax=u_pred.max())
        fig.canvas.draw_idle()

    s_vx.on_changed(update_prediction)
    s_vy.on_changed(update_prediction)
    s_k.on_changed(update_prediction)

    # Mouse Drawing (uguale a prima)
    is_drawing, mouse_button = False, 1
    def add_force_blob(event):
        nonlocal f_data
        if event.inaxes != ax1: return
        x_idx, y_idx = int(event.xdata), int(event.ydata)
        y_g, x_g = np.ogrid[0:resolution, 0:resolution]
        dist = np.sqrt((x_g - x_idx)**2 + (y_g - y_idx)**2)
        # Rendiamo la dimensione del pennello proporzionale alla risoluzione (prima era 8.0 fissa per 64x64)
        brush_radius_sq = (resolution / 16.0)**2 
        blob = np.exp(-(dist**2) / brush_radius_sq)
        
        if mouse_button == 1: f_data += blob
        elif mouse_button == 3: f_data -= blob
        f_data = np.clip(f_data, -5, 5)
        update_prediction()

    def on_press(event):
        nonlocal is_drawing, mouse_button
        if event.inaxes == ax1:
            is_drawing = True
            mouse_button = event.button
            add_force_blob(event)

    def on_release(event):
        nonlocal is_drawing
        is_drawing = False

    def on_motion(event):
        if is_drawing: add_force_blob(event)

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    # Pulsante Reset
    ax_reset = fig.add_axes([0.85, 0.05, 0.1, 0.04])
    btn_reset = Button(ax_reset, 'Pulisci Tela', hovercolor='0.975')
    def reset(event):
        nonlocal f_data
        f_data.fill(0)
        update_prediction()
    btn_reset.on_clicked(reset)

    plt.show()

if __name__ == "__main__":
    main()
