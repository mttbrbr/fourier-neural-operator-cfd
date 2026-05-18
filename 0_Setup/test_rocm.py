import torch

print("=== Test PyTorch con ROCm ===")
print(f"Versione di PyTorch installata: {torch.__version__}")
print(f"CUDA (ROCm) è disponibile? {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Numero di GPU rilevate: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Test di allocazione e calcolo su GPU
    print("\nEseguo un test di calcolo tensoriale sulla GPU...")
    try:
        x = torch.rand(5000, 5000).cuda()
        y = torch.rand(5000, 5000).cuda()
        z = torch.matmul(x, y)
        print("Calcolo completato con successo sulla GPU!")
    except Exception as e:
        print(f"Errore durante il calcolo su GPU: {e}")
else:
    print("\nATTENZIONE: PyTorch non rileva la GPU. Verifica l'installazione di ROCm.")
