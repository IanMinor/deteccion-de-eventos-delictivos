from torch.utils.data import DataLoader

import torch
import os

from model import Model
from dataset import Dataset
from test import test
import option
from utils import Visualizer
# viz = Visualizer(env='DeepMIL', use_incoming_socket=False)
# torch.set_default_tensor_type('torch.cuda.FloatTensor')  # Movido al main

if __name__ == '__main__':
    # Ajustar para usar GPU disponible o CPU si no hay GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # configurar CUDA si está disponible
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Usar GPU 0 si está disponible
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("CUDA no disponible, usando CPU")
        torch.set_default_tensor_type('torch.FloatTensor')
    
    args = option.parser.parse_args()

    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    model = Model(args.feature_size)
    for name, value in model.named_parameters():
        print(name)

    model_dict = model.load_state_dict(
        {k.replace('module.', ''): v for k, v in torch.load('ckpt/deepmilfinal.pkl', map_location=device).items()})

    viz = None  # Temporalmente deshabilitado
    auc, ap = test(test_loader, model, args, viz, device)
    print(f"AUC: {auc}, AP: {ap}")

