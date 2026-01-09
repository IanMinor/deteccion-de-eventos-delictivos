import matplotlib.pyplot as plt
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import numpy as np

def test(dataloader, model, args, viz, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).to(device)
        for i, input in enumerate(dataloader):
            input = input.to(device)
            logits = model(inputs=input)
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)
            sig = logits
            pred = torch.cat((pred, sig))
        
        gt = np.load(args.gt)
        pred = list(pred.cpu().detach().numpy())
        
        # Calcular cuántos segmentos por video basado en los datos reales
        segments_per_video = len(gt) // len(pred)
        print(f"Videos de test: {len(pred)}, Segmentos totales: {len(gt)}, Segmentos por video: {segments_per_video}")
        
        # Expandir las predicciones para que coincidan con gt
        pred_expanded = np.repeat(np.array(pred), segments_per_video)
        
        # Asegurar que tengan la misma longitud
        min_length = min(len(gt), len(pred_expanded))
        gt = gt[:min_length]
        pred_expanded = pred_expanded[:min_length]
        
        print(f"Longitudes finales - GT: {len(gt)}, Pred: {len(pred_expanded)}")
        
        fpr, tpr, threshold = roc_curve(list(gt), pred_expanded)  ###计算真正率和假正率
        np.save('fpr.npy', fpr)
        np.save('tpr.npy', tpr)
        rec_auc = auc(fpr, tpr)  ###计算auc的值
        precision, recall, th = precision_recall_curve(list(gt), pred_expanded)
        pr_auc = auc(recall, precision)
        np.save('precision.npy', precision)
        np.save('recall.npy', recall)
        
        # Solo usar visualizador si está disponible
        if viz is not None:
            viz.plot_lines('pr_auc', pr_auc)
            viz.plot_lines('auc', rec_auc)
            viz.lines('scores', pred_expanded)
            viz.lines('roc', tpr, fpr)
        
        print(f"ROC AUC: {rec_auc:.4f}")
        print(f"PR AUC: {pr_auc:.4f}")
        
        return rec_auc, pr_auc

