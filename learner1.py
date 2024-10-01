import torch

checkpoint = torch.load('./checkpoint/ckpt.pth')

# Assuming you saved other metrics in the checkpoint
loaded_auc = checkpoint['best_auc']
loaded_precision = checkpoint['best_precision']
loaded_recall = checkpoint['best_recall']
loaded_f1_score = checkpoint['best_f1']

print("Loaded AUC:", loaded_auc)
print("Loaded Precision:", loaded_precision)
print("Loaded Recall:", loaded_recall)
print("Loaded F1-score:", loaded_f1_score)