import config
import torch
import numpy as np
from unet import UNet
from data_loading import BasicDataset
from torch.utils.data import DataLoader, Subset
from utils import Score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def evaluate(model, loader, n_classes):
    model.eval()
    score = Score(n_classes)
    with torch.no_grad():
        for data in loader:
            images = data['image'].to(device)
            labels = data['mask'].to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            preds_np = preds.cpu().numpy().astype(np.uint8).squeeze(1)
            labels_np = labels.cpu().numpy().astype(np.uint8).squeeze(1)
            score.update(labels_np, preds_np)
    return score.get_scores()

if __name__ == '__main__':
    model = UNet(n_channels=3, n_classes=config.n_classes).to(device)
    model.load_state_dict(torch.load(config.model_path, map_location=device))

    dataset = BasicDataset(config.X_path, config.y_path, config.img_scale)
    indices = list(range(min(10, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=config.batch_size,
                        shuffle=False, num_workers=config.num_workers)

    results = evaluate(model, loader, n_classes=2)
    print("Evaluation results:")
    for k, v in results.items():
        if isinstance(v, dict):
            print(f"{k}:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv:.4f}")
        elif isinstance(v, np.ndarray):
            # 处理数组，如 class_acc
            print(f"{k}:", ' '.join([f'{x:.4f}' for x in v]))
        else:
            print(f"{k}: {v:.4f}")