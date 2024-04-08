import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from networks.DualNet import DualNet
from networks.resnet import resnet50, ResNet
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

models_list = [
    {'model': DualNet(), 'name': '256dualnetbs26lr0001'},
    #{'model': DualNet(), 'name': '256dualnet_dropout0.6'},
    #{'model': resnet50(num_classes=1, gap_size=1, stride0=1), 'name': '512ft_bs12_lr0001'},
    #{'model': DualNet(), 'name': '512dualnet_bs4_lr0001'},
    #{'model': resnet50(num_classes=1, gap_size=1, stride0=1), 'name': '256ft_bs56_lr0001'}
]

if __name__ == '__main__':
    with torch.no_grad():
        transform = transforms.Compose([transforms.ToTensor()])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_dataset = ImageDataset('test', transform=transform, label_type=torch.long)
        #test_dataset = ImageDataset('test', transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
        test_num = len(test_dataset)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        for model_dict in models_list:
            model = model_dict['model']
            model.load_state_dict(torch.load(f'./models/{model_dict["name"]}/model_best_loss.pth')['net'])
            model.to(device)
            model.eval()
            step = 0
            loss_sum = 0.0
            pred_list = []
            labels_list = []
            result = None
            for _, (images, labels) in enumerate(test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                if isinstance(model, ResNet):
                    outputs = torch.mean(outputs[:, 0], (1, 2))
                    result = torch.round(torch.sigmoid(outputs))
                else:
                    result = torch.max(outputs, dim=1)[1]
                loss1 = criterion(outputs, labels)
                loss_sum += loss1.item()
                step += 1
                pred_list.append(result.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
            test_loss = loss_sum / step
            pred = np.concatenate(pred_list) 
            label = np.concatenate(labels_list)
            acc = accuracy_score(label, pred)
            f1 = f1_score(label, pred)
            auc = roc_auc_score(label, pred)
            print(f'{model_dict["name"]} test_loss: {test_loss:.4f}, acc: {acc:.4f}, f1: {f1:.4f}, auc: {auc:.4f}')