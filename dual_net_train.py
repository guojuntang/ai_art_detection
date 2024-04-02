import argparse
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ImageDataset
from networks.DualNet import DualNet
from networks.residual_stream import ResidualStream
from networks.content_stream import ContentStream

class Save_best:
    def __init__(self):
        self.best_acc = 0.0
        self.best_loss = 10000

def train(model,  train_loader, val_loader, criterion, optimizer, output_path, lr_scheduler, save_best, device, epochs=10):
    train_num = len(train_loader.dataset)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        step = 0
        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            predict = torch.max(outputs, dim=1)[1]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += (predict == labels).sum().item()
            step += 1
        lr_scheduler.step()
        train_loss = running_loss / step
        train_acc = running_corrects / train_num
        print(f'Training Epoch {epoch+1}/{epochs}, Epoch Loss: {train_loss:.4f}, Epoch Acc: {train_acc:.4f}')
        validate(model, val_loader, epoch, output_path, optimizer, criterion, save_best, device)

def validate(model, val_loader, epoch, output_path, optimizer, criterion, save_best, device):
    model.eval()
    val_loss_sum = 0.0
    acc = 0.0
    with torch.no_grad():
        step = 0
        val_num = len(val_loader.dataset)
        for _, (images, labels) in enumerate(val_loader):
            val_images, val_labels = images.to(device), labels.to(device)
            outputs = model(val_images)
            loss1 = criterion(outputs, val_labels)
            result = torch.max(outputs, dim=1)[1]
            val_loss_sum += loss1.item()
            acc += (result == val_labels.cuda()).sum().item()
            step += 1
        val_loss = val_loss_sum / step
        val_acc = acc / val_num
        print(f"Val epoch {epoch}: "
              f" val_loss: {val_loss:.3f} |"
              f" val_acc: {val_acc:.3f} |\n")
        if val_acc > save_best.best_acc:
            save_best.best_acc = val_acc
            torch.save(
                {'net': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss,
                    'acc': save_best.best_acc}, output_path + 'model_best_acc.pth'
            )

        if val_loss < save_best.best_loss:
            save_best.best_loss = val_loss
            torch.save(
                {'net': model.state_dict(),
                    'opt': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': save_best.best_loss,
                    'acc': val_acc
                    }, output_path + 'model_best_loss.pth'
            )

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    weight_decay = args.weight_decay
    decay_steps = args.decay
    decay_rate = args.decay_rate
    model_type = args.model
    output_path = args.output_path
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = ImageDataset('train', transform=transform, label_type=torch.long)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = ImageDataset('val', transform=transform, label_type=torch.long)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    save_best = Save_best()

    model = None
    if model_type == 'DualNet':
        model = DualNet()
    elif model_type == 'Residual':
        model = ResidualStream()
    elif model_type == 'Content':
        model = ContentStream()
    else:
        raise ValueError('Model not supported')
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = decay_steps,
        gamma = decay_rate
    )
    train(model, train_loader, val_loader, criterion, optimizer, output_path, lr_scheduler, save_best, device, epochs)
    # save the last model
    torch.save(
        {'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            }, output_path + 'model_last_round.pth'
    )

def parse_args():
    parser = argparse.ArgumentParser(description='XNet')
    parser.add_argument('--output_path', default='models/', required=False)
    parser.add_argument('--batch_size', default=26, type=int, required=False)
    parser.add_argument('--epochs', default=120, type=int, required=False)
    parser.add_argument('--lr', default=2e-4, type=float, required=False)
    parser.add_argument('--model', default='DualNet', required=False)
    parser.add_argument('--decay', default=30, type=int, required=False)
    parser.add_argument('--decay_rate', default=0.1, type=float, required=False)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args =  parse_args()
    main(args)