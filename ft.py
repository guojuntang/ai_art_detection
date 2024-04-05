import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
from networks.resnet import resnet50
from dataset import ImageDataset

class Save_best:
    def __init__(self):
        self.best_acc = 0.0
        self.best_loss = 10000

def fine_tune(model,  train_loader, val_loader, criterion, optimizer, lr_scheduler, save_best, output_path, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        loss = 0.0
        total_loss = 0.0
        steps = 0
        for _, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.mean(outputs[:, 0], (1, 2))
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            steps += labels.size(0)
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        print(f'Training Epoch {epoch+1}/{epochs}, Epoch Loss: {total_loss / steps:.4f}')
        validate(model, val_loader, criterion, epoch, save_best, output_path, device)
    return model

def validate(model, val_loader, criterion, epoch, save_best, output_path, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        total = 0
        for _, (images, labels) in enumerate(val_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.mean(outputs[:, 0], (1, 2))
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total += labels.size(0)
        val_loss = total_loss / total
        print(f"Val epoch {epoch}: "
              f" val_loss: {val_loss:.3f}")
        if val_loss < save_best.best_loss:
            save_best.best_loss = val_loss
            torch.save(
                {'net': model.state_dict(),
                    'epoch': epoch,
                    'loss': save_best.best_loss,
                    }, output_path + 'model_best_loss.pth'
            )

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_file = args.weights_file
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    output_path = args.output_path
    pin_memory = args.pin_memory

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    train_dataset = ImageDataset('train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    val_dataset = ImageDataset('val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    criterion = torch.nn.CrossEntropyLoss()
    save_best = Save_best()


    model = resnet50(num_classes=1, gap_size=1, stride0=1)
    model.load_state_dict(torch.load(weights_file)['model'])
    for param in model.parameters():
        param.requires_grad = False
    # reset fc
    model.change_output(1)

    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
        params,
        lr=lr,
        momentum = momentum,
        weight_decay = weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size = 5,
        gamma = 0.1
    )

    model.to(device)
    fine_tune(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, save_best, output_path, device, epochs)
    torch.save({'model': model.state_dict()}, output_path + 'model_last_round.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=56)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--weights_file', type=str, default='models/weights/Grag2021_latent/model_epoch_best.pth')
    parser.add_argument('--pin_memory', type=bool, default=False)
    parser.add_argument('--output_path', type=str, default='models/')
    args = parser.parse_args()
    main(args)
    