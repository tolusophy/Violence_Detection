import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
from numpy import random
from tqdm import tqdm
import os
import torch.nn.functional as F
import torch.optim as optim
from Scripts.salientclassifier import SalientClassifier
from torchsummary import summary
from Scripts.ssi import SalientSuperImage
torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import average_precision_score


##################################################################################################################################################################################

color_jitter = transforms.ColorJitter(random.uniform(0.1, 0.5),random.uniform(0.1, 0.5),random.uniform(0.1, 0.5),random.uniform(0.01, 0.15))

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([color_jitter], p=0.5),
    transforms.RandomAutocontrast(p=0.5),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

###################################################################################################################################################################################

train_ds =  SalientSuperImage(root_dir='/home/toluwani/Desktop/SIViDet/Dataset/SCVD/Train', num_secs=1, k=12, sampler='uniform', aspect_ratio='480p_A', grid_shape=(4,3), transform=train_transform)
test_ds =  SalientSuperImage(root_dir='/home/toluwani/Desktop/SIViDet/Dataset/SCVD/Test', num_secs=1, k=12, sampler='uniform', aspect_ratio='480p_A', grid_shape=(4,3), transform=test_transform)

batch_size = 32

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=8)

####################################################################################################################################################################################

def main(model, train_loader, test_loader, n_epochs=100, num_classes=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device=device)

    # Define loss functions
    ce_loss = nn.CrossEntropyLoss()

    # Define optimizer and learning rate scheduler
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()

    avgpr = 0.0
    best_accuracy = 0.0
    best_model_state_dict = None

    for epoch in range(1, n_epochs+1):
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Train loop
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device=device)
            targets = targets.to(device=device)
            optimizer.zero_grad()

            # Compute image and text embeddings
            output = model(data)

            # Compute losses
            loss = ce_loss(output, targets)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # Compute training accuracy
            _, predicted = torch.max(output.data, 1)
            train_correct += (predicted == targets).sum().item()
            train_total += targets.size(0)

            train_loss += loss.item()

        # calculate average losses
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Update learning rate
        lr_scheduler.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        y_true = [[] for i in range(num_classes)]
        y_scores = [[] for i in range(num_classes)]

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(tqdm(test_loader)):
                data = data.to(device=device)
                targets = targets.to(device=device)

                # Compute image embeddings for test set
                scores = model(data)

                # Compute validation accuracy
                _, predicted = torch.max(scores.data, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)

                # Compute validation loss
                val_loss += ce_loss(scores, targets).item()

                # Convert targets to binary format using one-hot encoding
                targets_onehot = torch.eye(num_classes)[targets]

                # Append true and predicted labels for computing average precision
                for i in range(num_classes):
                    y_true[i].extend(targets_onehot[:, i].cpu().numpy().tolist())
                    y_scores[i].extend(scores[:, i].cpu().numpy().tolist())

            val_loss /= len(test_loader)
            val_acc = 100. * val_correct / val_total

        # Compute average precision for each class separately
        ap = []
        for i in range(num_classes):
            ap_i = average_precision_score(y_true[i], y_scores[i])
            ap.append(ap_i)

        # Compute mean average precision (mAP) over all classes
        mAP = (np.mean(ap)) * 100
        # Print epoch results and save best model
        save_path = f"weights/{type(model).__name__}.pth"
        if mAP > avgpr and val_acc > best_accuracy:

            torch.save(model.state_dict(), save_path)
            avgpr = mAP
            best_loss = train_loss
            best_accuracy = val_acc
            best_model_state_dict = model.state_dict()

        print(f"Epoch {epoch}/{n_epochs}: Train Loss: {train_loss:.4f} || Train Acc: {train_acc:.2f}% || Val Loss: {val_loss:.4f} || Val Acc: {val_acc:.2f}%  || AP: {mAP:.4f}%")

    return model, best_loss, best_accuracy, avgpr

torch.manual_seed(42)

model = SalientClassifier(salinet_arch="salinet2m", num_classes=3)

# req = []
# # for name, param in img_model.named_parameters():
# #     if param.requires_grad:
# #         req.append(name)
# # print(req)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

model, best_loss, best_accuracy, avgpr = main(model, train_loader, test_loader, num_classes=3, n_epochs=30)
print(best_loss, best_accuracy, avgpr)