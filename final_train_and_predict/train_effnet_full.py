import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn as nn
import csv, time, gc
import random
import numpy as np
import os
import matplotlib.pyplot as plt
import timm


def set_seed(seed_value):
    random.seed(seed_value)      
    np.random.seed(seed_value)    

def split_sequence(array_indices, train_fraction=0.9):
    size = len(array_indices)
    train_size = int(train_fraction * size)    
    train_indices = array_indices[:train_size]
    test_indices = array_indices[train_size:]
    return train_indices, test_indices


class CaxtonDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_dir /  f"id_{self.img_labels.at[idx, 'print_id']}_{self.img_labels.at[idx, 'image_name']}"
        image = Image.open(img_path)
        label = self.img_labels.loc[idx, ['flow_rate_class', 'feed_rate_class', 
                                                 'z_offset_class', 'hotend_class']].values
        label = label.astype(np.float32) 
        label = torch.tensor(label, dtype=torch.long)  
        image = self.transform(image)
        return image, label
    

class Effnet(nn.Module):
    def __init__(self):
        super(Effnet, self).__init__()
        self.effnet_base = timm.create_model(
            'efficientnet_b0.ra_in1k',
            pretrained=True,
            num_classes=0  
        )
        self.fc_task1 = nn.Linear(1280, 3)
        self.fc_task2 = nn.Linear(1280, 3)
        self.fc_task3 = nn.Linear(1280, 3)
        self.fc_task4 = nn.Linear(1280, 3)

    def forward(self, x):
        features = self.effnet_base(x) 
        output_task1 = self.fc_task1(features)
        output_task2 = self.fc_task2(features)
        output_task3 = self.fc_task3(features)
        output_task4 = self.fc_task4(features)
        return output_task1, output_task2, output_task3, output_task4


def train(model, opt, loss_fn, epochs, history, device, train_dataloder, test_dataloder, id_model):
    start_time = time.time()
    for epoch in range(epochs):
        print('----- Epoch %d/%d -----' % (epoch+1, epochs))
        avg_loss_tr = 0
        model.train()
        for X_batch, Y_batch in train_dataloder:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            opt.zero_grad()
            Y_pred_0, Y_pred_1, Y_pred_2, Y_pred_3 = model(X_batch)
            loss = loss_fn(Y_pred_0, Y_batch[:, 0]) + loss_fn(Y_pred_1, Y_batch[:, 1]) + loss_fn(Y_pred_2, Y_batch[:, 2]) + loss_fn(Y_pred_3, Y_batch[:, 3])
            loss.backward()
            opt.step()
            avg_loss_tr += loss / len(train_dataloder)
        history['train loss'].append(avg_loss_tr.detach().cpu().numpy())
        print('train loss: %f \n' % avg_loss_tr, end='  ')
        model.eval()
        with torch.no_grad():
            avg_loss_val = 0
            for X_batch, Y_batch in test_dataloder:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                Y_pred_0, Y_pred_1, Y_pred_2, Y_pred_3 = model(X_batch)
                loss_val = loss_fn(Y_pred_0, Y_batch[:, 0]) + loss_fn(Y_pred_1, Y_batch[:, 1]) + loss_fn(Y_pred_2, Y_batch[:, 2]) + loss_fn(Y_pred_3, Y_batch[:, 3])
                avg_loss_val += loss_val / len(test_dataloder)
        history['val loss'].append(avg_loss_val.detach().cpu().numpy())
        print('val loss: %f \n' % avg_loss_val)
        weights_filename = f'resnet_{id_model}_weights_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), weights_filename)
        print("--- %i seconds ---" % (time.time() - start_time))


model_effent = Effnet()
data_config = timm.data.resolve_model_data_config(model_effent)
transforms = timm.data.create_transform(**data_config, is_training=False)
dataset = CaxtonDataset(
    annotations_file=Path("/kaggle/input/caxton-224/caxton_224/caxton_united.csv"),  
    img_dir=Path('/kaggle/input/caxton-224/caxton_224'),
    transform=transforms
)

set_seed(42)
size_subdataset = len(dataset)  
random_indices =  np.random.choice(len(dataset), size_subdataset, replace=False)
train_indices, test_indices = split_sequence(random_indices) 

train_dataset_1 = Subset(dataset, train_indices)
test_dataset_1 = Subset(dataset, test_indices)
train_loader_1 = DataLoader(train_dataset_1, batch_size=32, shuffle=True, num_workers=os.cpu_count()-1)
test_loader_1 = DataLoader(test_dataset_1, batch_size=32, shuffle=False, num_workers=os.cpu_count()-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loss_fn = nn.CrossEntropyLoss()
max_epochs = 5
print(device)
optim = torch.optim.Adam(model_effent.parameters(), lr=5e-5)
model_effent.to(device)
history = {'train loss':[], 'val loss':[]}
train(model_effent, optim, loss_fn, max_epochs, history, device, train_loader_1, test_loader_1, '1000k')
with open(f'history.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    headers = list(history.keys())
    writer.writerow(headers)
    for values in zip(*history.values()):
        writer.writerow(values)

