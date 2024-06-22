import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import torch.nn as nn
import csv, time, gc
import random
import numpy as np
import tqdm as notebook_tqdm
import os
import matplotlib.pyplot as plt

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
        label_values = self.img_labels.loc[idx, ['flow_rate_class', 'feed_rate_class', 
                                                 'z_offset_class', 'hotend_class']].values
        label_values = label_values.astype(np.float32) 
        label = torch.tensor(label_values, dtype=torch.long)  
        image = self.transform(image)
        return image, label
    
def set_seed(seed_value):
    random.seed(seed_value)      
    np.random.seed(seed_value)    
    torch.manual_seed(seed_value) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

''' для полного датасета
def random_split_indices(total_size, train_fraction=0.8):
    indices = np.arange(total_size) 
    np.random.shuffle(indices)       

    train_size = int(train_fraction * total_size)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    return train_indices, test_indices'''

#ля части датасета
def random_split_indices(array_indices, train_fraction=0.8):
    size = len(array_indices)
    train_size = int(train_fraction * size)
    train_indices = array_indices[:train_size]
    test_indices = array_indices[train_size:]

    return train_indices, test_indices


class MultiTaskResNet50(nn.Module):
    def __init__(self):
        super(MultiTaskResNet50, self).__init__()
        self.resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.resnet_base.fc = nn.Identity()

        self.fc_task1 = nn.Linear(2048, 3)
        self.fc_task2 = nn.Linear(2048, 3)
        self.fc_task3 = nn.Linear(2048, 3)
        self.fc_task4 = nn.Linear(2048, 3)

    def forward(self, x):
        features = self.resnet_base(x)
        output_task1 = self.fc_task1(features)
        output_task2 = self.fc_task2(features)
        output_task3 = self.fc_task3(features)
        output_task4 = self.fc_task4(features)
        return output_task1, output_task2, output_task3, output_task4


def train(model, opt, loss_fn, epochs, history, device, train_dataloder, test_dataloder):
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
        #gc.collect()
        history['train loss'].append(avg_loss_tr.detach().cpu().numpy())
        print('train loss: %f \n' % avg_loss_tr, end='  ')
        #torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            avg_loss_val = 0
            for X_batch, Y_batch in test_dataloder:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                Y_pred_0, Y_pred_1, Y_pred_2, Y_pred_3 = model(X_batch)
                loss_val = loss_fn(Y_pred_0, Y_batch[:, 0]) + loss_fn(Y_pred_1, Y_batch[:, 1]) + loss_fn(Y_pred_2, Y_batch[:, 2]) + loss_fn(Y_pred_3, Y_batch[:, 3])
                avg_loss_val += loss_val / len(test_dataloder)
        #gc.collect()
        history['val loss'].append(avg_loss_val.detach().cpu().numpy())
        print('val loss: %f \n' % avg_loss_val)
        weights_filename = f'resnet_50_weights_p1_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), weights_filename)
        print("--- %i seconds ---" % (time.time() - start_time))

#нормализация относительно датасета imagenet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
#нормализация относительно датасета caxton
'''transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.2915257, 0.27048784, 0.14393276], std=[0.066747, 0.06885352, 0.07679665])
])'''
    #!!!!!!!!!!!!!!!!!!!!!!!!!путь!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dataset = CaxtonDataset(
    annotations_file=Path('/kaggle/input/caxton-224/caxton_224/caxton_united.csv'),
    img_dir=Path('/kaggle/input/caxton-224/caxton_224'),
    transform=transform
)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
set_seed(42)

#для полного датасета
#train_indices, test_indices = random_split_indices(len(dataset))

#для части датасета
random_indices =  np.random.choice(len(dataset), len(dataset) // 10, replace=False)
train_indices, test_indices = random_split_indices(random_indices)
train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=os.cpu_count()-1)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count()-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_resnet50 = MultiTaskResNet50()
model_resnet50.to(device)
max_epochs = 15
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model_resnet50.parameters(), lr=1e-3)
history = {'train loss':[], 'val loss':[]}
train(model_resnet50, optim, loss_fn, max_epochs, history, device, train_loader, test_loader)
with open('history.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    headers = list(history.keys())
    writer.writerow(headers)
    for values in zip(*history.values()):
        writer.writerow(values)

def predict(model, test_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    count_correct = 0
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for X_batch, Y_batch in test_dataloader:
            X_batch = X_batch.to(device) 
            Y_pred_0, Y_pred_1, Y_pred_2, Y_pred_3 = model(X_batch)
            Y_pred_0_label = torch.argmax(Y_pred_0, dim=1)
            Y_pred_1_label = torch.argmax(Y_pred_1, dim=1)
            Y_pred_2_label = torch.argmax(Y_pred_2, dim=1)
            Y_pred_3_label = torch.argmax(Y_pred_3, dim=1)
            count_correct += torch.sum((Y_pred_0_label.cpu() == Y_batch[:, 0]))
            count_correct += torch.sum((Y_pred_1_label.cpu() == Y_batch[:, 1]))  
            count_correct += torch.sum((Y_pred_2_label.cpu() == Y_batch[:, 2]))
            count_correct += torch.sum((Y_pred_3_label.cpu() == Y_batch[:, 3]))
    return count_correct

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    #!!!!!!!!!!!!!!!!!!!!!!!!!путь!!!!!!!!!!!!!!!!!!!!!!!!!!!!
dataset = CaxtonDataset(
        annotations_file=Path('/kaggle/input/caxton-224/caxton_224/caxton_united.csv'),
        img_dir=Path('/kaggle/input/caxton-224/caxton_224'),
        transform=transform
    )
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
set_seed(42)

#для полного датасета
#train_indices, test_indices = random_split_indices(len(dataset))

#для части датасета
random_indices =  np.random.choice(len(dataset), len(dataset) // 10, replace=False)
train_indices, test_indices = random_split_indices(random_indices)

train_dataset = Subset(dataset, train_indices)
test_dataset = Subset(dataset, test_indices)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=os.cpu_count()-1)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=os.cpu_count()-1)

#тестирование - выбраны эпохи, где меньший лосс на валидационной выборке, и еще не началось переобучение

corr = dict()
model_0 = MultiTaskResNet50()  
corr_0 = predict(model_0, test_loader)
corr['default'] = corr_0 / (len(test_dataset)*4)
model = MultiTaskResNet50()
model.load_state_dict(torch.load(f'/kaggle/input/resnet50_caxton/pytorch/len_0.1/1/resnet_50_weights_p1_epoch6_len_0.1.pth'))
corr_i = predict(model, test_loader).item()
corr['0.1*len'] = corr_i / (len(test_dataset)*4)
model = MultiTaskResNet50()
model.load_state_dict(torch.load(f'/kaggle/input/resnet50_caxton/pytorch/len_0.2/1/resnet_50_weights_p1_epoch3_len_0.2.pth'))
corr_i = predict(model, test_loader).item()
corr['0.2*len'] = corr_i / (len(test_dataset)*4)
model = MultiTaskResNet50()
model.load_state_dict(torch.load(f'/kaggle/input/resnet50_caxton/pytorch/len_0.05/1/resnet_50_weights_p1_epoch9_len_0.05.pth'))
corr_i = predict(model, test_loader).item()
corr['0.05*len'] = corr_i / (len(test_dataset)*4)



