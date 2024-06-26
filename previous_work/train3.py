import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
from skimage.io import imread
from torchvision import transforms, models
import torch
import torch.nn as nn
import gc
import csv
import re
import sys
import time

#функция, обработки изображения и формирования меток для них для одной дикректории print{i}
def create_printset(n, transform):
    df = pd.DataFrame(columns=['image', 'flow_rate_class','feed_rate_class',
                             'z_offset_class', 'hotend_class'])
    images = []
    name_data_dirname = 'print_log_filtered_classification3_gaps.csv'
    id_print = f'print{n}/'
    print(id_print)
    for dirname, _, filenames in os.walk(os.path.join('C:/Users/Vik/Neuron', id_print)):   #!!! поменять на путь к датасету
        if not filenames:
            continue
        data_dirname = pd.read_csv(os.path.join(dirname, name_data_dirname)) #таблица с данными печати для текущего архива
        image_index = 0   #индекс для обхода столбцов в датафрейме data_dirname
        try: 
            filenames.sort(key=lambda x: int(re.search(r'\d+', x).group()) if x[-1] == 'g' else -1) #сортировка по номеру изображения 'image-xxxx.jpg'
        except ValueError:
            print('Unexpected error')
            pass
        #filenames.sort( key=lambda x: int(x[6:x.index('.')]) if x[-1]=='g' else -1 )  #сортировка по номеру изображения 'image-xxxx.jpg'
        #filenames.sort(key=lambda x: int(re.search(r'\d+', x).group()) if x[-1] == 'g' else -1)
        for file in filenames:
            if (file == data_dirname.loc[image_index, 'img_path'].split(sep='/')[-1]):
                try:
                    img = imread(os.path.join(dirname, file))
                    img = transform(img)
                    images.append(img)
                    df.loc[image_index,'image'] = data_dirname.iloc[image_index, 0]
                    df.loc[image_index, ['flow_rate_class', 'feed_rate_class',
                                'z_offset_class', 'hotend_class']] = data_dirname.iloc[image_index, 12:]
                    image_index += 1
                except OSError:   #обработка исключения. в ходе загрузки выяснилось, что некоторые изображения повреждены и не загружаются
                    print(f'{os.path.join(dirname, file)} did not read')
                    gc.collect()
                    time.sleep(60)
                    image_index += 1
            if image_index == data_dirname.shape[0]:  #выход из цикла, если достигнут конец внутренней таблицы
                break
    X = torch.stack(images)
    Y = df.iloc[:,1:].to_numpy(dtype=np.uint8)
    return X, Y

#функция обучения нейросети
def train(model, opt, loss_fn, epochs, tr, val, batch_size, history, device):
    for epoch in range(epochs):
        np.random.shuffle(tr)
        np.random.shuffle(val)
        print('----- Epoch %d/%d -----' % (epoch+1, epochs))
        avg_loss_tr = 0
        model.train()  # train mode
        for print_id in tr:
            data_tr = DataLoader(list(zip(*create_printset(print_id, transform))),
                     batch_size=batch_size, shuffle=True)
            for X_batch, Y_batch in data_tr:
                # data to device
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                opt.zero_grad()
                Y_pred_0, Y_pred_1, Y_pred_2, Y_pred_3 = model(X_batch)
                loss = loss_fn(Y_pred_0, Y_batch[:, 0]) + loss_fn(Y_pred_1, Y_batch[:, 1]) + loss_fn(Y_pred_2, Y_batch[:, 2]) + loss_fn(Y_pred_3, Y_batch[:, 3])
                loss.backward()
                opt.step()
                avg_loss_tr += loss / len(data_tr)
            del data_tr, X_batch, Y_batch, Y_pred_0, Y_pred_1, Y_pred_2, Y_pred_3
            gc.collect()
        history['train loss'].append(avg_loss_tr.detach().cpu().numpy())
        print('train loss: %f \n' % avg_loss_tr, end='  ')
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            avg_loss_val = 0
            for print_id in val:
                data_val = DataLoader(list(zip(*create_printset(print_id, transform))),
                    batch_size=batch_size, shuffle=True)
                for X_batch, Y_batch in data_val:
                    Y_batch = Y_batch.to(device)
                    X_batch = X_batch.to(device)
                    Y_pred_0, Y_pred_1, Y_pred_2, Y_pred_3 = model(X_batch)
                    loss_val = loss_fn(Y_pred_0, Y_batch[:, 0]) + loss_fn(Y_pred_1, Y_batch[:, 1]) + loss_fn(Y_pred_2, Y_batch[:, 2]) + loss_fn(Y_pred_3, Y_batch[:, 3])
                    avg_loss_val += loss_val / len(data_val)
                del data_val, X_batch, Y_batch, Y_pred_0, Y_pred_1, Y_pred_2, Y_pred_3
                gc.collect()
            history['val loss'].append(avg_loss_val.detach().cpu().numpy())
            print('val loss: %f \n' % avg_loss_val)
        weights_filename = f'resnet_50_weights_p1_epoch{epoch+1}.pth'
        torch.save(model.state_dict(), weights_filename)
        print("--- %s seconds ---" % (time.time() - start_time))
            
class MultiTaskResNet50(nn.Module):
    def __init__(self):
        super(MultiTaskResNet50, self).__init__()
        # Загружаем предварительно обученную модель ResNet50
        self.resnet_base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # Удаляем последний полносвязный слой в ResNet50
        self.resnet_base.fc = nn.Identity()
        
        # Добавляем новые полносвязные слои для наших задач
        # Для ResNet50 размер входа в последний слой равен 2048
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


start_time = time.time()
print(time.time())
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(240, antialias=True),
    transforms.CenterCrop(224),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
list_print = np.random.choice(183, 183, replace=False)
val_size = int(0.1 * 183)
val, ts, tr = np.split(list_print, [val_size, val_size*2])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_resnet50 = MultiTaskResNet50()
model_resnet50.to(device)
max_epochs = 5
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model_resnet50.parameters(), lr=3e-5)
history = {'train loss':[], 'val loss':[]}
train(model_resnet50, optim, loss_fn, max_epochs, tr, val, 32, history, device)
with open('history.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    headers = list(history.keys())
    writer.writerow(headers)
    for values in zip(*history.values()):
        writer.writerow(values)       
        
