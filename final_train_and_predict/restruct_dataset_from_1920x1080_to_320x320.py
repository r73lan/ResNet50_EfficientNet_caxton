import pandas as pd
from PIL import Image
import os
import re
from torchvision import transforms
from pathlib import Path

def create_printset(df, n, path_dataset, out_path_dataset):
    name_data_dirname = 'print_log_filtered_classification3_gaps.csv'
    id_print = f'print{n}/'
    data_dirname = pd.read_csv(os.path.join(path_dataset, id_print, id_print, name_data_dirname)) #таблица с данными печати для текущего архива
    center_x = data_dirname.loc[0, 'nozzle_tip_x']
    center_y = data_dirname.loc[0, 'nozzle_tip_y']
    crop_size = 320
    left = center_x - crop_size // 2
    top = center_y - crop_size // 2
    for dirname, _, filenames in os.walk(os.path.join(path_dataset, id_print, id_print)):   #!!! поменять на путь к датасету
        image_index = 0   #индекс для обхода столбцов в датафрейме data_dirname
        try: 
            filenames.sort(key=lambda x: int(re.search(r'\d+', x).group()) if x[-1] == 'g' else -1) #сортировка по номеру изображения 'image-xxxx.jpg'
        except ValueError:
            print('Unexpected error')
            pass
        for file in filenames:
            if (file == data_dirname.loc[image_index, 'img_path'].split(sep='/')[-1]):
                try:
                    img = Image.open(os.path.join(dirname, file))
                    img = transforms.functional.crop(img, top, left, crop_size, crop_size)
                    df.loc[len(df.index)] = [file, n, *data_dirname.iloc[image_index, 12:]]
                    image_index += 1
                    new_name = os.path.join(out_path_dataset, f'id_{n}_'+file)
                    img.save(new_name)
                except OSError:   #обработка исключения. в ходе загрузки выяснилось, что некоторые изображения повреждены и не загружаются
                    print(f'{os.path.join(dirname, file)} did not read')
                    image_index += 1
            if image_index == data_dirname.shape[0]:  #выход из цикла, если достигнут конец внутренней таблицы
                break
    return df


df = pd.DataFrame(columns=['image_name', 'print_id', 'flow_rate_class','feed_rate_class',
                             'z_offset_class', 'hotend_class'])
Path('C:/caxton_mod').mkdir(exist_ok=True)
for i in range(183):
    #df = pd.concat([df, create_printset(df, i, 'C:/caxton_text/', 'C:/caxton_mod/')], axis = 0, ignore_index=True)
    create_printset(df, i, 'C:/dataset_full/', 'C:/caxton_mod/')
    print(f"print{i} read")
df.to_csv(Path('C:/caxton_mod/caxton_united.csv'))
#датасет C:/caxton_full/ содержит подпапочные print'ы