from PIL import Image
from pathlib import Path
import random
import shutil

path_dataset = Path('C:/caxton_320_crop/')
path_out = Path('C:/caxton_224_half_flipped/')
path_out.mkdir(parents=True, exist_ok=True)
for path in path_dataset.rglob('*.jpg'):
    img = Image.open(path)
    new_size = (224, 224)
    resized_image = img.resize(new_size)
    if random.random() > 0.5:
        flipped_image = resized_image.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        flipped_image = resized_image
    flipped_image.save(path_out / path.name) 
for path in path_dataset.rglob('*.csv'):
    shutil.copy(path, path_out / path.name)