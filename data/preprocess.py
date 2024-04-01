import os
import random
from PIL import Image

def center_crop(img):
    w, h = img.size
    top, btm, left, right = 0, 0, 0, 0
    if w > h :
        top = 0 
        btm = h 
        left = w // 2 - h //2 
        right = w // 2 + h // 2
    else:
        top = h // 2 - w // 2
        btm = h // 2 + w // 2
        left = 0
        right = w
    return img.crop((left, top, right, btm))

def resize_image(img, width, height):
	return img.resize((width, height), Image.Resampling.LANCZOS)

if __name__ == '__main__':
    folders = ['train', 'val', 'test']
    for folder in folders:
        for label in ['images', 'images_ai']:
            path = f'{folder}/{label}/'
            for file in os.listdir(path):
                try:
                    quality = random.randint(75, 95) 
                    img = Image.open(path + file).convert('RGB')
                    img = center_crop(img)
                    img = resize_image(img, 512, 512)
                    save_path = f'{folder}_preprocessed/{label}/'
                    if os.path.exists(save_path) == False:
                        os.makedirs(save_path)
                    img.save(save_path + file, format='jpeg', quality=quality)
                except Exception as e:
                    print(e)
                    print(f'Error processing {path + file}')