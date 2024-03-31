import os
import random

folders = ['images', 'images_ai']
# Train, validation, test
ratios = [0.7, 0.2, 0.1]

for folder in folders:
    fol = os.listdir(folder)
    train_size = int(len(fol) * ratios[0])
    val_size = int(len(fol) * ratios[1])
    test_size = int(len(fol) * ratios[2])
    print (f'{folder}: {train_size} train, {val_size} val, {test_size} test')
    # Train
    os.mkdir('train/' + folder)
    train_samples = random.sample(fol, train_size)
    for sample in train_samples:
        os.rename(folder + '/' + sample, 'train/' + folder + '/' + sample)

    # Validation
    fol = os.listdir(folder)
    os.mkdir('val/' + folder)
    val_samples = random.sample(fol, val_size)
    for sample in val_samples:
        os.rename(folder + '/' + sample, 'val/' + folder + '/' + sample)
    
    # Save the rest into test set
    fol = os.listdir(folder)
    os.mkdir('test/' + folder)
    for sample in fol:
        os.rename(folder + '/' + sample, 'test/' + folder + '/' + sample)
