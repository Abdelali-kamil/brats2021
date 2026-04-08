import os

DATA_ROOT = "/home/kamilabdelali/brats2021/data"

BRATS_SUR_FOLDER = DATA_ROOT 

def get_brats_folder(mode):
    if mode == 'train':
        return os.path.join(DATA_ROOT, "BraTS2021_Training_Data")
    elif mode == 'val':
        return os.path.join(DATA_ROOT, "val")
    elif mode == 'test':
        return os.path.join(DATA_ROOT, "test")
    else:
        return DATA_ROOT