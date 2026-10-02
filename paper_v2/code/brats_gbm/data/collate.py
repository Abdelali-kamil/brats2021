import torch
import torch.nn.functional as F
import random
from torch.utils.data._utils.collate import default_collate

def custom_collate(batch):
    # تم إزالة default_collate لحل مشكلة KeyError: 0
    # إرجاع القاموس الجاهز للتدريب (مع البادينج العشوائي)
    return pad_batch_to_max_shape_fast(batch, deterministic=False)

def determinist_collate(batch):
    # تم إزالة default_collate لحل مشكلة KeyError: 0
    # إرجاع القاموس الجاهز للتقييم (مع البادينج الثابت)
    return pad_batch_to_max_shape_fast(batch, deterministic=True)

def pad_batch_to_max_shape_fast(batch, deterministic=False):
    # 1. Find the max dimensions in the current batch
    shapes = [sample['label'].shape for sample in batch]
    _, z_sizes, y_sizes, x_sizes = zip(*shapes)
    
    max_stride = 16
    # Fast math to find the next multiple of 16
    zmax = ((int(max(z_sizes)) - 1) // max_stride + 1) * max_stride
    ymax = ((int(max(y_sizes)) - 1) // max_stride + 1) * max_stride
    xmax = ((int(max(x_sizes)) - 1) // max_stride + 1) * max_stride

    B = len(batch)
    C_img = batch[0]['image'].shape[0]
    C_lbl = batch[0]['label'].shape[0]

    # 2. PRE-ALLOCATE memory once (This eliminates the CPU bottleneck)
    batched_images = torch.zeros((B, C_img, zmax, ymax, xmax), dtype=batch[0]['image'].dtype)
    batched_labels = torch.zeros((B, C_lbl, zmax, ymax, xmax), dtype=batch[0]['label'].dtype)

    # Collate any metadata (e.g., patient IDs) if they exist
    metadata_batch = [{k: v for k, v in b.items() if k not in ['image', 'label']} for b in batch]
    batched_dict = default_collate(metadata_batch) if metadata_batch and metadata_batch[0] else {}

    # 3. Paste data directly into pre-allocated memory
    for i, elem in enumerate(batch):
        img, lbl = elem['image'], elem['label']
        _, z, y, x = img.shape
        
        zpad, ypad, xpad = zmax - z, ymax - y, xmax - x
        
        if deterministic:
            # Stable, centered padding for validation
            z_start, y_start, x_start = zpad // 2, ypad // 2, xpad // 2
        else:
            # Free data augmentation via random translation for training
            z_start = random.randint(0, zpad)
            y_start = random.randint(0, ypad)
            x_start = random.randint(0, xpad)
        
        # Slicing is exponentially faster than F.pad
        batched_images[i, :, z_start:z_start+z, y_start:y_start+y, x_start:x_start+x] = img
        batched_labels[i, :, z_start:z_start+z, y_start:y_start+y, x_start:x_start+x] = lbl

    batched_dict['image'] = batched_images
    batched_dict['label'] = batched_labels
    
    return batched_dict


def pad_batch1_to_compatible_size(batch):
    # Kept mostly intact for single-volume inference
    print(batch.shape)
    shape = batch.shape
    zyx = list(shape[-3:])
    for i, dim in enumerate(zyx):
        max_stride = 16
        if dim % max_stride != 0:
            # Make it divisible by 16
            zyx[i] = ((dim // max_stride) + 1) * max_stride
            
    zmax, ymax, xmax = zyx
    
    # Using negative indices ensures it works for both 4D and 5D tensors
    zpad = zmax - batch.size(-3)
    ypad = ymax - batch.size(-2)
    xpad = xmax - batch.size(-1)
    
    assert all(pad >= 0 for pad in (zpad, ypad, xpad)), "Negative padding value error !!"
    
    pads = (0, xpad, 0, ypad, 0, zpad)
    batch = F.pad(batch, pads)
    
    return batch, (zpad, ypad, xpad)