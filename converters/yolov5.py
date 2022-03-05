from tqdm import tqdm
import os
from shutil import copyfile
import yaml

from model import DetectionInfo, ImageInfo, ImagesInfo


def to_yolov5(images_info: ImagesInfo, save_path: str):
    train_path = os.path.join(save_path, 'train')

    train_path_images = os.path.join(train_path, 'images')
    train_path_labels = os.path.join(train_path, 'labels')

    val_path = os.path.join(save_path, 'val')

    val_path_images = os.path.join(val_path, 'images')
    val_path_labels = os.path.join(val_path, 'labels')

    dataset_path = os.path.join(save_path, 'dataset.yaml')

    os.makedirs(train_path_images, exist_ok=True)
    os.makedirs(train_path_labels, exist_ok=True)
    os.makedirs(val_path_images, exist_ok=True)
    os.makedirs(val_path_labels, exist_ok=True)

    cat_code = dict()

    for image in tqdm(images_info.images):
        image: ImageInfo
        if image.split == 'train':
            filepath = os.path.join(train_path_images, image.filename)
            labels_path = os.path.join(train_path_labels, image.filename.split('.')[0]+'.txt')
        elif image.split == 'val':
            filepath = os.path.join(val_path_images, image.filename)
            labels_path = os.path.join(val_path_labels, image.filename.split('.')[0] + '.txt')
        else:
            raise ValueError("ImageInfo.split not in [train, val]")
        copyfile(image.path, filepath)

        lines = []
        for detection in image.detections:
            detection: DetectionInfo
            center_x = str(detection.x)
            center_y = str(detection.y)
            w = str(detection.w)
            h = str(detection.h)
            name = detection.name
            if name not in cat_code.keys():
                cat_code[name] = len(cat_code.keys())
            name_code = str(cat_code[name])
            res_str = '\t'.join([name_code, center_x, center_y, w, h])
            lines.append(res_str)
        res_data = '\n'.join(lines)
        with open(labels_path, 'w') as f:
            f.write(res_data)

    dict_file = {
        'path': save_path,
        'train': train_path,
        'val': val_path,
        'nc': len(cat_code.keys()),
        'names': list(cat_code.keys()),
    }
    # print(dict_file)
    with open(dataset_path, 'w') as f:
        yaml.dump(dict_file, f)
