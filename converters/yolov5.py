from tqdm import tqdm
import os
from shutil import copyfile
import yaml
import cv2

from model import DetectionInfo, ImageInfo, ImagesInfo


def to_yolov5(images_info: ImagesInfo, save_path: str, name_to_code: dict = None):
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

    if name_to_code is None:
        name_to_code = dict()

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
            if name not in name_to_code.keys():
                name_to_code[name] = len(name_to_code.keys())
            name_code = str(name_to_code[name])
            res_str = '\t'.join([name_code, center_x, center_y, w, h])
            lines.append(res_str)
        res_data = '\n'.join(lines)
        with open(labels_path, 'w') as f:
            f.write(res_data)

    dict_file = {
        'path': save_path,
        'train': train_path,
        'val': val_path,
        'nc': len(name_to_code.keys()),
        'names': list(name_to_code.keys()),
    }

    with open(dataset_path, 'w') as f:
        yaml.dump(dict_file, f)


def from_yolov5_detections(images_folder, txt_folder, code_to_name, split='test'):
    image_names = os.listdir(images_folder)
    images = []
    for image_name in tqdm(image_names):
        filename = image_name.split('.')[0]
        txt_name = filename + '.txt'

        txt_path = os.path.join(txt_folder, txt_name)
        im_path = os.path.join(images_folder, image_name)

        im = cv2.imread(im_path)
        im_width = im.shape[1]
        im_height = im.shape[0]
        detections = []
        with open(txt_path, 'r') as f:
            data = f.read()
            lines = data.split('\n')[:-1]
            for line in lines:
                print(line)
                name_code, center_x, center_y, w, h, conf = line.split('\t')
                name_code = int(name_code)
                center_x = float(center_x)
                center_y = float(center_y)
                w = float(w)
                h = float(h)
                conf = float(conf)
                name = code_to_name[name_code]
                det = DetectionInfo(name=name, x=center_x, y=center_y, w=w, h=h)
                detections.append(det)
        images.append(
            ImageInfo(
                name=None,
                filename=image_name,
                path=im_path,
                width=im_width,
                height=im_height,
                detections=detections,
                split=split,
            )
        )
        images_info = ImagesInfo(name='yolov5_detections', images=images)
        return images_info


if __name__ == '__main__':
    from_yolov5_detections(

    )
