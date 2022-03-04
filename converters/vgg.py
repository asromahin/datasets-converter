import pandas as pd
from tqdm import tqdm
import os
import cv2
import json

from model import DetectionInfo, ImageInfo, ImagesInfo


def from_vgg_csv(csv_path, images_path, split='train'):
    res_images = []
    df = pd.read_csv(csv_path)
    df['region_shape_attributes'] = df['region_shape_attributes'].apply(json.loads)
    df['region_attributes'] = df['region_attributes'].apply(json.loads)
    df['file_attributes'] = df['file_attributes'].apply(json.loads)
    filenames = list(df['filename'].unique())
    groups = df.groupby('filename')
    for i in tqdm(range(len(filenames))):
        filename = filenames[i]
        image_path = os.path.join(images_path, filename)
        im = cv2.imread(image_path)
        group = groups.get_group(filename)
        first_row = group.iloc[0]
        detections = []
        for j in range(len(group)):
            row = group.iloc[j]
            # print(row['region_shape_attributes'])
            if row['region_shape_attributes']['name'] == 'rect':
                x = (row['region_shape_attributes']['x'] + row['region_shape_attributes']['width'] // 2) / im.shape[1]
                y = (row['region_shape_attributes']['y'] + row['region_shape_attributes']['height'] // 2) / im.shape[0]
                w = row['region_shape_attributes']['width'] / im.shape[1]
                h = row['region_shape_attributes']['height'] / im.shape[0]
                region_attributes = row['region_attributes']
                # for key, val in region_attributes.items():
                # print(region_attributes)
                if len(region_attributes) > 0:
                    det = DetectionInfo(id=j, name=list(region_attributes.values())[-1], x=x, y=y, w=w, h=h)
                else:
                    det = DetectionInfo(id=j, name=None, x=x, y=y, w=w, h=h)
                detections.append(det)
        # print(first_row['file_attributes'])
        file_attributes = first_row['file_attributes']
        if len(file_attributes) > 0:
            image_info = ImageInfo(
                name=list(first_row['file_attributes'].values())[-1],
                id=i,
                filename=filename,
                width=im.shape[1],
                height=im.shape[0],
                detections=detections,
                path=image_path,
                split=split,
            )
        else:
            image_info = ImageInfo(
                name=None,
                id=i,
                filename=filename,
                width=im.shape[1],
                height=im.shape[0],
                detections=detections,
                path=image_path,
                split=split,
            )
        res_images.append(image_info)

    return ImagesInfo(name='vgg_dataset', id=0, images=res_images)


if __name__ == '__main__':
    im_info = from_vgg_csv(
        csv_path=os.path.join(r'C:\datasets\kaggle\happy-whale-and-dolphin\part1_train', 'det.csv'),
        images_path=r'C:\datasets\kaggle\happy-whale-and-dolphin\part1_train',
    )


