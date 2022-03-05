import pandas as pd
from tqdm import tqdm
import os
import cv2
import json
from shutil import copyfile

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
        if im is None:
            continue
        group = groups.get_group(filename)
        first_row = group.iloc[0]
        detections = []
        for j in range(len(group)):
            row = group.iloc[j]
            # print(row['region_shape_attributes'])
            region_name = row['region_shape_attributes'].get('name')
            if region_name is None:
                continue
            elif region_name == 'rect':
                x = (row['region_shape_attributes']['x'] + row['region_shape_attributes']['width'] // 2) / im.shape[1]
                y = (row['region_shape_attributes']['y'] + row['region_shape_attributes']['height'] // 2) / im.shape[0]
                w = row['region_shape_attributes']['width'] / im.shape[1]
                h = row['region_shape_attributes']['height'] / im.shape[0]
                region_attributes = row['region_attributes']
                # for key, val in region_attributes.items():
                # print(region_attributes)
                if len(region_attributes) > 0:
                    det = DetectionInfo(name=list(region_attributes.values())[-1], x=x, y=y, w=w, h=h)
                else:
                    det = DetectionInfo(name=None, x=x, y=y, w=w, h=h)
                detections.append(det)
        # print(first_row['file_attributes'])
        file_attributes = first_row['file_attributes']
        if len(file_attributes) > 0:
            image_info = ImageInfo(
                name=list(first_row['file_attributes'].values())[-1],
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
                filename=filename,
                width=im.shape[1],
                height=im.shape[0],
                detections=detections,
                path=image_path,
                split=split,
            )
        res_images.append(image_info)

    return ImagesInfo(name='vgg_dataset', images=res_images)


def to_vgg_csv(images_info: ImagesInfo, save_path: str):
    df_path = os.path.join(save_path, 'result.csv')
    images_path = os.path.join(save_path, 'images')

    os.makedirs(images_path, exist_ok=True)

    rows = []
    for image in tqdm(images_info.images):
        image: ImageInfo
        new_path = os.path.join(images_path, image.filename)
        copyfile(image.path, new_path)
        image_row = {
            "filename": image.filename,
            "file_size": os.path.getsize(image.path),
            "file_attributes": {},
        }
        if image.name is not None:
            image_row["file_attributes"]["name"] = image.name
        image_row["file_attributes"] = json.dumps(image_row["file_attributes"]).replace("'", '"')
        for i, detection in enumerate(image.detections):
            detection: DetectionInfo
            detection_row = image_row.copy()
            detection_row["region_count"] = len(image.detections)
            detection_row["region_id"] = i
            detection_row["region_shape_attributes"] = json.dumps({
                "name": "rect",
                "x": int((detection.x-detection.w/2)*image.width),
                "y": int((detection.y-detection.h/2)*image.height),
                "width": int(detection.w * image.width),
                "height": int(detection.h * image.height),
            }).replace("'", '"')
            detection_row["region_attributes"] = dict()
            if detection.name is not None:
                detection_row["region_attributes"]["name"] = detection.name
            detection_row["region_attributes"] = json.dumps(detection_row["region_attributes"]).replace("'", '"')
            rows.append(detection_row.copy())

    df = pd.DataFrame(rows)
    df.to_csv(path_or_buf=df_path, index=False)


if __name__ == '__main__':
    im_info = from_vgg_csv(
        csv_path=os.path.join(r'C:\datasets\kaggle\happy-whale-and-dolphin\part1_train', 'det.csv'),
        images_path=r'C:\datasets\kaggle\happy-whale-and-dolphin\part1_train',
    )
    to_vgg_csv(im_info, 'result')


