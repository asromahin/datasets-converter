import os


from converters.vgg import from_vgg_csv
from converters.yolov5 import to_yolov5

if __name__ == '__main__':
    iminfo = from_vgg_csv(
        csv_path=os.path.join(r'C:\datasets\kaggle\happy-whale-and-dolphin\part1_train', 'det.csv'),
        images_path=r'C:\datasets\kaggle\happy-whale-and-dolphin\part1_train',
        split='train',
    )
    to_yolov5(iminfo, 'result')

