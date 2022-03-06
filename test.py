from converters.yolov5 import from_yolov5_detections
from converters.vgg import to_vgg_csv

if __name__ == '__main__':
    res = from_yolov5_detections(
        r'C:\datasets\kaggle\happy-whale-and-dolphin\part1_train',
        r'C:\Users\asrom\Downloads\labels',
        {
            0: 'head',
            1: 'tail',
        }
    )
    to_vgg_csv(res, 'result')
