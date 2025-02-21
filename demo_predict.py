from ultralytics import YOLO

yolo=YOLO('C:\\Users\\User\\Desktop\\result\\no_load\\yolov8\\weights\\last.pt', task='detect')

result=yolo(source='pic/garbage_img_3105.png',save=True)

