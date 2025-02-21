from ultralytics import YOLO



# # train

# model=YOLO('ultralytics/cfg/models/v8/yolov8.yaml').load('yolov8n.pt')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=400)


model=YOLO('ultralytics/cfg/models/v10/yolov10m.yaml')
model.train(data='garbage.yaml',workers=0,batch=16,epochs=400,patience=0)


# model=YOLO('ultralytics/cfg/models/v8/yolov8_effientnetv2.yaml').load('yolov8n.pt')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=400)


# yolov8D:\download\ultralytics-main\ultralytics\cfg\models\v8\yolov8_gam_se.yaml
# model=YOLO('ultralytics/cfg/models/v8/yolov8.yaml').load('yolov8n.pt')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=400)
# fps:42

# # simam(没用)
# model=YOLO('ultralytics/cfg/models/v8/YOLOv8_SimAM.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=64,epochs=500)

# # odconv(map提升比原来多):第二个卷积换动态卷积
# model=YOLO('ultralytics/cfg/models/v8/yolov8_odconv.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)
# fps:39

# # mobilenetv3(没用)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_moblienetv3.yaml')
# model.train(data='garbage.yaml',workers=0,batch=64,epochs=500)

# # c2f_faster(faster)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_c2f_fast.yaml').load('yolov8n.pt')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=500)
# YOLOv8_c2f_fast summary: 245 layers, 2,307,013 parameters, 2,306,997 gradients, 6.4 GFLOPs
# Speed: 0.2ms preprocess, 2.0ms inference, 0.0ms loss, 0.5ms postprocess per image
# fps:51

# # eca
# model=YOLO('ultralytics/cfg/models/v8/yolov8_eca.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)
# fps:29

# model=YOLO('ultralytics/cfg/models/v8/yolov8_gam.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)


# gam(best attention)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_gam.yaml')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=500)
# YOLOv8_gam summary: 243 layers, 9,056,501 parameters, 9,056,485 gradients, 23.9 GFLOPs
# Speed: 0.1ms preprocess, 3.9ms inference, 0.0ms loss, 0.6ms postprocess per image
# fps:
#
# # ca
# model=YOLO('ultralytics/cfg/models/v8/yolov8_ca.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)
# # YOLOv8_ca summary: 245 layers, 5,790,501 parameters, 5,790,485 gradients, 21.3 GFLOPs
# # Speed: 0.1ms preprocess, 3.6ms inference, 0.0ms loss, 0.7ms postprocess per image
#
# # cbam
# model=YOLO('ultralytics/cfg/models/v8/yolov8_cbam.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)
# # YOLOv8_cbam summary: 241 layers, 5,908,921 parameters, 5,908,905 gradients, 21.4 GFLOPs
# # Speed: 0.2ms preprocess, 3.6ms inference, 0.0ms loss, 0.5ms postprocess per image

# # mlla
# model=YOLO('ultralytics/cfg/models/v8/yolov8_mlla.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)

# # gam
# model=YOLO('ultralytics/cfg/models/v8/yolov8_gam_2.yaml')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=500)
#
# # slim-neck(相较于原生yolov8有提升)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_slim_neck.yaml')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=500)
#
# # BiFPN_concat(相较于原生yolov8有提升)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_bifpn_concat.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)


"""单个卷积没有用"""
# # odconv(第一个卷积)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_odconv1.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)
#
# # odconv(最后一个卷积)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_odconv2.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)
#
# # odconv(中间的卷积)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_odconv3.yaml')
# model.train(data='yolo-bvn.yaml',workers=0,batch=16,epochs=500)


# model=YOLO('garbage_model.yaml')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=500)

# # odconv(两个卷积)(提升跟放在第二个差不多)
# model=YOLO('ultralytics/cfg/models/v8/yolov8_odconv_2.yaml')
# model.train(data='garbage.yaml',workers=0,batch=16,epochs=500)



