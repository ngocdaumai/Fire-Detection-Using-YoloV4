import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320

confThreshold =0.2
nmsThreshold= 0.5

classesFile = "coco_fire.names"
# classesFile = "coco.names"
classNames = []
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
# print(classNames)
# print(len(classNames))

# modelConfiguration = "yolov3-tiny.cfg"
# modelWeights = "yolov3-tiny.weights"

# modelConfiguration = "yolov3.cfg"
# modelWeights = "yolov3.weights"

# modelConfiguration = "yolov3-tiny-obj.cfg"
# modelWeights = "yolov3-tiny-obj_final.weights"

# modelConfiguration = "yolov4-custom.cfg"
# modelWeights = "yolov4-custom_last.weights"

# modelConfiguration = "yolov4-tiny_custom.cfg"
# modelWeights = "yolov4-tiny_custom_last_2.weights"

modelConfiguration = "yolov4-tiny_custom.cfg"
modelWeights = "yolov4-tiny_custom_last_3.weights"


# net = cv2.dnn.readNet(modelConfiguration, modelWeights)
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
path = "E:/Daumn/1.PythonCode/Computervision/train_data/images/train/"

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(  det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))
    # print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    # print(indices[0])

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    success, img = cap.read()
    # img = cv2.imread(path+"img (56).jpg")
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    # print(layersNames)
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    # print(outputNames)
    outputs = net.forward(outputNames)
    # print(outputs[0].shape)
    # print(outputs[1].shape)
    # print(outputs[2].shape)
    # print(outputs[0][0])
    findObjects(outputs, img)
    cv2.imshow("original", img)
    if cv2.waitKey(1)==ord("q"):
        break