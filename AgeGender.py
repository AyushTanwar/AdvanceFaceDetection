# Import required modules
import cv2 as cv
import time
import argparse

def boxmaker(net, frame, conf_threshold=0.7):
    opencvframe = frame.copy()
    heightofFrame = opencvframe.shape[0]
    widthofFrame = opencvframe.shape[1]
    blob = cv.dnn.blobFromImage(opencvframe, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * widthofFrame)
            y1 = int(detections[0, 0, i, 4] * heightofFrame)
            x2 = int(detections[0, 0, i, 5] * widthofFrame)
            y2 = int(detections[0, 0, i, 6] * heightofFrame)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(opencvframe, (x1, y1), (x2, y2), (255, 255, 255), int(round(heightofFrame/150)), 2)
    return opencvframe, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceprototype = "face_detector.pbtxt"
faceModel = "face_detector_ui.pb"

ageprototype = "agecalc.prototxt"
ageModel = "ageavg.caffemodel"

genderProto = "genderclassifier.prototxt"
genderModel = "genderavg.caffemodel"

meanmodelvalues = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-8)', '(8-12)', '(15-25)', '(25-35)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageprototype)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceprototype)

# Open a video file or an image file or a camera stream
# " args.input if args.input else 0 " use this for webcam
# "'sample1.jpg' or 'srk.jpg'" for images
cap = cv.VideoCapture(args.input if args.input else 0 )
padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameFace, tempboxes = boxmaker(faceNet, frame)
    if not tempboxes:
        print("No face Detected, Checking next frame")
        continue

    for ibox in tempboxes:
        # print(bbox)
        face = frame[max(0,ibox[1]-padding):min(ibox[3]+padding,frame.shape[0]-1),max(0,ibox[0]-padding):min(ibox[2]+padding, frame.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), meanmodelvalues, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print("Age Output : {}".format(agePreds))
        print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))
        
        if gender == 'Male':
            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (ibox[0], ibox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 210, 0), 2, cv.LINE_AA)
            cv.imshow("Age Gender Demo", frameFace)
        else :
            label = "{},{}".format(gender, age)
            cv.putText(frameFace, label, (ibox[0], ibox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 153, 255), 2, cv.LINE_AA)
            cv.imshow("Age Gender Demo", frameFace)

        
    print("time : {:.3f}".format(time.time() - t))
