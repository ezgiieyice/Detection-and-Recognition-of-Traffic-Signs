import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import glob
import random
import pandas as pd

# Load Yolo model eğittim
net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

# Name custom object
classes = ["TrafficSign"]

# Images path
images_path = glob.glob(r"C:\Users\Ezgi\Desktop\yolo\train_yolo_to_detect_custom_object"
                        r"\yolo_custom_detection\test_images\*.jpg")

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Insert here the path of your images
random.shuffle(images_path)

# loop through all the images
for img_path in images_path:
    # Loading image
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # Detecting objects using 416x416 modelini kullanarak
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # Object detected
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print("İndexes: ", indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    labels = pd.read_csv("C:/Users/Ezgi/Desktop/datasets/Traffic Sign Images From Turkey/labels.csv",
                        encoding="ISO-8859-1")
    model = keras.models.load_model("model_with_97_accurcy.h5")
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

            # CONVERTİNG OF THE LOCALIZATED REGION TO GRAYSCALE TO USE IN TRAINED MODEL
            roi = img[y:y+h,x:x+w]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi = cv2.resize(gray_roi, (32, 32))
            gray_roi = gray_roi / 255.0

            print(type(roi))
            # print(type(gray_roi))
            print(roi.shape)
            # print(gray_roi.shape)

            plt.imshow(roi)
            plt.imshow(gray_roi)
            plt.imsave("C:/Users/Ezgi/Desktop/resim.png",gray_roi)
            pred = model.predict(np.array([gray_roi]))
            # cv2.putText()
            # cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

            pred = np.argmax(pred, axis=1)
            # Get the class label with highest probability
            # Convert the class index to class label
            # class_label = labels[class_index]
            # Define the position of the class label text
            text_x, text_y = x + w - 100, y

            # Add the class label text to the image
            cv2.putText(img, labels['Name'][int(pred)], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # # Display the image
            # cv2.imshow("Image with class label", img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            pred = None

    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

cv2.destroyAllWindows()
