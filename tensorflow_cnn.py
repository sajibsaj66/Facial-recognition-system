import cv2
import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Layer, Conv2D, MaxPooling2D, Flatten, Dense
from retinaface import RetinaFace


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

start_time = time.time()


def imPreprocess(imgArr):
    pre_time = time.time()
    # input_shape = (100, 100, 3)  # Define the dimensions of your input image
    # Load and preprocess the image using PIL (Python Imaging Library)
    # img=cv2.imread(path)

    faces = RetinaFace.detect_faces(imgArr)

    if len(faces) > 0:

        face = faces["face_1"]
        # value=face["facial_area"]
        print(f"face-------{face}")
        coordinates = face["facial_area"]
        imgArr = imgArr[coordinates[1]:coordinates[3],
                        coordinates[0]:coordinates[2]]
        print(
            f"face detection---------{time.time()-pre_time}----------------")
        # cv2.rectangle(img, (points[2], points[3]), (points[0], points[1]), (0,255,255), 2)
        # cv2.imshow('Image with Bounding Box', imgArr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Since image_array has shape (1, height, width, channels)
        # plt.imshow(imgArr)
        # plt.axis('off')  # Turn off axis labels
        # plt.show()

        # image = Image.open(path)

        # image = image.resize((100,100))  # Resize to match the input shape
        image_array = np.array(imgArr)
        # print(image_array)
        # Normalize pixel values to [0, 1]
        image_array = image_array.astype('float32') / 255.0
        # cv2.imshow(image_array)
        image_array = np.expand_dims(image_array, axis=0)

        # plt.imshow(image_array[0])  # Since image_array has shape (1, height, width, channels)
        # plt.axis('off')  # Turn off axis labels
        # plt.show()
        pre_time = time.time()-pre_time
        print(
            f"preprocess time----------------{pre_time}--------------------")
        return image_array, True
    else:
        print("no face detected")
        return None, False


def embedding(imgArr):
    img, isFace = imPreprocess(imgArr)
    if isFace:
        emb_time = time.time()

        # print("111111111111111111111111111111111111111111111111111111111111111")
        t = time.time()
        c1 = Conv2D(64, (10, 10), activation="relu")(img)
        print(f"embedding c1---------{time.time()-t}----------------")
        t = time.time()
        m1 = MaxPooling2D(64, (2, 2), padding="same")(c1)
        print(f"embedding m1---------{time.time()-t}----------------")

        t = time.time()
        c2 = Conv2D(128, (7, 7), activation="relu")(m1)
        print(f"embedding c2---------{time.time()-t}----------------")
        t = time.time()
        m2 = MaxPooling2D(64, (2, 2), padding="same")(c2)
        print(f"embedding m2---------{time.time()-t}----------------")

        t = time.time()
        c3 = Conv2D(128, (4, 4), activation="relu")(m2)
        print(f"embedding c3---------{time.time()-t}----------------")
        t = time.time()
        m3 = MaxPooling2D(64, (2, 2), padding="same")(c3)
        print(f"embedding m3---------{time.time()-t}----------------")

        t = time.time()
        c4 = Conv2D(256, (4, 4), activation="relu")(m3)
        print(f"embedding c4---------{time.time()-t}----------------")
        t = time.time()
        f1 = Flatten()(c4)
        print(f"embedding f1---------{time.time()-t}----------------")
        t = time.time()
        d1 = Dense(4096, activation="sigmoid")(f1)
        print(f"embedding d1---------{time.time()-t}----------------")
        emb_time = time.time()-emb_time
        print(f"embeddeding time--------------{emb_time}---------------")
        return d1, True
    else:
        return None, False


def embedding_distance(actualEmbedding, validationEmbedding):
    return tf.math.abs(actualEmbedding - validationEmbedding)


# img = cv2.imread("fahim_front.jpg")
# imPreprocess(img)


def siamese():

    path = "fahim_front.jpg"

    img = cv2.imread(path)
    actualEmbedding, value = embedding(img)

    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        validationEmbedding, isEmbedding = embedding(frame)
        if isEmbedding:
            distance = embedding_distance(actualEmbedding, validationEmbedding)
            dence = Dense(1, activation="sigmoid")(distance)
            print(f"distance-----------{dence}--------------")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    video.release()
    cv2.destroyAllWindows()


siamese()
