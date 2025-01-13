import json
import random
from datetime import datetime, timedelta
from io import BytesIO
from time import sleep
import os

import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt


CAMERA_STREAM = "http://192.168.42.17/tmpfs/auto.jpg"
CAMERA_AUTH = ("admin", "n00B!cgp")


def collect_images(count=5, rotate=None):
    try:
        resps = []
        for i in range(count):
            resp = requests.get(CAMERA_STREAM, auth=CAMERA_AUTH, stream=True).raw
            resps.append(resp)
        images = [np.asarray(bytearray(resp.read()), dtype="uint8") for resp in resps]
        [resp.close() for resp in resp]
        images = [cv2.imdecode(image, cv2.IMREAD_COLOR) for image in images]
        images = [cv2.resize(camImage, (1920, 1080)) for camImage in images]
        if rotate:
            images = [cv2.rotate(camImage, rotate) for camImage in images]

        return images
    except Exception as e:
        print(f"Failed to collect image: {e}")
    return []


if __name__ == "__main__":
    image = collect_images(count=1)[0]
    path = "/home/musengdir/oot/grabber_directory"
    if not os.path.exists(path):
        os.makedirs(path)
    elif not os.path.isdir(path):
        os.remove(path)
    cv2.imwrite(f'{path}/{datetime.utcnow().isoformat()}.jpg', image)
