import cv2
import os

video_list = os.listdir("videos")

for video in video_list:
    vidcap = cv2.VideoCapture('videos/{}'.format(video))

    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite("frames/{}-{}.jpg".format(video[:-4], count), image)     # save frame as JPEG file
        success, image = vidcap.read()
        #print('Read a new frame: ', success)
        count += 1

    print(video)
    print(count)