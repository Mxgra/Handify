import cv2
import time

cap= cv2.VideoCapture(0)

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer= cv2.VideoWriter('videos/rock_left.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))


start = time.time()

while True:
    ret,frame= cap.read()

    writer.write(frame)

    cv2.imshow('frame', frame)

    print(time.time()-start)

    if time.time() - start > 15:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
writer.release()
cv2.destroyAllWindows()