import my_canny as mc
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

# origin_img = Image.open('/Users/jewonrho/Documents/ComputerVision/final_project/sample.jpg')
# origin_img = Image.open('final_project/sample2.png')
# origin_img = Image.open('final_project/sample3.jpeg')
origin_img = Image.open('final_project/road1.jpeg')
# origin_img = Image.open('final_project/road2.jpeg')

gray_img = ImageOps.grayscale(origin_img)
img = np.array(gray_img)

blur_img = mc.gaussian_blur(np.copy(img),k_size=5)
opencv_blur = cv2.GaussianBlur(np.copy(img),(5,5),1)
print(blur_img.shape)

res1 = mc.my_canny(np.copy(img),100,150)
starttime = time.time_ns()
res2 = mc.my_canny(np.copy(blur_img),10,100)
endtime = time.time_ns()
my_process_time = endtime - starttime

starttime = time.process_time_ns()
opencv_canny = cv2.Canny(np.copy(blur_img),50,150)
endtime = time.process_time_ns()
opencv_porcess_time = endtime - starttime
print("Numpy Canny edge process time  : ",my_process_time,"ns")
print("OpenCV Canny edge process time : ",opencv_porcess_time,"ns")
plt.figure(figsize=(20,10))

plt.subplot(2,3,1).set_title("original image")
plt.imshow(img,cmap='gray')
plt.subplot(2,3,2).set_title("blur image")
plt.imshow(blur_img,cmap='gray')
plt.subplot(2,3,3).set_title("OpenCv blur image")
plt.imshow(opencv_blur,cmap='gray')
plt.subplot(2,3,4).set_title("Canny Edge detect to original image")
plt.imshow(res1,cmap='gray')
plt.subplot(2,3,5).set_title("Canny Edge detect to blured image")
plt.imshow(res2,cmap='gray')
plt.subplot(2,3,6).set_title("OpenCV Canny Edge detect")
plt.imshow(opencv_canny,cmap='gray')
plt.show()
