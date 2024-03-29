import numpy as np
import cv2
from numpy import array
import matplotlib.pyplot as plt



 
def divideImage(imageFile):
  picture = []
  image = imageFile 
  height = image.shape[0]

  width = image.shape[1]

  for x in range(0, height, 10):
      for n in range(0, width, 10):
          
          print('x', x)
          a = image[x:x+10, n:n+10]
          picture.append(a)
  return picture

cropped = []
file = np.load('file_drawing.npy')
plt.figure()
plt.imshow(file)
fig = plt.figure(figsize=(10,10))

cropped.append(divideImage(file[:,:,0]))
for i in range(len(cropped[0])):
    fig.add_subplot(10,10,i+1)
    plt.imshow(cropped[0][i])

