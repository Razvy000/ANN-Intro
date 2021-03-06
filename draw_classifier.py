import cv2
import numpy as np
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer
#from NeuralNetwork import NeuralNetwork
from ann import ANN
# http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html
import pylab as pl

import random
import time

serialized_name = 'models/28_1600000.pickle'#'60000.pickle'#'nn_mnist_8by8_10epochs.pickle'
nn = ANN([1, 1]).deserialize(serialized_name)


# https://anaconda.org/menpo/opencv
# conda install -c https://conda.anaconda.org/menpo opencv


drawing = False  # true if mouse is pressed
mode = False  # if True, draw rectangle. Press 'm' to toggle to curve
ix, iy = -1, -1

radius = 25 # aprox 1/8  1/8 * 512 = 64
img_size = (28, 28) #(8, 8)#(28, 28)
# mouse callback function

color = (255)
X=[]
y=[]
with open('data/mnist_test_data.txt', 'r') as fd, open('data/mnist_test_label.txt', 'r') as fl:
	for line in fd:
		img = line.split()
		pixels = [int(pixel) for pixel in img]
		X.append(pixels)
	for line in fl:
		pixel = int(line)
		y.append(pixel)
X = np.array(X, np.float)
y = np.array(y, np.float)


def draw_image(img):
	img2 = np.array(img, np.uint8)
	img2 = img2.reshape((28,28))
	#print type(img2), img2.dtype, img2.shape
	rzimg = cv2.resize(img2, (512, 512), interpolation=cv2.INTER_AREA)
	cv2.imshow('random existing image', rzimg)
	cv2.imshow('random existing image 2', img2)


draw_image(X[0])

def draw_circle(event, x, y, flags, param):
	global ix, iy, drawing, mode

	if event == cv2.EVENT_LBUTTONDOWN:
		drawing = True
		ix, iy = x, y

	elif event == cv2.EVENT_MOUSEMOVE:
		if drawing == True:
			if mode == True:
				cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
			else:
				cv2.circle(img, (x, y), radius, color, -1)

	elif event == cv2.EVENT_LBUTTONUP:
		drawing = False
		if mode == True:
			cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
		else:
			cv2.circle(img, (x, y), radius, color, -1)

		font = cv2.FONT_HERSHEY_SIMPLEX

		#rzimg = cv2.resize(img, (32, 32))#, interpolation=cv2.INTER_AREA) #
		#rzimg = cv2.blur(rzimg, (8, 8))
		#cv2.imshow('resize32x32', rzimg)

		#rzimg = cv2.resize(img, (32, 32))
		#rzimg = cv2.blur(rzimg, (6, 6))
		#cv2.imshow('resize32x32', rzimg)

		rzimg = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
		#rzimg = cv2.blur(rzimg, (2, 2))
		# pl.gray()
		# pl.matshow(rzimg)
		# pl.show()

		#rzimg = cv2.blur(rzimg, (2,2))
		zoom = cv2.resize(rzimg, (512, 512), interpolation=cv2.INTER_NEAREST)
		cv2.imshow('resize8x8', rzimg)
		#print type(rzimg), rzimg.dtype, rzimg.shape
		#print rzimg
		cv2.imshow('resize8x8zoom', zoom)
		'''
		'''
		#print rzimg
		#print type(rzimg)
		c = rzimg.reshape((img_size[0]*img_size[1],))
		c = c.astype(float)
		c -= c.min()  # normalize the values to bring them into the range 0-1
		c /= c.max()
		c = c * 0.99
		# print c
		out = nn.predict(c)

		# compute a softmax

		out = np.array(out)
		softmax = np.exp(out) / np.sum(np.exp(out), axis= 0)
		predictNo = np.argmax(softmax)
		print "lbls   0  1  2  3  4  5  6  7  8  9"
		print "prob", (100 * softmax).astype(int)
		print "i predict", predictNo
		print ""

	#cv2.putText(img,''+ str(predictNo),(10,500), font, 1,(128,128,128),2)

img = np.zeros((512, 512, 1), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

oldepoch = time.time()
while(1):
	cv2.imshow('image', img)
	k = cv2.waitKey(1) & 0xFF

	passed = time.time() - oldepoch
	if passed > 2:
		oldepoch = time.time()
		draw_image(X[random.uniform(0, X.shape[0])])

	if k == ord(' '):
		#mode = not mode
		# reset image on space
		img = np.zeros((512, 512, 1), np.uint8)
		
	elif k == 27:
		break

	#rzimg = cv2.resize(img, (8, 8))
	#cv2.imshow('resize', rzimg)

cv2.destroyAllWindows()
