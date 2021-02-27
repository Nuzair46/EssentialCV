import cv2 as cv
import numpy as np

class Edge():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def CannyEdge(frame, th1, th2):
		gray = Colorspace.toGray(frame)
		result =  cv.Canny(gray, th1, th2)
		return result

	def DilateEdge(frame, th1, th2, iterations,  strength = (7,7)):
		gray = Colorspace.toGray(frame)
		result = cv.dilate(Edge.CannyEdge(gray, th1, th2), strength, iterations)
		return result

	def LapEdge(frame):
		gray = Colorspace.toGray(frame)
		lap = cv.Laplacian(gray, cv.CV_64F)
		lap = np.uint8(np.absolute(lap))
		return lap

	def SobelEdge(frame):
		gray = Colorspace.toGray(frame)
		sobX = cv.Sobel(gray, cv.CV_64F, 1, 0)
		sobY = cv.Sobel(gray, cv.CV_64F, 0, 1)
		sobXY = Bitwise.bitOr(sobX, sobY)
		return sobXY

class Threshold():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def simpleThresh(frame, th1, th2):
		ret, thresh = cv.threshold(frame, th1, th2, cv.THRESH_BINARY)
		return ret, thresh

	def simpleThresh_inv(frame, th1, th2):
		ret, thresh_inv = cv.threshold(frame, th1, th2, cv.THRESH_BINARY_INV)
		return ret, thresh_inv

	def adaptiveThresh(frame, th1, kernel = 11, mean_tune = 3):
		adaptive = cv.adaptiveThreshold(gray, th1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, kernel, mean_tune)
		return adaptive

	def adaptiveThresh_inv(frame, th1, kernel = 11, mean_tune = 3):
		adaptive_inv = cv.adaptiveThreshold(gray, th1, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, kernel, mean_tune)
		return adaptive_inv

	def adaptiveThresh_gauss(frame, th1, kernel = 11, mean_tune = 3):
		adaptive = cv.adaptiveThreshold(gray, th1, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, kernel, mean_tune)
		return adaptive

class Blur():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def avgBlur(img, strength = (3,3)):
		result = cv.blur(img, strength)
		return result

	def GaussBlur(img, strength = (3,3)):
		result = cv.GaussianBlur(img, strength, cv.BORDER_DEFAULT)
		return result

	def medBlur(img, strength = 3):
		result = cv.medianBlur(img, strength)
		return result

	def biBlur(img, diameter = 10, color = 35, space = 25):
		result = cv.bilateralFilter(img, diameter, sigma, space)
		return result

class Transform():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def translate(img, x, y):
		transMat = np.float32([[1,0,x],[0,1,y]])
		dimension = (img.shape[1], img.shape[0])
		return cv.warpAffine(img, transMat, dimension)

	def rotate(img, angle, rotPoint = None):
		(height, width) = img.shape[:2]

		if rotPoint is None:
			rotPoint = (width//2, height//2)

		rotMat = cv.getRotationMatrix2D(rotPoint, angle, scale = 1.0)
		dimension = (width, height)

		return cv.warpAffine(img, rotMat, dimension)

	def flip(img, axis):
		return cv.flip(img, axis)

class Rescale():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def rescaleframe(frame, scale):
		width = int(frame.shape[1] * scale)
		height = int(frame.shape[0] * scale)
		dimension = (width,height)
		return cv.resize(frame, dimension, interpolation = cv.INTER_CUBIC if scale > 1 else cv.INTER_AREA)

	def changeRes(frame, width, height):
		frame.set(3, width)
		frame.set(4, height)

	"""
	def crop(frame):
		'''
		INCOMPLETE
		Crop image.
		This takes 1 values. The Image.
		returns image.
		'''
		result = frame[50:200,200:400]
		return result
	"""
class Colorspace():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def toGray(frame):
		result =  cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
		return result

	def toHSV(frame):
		result =  cv.cvtColor(frame, cv.COLOR_BGR2HSV)
		return result

	def toLAB(frame):
		result =  cv.cvtColor(frame, cv.COLOR_BGR2LAB)
		return result

	def toRGB(frame):
		result =  cv.cvtColor(frame, cv.COLOR_BGR2RGB)
		return result

	def fromGray(frame):
		result =  cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
		return result

	def fromHSV(frame):
		result =  cv.cvtColor(frame, cv.COLOR_HSV2BGR)
		return result

	def fromLAB(frame):
		result =  cv.cvtColor(frame, cv.COLOR_LAB2BGR)
		return result

	def fromRGB(frame):
		result =  cv.cvtColor(frame, cv.COLOR_RGB2BGR)
		return result

class Colors():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def colorsplit(img, depth):
		b,g,r = cv.split(img)
		if depth == "1":
			return b,g,r
		elif depth == ("2"):
			blank = np.zeros(img.shape, dtype='uint8')
			blue = colors.colormerge(b,blank,blank)
			green = colors.colormerge(blank,g.blank)
			red = colors.colormerge(blank,blank,r)
			return blue, green, red
		else:
			raise KeyError
			return

	def colormerge(b,g,r):
		merged = cv.merge([b,g,r])
		return merged

class Bitwise():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def bitAnd(one, two, dst = None, mask = None):
		result = cv.bitwise_and(one, two, dst, mask)
		return result 

	def bitOr(one, two, dst = None, mask = None):
		result = cv.bitwise_or(one, two, dst, mask)
		return result 

	def bitXor(one, two, dst = None, mask = None):
		result = cv.bitwise_xor(one, two, dst, mask)
		return result

	def bitNot(img, dst = None, mask = None):
		result = cv.bitwise_not(img, dst, mask)
		return result

class Mask():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def grayMasking(img, mask):
		masked = Bitwise.bitAnd(img, mask)
		return masked
	def colorMasking(img, mask):
		dst = np.zeros(img.shape[:2], dtype = 'uint8')
		masked = Bitwise.bitAnd(img, img, dst, mask)
		return masked

class MaskMaker():

	def __init__(self, *args, **kwargs):
		super().__init__(*args)

	def create(img, material):
		blank = np.zeros(img.shape[:2], dtype = 'uint8')
		if material == "rectangle": 
			shape = cv.rectangle(blank, (0,0), (blank.shape[1]//2, blank.shape[0]//2), 255, thickness=-1)
		elif material == "circle":
			shape = cv.circle(blank, (blank.shape[1]//2, blank.shape[0]//2), 100, 255, thickness=-1)
		else:
			raise KeyError

		return shape

if __name__ == '__main__':
	'''
		xD
	'''
	print("\n\t\tGo Back and call the functions. No point running this file.\n")
