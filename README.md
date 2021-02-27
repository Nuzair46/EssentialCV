EssentialCV
------------

[![PyPI](https://img.shields.io/pypi/v/EssentialCV.svg)](https://pypi.python.org/pypi/EssentialCV)

* This is a simplified module for the great OpenCV Library.
* Most of the essential functions in the library are condensed to a minimal and easy to understand code.
	
* This project is usefull for simple and small works only. 
* Some Functions use values that cannot be passed while calling. These values are from OpenCV library
	that do not have a huge impact on simple usage.
		example: `cv.threshold(frame, th1, th2, cv.THRESH_BINARY)`
		Here everything except cv.THRESH_BINARY can be passed while calling the function.

* That said, this project aims to simplify OpenCV functions. By using this module, you are expected to have some knowledge with OpenCV.

Installation:  
--------------

* `pip install EssentialCV`

In this module:
---------------

* Image will be sometimes reffered as Frame or img.  
* Threshold will be called thresh    
* Blank is a blank image used for masking.  
* My personal best methods for different functions will be mentioned.  
	
Requirements:  
--------------

* OpenCV  
* Matplotlib  
* Numpy (included with OpenCV)  

Documentation:
--------------

* You can ignore passing variables which already contain a default value. As this project is aimed to simplify things like that.  
* Default values are used if you don't pass the required values.  

* class Edge:  
	
	1. `Edge.CannyEdge(frame, th1, th2)`  
	
		* Canny Edge Detection.
		* This takes 3 values. The Image, and 2 Thresholds.
		* Returns image.

	2. `Edge.DilateEdge(frame, th1, th2, iterations,  strength = (7,7))`  
 
		* Dilate Edges.
		* This takes 5 values. The Image, and 2 Thresholds, iterations needed and strength with default (7,7). 
		* Change the default while calling according to your need. 
		* Returns image.

	3. `Edge.LapEdge(frame)`  

		* Laplacian Edge Detection.
		* This takes 1 value. The Image.
		* Returns image.

	4. `Edge.SobelEdge(frame)`  

		* Sobel Edge Detection. Best.
		* This takes 1 value. The Image.
		* Returns image.

* class Threshold:  
	
	1. `Threshold.simpleThresh(frame, th1, th2)`  

		* Simple Threshold.
		* This takes 3 values. The Image, and 2 Thresholds.
		* Returns ret -> th1 and thresholded image as thresh.

	2. `Threshold.simpleThresh_inv(frame, th1, th2):`  

		* Inverse Simple Threshold.
		* This takes 3 values. The Image, and 2 Thresholds.
		* Returns ret -> th1 and thresholded image as thresh.

	3. `Threshold.adaptiveThresh(frame, th1, kernel = 11, mean_tune = 3):`  

		* Adaptive Threshold. Best.
		* This takes 4 values. The Image, and max Threshold, kernel size default 11 and Mean_tune default 3.
		* Returns thresholded image.

	4. `Threshold.adaptiveThresh_inv(frame, th1, kernel = 11, mean_tune = 3)`  

		* Inverse Adaptive Threshold.
		* This takes 4 values. The Image, and max Threshold, kernel size default 11 and Mean_tune default 3.
		* Returns inverse thresholded image.

	5. `Threshold.adaptiveThresh_gauss(frame, th1, kernel = 11, mean_tune = 3)`  

		* Adaptive Threshold Gaussian.
		* This takes 4 values. The Image, and max Threshold, kernel size default 11 and Mean_tune default 3.
		* Returns thresholded image.

* class Blur:  
	
	1. `Blur.avgBlur(img, strength = (3,3))`  

		* Average Blur.
		* This takes 2 values. The Image, and strength default (3,3).
		* Returns blurred image.

	2. `Blur.GaussBlur(img, strength = (3,3))`  

		* Gaussian Blur.
		* This takes 2 values. The Image, and strength default (3,3).
		* Returns blurred image.

	3. `Blur.medBlur(img, strength = 3)`  

		* Median Blur.
		* This takes 2 values. The Image, and strength default 3.
		* Returns blurred image.

	4. `Blur.biBlur(img, diameter = 10, color = 35, space = 25)`  

		* Bilateral Blur. Best.
		* This takes 4 values. The Image, and diameter, sigmaColor, sigmaSpace.
		* The diameter, sigmaColor, sigmaSpace can be ignored if you dont need complications.
		* Returns blurred image.

* class Transform:
	
	1. `Transform.translate(img, x, y)`   

		* Translate (move) image to (x,y).
		* This takes 3 values. The Image, and x and y coordinates.
		* Returns image.

	2. `Transform.rotate(img, angle, rotPoint = None)`  

		* Rotate image to with angle along (x,y).
		* This takes 3 values. The Image, angle and rotPoint => (x,y) coordinates. rotPoint Default => None (0,0).
		* Returns image. 

	3. `Transform.flip(img, axis)`  

		* Flip along x or y or both axis.
		* This takes 2 values. The Image, axis.
		* axis : 
			 0  => x
			 1  => y
			 -1 => xy
		* Returns image.

* class Rescale:  
	
	1. `Rescale.rescaleframe(frame, scale)`  

		* Rescale image.
		* This takes 2 values. The Image, and scale factor.
		* Returns image.

	2. `Rescale.changeRes(frame, width, height)`  

		* Change resoltuion.
		* This takes 3 values. The Image, width and height.
		* Returns image.

* class Colorspace:  
	
	1. BGR to other color formats.  
		
		* This takes 1 values. The Image.
		* Returns image.

	* To Gray:  
		`Colorspace.toGray(frame)`
	
	* To HSV:  
		`Colorspace.toHSV(frame)`
	
	* To LAB:  
		`Colorspace.toLAB(frame)`
	
	* To RGB:  
		`Colorspace.toRGB(frame)`

	2. From other color formats BGR.  
		
		* This takes 1 values. The Image.
		* Returns image.

	* from Gray:  
		`Colorspace.fromGray(frame)`
	
	* from HSV:  
		`Colorspace.fromHSV(frame)`
	
	* from LAB:  
		`Colorspace.fromLAB(frame)`
	
	* from RGB:  
		`Colorspace.fromRGB(frame)`

* class Colors:  
	
	1. `Colors.colorsplit(img, depth)`  

		* Split colors of an image.
		* This takes 2 values. The Image and Depth.
		* Depth:
			1 --> GRAY
			2 --> BGR
		* Returns 3 image of blue, green and red of the specified depth.

	2. `Colors.colormerge(b,g,r)`  

		* Merge 3 color splitted image to get the final image.

* class Bitwise:  

	1. Bitwise Funtions.  
	
		* Overview of each values. Don't mind this if using the Mask class. It will do this automatically for you:
			* dst => output array. Specify a Blank image for color image input. Leave None for Gray.
			* mask => input array. Specify a Mask image for color image input. Leave None for Gray. 
			* one => Image to which mask is to be applied.
			* two=> Same as "one" in case of color image. If "one" is gray, use mask image.

		* For operations without mask image, Only need to pass images "one" and "two".

	* Bitwise AND:  
		`Bitwise.bitAnd(one, two, dst = None, mask = None)`

	* Bitwise OR:  
		`Bitwise.bitOr(one, two, dst = None, mask = None)`

	* Bitwise XOR:  
		`Bitwise.bitXor(one, two, dst = None, mask = None)`

	* Bitwise NOT:  
		`Bitwise.bitNot(one, dst = None, mask = None)`

* class Mask:  

	1. `Mask.grayMasking(img, mask)`  

		* Gray Masking.
		* Use only to mask gray image.
		* Takes 2 values,Gray Image and Mask image.
		* Returns masked image.

	2. `Mask.colorMasking(img, mask)`  

		* Color Masking.
		* Use only to mask Color image.
		* Takes 2 values,Color Image and Mask image.
		* Returns masked image.

* class MaskMaker:  

	1. `MaskMaker.create(img, material)`  

		* Creates basic rectanglur and circular mask in the centre if the image with half the size.
		* Takes image to use as masking reference and material (shape).
		* materials:
			rectangle
			circle
		* KeyError is raised if the input material is not valid.
		* Returns mask image.

LICENSE
-------

MIT License

Copyright (c) 2021 rednek46

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
