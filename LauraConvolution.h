//Copyright 2013 Laura Ekstrand <laura@jlekstrand.net>
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#ifndef __LAURACONVOLUTION_H__
#define __LAURACONVOLUTION_H__

#include <opencv2/opencv.hpp>
using cv::Mat;
using cv::Range;

class LauraConvolution
{
	//Function for convolution engine that performs convolution.
	static float convFunc(Mat& inhood, Mat& filter, Range xidx,
		Range yidx, void* varargs);
	//Function for convolution engine that performs hit and miss.
	static float hitAndMissFunc(Mat& inhood, 
		Mat& filter, Range xidx, Range yidx, void* varargs);
public:
	LauraConvolution();
	~LauraConvolution();

	//Visits each pixel in turn and applies the function in *func
	//varargs is a pointer to more data for passing into *func.
	static Mat convolutionEngine(Mat& img, Mat& filter, void* varargs,
		float (*func) (Mat& inhood, Mat& filter, Range xidx, 
			Range yidx, void* varargs));

	//Produces an image with mirror-padded boundaries.
	//left, right, top, and bottom are number of pixels
	//to pad.
	static Mat addMirroredBoundaries(Mat& img, 
		int left, int right, int top, int bottom);

	//Eat in from the boundaries.
	//left, right, top and bottom are number of pixels
	//to remove.
	static Mat removeBoundaries(Mat& img,
		int left, int right, int top, int bottom);

	//Convolve image img with filter.
	static Mat convolve(Mat& img, Mat& filter);

	//Hit and miss morphology. Blank is indicated by a value in the 
	//filter that is not 0 or 1.
	//WARNING: You may get a completely black image if you try to
	//hit and miss an image that is not made of 0s and 1s
	static Mat hitAndMiss(Mat& img, Mat& filter);
};

#endif //!defined __LAURACONVOLUTION_H__
