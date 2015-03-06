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

#include "LauraConvolution.h"
#include <assert.h>
using cv::Range;
using cv::Mat_;

LauraConvolution::LauraConvolution()
{

}

LauraConvolution::~LauraConvolution()
{

}

Mat
LauraConvolution::convolutionEngine(Mat& img, Mat& filter, 
	void* varargs,
	float (*func) (Mat& inhood, Mat& filter, Range xidx, Range yidx, 
	void* varargs))
{
	//If you passed a filter under 3x3, you get a 
	//3x3 mean filter. Sorry.
	if ((filter.rows < 3) & (filter.cols < 3))
	{
		filter = Mat::ones(3, 3, CV_32F);
		filter = filter/9.0f;
	}

	//Compute borders.
	int left = filter.cols/2;
	int right = filter.cols - left - 1;
	int top = filter.rows/2;
	int bottom = filter.rows - top - 1;
	
	//Add mirrored boundaries.
	Mat imgMir = addMirroredBoundaries(
		img, left, right, top, bottom);
	
	//Compute indices to examine.
	Range xidx = Range(-left, right + 1);
	Range yidx = Range(-top, bottom + 1);

	//Make conv output matrix.
	//Mat_ is templated Mat for element-wise ops
	Mat imgConv = Mat::zeros(imgMir.rows, imgMir.cols, CV_32F);
	Mat_<float> imgConv_ = imgConv;

	//Perform the filtering
	for (int i = top; i < imgMir.rows - bottom; ++i)
	{
		for (int j = left; j < imgMir.cols - right; ++j)
		{
			//Compute indices.
			Range xidxCurrent = xidx;
			xidxCurrent.start += j;
			xidxCurrent.end += j;
			Range yidxCurrent = yidx;
			yidxCurrent.start += i;
			yidxCurrent.end += i;

			//Get neighborhood
			Mat inhood = imgMir(yidxCurrent, xidxCurrent);

			assert(inhood.type() == filter.type());

			//Apply neighborhood operation.
			imgConv_(i, j) = (*func)(inhood, filter, 
				xidxCurrent, yidxCurrent, varargs);
		}
	}

	return removeBoundaries(imgConv, left, right,
		top, bottom);
}

float
LauraConvolution::convFunc(Mat& inhood, Mat& filter, 
	Range xidx, Range yidx, void* varargs)
{
	return inhood.dot(filter);
}

float
LauraConvolution::hitAndMissFunc(Mat& inhood, Mat& filter, 
	Range xidx, Range yidx, void* varargs)
{
	bool hit = true; //Assume the hit is true, until proven wrong.
	Mat_<float> inhood_ = inhood;
	Mat_<float> filter_ = filter;

	for (int i = 0; i < inhood.rows; ++i)
	{
		for (int j = 0; j < inhood.cols; ++j)
		{
			if ((0 != filter_(i, j)) && (1 != filter_(i, j)))
				continue;  //This is a skip pixel.

			if (filter_(i, j) != inhood_(i, j)) 
			{
				hit = false;
				break;
			}
		}
	}

	float ret = inhood_(inhood.rows/2, inhood.cols/2);
	if (hit) return !ret;
	else return ret;
}

Mat
LauraConvolution::addMirroredBoundaries(
	Mat& img, int left, int right, int top,
	int bottom)
{
	//Create new matrix.
	int rows = img.rows;
	int cols = img.cols;
	Mat ret = Mat::zeros(rows + top + bottom, 
		cols + left + right, CV_32F);
	int mrows = rows + top + bottom;
	int mcols = cols + left + right;
	
	//Copy the center.
	Mat center = ret(Range(top, rows + top), 
		Range(left, cols + left));
	img.copyTo(center);
	//Copy top left corner.
	Mat topLeft = ret(Range(0, top),
		Range(0, left));
	flip(img(Range(0, top), Range(0, left)), topLeft, -1);
	//Copy top right corner.
	Mat topRight = ret(Range(0, top), Range(mcols - right, mcols));
	flip(img(Range(0, top), Range(cols - right, cols)),
		 topRight, -1);
	//Copy bottom left corner.
	Mat bottomLeft = ret(Range(mrows - bottom, mrows), Range(0, left));
	flip(img(Range(rows - bottom, rows), Range(0, left)), bottomLeft, -1);
	//Copy bottom right corner.
	Mat bottomRight = ret(Range(mrows - bottom, mrows), Range(mcols - right,
		mcols));
	flip(img(Range(rows - bottom, rows), Range(cols - right, cols)),
		bottomRight, -1);
	//Copy top
	Mat topLine = ret(Range(0, top), Range(left, mcols - right));
	flip(img(Range(0, top), Range(0, cols)), topLine, 0);
	//Copy left
	Mat leftLine = ret(Range(top, mrows - bottom), Range(0, left));
	flip(img(Range(0, rows), Range(0, left)), leftLine, 1);
	//Copy right
	Mat rightLine = ret(Range(top, mrows - bottom), Range(mcols - right, mcols));
	flip(img(Range(0, rows), Range(cols - right, cols)), rightLine, 1);
	//Copy bottom
	Mat bottomLine = ret(Range(mrows - bottom, mrows), Range(left, mcols - right));
	flip(img(Range(rows - bottom, rows), Range(0, cols)), bottomLine, 0);

	return ret;
}

Mat
LauraConvolution::removeBoundaries(Mat& img,
	int left, int right, int top, int bottom)
{
	Mat ret = img(Range(top, img.rows - bottom),
		Range(left, img.cols - right));

	return ret;
}

Mat
LauraConvolution::convolve(Mat& img, Mat& filter)
{
	return convolutionEngine(img, filter, NULL, convFunc);
}

Mat
LauraConvolution::hitAndMiss(Mat& img, Mat& filter)
{
	return convolutionEngine(img, filter, NULL, hitAndMissFunc);
}
