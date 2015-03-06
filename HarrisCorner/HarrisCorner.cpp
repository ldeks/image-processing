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

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>
#include "../LauraConvolution.h"
#include "../LauraFilters.h"

using cv::Mat;
using cv::namedWindow;
using cv::imshow;
using cv::imread;
using cv::waitKey;
using cv::Range;

using std::cout;
using std::endl;
using std::string;
using std::vector;

#define EPS 1e-7

Mat HarrisCornerSignal(Mat& gx, Mat& gy, int fsize1, int fsize2);
float cornerSignal(Mat& inhood, Mat& filter, Range xidx,
	Range yidx, void* varargs);
void normalizeImage(Mat& img);
Mat removeMultiDots(Mat& img, int fsize);
float dotYield(Mat& inhood, Mat& filter, Range xidx, Range yidx,
	void* varargs);
void printMat(Mat& littleMat);

int
main(int argc, char** argv) {
	//Read image from command line.
	string fname;
	if (argc != 2) { //user did something wrong, correct them and exit
		cout << "Format: ./HarrisCorner [filename]." << endl;
		return 0;
	}
	else {
		fname = argv[1]; //grab filename
	}
	Mat img = imread(fname);
	if (!img.data) return -1; //Snippet from opencv 2.1 doc intro to make sure it loaded properly.

	//Convert to grayscale
	cv::cvtColor(img, img, CV_RGB2GRAY);
	//Convert to float
	img.convertTo(img, CV_32F);

	//Gaussian smooth the image.
	Mat gaussian = LauraFilters::gaussian(9, 9, 1.3);
	Mat smoothed = LauraConvolution::convolve(img, gaussian);

	//Remove the dots around the outside with grayscale morphology.
	//From looking at equations in Wikipedia:Mathematical morphology: 
	//Grayscale version is
	//essentially the same as binary version, just uses max and min
	//to eat in/spread out edges. This has a nice smoothing effect
	//on the colors, too.
	Mat rect = cv::getStructuringElement(cv::MORPH_RECT, 
		cv::Size(9, 9));
	cv::morphologyEx(smoothed, smoothed, cv::MORPH_OPEN, rect);

	//Find the gradients.
	Mat gxfilt = LauraFilters::gx3x3();
	Mat gyfilt = LauraFilters::gy3x3();
	Mat gx = LauraConvolution::convolve(smoothed, gxfilt);
	Mat gy = LauraConvolution::convolve(smoothed, gyfilt);

	//Calculate the corner signal.
	gy = LauraConvolution::addMirroredBoundaries(gy, 2, 2, 2, 2);
	Mat cimg = HarrisCornerSignal(gx, gy, 3, 3);
	gy = LauraConvolution::removeBoundaries(gy, 2, 2, 2, 2);

	//Nonmaxima suppression.
	//Carry out nonmaxima suppression.
	Mat thinned = LauraFilters::nonmaximaSuppression3x3(cimg);
	normalizeImage(thinned);
	thinned = LauraFilters::threshold(thinned, 50.0f);
	thinned = removeMultiDots(thinned, 7);

	//Convert back to uchar for display.
	img.convertTo(img, CV_8U);
	smoothed.convertTo(smoothed, CV_8U);
	normalizeImage(cimg);
	cimg.convertTo(cimg, CV_8U);
	thinned.convertTo(thinned, CV_8U);

	//Highlight the corners in red.  
	vector<Mat> channels;
	channels.push_back(0.35f*img);
	channels.push_back(0.35f*img);
	channels.push_back(0.35f*img + thinned);
	Mat final;
	cv::merge(channels, final);

	//Show image
	namedWindow(fname, CV_WINDOW_AUTOSIZE);
	imshow(fname, img);
	namedWindow("smoothed", CV_WINDOW_AUTOSIZE);
	imshow("smoothed", smoothed);
	namedWindow("cimg", CV_WINDOW_AUTOSIZE);
	imshow("cimg", cimg);
	namedWindow("thinned", CV_WINDOW_AUTOSIZE);
	imshow("thinned", thinned);
	namedWindow("final", CV_WINDOW_AUTOSIZE);
	imshow("final", final);
	waitKey(0);

	return 0;
}

Mat HarrisCornerSignal(Mat& gx, Mat& gy, int fsize1, int fsize2)
{
	Mat filter = Mat::ones(fsize1, fsize2, gx.type());
	void* varargs = (void*) &gy;
	return LauraConvolution::convolutionEngine(
		gx, filter, varargs, cornerSignal);
}

float cornerSignal(Mat& inhood, Mat& filter, Range xidx,
	Range yidx, void* varargs) 
{
	Mat gxinhood = inhood;
	//Unpackage gy and get its neighborhood
	Mat* gy = (Mat*) varargs;
	Mat gyinhood = (*gy)(yidx, xidx);

	//Calculate A.
	//Sum of I_x^2:
	//dot() will unroll into a vector
	float A11 = gxinhood.dot(gxinhood);
	//Sum of I_xI_y:
	float A12 = gxinhood.dot(gyinhood);
	//Sum of I_y^2:
	float A22 = gyinhood.dot(gyinhood);

	float traceA = A11 + A22;
	float detA = (A11*A22) - (A12*A12);

	return (2*detA)/(traceA + EPS);
}

//Normalize grayscale image values to be between 0 and 255.
void normalizeImage(Mat& img)
{
	double min, max;
	cv::minMaxLoc(img, &min, &max);
	img -= min;
	cv::minMaxLoc(img, &min, &max);
	img /= max;
	img *= 255.0f;
}

//Get rid of more than one dot on each corner.
Mat removeMultiDots(Mat& img, int fsize)
{
	Mat filter = Mat::ones(fsize, fsize, img.type());
	return LauraConvolution::convolutionEngine(
		img, filter, NULL, dotYield);
}

float dotYield(Mat& inhood, Mat& filter, Range xidx, Range yidx,
	void* varargs)
{
	cv::Mat_<float> inhood_ = inhood;
	float inhoodSum = inhood.dot(filter);
	int i = filter.rows/2;
	int j = filter.cols/2;
	float p0 = inhood_(i, j);

	//If there is another non-zero in the neighborhood
	//turn this pixel off.
	if (p0 && (inhoodSum > p0))
	{
		inhood_(i, j) = 0.0f; //Must do this to ensure in-place
		return 0.0f;
	}
	else
		return p0;
}

void printMat(Mat& littleMat)
{
	cv::Mat_<float> littleMat_ = littleMat;
	for (int i = 0; i < littleMat.rows; ++i)
	{
		for (int j = 0; j < littleMat.cols; ++j)
		{
			cout << littleMat_(i, j) << ", ";
		}
		cout << endl;
		
	}
}
