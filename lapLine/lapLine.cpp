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

using std::cout;
using std::endl;
using std::string;
using std::vector;

Mat removeSP(Mat& img);

int
main(int argc, char** argv) {
	//Read image from command line.
	string fname;
	if (argc != 2) { //user did something wrong, correct them and exit
		cout << "Format: ./lapLine [filename]." << endl;
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

	//Remove salt and pepper noise.
	Mat filtered = removeSP(img);

	/*** Get 1 pixel lines on fg or bg ***/
	//Apply the laplacian.
	Mat laplacian = LauraFilters::laplacian();
	Mat lapimg = LauraConvolution::convolve(
		filtered, laplacian);
	
	//Take absolute value image.
	Mat absimg = cv::abs(lapimg);

	//Dynamic thresholding
	//Find mean and median, ignoring the 0's
	//Using the image as its own mask works for ignoring 0's,
	//as long as it is converted to CV_8U
	cv::Scalar amean, astd;
	Mat mask = absimg;
	absimg.convertTo(mask, CV_8U);
	cv::meanStdDev(absimg, amean, astd, mask);
	//Threshold absimg to thin the lines.
	Mat thinned = LauraFilters::threshold(
		absimg, amean(0) + astd(0));

	//Remove salt and pepper noise.
	thinned = removeSP(thinned);

	//Convert back to uchar for display.
	img.convertTo(img, CV_8U);
	filtered.convertTo(filtered, CV_8U);
	lapimg.convertTo(lapimg, CV_8U);
	absimg.convertTo(absimg, CV_8U);
	thinned.convertTo(thinned, CV_8U);

	//Show image
	namedWindow(fname, CV_WINDOW_AUTOSIZE);
	imshow(fname, img);
	namedWindow("filtered", CV_WINDOW_AUTOSIZE);
	imshow("filtered", filtered);
	namedWindow("Laplacian", CV_WINDOW_AUTOSIZE);
	imshow("Laplacian", lapimg);
	namedWindow("abs(Laplacian)", CV_WINDOW_AUTOSIZE);
	imshow("abs(Laplacian)", absimg);
	namedWindow("thinned", CV_WINDOW_AUTOSIZE);
	imshow("thinned", thinned);
	waitKey(0);

	return 0;
}


//Filter to remove Salt & Pepper noise
Mat removeSP(Mat& img)
{
	Mat bw = img.clone();
	bw *= 1/255.0f;
	Mat pepper = (cv::Mat_<float>(5, 5)
		<< 1, 1, 1, 1, 1,
		   1, 2, 2, 2, 1,
		   1, 2, 0, 2, 1,
		   1, 2, 2, 2, 1,
		   1, 1, 1, 1, 1);
	Mat salt = (cv::Mat_<float>(5, 5)
		<< 0, 0, 0, 0, 0,
		   0, 2, 2, 2, 0,
		   0, 2, 1, 2, 0,
		   0, 2, 2, 2, 0,
		   0, 0, 0, 0, 0);
	Mat filtered = LauraConvolution::
		hitAndMiss(bw, salt);
	filtered = LauraConvolution::
		hitAndMiss(filtered, pepper);
	pepper = (cv::Mat_<float>(3, 3)
		<< 1, 1, 1, 1, 0, 1, 1, 1, 1);
	salt = (cv::Mat_<float>(3, 3)
		<< 0, 0, 0, 0, 1, 0, 0, 0, 0);
	filtered = LauraConvolution::
		hitAndMiss(filtered, salt);
	filtered = LauraConvolution::
		hitAndMiss(filtered, pepper);
	filtered = filtered * 255.0f;

	return filtered;
}
