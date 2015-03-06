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

int
main(int argc, char** argv) {
	//Read image from command line.
	string fname;
	if (argc != 2) { //user did something wrong, correct them and exit
		cout << "Format: ./logEdge [filename]." << endl;
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

	//LoG filter.
	Mat logfilt = LauraFilters::LoG(13, 2.0f);
	Mat img2 = LauraConvolution::convolve(
		img, logfilt);
	double min, max;
	cv::minMaxLoc(img2, &min, &max);
	cv::Scalar lmean = mean(img2);
	cout << "Min: " << min << endl;
	cout << "Max: " << max << endl;
	cout << "Mean: " << lmean(0) << endl;
	cv::add(img2, -lmean, img2);
	
	//Binary edge image.
	Mat bedge = LauraFilters::zeroCross3x3(img2);

	//Convert back to uchar for display.
	img.convertTo(img, CV_8U);
	img2 = cv::abs(img2);
	img2.convertTo(img2, CV_8U);
	bedge.convertTo(bedge, CV_8U);

	//Show image
	namedWindow(fname, CV_WINDOW_AUTOSIZE);
	imshow(fname, img);
	namedWindow("filtered", CV_WINDOW_AUTOSIZE);
	imshow("filtered", 15*img2);
	namedWindow("edges", CV_WINDOW_AUTOSIZE);
	imshow("edges", bedge);
	waitKey(0);

	return 0;
}
