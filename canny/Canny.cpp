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
		cout << "Format: ./Canny [filename]." << endl;
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

	//Gaussian filter.
	Mat gfilt = LauraFilters::gaussian(7, 7, 1.0f);
	Mat img2 = LauraConvolution::convolve(
		img, gfilt);

	//Gradient images.
	Mat dxfilt = LauraFilters::gx3x3();
	Mat dyfilt = LauraFilters::gy3x3();
	Mat dximg = LauraConvolution::convolve(
		img2, dxfilt);
	Mat dyimg = LauraConvolution::convolve(
		img2, dyfilt);
	
	//Compute magnitude image.
	Mat mag;
	cv::magnitude(dximg, dyimg, mag);
	//Compute angle image.
	Mat angimg;
	cv::phase(dximg, dyimg, angimg, true);

	//Carry out nonmaxima suppression.
	Mat thinned = LauraFilters::
		nonmaximaSuppression3x3(mag, angimg);
	//Make sure values are clamped to 
	//between 0 and 255.
	thinned.convertTo(thinned, CV_8U);
	thinned.convertTo(thinned, CV_32F);

	//Hysteresis thresholding.
	//Auto-compute the threshold.
	float tmean, tstd;
	LauraFilters::correctedMeanStdDev(
		thinned, &tmean, &tstd);
	float kl = 1.5f;
	float ku = 0.7f;
	float lthresh = tmean - (kl*tstd);
	if (lthresh < 0) lthresh = 0;
	float uthresh = tmean + (ku*tstd);
	if (uthresh > 255) lthresh = 255;
	cout << "mean = " << tmean << endl;
	cout << "std = " << tstd << endl;
	cout << "lthresh = " << lthresh << endl;
	cout << "uthresh = " << uthresh << endl;
	Mat lthreshed = LauraFilters::threshold(
		thinned, lthresh);
	Mat uthreshed = LauraFilters::threshold(
		thinned, uthresh);
	Mat threshed = LauraFilters::hysteresisThresholding(
		thinned, lthresh, uthresh);
	
	//Convert back to uchar for display.
	img.convertTo(img, CV_8U);
	img2.convertTo(img2, CV_8U);
	mag.convertTo(mag, CV_8U);
	angimg.convertTo(angimg, CV_8U);
	thinned.convertTo(thinned, CV_8U);
	lthreshed.convertTo(lthreshed, CV_8U);
	uthreshed.convertTo(uthreshed, CV_8U);
	threshed.convertTo(threshed, CV_8U);

	//Show image
	namedWindow(fname, CV_WINDOW_AUTOSIZE);
	imshow(fname, img);
	namedWindow("img2", CV_WINDOW_AUTOSIZE);
	imshow("img2", img2);
	namedWindow("mag", CV_WINDOW_AUTOSIZE);
	imshow("mag", mag);
	namedWindow("angimg", CV_WINDOW_AUTOSIZE);
	imshow("angimg", angimg);
	namedWindow("thinned", CV_WINDOW_AUTOSIZE);
	imshow("thinned", thinned);
	namedWindow("lthreshed", CV_WINDOW_AUTOSIZE);
	imshow("lthreshed", lthreshed);
	namedWindow("uthreshed", CV_WINDOW_AUTOSIZE);
	imshow("uthreshed", uthreshed);
	namedWindow("threshed", CV_WINDOW_AUTOSIZE);
	imshow("threshed", threshed);
	waitKey(0);

	return 0;
}
