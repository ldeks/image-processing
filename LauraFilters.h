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

#ifndef __LAURAFILTERS_H__
#define __LAURAFILTERS_H__

#include <opencv2/opencv.hpp>
using cv::Mat;

class LauraFilters
{
public:
	LauraFilters();
	~LauraFilters();

	/***** Filters for use with convolution *****/

	//Generates a 2D Gaussian filter
	//of size fsize1 rows x fsize2 cols
	//and std. dev. sigma.
	static Mat gaussian(int fsize1,
		int fsize2, float sigma);

	//Generates a 3x3 Laplacian filter.
	static Mat laplacian();

	//Generates a 2D Laplacian of Gaussian filter
	//of size fsize x fsize and std. dev.
	//sigma.
	static Mat LoG(int fsize, float sigma);

	//3x3 Sobel gradient estimate in x
	static Mat gx3x3();
	//3x3 Sobel gradient estimate in y
	static Mat gy3x3();

	/***** Filters for the whole image ****/

	//Finds zero-crossings in an image
	//by looking in a 3x3 neighborhood.
	static Mat zeroCross3x3(Mat& img);
	//Helper function for zeroCross3x3
	//Determines whether two intensities
	//are on opposite sides of I = 0.
	//eps is the threshold for a significant difference.
	static bool opposingPair(float p1, float p2, float eps);

	//Performs nonmaxima suppression
	//Requires a gradient magnitude and
	//a gradient angle (phase) image.
	//angle must be in degrees!!!!
	//Only examines a 3x3 neighborhood.
	static Mat nonmaximaSuppression3x3(
		Mat& mag, Mat& angle);
	//Performs nonmaxima suppression on blobs (all angles at once).
	//Only works on a 3x3.
	static Mat nonmaximaSuppression3x3(Mat& mag);
	//Helper for nonmaximaSuppression.
	//Determine whether p0 is a local
	//max in comparison with p1 and p2.
	static bool isLocalMax(float p0,
		float p1, float p2);

	//Performs hystersis thresholding
	//using upper threshold uthresh
	//and lower threshold lthresh.
	static Mat hysteresisThresholding(
		Mat& img, float lthresh,
		float uthresh);

	//Performs thresholding.
	//Points above thresh will be set to
	//255.0f, points below will be 
	//set to 0.0f.
	static Mat threshold(Mat& img,
		float thesh);

	/**** Functions returning scalars ***/
	//Computes mean and standard deviation
	//for the image and returns them as 
	//floats.  Zeros both the 0 and 255
	//contribution to avoid getting thrown off by
	//them.
	static void correctedMeanStdDev(
		Mat& img, float* mean, float* stddev);
};

#endif //!defined __LAURAFILTERS_H__
