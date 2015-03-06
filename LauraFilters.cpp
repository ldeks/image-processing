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

#include "LauraFilters.h"
#include <cmath>
#include <iostream>

#define PI 3.14159265358979323846264338327950288

LauraFilters::LauraFilters()
{

}

LauraFilters::~LauraFilters()
{

}

Mat
LauraFilters::gaussian(int fsize1, int fsize2, float sigma)
{
	//Make return matrix. _ for element access.
	Mat ret = Mat::zeros(fsize1, fsize2, CV_32F);
	cv::Mat_<float> ret_ = ret;
	
	int halfsize1 = fsize1/2;
	int halfsize2 = fsize2/2;
	for (int i = -halfsize1; i < halfsize1 + 1; ++i)
	{
		for	(int j = -halfsize2; j < halfsize2 + 1; ++j)
		{
			float y = (float) i;
			float x = (float) j;
			ret_(i + halfsize1, j + halfsize2) = 
				(1/(2*PI*sigma*sigma)) *
				exp(-1*((x*x + y*y)/(2*sigma*sigma)));
		}
	}

	return ret;
}

Mat
LauraFilters::laplacian()
{
	Mat ret = (cv::Mat_<float>(3, 3) 
		<< 1, 1, 1, 1, -8, 1, 1, 1, 1);
	return ret;
}

Mat
LauraFilters::LoG(int fsize, float sigma)
{
	//Make return matrix. _ for element access.
	Mat ret = Mat::zeros(fsize, fsize, CV_32F);
	cv::Mat_<float> ret_ = ret;
	
	int halfsize = fsize/2;
	for (int i = -halfsize; i < halfsize + 1; ++i)
	{
		for	(int j = -halfsize; j < halfsize + 1; ++j)
		{
			float y = (float) i;
			float x = (float) j;
			ret_(i + halfsize, j + halfsize) = 
				((x*x + y*y - (2*sigma*sigma))/
				(2*PI*pow(sigma, 6))) *
				exp(-1*((x*x + y*y)/(2*sigma*sigma)));
		}
	}

	return ret;
}

Mat
LauraFilters::gx3x3()
{
	Mat ret = (cv::Mat_<float>(3, 3) 
		<< -1, 0, 1, -2, 0, 2, -1, 0, 1);
	return ret;
}

Mat
LauraFilters::gy3x3()
{
	Mat ret = (cv::Mat_<float>(3, 3) 
		<< 1, 2, 1, 0, 0, 0, -1, -2, -1);
	return ret;
}

Mat
LauraFilters::zeroCross3x3(Mat& img)
{
	//For determining if pixel in process is
	//"small" in intensity value.
	double min, max;
	cv::minMaxLoc(img, &min, &max);
	//float eps = 0.1*((fabs(min) + fabs(max))/2.0f);
	cv::Scalar mean, stddev;
	cv::meanStdDev(img, mean, stddev);
	float eps = 2*stddev(0);//1.5*stddev(0);
	std::cout << "mean: " << mean(0) << std::endl;
	std::cout << "std: " << stddev(0) << std::endl;
	std::cout << "Zero epsilon: " << eps << std::endl;
	float deps = 0.5*stddev(0);//2.5*stddev(0);
	std::cout << "Difference epsilon: " << deps << std::endl;

	//Make return matrix and _ for element access.
	cv::Mat_<float> img_ = img;
	Mat ret = Mat::zeros(img.rows, img.cols, CV_32F);
	cv::Mat_<float> ret_ = ret;

	//For each pixel in process (not considering the boundary).
	for (int i = 1; i < img.rows - 1; ++i)
	{
		for (int j = 1; j < img.cols - 1; ++j)
		{
			//Check to see how many pairs of neighbors
			//have opposing signs. If between 1 and 3,
			//P0 is on the edge.

			//Using the book's numbering:
			//P4 P3 P2
			//P5 P0 P1
			//P6 P7 P8
			float p0 = img_(i, j);
			if (1e-30 > p0)
			{
				float p1 = img_(i, j+1);
				float p5 = img_(i, j-1);
				float p3 = img_(i-1, j);
				float p4 = img_(i-1, j-1);
				float p2 = img_(i-1, j+1);
				float p7 = img_(i+1, j);
				float p6 = img_(i+1, j-1);
				float p8 = img_(i+1, j+1);

				////For counting opposite pairs.
				int oppositePairs = 0;
				if(opposingPair(p1, p5, deps)) oppositePairs++;
				if(opposingPair(p2, p6, deps)) oppositePairs++;
				if(opposingPair(p3, p7, deps)) oppositePairs++;
				if(opposingPair(p4, p8, deps)) oppositePairs++;

				if ((0 < oppositePairs) && (4 > oppositePairs))
					ret_(i, j) = 255.0f;
			}
		}
	}

	return ret;
}

bool 
LauraFilters::opposingPair(float p1, float p2, float eps)
{
	//For determining if there is a 
	//significant difference.
	float zeps = 10e-30;

	if ((p1 * p2 < zeps) && (fabs(p2 - p1) > eps))
		return true;
	else
		return false;
}

Mat
LauraFilters::nonmaximaSuppression3x3(
	Mat& mag, Mat& angle)
{
	//Make return matrix and _ for element access.
	cv::Mat_<float> mag_ = mag;
	cv::Mat_<float> angle_ = angle;
	Mat ret = mag.clone();
	cv::Mat_<float> ret_ = ret;

	//For each pixel in process (not considering the boundary).
	for (int i = 1; i < mag.rows - 1; ++i)
	{
		for (int j = 1; j < mag.cols - 1; ++j)
		{
			//If the pixel is pure black, ignore it.
			if (!mag_(i, j)) continue;

			//Determine edge angle.
			float ang = angle_(i, j);
			while (0.0f > ang)
				ang += 360.0f;
			while (360.0f < ang)
				ang -= 360.0f;

			//Direction to thin (degrees).
			//Either -45, 0, 45, or 90.
			int thinDir;
			if ((22.5f > ang) || (337.5f <= ang))
				thinDir = 0;
			else if ((22.5f <= ang) && (67.5f > ang))
				thinDir = 45;
			else if ((67.5f <= ang) && (112.5f > ang))
				thinDir = 90;
			else if ((112.5f <= ang) && (157.5f > ang))
				thinDir = -45;
			else if ((157.5f <= ang) && (202.5f > ang))
				thinDir = 0;
			else if ((202.5f <= ang) && (247.5f > ang))
				thinDir = 45;
			else if ((247.5f <= ang) && (292.5f > ang))
				thinDir = 90;
			else if ((292.5f <= ang) && (337.5f > ang))
				thinDir = -45;

			//Determine whether or not pix in process
			//is a local maximum.
			//Using the book's numbering:
			//P4 P3 P2
			//P5 P0 P1
			//P6 P7 P8
			float p0 = mag_(i, j);
			bool isMax;
			if (-45 == thinDir)
			{
				float p4 = mag_(i-1, j-1);
				float p8 = mag_(i+1, j+1);
				isMax = isLocalMax(p0, p4, p8);
			}
			else if (0 == thinDir)
			{
				float p1 = mag_(i, j+1);
				float p5 = mag_(i, j-1);
				isMax = isLocalMax(p0, p1, p5);
			}
			else if (45 == thinDir)
			{
				float p2 = mag_(i-1, j+1);
				float p6 = mag_(i+1, j-1);
				isMax = isLocalMax(p0, p2, p6);
			}
			else //thinDir = 90
			{
				float p3 = mag_(i-1, j);
				float p7 = mag_(i+1, j);
				isMax = isLocalMax(p0, p3, p7);
			}

			if (!isMax)
				ret_(i, j) = 0.0f;
		}
	}

	return ret;
}

Mat
LauraFilters::nonmaximaSuppression3x3(Mat& mag)
{
	//Make return matrix and _ for element access.
	cv::Mat_<float> mag_ = mag;
	Mat ret = mag.clone();
	cv::Mat_<float> ret_ = ret;

	//For each pixel in process (not considering the boundary).
	for (int i = 1; i < mag.rows - 1; ++i)
	{
		for (int j = 1; j < mag.cols - 1; ++j)
		{
			//If the pixel is pure black, ignore it.
			if (!mag_(i, j)) continue;

			//Determine whether or not pix in process
			//is a local maximum in the blob.
			//Using the book's numbering:
			//P4 P3 P2
			//P5 P0 P1
			//P6 P7 P8
			bool isMax;
			int count = 0;
			float p0, p1, p2, p3, p4, p5, p6, p7, p8;
			p0 = mag_(i, j);
			p4 = mag_(i-1, j-1);
			p8 = mag_(i+1, j+1);
			isMax = isLocalMax(p0, p4, p8);
			if(isMax) count++;
			p1 = mag_(i, j+1);
			p5 = mag_(i, j-1);
			isMax = isLocalMax(p0, p1, p5);
			if(isMax) count++;
			p2 = mag_(i-1, j+1);
			p6 = mag_(i+1, j-1);
			isMax = isLocalMax(p0, p2, p6);
			if(isMax) count++;
			p3 = mag_(i-1, j);
			p7 = mag_(i+1, j);
			isMax = isLocalMax(p0, p3, p7);
			if(isMax) count++;

			if (count < 4)
				ret_(i, j) = 0.0f;
		}
	}

	return ret;
}

bool
LauraFilters::isLocalMax(float p0,
	float p1, float p2)
{
	float eps = -1e-32;
	//If p0 == p1 or p0 == p2,
	//this will probably return true and may ultimately give you
	//a line of double thickness.
	//But without a bigger neighborhood and interpolation, this
	//will have to do.
	if ((p0 - p1 > eps) && (p0 - p2 > eps))
		return true;
	else
		return false;
}


Mat 
LauraFilters::hysteresisThresholding(
	Mat& img, float lthresh,
	float uthresh)
{
	//Threshold the image
	//ubin = upper thresholded image.
	//lbin = lower thresholded image.
	Mat ubin = threshold(img, uthresh);
	cv::Mat_<float> ubin_ = ubin;
	Mat lbin = threshold(img, lthresh);
	cv::Mat_<float> lbin_ = lbin;

	//Examine 3x3 neighborhoods.
	//Operate directly on ubin, since this
	//will act to grow more edges.
	for (int i = 1; i < img.rows - 1; ++i)
	{
		for (int j = 1; j < img.cols - 1; ++j)
		{
			//If this pixel is white in lbin
			//and black in ubin
			if (lbin_(i, j) && (!ubin_(i, j)))
			{
				//Are any in the neighborhood 
				//in ubin white?
				//If so, make this one white.
				float intensitySum = 0;
				intensitySum = ubin_(i-1, j-1)
				 			 + ubin_(i-1, j)
				 			 + ubin_(i-1, j+1)
							 + ubin_(i, j-1)
							 + ubin_(i, j+1)
				 			 + ubin_(i+1, j-1)
				 			 + ubin_(i+1, j)
				 			 + ubin_(i+1, j+1);
				if ((0 < intensitySum) && (5*255.0f > intensitySum))
					ubin_(i, j) = 255.0f;
			}
		}
	}

	return ubin;
}

Mat 
LauraFilters::threshold(Mat& img,
		float thresh)
{
	//Make return matrix and _ for element access.
	cv::Mat_<float> img_ = img;
	Mat ret = Mat::zeros(img.rows, img.cols, CV_32F);
	cv::Mat_<float> ret_ = ret;

	for (int i = 0; i < img.rows; ++i)
	{
		for (int j = 0; j < img.cols; ++j)
		{
			if (thresh < img_(i, j))
				ret_(i, j) = 255.0f;
		}
	}

	return ret;
}

void
LauraFilters::correctedMeanStdDev(
	Mat& img, float* mean, float* stddev)
{
	//Generate a mask from img where 0 values are 0 (trivial, of
	//course) and 255 values are also 0.
	Mat img8u;
	img.convertTo(img8u, CV_8U); //Mask must be CV_8U.
	cv::Mat_<unsigned char> img8u_ = img8u;
	for (int i = 0; i < img8u.rows; ++i)
	{
		for (int j = 0; j < img8u.cols; ++j)
		{
			if (255 == img8u_(i, j))
				img8u_(i, j) = 0;
		}
	}
	
	//Ask OpenCV to compute mean and stddev without including
	//the masked points.
	cv::Scalar tmean, tstd;
	cv::meanStdDev(img, tmean, tstd, img8u);

	//Convert to float and store in pointers.
	*mean = (float) tmean(0);
	*stddev = (float) tstd(0);
}
