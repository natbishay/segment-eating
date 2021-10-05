#include "opencv2/core/core.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/contrib/contrib.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"

#include <iostream>
#include <fstream>
#include <string>


using namespace cv;
using namespace std;


/// Global variables  for TRESHOLDING ------------------------------------------------------------------------------------------

int threshold_value = 51;
int threshold_type = 0;;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

#define MAX_HISTOGRAM_SIZE 2000
#define CANONICAL_SIZE 150
#define PLOT_VERTICAL_AXIS_FOR_HISTOGRAM 2000

CvMemStorage* storage;
CvSeq* firstContour = NULL;

Mat src1, src1_gray, dst1;
char* window_name = "Threshold Demo";

char* trackbar_type = "Type: \n 0: Binary \n 1: Binary Inverted \n 2: Truncate \n 3: To Zero \n 4: To Zero Inverted";
char* trackbar_value = "Value";

void InterpolateUp(int* inputVector, int* expandedOutputVector, int inputSize, int outputSize);
void decimateDown(int* longInputVector, int* outputVector, int inputSize, int outputSize);
//inputSize > CANONICAL_SIZE

/// Function headers
void Threshold_Demo(int, void*);
RNG rng(12345);
void calculateHistogram(int x1, int y1, int xc, int yc, int height, int width, int* maxR, int* histogramPeak, int* histogram);
cv::Mat rawRadialHistogram = cv::Mat::zeros(1, MAX_HISTOGRAM_SIZE, CV_16U);
cv::Mat canonicalRadialHistogramMat = cv::Mat::zeros(1, MAX_HISTOGRAM_SIZE, CV_16U);
cv::Mat rawRadialHistogramPlot = cv::Mat::zeros(PLOT_VERTICAL_AXIS_FOR_HISTOGRAM, MAX_HISTOGRAM_SIZE, CV_16U);

cv::Mat canonicalRadialHistogramMatPlot = cv::Mat::zeros(PLOT_VERTICAL_AXIS_FOR_HISTOGRAM, CANONICAL_SIZE, CV_8U);
int canonicalHistogram[CANONICAL_SIZE];
//--------------------------------------------------------------------------------------------------------------------------------
/// Global variables
Mat src, src_gray;
int thresh = 200;
int max_thresh = 255;

char* source_window = "Source image";
char* corners_window = "Corners detected";



/// Function header
void cornerHarris_demo(int, void*);
void findColorInImage(char* inputImageName, int color, Vec3b bgrPixel, cv::Scalar minBGR, cv::Scalar maxBGR, cv::Scalar minHSV, cv::Scalar maxHSV);

Mat dst, detected_edges;
Mat frame;


void writeMatToFile(cv::Mat& m, const char* filename, int datalength)  // to get rid of the trailing zeros in the Mat (1-dim mat)
																	// 
{
	ofstream fout(filename);

	if (!fout)
	{
		cout << "File Not Opened" << endl;  return;
	}

	for (int i = 0; i<m.rows; i++)
	{
		//for (int j = 0; j<m.cols; j++)
		for (int j = 0; j<datalength; j++)
		{
			fout << m.at<short>(i, j) << endl;
		}
		fout << endl;
	}

	fout.close();
}

/*
int main()
{
// VIDEO

VideoCapture cap(1); // open the default camera
if (!cap.isOpened())  // check if we succeeded
return -1;


for (; ; ) // video loop
{

cap >> frame; // get a new frame from camera
// do any processing
//imwrite("path/to/image.png", frame);
if (waitKey(30) >= 0) break;   // you can increase delay to 2 seconds here
cv::imshow("grabbed Image", frame);
}

cv::imshow("grabbed Image", frame);



cv::imwrite("objects.raw", frame);

waitKey(0);

Mat bright;
Mat dark;

//bright = imread("brightChart.png", CV_LOAD_IMAGE_UNCHANGED);
//dark = imread("darkChart.png", CV_LOAD_IMAGE_UNCHANGED);

bright = imread("brightChart.jpg", CV_LOAD_IMAGE_UNCHANGED);
dark = imread("darkChart.jpg", CV_LOAD_IMAGE_UNCHANGED);
//cvtColor(src, src_gray, CV_BGR2GRAY);
cv::imshow("bright Image", bright);

if (!bright.data)                              // Check for invalid input
{
cout << "Could not open or find the bright image" << std::endl;
return -1;
}


//-----------------------------------------------
//Converting image from BGR to HSV color space.
//	Mat hsv1;
//	cvtColor(frame, hsv1, COLOR_BGR2HSV);

//	Mat mask1, mask2;
// Creating masks to detect the upper and lower red color.
//	inRange(hsv1, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
//	inRange(hsv1, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);

// Generating the final mask
//	mask1 = mask1 + mask2;
//------------------------------------------------------
// color selection of green
Mat brightHSV, darkHSV;

//C++
cv::cvtColor(bright, brightHSV, cv::COLOR_BGR2HSV);
cv::cvtColor(dark, darkHSV, cv::COLOR_BGR2HSV);

//C++ code
//cv::Vec3b bgrPixel(40, 158, 16);   // green pixel in the macbeth chart - works
cv::Vec3b bgrPixel(50, 170, 170);
// Create Mat object from vector since cvtColor accepts a Mat object
Mat3b bgr(bgrPixel);

//Convert pixel values to other color spaces.
Mat3b hsv, ycb, lab;
cvtColor(bgr, ycb, COLOR_BGR2YCrCb);
cvtColor(bgr, hsv, COLOR_BGR2HSV);
cvtColor(bgr, lab, COLOR_BGR2Lab);
//Get back the vector from Mat
Vec3b hsvPixel(hsv.at<Vec3b>(0, 0));
Vec3b ycbPixel(ycb.at<Vec3b>(0, 0));
Vec3b labPixel(lab.at<Vec3b>(0, 0));

int thresh = 40;
int hsvThresh = 20;

cv::Scalar minBGR = cv::Scalar(bgrPixel.val[0] - thresh, bgrPixel.val[1] - thresh, bgrPixel.val[2] - thresh);
cv::Scalar maxBGR = cv::Scalar(bgrPixel.val[0] + thresh, bgrPixel.val[1] + thresh, bgrPixel.val[2] + thresh);

cv::Mat maskBGR, resultBGR;
cv::inRange(bright, minBGR, maxBGR, maskBGR);
imshow("main:bright before bitwise-and", bright);
cv::bitwise_and(bright, bright, resultBGR, maskBGR);
imshow("main:bright AFTER bitwise-and", bright);


imshow("main:maskBGR", maskBGR);

cv::Scalar minHSV = cv::Scalar(hsvPixel.val[0] - hsvThresh, hsvPixel.val[1] - hsvThresh, hsvPixel.val[2] - hsvThresh);
cv::Scalar maxHSV = cv::Scalar(hsvPixel.val[0] + hsvThresh, hsvPixel.val[1] + hsvThresh, hsvPixel.val[2] + hsvThresh);

cv::Mat maskHSV, resultHSV;
cv::inRange(brightHSV, minHSV, maxHSV, maskHSV);
cv::bitwise_and(brightHSV, brightHSV, resultHSV, maskHSV);

//cv::Scalar minYCB = cv::Scalar(ycbPixel.val[0] - thresh, ycbPixel.val[1] - thresh, ycbPixel.val[2] - thresh);
//cv::Scalar maxYCB = cv::Scalar(ycbPixel.val[0] + thresh, ycbPixel.val[1] + thresh, ycbPixel.val[2] + thresh);

//	cv::Mat maskYCB, resultYCB;
//cv::inRange(brightYCB, minYCB, maxYCB, maskYCB);
//cv::bitwise_and(brightYCB, brightYCB, resultYCB, maskYCB);

//cv::Scalar minLAB = cv::Scalar(labPixel.val[0] - thresh, labPixel.val[1] - thresh, labPixel.val[2] - thresh);
//cv::Scalar maxLAB = cv::Scalar(labPixel.val[0] + thresh, labPixel.val[1] + thresh, labPixel.val[2] + thresh);

//	cv::Mat maskLAB, resultLAB;
//cv::inRange(brightLAB, minLAB, maxLAB, maskLAB);
//cv::bitwise_and(brightLAB, brightLAB, resultLAB, maskLAB);

imshow("Result BGR", resultBGR);
imshow("Result HSV", resultHSV);
//imshow("Result YCB", resultYCB);
//imshow("Output LAB", resultLAB);

// find color in Image
//cv::Vec3b bgrPixel(40, 158, 16);   // green pixel in the macbeth chart
int color = 1;
findColorInImage("test-image.jpg", color, bgrPixel, minBGR, maxBGR, minHSV, maxHSV);

// the camera will be deinitialized automatically in VideoCapture destructor
waitKey(0);


return 0;

}
*/

void findColorInImage(char* inputImageName, int color, Vec3b bgrPixel, cv::Scalar minBGR, cv::Scalar maxBGR, cv::Scalar minHSV, cv::Scalar maxHSV)
{
	Mat inputImage, resultImage;
	inputImage = imread(inputImageName, CV_LOAD_IMAGE_UNCHANGED);

	// Create Mat object from vector since cvtColor accepts a Mat object
	Mat3b bgr(bgrPixel);

	//Convert pixel values to other color spaces.
	Mat3b hsv;
	cvtColor(bgr, hsv, COLOR_BGR2HSV);
	//Get back the vector from Mat
	Vec3b hsvPixel(hsv.at<Vec3b>(0, 0));

	imshow("Input Image", inputImage);

	cv::Mat maskBGR, resultBGR;
	cv::inRange(inputImage, minBGR, maxBGR, maskBGR);
	imshow("maskBGR", maskBGR);
	cv::bitwise_and(inputImage, inputImage, resultImage, maskBGR);

	imshow("resulting Image", resultImage);
	/*
	cv::cvtColor(bright, brightHSV, cv::COLOR_BGR2HSV);
	//cv::cvtColor(dark, darkHSV, cv::COLOR_BGR2HSV);
	cv::Mat maskHSV, resultHSV;
	cv::inRange(brightHSV, minHSV, maxHSV, maskHSV);
	cv::bitwise_and(brightHSV, brightHSV, resultHSV, maskHSV);
	*/
}

/*
int main()
{
/// Load source image and convert it to gray

//src = imread("thresholded2.jpg", CV_LOAD_IMAGE_UNCHANGED);
src = imread("white-utentils.jpg", CV_LOAD_IMAGE_UNCHANGED);
//cvtColor(src, src_gray, CV_BGR2GRAY);

if (!dst.data)
waitKey(0);
/// Create a window and a trackbar
namedWindow(source_window, CV_WINDOW_AUTOSIZE);
createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
imshow(source_window, src);

cornerHarris_demo(0, 0);

waitKey(0);
return(0);
}
*/
void cornerHarris_demo(int, void*)
{

	Mat dst, dst_norm, dst_norm_scaled;
	dst = Mat::zeros(src.size(), CV_32FC1);

	/// Detector parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	/// Detecting corners
	cornerHarris(src, dst, blockSize, apertureSize, k, BORDER_DEFAULT);

	/// Normalizing
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(dst_norm, dst_norm_scaled);

	/// Drawing a circle around corners
	for (int j = 0; j < dst_norm.rows; j++)
	{
		for (int i = 0; i < dst_norm.cols; i++)
		{
			if ((int)dst_norm.at<float>(j, i) > thresh)
			{
				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
			}
		}
	}
	/// Showing the result
	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
	imwrite("corner_found.jpg", dst_norm_scaled);
	imshow(corners_window, dst_norm_scaled);
}

////////////////////////////////////// main.c THRESHOLDING WHITE OBJECTS ON BLACK BACKGROUND/////////////
int main()
{
	/// Load an image
	//src1 = imread("white-utentils.jpg", CV_LOAD_IMAGE_UNCHANGED);
	//src1 = imread("two-spoons-knife.png", CV_LOAD_IMAGE_UNCHANGED);  //plate-small-spoon
	src1 = imread("plate-small-spoon.png", CV_LOAD_IMAGE_UNCHANGED);
	storage = cvCreateMemStorage();

	imshow("Source Image", src1);
	
	//Mat src1_gray = cv::Mat::zeros(src1.size(), CV_8U);
	src1_gray.create(src1.size(), CV_8U);



	/// Convert the image to Gray
	cvtColor(src1, src1_gray, CV_BGR2GRAY);

	//imshow("gray image", src1_gray);
	// file to save historgram
	// Declare what you need
	//cv::FileStorage file("some_name.ext", cv::FileStorage::WRITE);

	/// Create a window to display results
	namedWindow(window_name, CV_WINDOW_AUTOSIZE);

	/// Create Trackbar to choose type of Threshold
	createTrackbar(trackbar_type,
		window_name, &threshold_type,
		max_type, Threshold_Demo);

	createTrackbar(trackbar_value,
		window_name, &threshold_value,
		max_value, Threshold_Demo);

	/// Call the function to initialize
	Threshold_Demo(0, 0);
	// saving to file is at the end of Threshold_Demo

	/*
		const char* filename = "rawRadialHistogram.txt";
	string filenameString;
	filename = filenameString.c_str();
	string objectNumberString;

	int i;

	objectNumberString = std::to_string(i);

	filenameString = filename + objectNumberString;

		writeMatToFile(rawRadialHistogram, filename);

		writeMatToFile(canonicalRadialHistogramMat, "canonicalRadialHistogram.txt");
		*/
	/// Wait until user finishes program
	while (true)
	{
		int c;
		c = waitKey(20);
		if ((char)c == 27)
		{
			break;
		}
	}

}



/**
* @function Threshold_Demo
*/
void Threshold_Demo(int, void*)
{
	/* 0: Binary
	1: Binary Inverted
	2: Threshold Truncated
	3: Threshold to Zero
	4: Threshold to Zero Inverted
	*/
	

	threshold(src1_gray, dst1, threshold_value, max_BINARY_value, threshold_type);

	imshow(window_name, dst1);

	

	// Setup SimpleBlobDetector parameters.
	SimpleBlobDetector::Params params;

	// Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 256;

	// Filter by Area.
	params.filterByArea = false; // false
	params.minArea = 1000;
	params.maxArea = 1000000;

	// filter my min distance
	//params.minDistBetweenBlobs=100;

	// Filter by Circularity
	params.filterByCircularity = false;
	params.minCircularity = 0.8;

	// Filter by Convexity
	params.filterByConvexity = false;
	params.minConvexity = 0.5;

	// Filter by Inertia
	params.filterByInertia = false;
	params.minInertiaRatio = 0.01;

	//filter by colour
	params.filterByColor = false;
	params.blobColor = 255;

	// Storage for blobs
	vector<KeyPoint> keypoints;



	// Set up detector with params
	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

	// Detect blobs
	detector->detect(dst1, keypoints);

	//the total no of blobs detected are:
	size_t x = keypoints.size();
	cout << "total no of circles detected are:" << x << endl;

	if (x < 5) // max number of detected Objects is 5
	{
		for (int i = 0; i < x; i++)
		{
		//location of first blob
		Point2f point1 = keypoints.at(i).pt;  // CG for the object
		float x1 = point1.x;
		float y1 = point1.y;
		cout << "location of the" << i << "th" << " blob is " << x1 << "," << y1 << endl;
		}
	}


	// Draw detected blobs as red circles.
	// DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures
	// the size of the circle corresponds to the size of blob
	int g, h;
	Mat im_with_keypoints;
	drawKeypoints(dst1, keypoints, im_with_keypoints, Scalar(0, 0, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	Point point;
	point.x = 2;
	point.y = 4;
	int hh = dst.cols;
	int w = dst.rows;

	int m = dst1.at<uchar>(point);

	// Show blobs
	imshow("keypoints", im_with_keypoints);


	imshow(window_name, dst1);

	
	Mat canny_output;
	Canny(src1_gray, canny_output, threshold_value, thresh * 2);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	//findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	findContours(dst1, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

	imshow("canny output", canny_output);

	
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f>centers(contours.size());
	vector<float>radius(contours.size());
	vector<Point2f> topLeft(contours.size()); vector<Point2f> bottomRight(contours.size());
	vector<int> width(contours.size()); vector<int> height(contours.size()); vector<int> area(contours.size());
	vector<int> maxR(contours.size());
	vector<int> histogramPeak(contours.size());

	int histogram[MAX_HISTOGRAM_SIZE];

	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

	for (size_t i = 0; i < contours.size(); i++)
	{
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		boundRect[i] = boundingRect(contours_poly[i]);
		minEnclosingCircle(contours_poly[i], centers[i], radius[i]);


		//Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);

		cout << "contours.size =  " << contours.size() << endl;


		Scalar color = Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
		drawContours(drawing, contours, (int)i, color, 2, LINE_8, hierarchy, 0);
		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2); // tl = top-left  br = bottom-right
		area[i] = boundRect[i].area();
		topLeft[i] = boundRect[i].tl();
		bottomRight[i] = boundRect[i].br();
		width[i] = boundRect[i].width;
		height[i] = boundRect[i].height;

		cout << "width, height, area of Rect[" << i << "]=" << width[i] << " " << height[i] << " " << area[i] << ", center = " << centers[i].x <<"," << centers[i].y << endl;

		circle(drawing, centers[i], (int)radius[i], color, 2);

		imshow("Contours", drawing);

		// clear the data structures
		for (int s = 0; s < MAX_HISTOGRAM_SIZE; s++)
		{
			rawRadialHistogram.at<short>(0, s) = 0; // short = CV_16U
			canonicalRadialHistogramMat.at<short>(0, s) = 0; 
			histogram[s] = 0;
		}
		for (int s = 0; s < CANONICAL_SIZE; s++)
		{
			canonicalHistogram[s] = 0;
		}

		for (int m = 0; m < PLOT_VERTICAL_AXIS_FOR_HISTOGRAM; m++)
		{
			for (int s = 0; s < CANONICAL_SIZE; s++)
			{
				canonicalRadialHistogramMatPlot.at<uchar>(m, s) = 0; // to zero out the vertical bars at each point s
			}
		}

		
		//} // on the i-loop to check the bounding boxes result  // for testing it is here , it has to be removed to let the histogram be done
		
		//Point2f point1 = keypoints.at(i).pt;  // CG for the object
		//float x1 = point1.x;
		//float y1 = point1.y;
#define MIN_OBJECT_AREA 1000  // the are of the detected object must be higher 16x16 pixels is 256

		if (area[i] > MIN_OBJECT_AREA)
		{

			//Point2f cg = keypoints.at(i).pt;

			// } // on the i-loop to check the bounding boxes result  // for testing it is here , it has to be removed to let the histogram be done


			int histPeak;
			int MaxR;
			
		

			//calculateHistogram(topLeft[i].x, topLeft[i].y, (int)(keypoints.at(i).pt.x), (int)(keypoints.at(i).pt.y), height[i], width[i], &MaxR, &histPeak, histogram);
			calculateHistogram(topLeft[i].x, topLeft[i].y, (int)(centers[i].x), (int)(centers[i].y), height[i], width[i], &MaxR, &histPeak, histogram);

			histogramPeak[i] = histPeak ;

			maxR[i] = MaxR;

			if ((maxR[i]) < CANONICAL_SIZE)
			{
				// stretch to the canonical size
				InterpolateUp(histogram, canonicalHistogram, maxR[i], CANONICAL_SIZE);
				// inputSize < CANONICAL_SIZE
			}
			else
			{
				decimateDown(histogram, canonicalHistogram, maxR[i], CANONICAL_SIZE);
			}

			cout << "object " << i << " has peak = " << histPeak << " " << histogramPeak[i] <<  endl;


			 //the second item  // it will be put inside the loop after we implement dynamic allocaiton
			int plotVerticalSpan;

			//cv::Mat canonicalRadialHistogramMatPlot1 = cv::Mat::zeros(histogramPeak[i]+10, MAX_HISTOGRAM_SIZE, CV_8U);
			//cv::Mat canonicalRadialHistogramMatPlot1 = cv::Mat::zeros(500, MAX_HISTOGRAM_SIZE, CV_8U);

			int histPeakThisTime;
			histPeakThisTime = histPeak;

			// Plot the Canonical Histogram
			for (int j = 0; j < CANONICAL_SIZE; j++)
			{
				canonicalRadialHistogramMat.at<short>(0, j) = canonicalHistogram[j];
				for (int k = 0; k < canonicalHistogram[j]; k++)
					canonicalRadialHistogramMatPlot.at<uchar>(histPeakThisTime - k, j) = (uchar)255;  // to invert it vertically

			}

			// plot the raw data
			for (int j = 0; j < maxR[i]; j++)
			{
				for (int k = 0; k < histogram[j]; k++)
					rawRadialHistogramPlot.at<short>(histPeakThisTime - k, j) = (uchar)255;  // to invert it vertically

			}

			//imshow("histogram", canonicalRadialHistogramMatPlot);



			// Read image
			//Mat im = imread("24.jpg", IMREAD_GRAYSCALE);
			//Size s(400, 300);
			//resize(im, im, s);
			//imshow("original", im);

			// write the result to files 
			const char* RDfilename;
			const char* canonicalRDfilename;
			string rdH = ("rawRadialHistogram");
			string canonicalRDH("canonicalRadialHistorgram");
			string filenameString;
			
			string objectNumberString;

			objectNumberString = std::to_string(i);
			//  code example
			//std::string firstlevel("com");
			//std::string secondlevel("cplusplus");
			//std::string scheme("http://");
			//std::string hostname;
			//std::string url;

			//hostname = "www." + secondlevel + '.' + firstlevel;
			//url = scheme + hostname;


			
			filenameString = rdH + objectNumberString + "-" + std::to_string(maxR[i]) + ".txt";
			RDfilename = filenameString.c_str();
			writeMatToFile(rawRadialHistogram, RDfilename, maxR[i]);

			filenameString = canonicalRDH + objectNumberString + "-" + std::to_string(CANONICAL_SIZE) + ".txt";
			canonicalRDfilename = filenameString.c_str();
			writeMatToFile(canonicalRadialHistogramMat, canonicalRDfilename, CANONICAL_SIZE);

			imshow(RDfilename, rawRadialHistogramPlot);
			imshow(canonicalRDfilename, canonicalRadialHistogramMatPlot);




		}
		
	}
	
	
}

void calculateHistogram(int x1, int y1, int xc, int yc, int height, int width, int* maxR, int* histPeak, int* histogram)
{
	//int histogram[MAX_HISTOGRAM_SIZE];
	for (int h = 0; h < MAX_HISTOGRAM_SIZE; h++)
		histogram[h] = 0;

	//cv::Mat rawRadialHistogram = cv::Mat::zeros(1, MAX_HISTOGRAM_SIZE, CV_16U);

	int y1Int = (int)y1;
	int i, j;
	(*maxR) = (int) 0;
	(*histPeak) = (int) 0;  // peak of the histogram curve

	for (j = y1; j < (y1 + height); j++) {// can increment in a different way if by 1 isn't accurate enough, could consider rescaling
		float y_dist = (j - yc) * (j - yc);
		for (i = x1; i < (x1 + width); i++) {
			// i and j are now the locations
			float x_dist = (i - xc) * (i - xc);
			int r = (sqrt(x_dist + y_dist) + (0.5));
			if ((r > *maxR) && (histogram[r]>1)) { *maxR = r; }
			if ((dst1.at <uchar>(j, i)) > 0)
			{
				histogram[r] ++; //(dst1.at<uchar>(j, i))/230;  // (row, col) index (j, i) not (i, j)
				rawRadialHistogram.at<short>(0, r) = (short)histogram[r];
				if (histogram[r] > (*histPeak)) { *histPeak = histogram[r]; }
			}
		}
	}

	int mr = *maxR;

	//return histogram;
}

void InterpolateUp(int* inputVector, int* expandedOutputVector, int inputSize, int outputSize)
// inputSize < CANONICAL_SIZE
{
	float upScaleFactor;
	upScaleFactor = (float)outputSize / ((float)inputSize);

	// let us say our standard is 300 samples (outputSize) and our input samples are 170 (inputSize)
	// so we want to interpolate up the 170 samples to 300 samples i.e. map a 170-points line to a 300-points line. 
	// upScaleFactor is that ratio - by which we stretch out the samples

	int intervalx10 = (int)(upScaleFactor * 10);  // so as to convert the ratio into the  spacing between the samples but with avoiding to use floating point
	// so we multiply by 10 exp.  300/170 = 1.76  x 10 will give 17 and we shall stretch to 300 x 10 = 3000 , then map the 170 samples to a scale on 3000
	// then find the values at 0, 10, 20, 30, ... on the 3000 scale (by interpolation), they will become the samples at 0, 1, 2, 3, ... on the 300 scale
	// so sample number 0 of the 170 will go to 0 on the 3000, sample at 1 will go to 17, sample at 2 will go to 17*2 = 34, and so on
	// then we interpolate at 10, 20, 30, using these samples at 17, 17*2, 17*3, ...

	int* expandedOutputVectorx10;
	expandedOutputVectorx10 = new int[outputSize * 10];  // for canonical=300, this vector is 300x10 = 3000 SPREADING OUT

	//int expandedOutputVectorx10[CANONICAL_SIZE*10];  // not dynamic - easier for debugging, put it back to dynamic - there is a delete[] at the end of the function PUT IT BACK!!
	// the easy part - map the 170 to 3000

	// FIRST and LAST samples of input map to the first and last of output, and of the extended one becausethey will be used in Interpolation
	expandedOutputVectorx10[0] = inputVector[0];
	expandedOutputVector[0] = inputVector[0];

	//expandedOutputVectorx10[outputSize * 10 - 1] = inputVector[inputSize - 1];
	//expandedOutputVector[outputSize - 1] = inputVector[inputSize - 1];

	//STEP 1
	for (int i = 1; i < inputSize; i++)  // map the INPUT to the SPREAD-OUT vector at the new interval "intervalx10"
		// 0, 17, 34, 51, 68, ... i * intervalx10
	{
		expandedOutputVectorx10[i * intervalx10] = inputVector[i];
	}

	// STEP 2 : INTERPOLATE between the SPREADOUT samples obtained in STEP1 in order to find the values at 0, 10, 20, 30, in the iextended one
	// on a 3000 samples range,  those samples at 0, 10, 20, 30, ... on a 3000 range
	// will be used the samples at 0, 1, 2, 3, ... on the 300 (CANONICAL) range. Voila!!

	float deltaX;
	float deltaY;
	float deltaH;

	for (int i = 1; i < inputSize - 1; i++)
	{
		// find the k which marks the sample after thr current i
		int k = 0;
		do
		{
			k++;
		} while ((k * intervalx10) < (i * 10)); // which sample "number" is right AFTER the current sample i*10?  so at i = 3, we are at sample 30 but k needs to be 2 because
		// sample 30 needs to be computed using the sample at 34 which 2 (not 3) * 17 = 34.  this happens because more than one sample such as 20 and 30 lies in the span
		// between 17 and 34 which is k=1 and k=2 , i.e. withing ONE PERIOD intervalx10

		if (((k - 1)* intervalx10) == (i * 10))
		{
			expandedOutputVectorx10[i * 10] = expandedOutputVectorx10[(k - 1)*intervalx10];
		}
		else
		{

			// our current sample to be computed is at i*10 
			// it will be computed using the sample on the expanded scale that is before it and that is after it - this is where k is pointing. 
			//linear interpolation : using the value before and after the sample, which are located at intervals "intervalx10" i.e. at 0 , 17, 34, ...
			deltaH = (float)expandedOutputVectorx10[k*intervalx10] - (float)expandedOutputVectorx10[(k - 1)*intervalx10];  //vertical delta between samples at intervalx10 distance :  vertial
			deltaX = (float)(i * 10) - (float)((k - 1)*intervalx10);  // our current sample to be computed is at i*10
			deltaY = (deltaH * deltaX) / (float)intervalx10;  // by similar triangles
			expandedOutputVectorx10[i * 10] = expandedOutputVectorx10[(k - 1)*intervalx10] + (int)(deltaY + 0.5);

			if (expandedOutputVectorx10[i * 10] < 0)
			{
				int somethingwrong = 1;
			}


			//linear interpolation : using the value before and after the sample, which are located at intervals "intervalx10" i.e. at 0 , 17, 34, ...
			//expandedOutputVectorx10[i * 10] = (expandedOutputVectorx10[(i)*intervalx10] - expandedOutputVectorx10[(i - 1)*intervalx10]) / ((i*10) - (i-1)*intervalx10); //linear interpolation
			/*
			if (   ( (i * 10) >((i - 1)*intervalx10) )   &&   ( (i * 10) < (i*intervalx10) )   )
			{
			deltaH = expandedOutputVectorx10[i*intervalx10] - expandedOutputVectorx10[(i - 1)*intervalx10];  //vertical delta between samples at intervalx10 distance
			deltaX = (deltaH * ((i * 10) - ((i - 1)*intervalx10))) / (float)intervalx10;
			expandedOutputVectorx10[i * 10] = expandedOutputVectorx10[(i - 1)*intervalx10] + (int)deltaX;
			}
			else  // we have two samples lying between "one intervalx10" interval of the expanded samples, so we have to use the one before , which would be i-2 away not i-1
			{
			if ((i * 10) < ((i - 1)*intervalx10))  // I am at sample 30 (i=3) but it lies before 34 which is 2 * intervalx10 = 2 * 17 = 34
			{
			deltaH = expandedOutputVectorx10[(i-1)*intervalx10] - expandedOutputVectorx10[(i - 2)*intervalx10];  //vertical delta between samples at intervalx10 distance
			deltaX = (deltaH * ((i * 10) - ((i - 2)*intervalx10))) / (float)intervalx10;
			expandedOutputVectorx10[i * 10] = expandedOutputVectorx10[(i - 2)*intervalx10] + (int)deltaX;
			}
			}
			*/

			//those samples at 0, 10, 20, 30, ... on a 3000 range
			// will be used the samples at 0, 1, 2, 3, ... on the 300 (CANONICAL) range. Voila!!
			expandedOutputVector[i] = expandedOutputVectorx10[i * 10];
		}
	}

	delete[] expandedOutputVectorx10;
}

void decimateDown(int* longInputVector, int* outputVector, int inputSize, int outputSize)
//inputSize > CANONICAL_SIZE
{
	// outputSize is less than inputSize
	float upScaleFactor;  // needs to be float because we shall use one digit after the decimal point in intervalx10

	upScaleFactor = (float)inputSize / ((float)outputSize);  // 
	int remainder;
	remainder = inputSize%outputSize;

	int inputVectorCounter = 0;
	int outputVectorCounter = 0;

	int averageOfsetofSamples;
	int upScaleFactorinInt;
	upScaleFactorinInt = (int)upScaleFactor;  // just to use int variable type in the loop 


	if (remainder == 0)  // input is a whole multiple of the output 
		//==> average every number of upScaleFactor of inputSamples to give one output sample
	{

		for (int i = 0; i < outputSize; i++)
		{
			averageOfsetofSamples = 0;
			for (int k = inputVectorCounter; k < (inputVectorCounter + upScaleFactorinInt); k++)
			{
				averageOfsetofSamples += longInputVector[k];

			}
			inputVectorCounter += upScaleFactorinInt;

			outputVector[i] = averageOfsetofSamples / upScaleFactorinInt;
		}

	}

	else
	{

		// let us say our standard is 300 samples (outputSize) and our input samples are 421 (inputSize)
		// so we want to interpolate i.e. "decimate" down the 421 samples to 300 samples i.e. map a 421-points line to a 300-points line. 
		// upScaleFactor is that ratio - by which we squeeze the samples

		int intervalx10 = (int)(upScaleFactor * 10);

		// 421/300 = 1.4 , x 10 = 14.  So the idea is to map a set of the input samples to 10 output samples, till you fill the 300. 
		// 14/2 (=7) maps to 10/5 (=5) --- middle to middle always
		// in this example we need to map every 14 samples to 10 samples 
		// input 0 1 2 3 4 5 6 |7| 8 9 10 11 12 13  
		//output 0 1 2  3   4  |5|  6    7   8   9

		// we need a case statement here : based on the value of intervalx10

		// 11 --> 10
		// input 0 1 2 3 4 |5| 6 7 8 9 10 
		//output 0 1 2 3 4 |5| 6 7 8  9

		switch (intervalx10)
		{
		case 11:
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = longInputVector[inputVectorCounter + 1];
				outputVector[i + 2] = longInputVector[inputVectorCounter + 2];
				outputVector[i + 3] = longInputVector[inputVectorCounter + 3];
				outputVector[i + 4] = longInputVector[inputVectorCounter + 4];
				outputVector[i + 5] = longInputVector[inputVectorCounter + 5];
				outputVector[i + 6] = longInputVector[inputVectorCounter + 6];
				outputVector[i + 7] = longInputVector[inputVectorCounter + 7];
				outputVector[i + 8] = longInputVector[inputVectorCounter + 8];
				outputVector[i + 9] = (longInputVector[inputVectorCounter + 9] + longInputVector[inputVectorCounter + 10]) >> 1;  // shift right , i.e. divide by 2

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		case 12:
			// 12 --> 10
			// input 0 1 2 3 4 5 |6| 7  8 9  10 11
			//output 0 1 2 3  4  |5| 6   7   8  9
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = longInputVector[inputVectorCounter + 1];
				outputVector[i + 2] = longInputVector[inputVectorCounter + 2];
				outputVector[i + 3] = longInputVector[inputVectorCounter + 3];
				outputVector[i + 4] = (longInputVector[inputVectorCounter + 4] + longInputVector[inputVectorCounter + 5]) >> 1;
				outputVector[i + 5] = longInputVector[inputVectorCounter + 6];
				outputVector[i + 6] = longInputVector[inputVectorCounter + 7];
				outputVector[i + 7] = (longInputVector[inputVectorCounter + 8] + longInputVector[inputVectorCounter + 9]) >> 1;
				outputVector[i + 8] = longInputVector[inputVectorCounter + 10];
				outputVector[i + 9] = longInputVector[inputVectorCounter + 11];

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		case 13:
			// 13 --> 10
			// input 0 1 2 3 4 5 |6| 7 8 9 10 11 12
			//output 0 1 2 3  4  |5|  6   7    8  9
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = longInputVector[inputVectorCounter + 1];
				outputVector[i + 2] = longInputVector[inputVectorCounter + 2];
				outputVector[i + 3] = longInputVector[inputVectorCounter + 3];
				outputVector[i + 4] = (longInputVector[inputVectorCounter + 4] + longInputVector[inputVectorCounter + 5]) >> 1;
				outputVector[i + 5] = longInputVector[inputVectorCounter + 6];
				outputVector[i + 6] = (longInputVector[inputVectorCounter + 7] + longInputVector[inputVectorCounter + 8]) >> 1;
				outputVector[i + 7] = (longInputVector[inputVectorCounter + 9] + longInputVector[inputVectorCounter + 10]) >> 1;
				outputVector[i + 8] = longInputVector[inputVectorCounter + 11];
				outputVector[i + 9] = longInputVector[inputVectorCounter + 12];

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		case 14:
			// 14 --> 10
			// input 0 1 2 3 4 5 6 |7| 8 9 10 11 12 13  
			//output 0 1 2  3   4  |5|  6    7   8   9
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = longInputVector[inputVectorCounter + 1];
				outputVector[i + 2] = longInputVector[inputVectorCounter + 2];
				outputVector[i + 3] = (longInputVector[inputVectorCounter + 3] + longInputVector[inputVectorCounter + 4]) >> 1;
				outputVector[i + 4] = (longInputVector[inputVectorCounter + 5] + longInputVector[inputVectorCounter + 6]) >> 1;
				outputVector[i + 5] = longInputVector[inputVectorCounter + 7];
				outputVector[i + 6] = (longInputVector[inputVectorCounter + 8] + longInputVector[inputVectorCounter + 9]) >> 1;
				outputVector[i + 7] = (longInputVector[inputVectorCounter + 10] + longInputVector[inputVectorCounter + 11]) >> 1;
				outputVector[i + 8] = longInputVector[inputVectorCounter + 12];
				outputVector[i + 9] = longInputVector[inputVectorCounter + 13];

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		case 15:
			// 15 --> 10
			// input 0 1 2 3 4 5 6 |7| 8 9 10 11 12 13 14
			//output 0 1 2  3   4  |5|  6    7     8    9
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = longInputVector[inputVectorCounter + 1];
				outputVector[i + 2] = longInputVector[inputVectorCounter + 2];
				outputVector[i + 3] = (longInputVector[inputVectorCounter + 3] + longInputVector[inputVectorCounter + 4]) >> 1;
				outputVector[i + 4] = (longInputVector[inputVectorCounter + 5] + longInputVector[inputVectorCounter + 6]) >> 1;
				outputVector[i + 5] = longInputVector[inputVectorCounter + 7];
				outputVector[i + 6] = (longInputVector[inputVectorCounter + 8] + longInputVector[inputVectorCounter + 9]) >> 1;
				outputVector[i + 7] = (longInputVector[inputVectorCounter + 10] + longInputVector[inputVectorCounter + 11]) >> 1;
				outputVector[i + 8] = (longInputVector[inputVectorCounter + 12] + longInputVector[inputVectorCounter + 13]) >> 1;
				outputVector[i + 9] = longInputVector[inputVectorCounter + 14];

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		case 16:
			// 16 --> 10
			// input 0 1 2 3 4 5 6 7 |8| 9 10 11 12 13 14 15
			//output 0 1  2   3   4  |5|  6     7     8    9
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = longInputVector[inputVectorCounter + 1];
				outputVector[i + 2] = (longInputVector[inputVectorCounter + 2] + longInputVector[inputVectorCounter + 3]) >> 1;
				outputVector[i + 3] = (longInputVector[inputVectorCounter + 4] + longInputVector[inputVectorCounter + 5]) >> 1;
				outputVector[i + 4] = (longInputVector[inputVectorCounter + 6] + longInputVector[inputVectorCounter + 7]) >> 1;
				outputVector[i + 5] = longInputVector[inputVectorCounter + 8];
				outputVector[i + 6] = (longInputVector[inputVectorCounter + 9] + longInputVector[inputVectorCounter + 10]) >> 1;
				outputVector[i + 7] = (longInputVector[inputVectorCounter + 11] + longInputVector[inputVectorCounter + 12]) >> 1;
				outputVector[i + 8] = (longInputVector[inputVectorCounter + 13] + longInputVector[inputVectorCounter + 14]) >> 1;
				outputVector[i + 9] = longInputVector[inputVectorCounter + 15];

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		case 17:
			// 17 --> 10
			// input 0 1 2 3 4 5 6 7 |8| 9 10 11 12 13 14 15 16
			//output 0 1  2   3   4  |5|  6     7     8     9
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = longInputVector[inputVectorCounter + 1];
				outputVector[i + 2] = (longInputVector[inputVectorCounter + 2] + longInputVector[inputVectorCounter + 3]) >> 1;
				outputVector[i + 3] = (longInputVector[inputVectorCounter + 4] + longInputVector[inputVectorCounter + 5]) >> 1;
				outputVector[i + 4] = (longInputVector[inputVectorCounter + 6] + longInputVector[inputVectorCounter + 7]) >> 1;
				outputVector[i + 5] = longInputVector[inputVectorCounter + 8];
				outputVector[i + 6] = (longInputVector[inputVectorCounter + 9] + longInputVector[inputVectorCounter + 10]) >> 1;
				outputVector[i + 7] = (longInputVector[inputVectorCounter + 11] + longInputVector[inputVectorCounter + 12]) >> 1;
				outputVector[i + 8] = (longInputVector[inputVectorCounter + 13] + longInputVector[inputVectorCounter + 14]) >> 1;
				outputVector[i + 9] = (longInputVector[inputVectorCounter + 15] + longInputVector[inputVectorCounter + 16]) >> 1;

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		case 18:
			// 18 --> 10
			// input 0 1 2 3 4 5 6 7 8 |9| 10 11 12 13 14 15 16 17
			//output 0  1   2   3   4  |5|   6     7     8     9
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = (longInputVector[inputVectorCounter + 1] + longInputVector[inputVectorCounter + 2]) >> 1;
				outputVector[i + 2] = (longInputVector[inputVectorCounter + 3] + longInputVector[inputVectorCounter + 4]) >> 1;
				outputVector[i + 3] = (longInputVector[inputVectorCounter + 5] + longInputVector[inputVectorCounter + 6]) >> 1;
				outputVector[i + 4] = (longInputVector[inputVectorCounter + 7] + longInputVector[inputVectorCounter + 8]) >> 1;
				outputVector[i + 5] = longInputVector[inputVectorCounter + 9];
				outputVector[i + 6] = (longInputVector[inputVectorCounter + 10] + longInputVector[inputVectorCounter + 11]) >> 1;
				outputVector[i + 7] = (longInputVector[inputVectorCounter + 12] + longInputVector[inputVectorCounter + 13]) >> 1;
				outputVector[i + 8] = (longInputVector[inputVectorCounter + 14] + longInputVector[inputVectorCounter + 15]) >> 1;
				outputVector[i + 9] = (longInputVector[inputVectorCounter + 16] + longInputVector[inputVectorCounter + 17]) >> 1;

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		case 19:
			// 19 --> 10
			// input 0 1 2 3 4 5 6 7 8 |9| 10 11 12 13 14 15 16 17 18
			//output 0  1   2   3   4     |5|   6     7     8     9
			for (int i = 0; i < outputSize; i += 10)
			{
				outputVectorCounter = i;
				outputVector[i] = longInputVector[inputVectorCounter];
				outputVector[i + 1] = (longInputVector[inputVectorCounter + 1] + longInputVector[inputVectorCounter + 2]) >> 1;
				outputVector[i + 2] = (longInputVector[inputVectorCounter + 3] + longInputVector[inputVectorCounter + 4]) >> 1;
				outputVector[i + 3] = (longInputVector[inputVectorCounter + 5] + longInputVector[inputVectorCounter + 6]) >> 1;
				outputVector[i + 4] = (longInputVector[inputVectorCounter + 7] + longInputVector[inputVectorCounter + 8]) >> 1;
				outputVector[i + 5] = (longInputVector[inputVectorCounter + 9] + longInputVector[inputVectorCounter + 10]) >> 1;
				outputVector[i + 6] = (longInputVector[inputVectorCounter + 11] + longInputVector[inputVectorCounter + 12]) >> 1;
				outputVector[i + 7] = (longInputVector[inputVectorCounter + 13] + longInputVector[inputVectorCounter + 14]) >> 1;
				outputVector[i + 8] = (longInputVector[inputVectorCounter + 15] + longInputVector[inputVectorCounter + 16]) >> 1;
				outputVector[i + 9] = (longInputVector[inputVectorCounter + 17] + longInputVector[inputVectorCounter + 18]) >> 1;

				inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
			}
			break;

		}

		//inputVectorCounter += intervalx10; // point to the next batch of samples in the longInputVector;
	}
}

/*
int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
char* window_name = "Edge Map";

void CannyThreshold(int, void*)
{
/// Reduce noise with a kernel 3x3
blur(src_gray, detected_edges, Size(3, 3));

/// Canny detector
Canny(detected_edges, detected_edges, 47.0 , 47.0*ratio, kernel_size);

/// Using Canny's output as a mask, we display our result
dst = Scalar::all(0);

src.copyTo(dst, detected_edges);
imshow(window_name, dst);
}

int main(int argc, char** argv)
{
/// Load an image
src = imread("test1.bmp");

if (!src.data)
{
return -1;
}

/// Create a matrix of the same type and size as src (for dst)
dst.create(src.size(), src.type());

/// Convert the image to grayscale
cvtColor(src, src_gray, CV_BGR2GRAY);

/// Create a window
namedWindow(window_name, CV_WINDOW_AUTOSIZE);

/// Create a Trackbar for user to enter threshold
createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);

/// Show the image
CannyThreshold(0, 0);

detectHarris();
/// Wait until user exit program by pressing a key
waitKey(0);

return 0;
}



/// Global variables
Mat src, src_gray;
Mat myHarris_dst; Mat myHarris_copy; Mat Mc;
Mat myShiTomasi_dst; Mat myShiTomasi_copy;

int myShiTomasi_qualityLevel = 50;
int myHarris_qualityLevel = 50;
int max_qualityLevel = 100;

double myHarris_minVal; double myHarris_maxVal;
double myShiTomasi_minVal; double myShiTomasi_maxVal;

RNG rng(12345);

const char* myHarris_window = "My Harris corner detector";
const char* myShiTomasi_window = "My Shi Tomasi corner detector";

/// Function headers
void myShiTomasi_function(int, void*);
void myHarris_function(int, void*);


int main()
{
/// Load source image and convert it to gray
src = imread("thresholded.jpg");
cvtColor(src, src_gray, COLOR_BGR2GRAY);

/// Set some parameters
int blockSize = 3; int apertureSize = 3;

/// My Harris matrix -- Using cornerEigenValsAndVecs
myHarris_dst = Mat::zeros(src_gray.size(), CV_32FC(6));
Mc = Mat::zeros(src_gray.size(), CV_32FC1);

cornerEigenValsAndVecs(src_gray, myHarris_dst, blockSize, apertureSize, BORDER_DEFAULT);


for (int j = 0; j < src_gray.rows; j++)
{
for (int i = 0; i < src_gray.cols; i++)
{
float lambda_1 = myHarris_dst.at<Vec6f>(j, i)[0];
float lambda_2 = myHarris_dst.at<Vec6f>(j, i)[1];
Mc.at<float>(j, i) = lambda_1*lambda_2 - 0.04f*pow((lambda_1 + lambda_2), 2);
}
}

minMaxLoc(Mc, &myHarris_minVal, &myHarris_maxVal, 0, 0, Mat());

namedWindow(myHarris_window, WINDOW_AUTOSIZE);
createTrackbar(" Quality Level:", myHarris_window, &myHarris_qualityLevel, max_qualityLevel, myHarris_function);
myHarris_function(0, 0);

/// My Shi-Tomasi -- Using cornerMinEigenVal
myShiTomasi_dst = Mat::zeros(src_gray.size(), CV_32FC1);
cornerMinEigenVal(src_gray, myShiTomasi_dst, blockSize, apertureSize, BORDER_DEFAULT);

minMaxLoc(myShiTomasi_dst, &myShiTomasi_minVal, &myShiTomasi_maxVal, 0, 0, Mat());

namedWindow(myShiTomasi_window, WINDOW_AUTOSIZE);
createTrackbar(" Quality Level:", myShiTomasi_window, &myShiTomasi_qualityLevel, max_qualityLevel, myShiTomasi_function);
myShiTomasi_function(0, 0);

waitKey(0);
return(0);
}

void myShiTomasi_function(int, void*)
{
myShiTomasi_copy = src.clone();

if (myShiTomasi_qualityLevel < 1) { myShiTomasi_qualityLevel = 1; }

for (int j = 0; j < src_gray.rows; j++)
{
for (int i = 0; i < src_gray.cols; i++)
{
if (myShiTomasi_dst.at<float>(j, i) > myShiTomasi_minVal + (myShiTomasi_maxVal - myShiTomasi_minVal)*myShiTomasi_qualityLevel / max_qualityLevel)
{
circle(myShiTomasi_copy, Point(i, j), 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
}
}
}
imshow(myShiTomasi_window, myShiTomasi_copy);
}

void myHarris_function(int, void*)
{
myHarris_copy = src.clone();

if (myHarris_qualityLevel < 1) { myHarris_qualityLevel = 1; }

for (int j = 0; j < src_gray.rows; j++)
{
for (int i = 0; i < src_gray.cols; i++)
{
if (Mc.at<float>(j, i) > myHarris_minVal + (myHarris_maxVal - myHarris_minVal)*myHarris_qualityLevel / max_qualityLevel)
{
circle(myHarris_copy, Point(i, j), 4, Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1, 8, 0);
}
}
}
imshow(myHarris_window, myHarris_copy);
}


int main(int argc, char** argv)
{

namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

int iLowH = 0;
int iHighH = 179;

int iLowS = 0;
int iHighS = 255;

int iLowV = 0;
int iHighV = 255;

//Create trackbars in "Control" window
cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
cvCreateTrackbar("HighH", "Control", &iHighH, 179);

cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
cvCreateTrackbar("HighS", "Control", &iHighS, 255);

cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)
cvCreateTrackbar("HighV", "Control", &iHighV, 255);

while (true)
{
Mat imgOriginal;

imgOriginal = imread("redPic2.bmp");

//resize(imgOriginal, imgOriginal, cv::Size(640, 480));

if (!imgOriginal.data) //if not success, break loop
{
cout << "Cannot read a frame from video stream" << endl;
return 0;
break;
}

Mat imgHSV;

cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

Mat imgThresholded;

inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image

//morphological opening (remove small objects from the foreground)
erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

//morphological closing (fill small holes in the foreground)
dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

imshow("Thresholded Image", imgThresholded); //show the thresholded image
imshow("Original", imgOriginal); //show the original image

if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
{
cout << "esc key is pressed by user" << endl;
break;
}

if (waitKey(30) == 112)
{
imwrite("thresholded2.jpg", imgThresholded);
cout << "picture is written\n" << endl;
}
}

return 0;

}
*/