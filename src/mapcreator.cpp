#define BLUE_CHANNEL 0

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "tools.cpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#define BLOB_SIZE_THRESHOLD 10  // blobs smaller than (1/BLOB_SIZE_THRESHOLD)*SIZE_OF_THE BIGGEST_BLOB will be ignored
#define BLUR_SIZE 2

// names of the windows
const char* orig_image_window_name = "Original Image";
const char* blurred_image_window_name = "Blurred";
const char* blackpixels_window_name = "BLACK PIXELS from blurred image";
const char* blobs_image_window_name = "BLOBS";
const char* purged_blobs_image_window_name = "BIG BLOBS";
const char* centroids_image_window_name = "CENTROIDS";

int x_offset = 320; // these offsets are used to place the windows
int y_offset = 290;

inline std::string stringify(double x)
{
  std::ostringstream o;
  if (!(o << x)){
	  std::cout << "Bad Conversion: stringify(double)\n";
  }
  return o.str();
}

int main(int argc, char** argv){

	cvStartWindowThread();

	if(argc<2){
		std::cout << "Usage: MapCreator <image file name>\n";
		exit(0);
	}

	std::cout << "\n<<<<< MapCreator >>>>>\n\n";

	std::cout << "Reading filter \n";
	cv::Mat filter;
	filter = cv::imread("filter.png", 0);

	std::cout << "Reading image \"" << argv[1] << "\"...\n";
	cv::Mat orig_image;
	orig_image = cv::imread(argv[1], 0);

	if(!orig_image.data){
		std::cout << "cannot read image \"" << argv[1] << "\"\nExiting...\n";
		exit(1);
	}

	// blur the image
	std::cout << "Blurring the image...\n";
	cv::Mat *blurred_image = new cv::Mat;
//	tools::blur(&orig_image, blurred_image, BLUR_SIZE);

	// select black pixels;
	std::cout << "Selecting black pixels...\n";
	cv::Mat * blackpixels = tools::getBlackPixels(&orig_image);

	// extract blobs
	std::cout << "Extracting the blobs...\n";
	std::vector<Blob*> blobs;
	tools::getBlobsIterative(blackpixels, &blobs);

	// purge the blobs vector
	std::cout << "Removing small blobs...\n";
	std::vector<Blob*> purged_blobs = blobs;
	tools::purgeBlobs(&purged_blobs,BLOB_SIZE_THRESHOLD);

	// create the two images showing the blobs (one shows them all, the other shows only the big ones)
	cv::Mat blobs_image(orig_image.rows, orig_image.cols, CV_8UC3, cv::Scalar(0,0,0));
	tools::blobsPainter(&blobs_image, &blobs);
	cv::Mat big_blobs_image(orig_image.rows, orig_image.cols, CV_8UC3, cv::Scalar(0,0,0));
	tools::blobsPainter(&big_blobs_image, &purged_blobs);

	// compute the centroids
	std::cout << "Computing the centroids...\n";
	std::vector<FloatCouple> centroids;
	tools::getCentroids(&purged_blobs, &centroids);

	// create an image showing the centroids
	cv::Mat centroids_image(orig_image.rows, orig_image.cols, CV_8UC3, cv::Scalar(0,0,0));
	tools::centroidsPainter(&centroids_image, &centroids);

	// display the images
	std::cout << "Displaying the resulting images...\n";

	cv::namedWindow( orig_image_window_name, CV_WINDOW_NORMAL );
	cvMoveWindow(orig_image_window_name,0,0);
	cv::imshow( orig_image_window_name, orig_image );

	cv::namedWindow( blackpixels_window_name, CV_WINDOW_NORMAL );
	cvMoveWindow(blackpixels_window_name,0,y_offset);
	cv::imshow( blackpixels_window_name, *blackpixels );

	cv::namedWindow( blobs_image_window_name, CV_WINDOW_NORMAL );
	cvMoveWindow(blobs_image_window_name,x_offset,y_offset);
	cv::imshow( blobs_image_window_name, blobs_image );

	cv::namedWindow( purged_blobs_image_window_name, CV_WINDOW_NORMAL );
	cvMoveWindow(purged_blobs_image_window_name,2*x_offset,y_offset);
	cv::imshow( purged_blobs_image_window_name, big_blobs_image );

	cv::namedWindow( centroids_image_window_name, CV_WINDOW_NORMAL );
	cvMoveWindow(centroids_image_window_name,0,2*y_offset);
	cv::imshow( centroids_image_window_name, centroids_image );

	std::cout << "Press ANY KEY to quit\n";

	int button = cv::waitKey(0);

	// create the output file
	// set capture directory
	std::string dirname = "created_maps";
	int exists = open((dirname).c_str(), O_RDONLY);
	if(exists==-1){	//	the directory doesn't exist, must create it
		mkdir(dirname.c_str(), O_RDWR|S_IRWXU|S_IRWXG|S_IRWXO);
	}
	std::string mapname = dirname+"/" + basename(argv[1]);
	mapname = mapname.substr(0, mapname.length()-4);
	mapname = mapname+".txt";
	std::cout << "saving map to " << mapname << '\n';

	std::ofstream fout(mapname.c_str(), std::ios_base::trunc);

	if (!fout.is_open()){
		std::cout << "Error in creating the file, aborting...\n";
		exit(1);
	}

	for(unsigned int i = 0; i<centroids.size(); i++){
		fout << centroids[i].x << ' ' << orig_image.rows-centroids[i].y << '\n';
	}

	return 0;
}
