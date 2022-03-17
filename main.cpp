#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

//using namespace cv;

int main(int argc, char** argv )
{
/*
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <image path>\n");
        return -1;
    }
    
    cv::Mat image;
    image = cv::imread( argv[1], 1 );
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", image);
    cv::waitKey(0);
    cv::destroyAllWindows(); //Destroy all opened windows
*/

	// image processing variables
	cv::Mat imgHSV  ;      // HSV convert
	cv::Mat imgLines;      // empty image + tracking lines from colored object
	//cv::Mat imgGr   ;    // grayscale image
	cv::Mat imgdraw ;
	cv::VideoCapture cap(0);

	//Define names of the window
	cv::String win_control = "Control";
	cv::String win_orig = "Original";

	// Create a window with above names
	cv::namedWindow(win_control, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(win_orig, cv::WINDOW_AUTOSIZE);

	if ( cap.isOpened() == false )
	{
		std::cout << "Cannot open the video cam " << std::endl;
		return -1;
	}

    if ( cap.isOpened() == true )
	{
		std::cout << "Video is open " << std::endl;
	}

    // Capture a temporary image from the camera
	cv::Mat imgTmp;
	if ( cap.read(imgTmp) == false ) {
        std::cout << "No frame captured" << std::endl ;
        cv::waitKey() ;
        return -1 ;
    };

	// Create a black image with the size as the camera output
	imgLines = cv::Mat::zeros(imgTmp.size(), CV_8UC3);

	// setup trackbar - used for manual calibration ----------------------------------------
	//colorhsvtrackbar(win_control,iLowH, iHighH, iLowS, iHighS, iLowV, iHighV);

	// start frame -------------------------------------------------------------------------
	unsigned int fcnt = 0; // frame counter: used to send data to arduino at every nth frame
	while ( true ) {

		// get video
		cv::Mat imgOriginal;
		bool bSuccess = cap.read(imgOriginal); // read a new frame from video 

		//Breaking the while loop at the end of the video
		if ( bSuccess == false )
		{
			std::cout << "Video camera is disconnected" << std::endl;
			break;
		}

		// show video with tracking line
		cv::imshow("Original", imgOriginal); //show the original image

		//-------- Send the position to Arduino -----------------------------------------------

		// exit -------------------------------------------------------------------------------
		int comm = cv::waitKey(10);

		fcnt++;

		// exit -------------------------------------------------------------------------------
		if ( comm == 27 ) {
			std::cout << "Esc key is pressed by user. Stoppig the video" << std::endl;
			break;
		}
	}

	cv::destroyAllWindows(); //Destroy all opened windows

    return 0;
}