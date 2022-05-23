#include <stdio.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
//#include <opencv2/xobjdetect.hpp>
#include "opencv2/ml.hpp"
#include "opencv2/videoio.hpp"

///home/woodrat/opencv/opencv/modules/objdetect/src/hog.cpp

void ResizeBox(cv::Rect& box){
    box.x += cvRound(box.width*0.1) ;
    box.width = cvRound(box.width*0.8) ;
    box.y += cvRound(box.height*0.06) ;
    box.height = cvRound(box.height*0.8) ;
}

std::vector< float > get_svm_detector( const cv::Ptr< cv::ml::SVM >& svm )
{
    // get the support vectors
    cv::Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    cv::Mat alpha, svidx;
    double rho = svm->getDecisionFunction( 0, alpha, svidx );
    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    std::vector< float > hog_detector( sv.cols + 1 );
    memcpy( &hog_detector[0], sv.ptr(), sv.cols*sizeof( hog_detector[0] ) );
    hog_detector[sv.cols] = (float)-rho;
    return hog_detector;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
// reads all images from dirname to vector img_list
void load_images( const cv::String & dirname, std::vector< cv::Mat > & img_lst, bool showImages = false )
{
    std::vector< cv::String > files;
    cv::glob( dirname, files );

    for ( size_t i = 0; i < files.size(); ++i )
    {
        cv::Mat img = cv::imread( files[i] ); // load the image
        //std::clog << "check " << files[i] << std::endl ; CV_Assert(img.cols==64 || img.rows==128) ;
        if ( img.empty() )
        {
            std::cout << files[i] << " is invalid!" << std::endl; // invalid image, skip it.
            continue;
        }
        if ( showImages )
        {
            cv::imshow( "image", img );
            cv::waitKey( 1 );
        }
        img_lst.push_back( img );
    }
};

// image greater than detector then take sample from image
// otherwise take full image
// sample_neg( full_neg_lst, neg_lst, pos_image_size ); // [64x128]
void sample_neg( const std::vector< cv::Mat > & full_neg_lst, std::vector< cv::Mat > & neg_lst, const cv::Size & size )
{
    cv::Rect box;
    box.width = size.width; //64 320
    box.height = size.height; //128 240
    srand( (unsigned int)time( NULL ) );
    for ( size_t i = 0; i < full_neg_lst.size(); i++ )
        if ( full_neg_lst[i].cols > box.width && full_neg_lst[i].rows > box.height )
        {
            box.x = rand() % ( full_neg_lst[i].cols - box.width );//320-64=[0,256]
            box.y = rand() % ( full_neg_lst[i].rows - box.height );//240-128=[0,112]
            cv::Mat roi = full_neg_lst[i]( box );
            neg_lst.push_back( roi.clone() );
        }
        // save image with same size as detector
        else{
            neg_lst.push_back( full_neg_lst[i].clone() );
        }
        
}

//computeHOGs( pos_image_size, pos_lst, gradient_lst, false );
//computeHOGs( pos_image_size, neg_lst, gradient_lst, false );
void computeHOGs( const cv::Size wsize, const std::vector< cv::Mat > & img_lst, std::vector< cv::Mat > & gradient_lst, bool use_flip )
{
    cv::HOGDescriptor hog;
    hog.winSize = wsize; //64x128
    cv::Mat gray;
    std::vector< float > descriptors;
    for( size_t i = 0 ; i < img_lst.size(); i++ ) {
        if ( img_lst[i].cols >= wsize.width && img_lst[i].rows >= wsize.height ){
            // crop from middle of image
            cv::Rect r ;
            r = cv::Rect(( img_lst[i].cols - wsize.width ) / 2,
                        ( img_lst[i].rows - wsize.height ) / 2,
                        wsize.width, wsize.height);
            gray = img_lst[i](r);
        }

        //cv::cvtColor( img_lst[i](r), gray, cv::COLOR_BGR2GRAY );
        gray = img_lst[i];
            
        hog.compute( gray, descriptors, cv::Size( 8, 8 ), cv::Size( 0, 0 ) );
        gradient_lst.push_back( cv::Mat( descriptors ).clone() );

        if ( use_flip )
        {
            flip( gray, gray, 1 );
            hog.compute( gray, descriptors, cv::Size( 8, 8 ), cv::Size( 0, 0 ) );
            gradient_lst.push_back( cv::Mat( descriptors ).clone() );
        }
    }
}

//convert_to_ml( gradient_lst, train_data );
void convert_to_ml( const std::vector< cv::Mat > & train_samples, cv::Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    cv::Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = cv::Mat( rows, cols, CV_32FC1 );
    for( size_t i = 0 ; i < train_samples.size(); ++i )
    {
        CV_Assert( train_samples[i].cols == 1 || train_samples[i].rows == 1 );
        if( train_samples[i].cols == 1 )
        {
            transpose( train_samples[i], tmp ); 
            tmp.copyTo( trainData.row( (int)i ) ); 
        }
        else if( train_samples[i].rows == 1 )
        {
            train_samples[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

int main(int argc, char** argv )
{
	cv::String pos_dir = "/home/woodrat/projects/szeuni/computervision/peoplecounter/hogpedestrians/pos_resized_20220512/";
    cv::String neg_dir = "/home/woodrat/projects/szeuni/computervision/peoplecounter/hogpedestrians/neg_resized_20220522/";
    
    // testing separate code
    //cv::String test_dir = "/home/woodrat/projects/szeuni/computervision/peoplecounter/hogpedestrians/test/";

	cv::String obj_det_filename = "../detectionoutput.yml";
    int detector_width =64;//= 64 320 ; // parser.get< int >( "dw" );
    int detector_height =128;//= 128 240 ; // parser.get< int >( "dh" );
	
    std::vector< cv::Mat > pos_lst, full_neg_lst, neg_lst, gradient_lst;
    std::vector< int > labels;

    // ----------------------------------------- positive image processing START
	std::clog << "Positive images are being loaded..." ;
    load_images( pos_dir, pos_lst, false );
    if ( pos_lst.size() > 0 )
    {
        std::clog << "...[done] " << pos_lst.size() << " files." << std::endl;
    }
    else
    {
        std::clog << "no image in " << pos_dir << std::endl;
        return 1;
    }

    cv::Size pos_image_size ; 

    if ( detector_width && detector_height )
    {
        pos_image_size = cv::Size( detector_width, detector_height );
    }
    else
    {
        for ( size_t i = 0; i < pos_lst.size(); ++i )
        {
            if( pos_lst[i].size() != pos_image_size )
            {
                std::cout << "All positive images should be same size!" << std::endl;
                exit( 1 );
            }
        }
        pos_image_size = pos_image_size / 8 * 8;
    }

    std::cout << "pos_image_size: " << pos_image_size << std::endl ;
    // ----------------------------------------- positive image processing DONE

    // ----------------------------------------- negative image processing START
    std::clog << "Negative images are being loaded...";
    load_images( neg_dir, full_neg_lst, false );
    std::clog << "...[done] " << full_neg_lst.size() << " files." << std::endl;

    std::clog << "Negative images are being processed...";
    sample_neg( full_neg_lst, neg_lst, pos_image_size ); // [64x128]
    std::clog << "...[done] " << neg_lst.size() << " files." << std::endl;
    // ----------------------------------------- negative image processing DONE

    // ----------------------------------------- positive image HOG START
    std::clog << "Histogram of Gradients are being calculated for positive images...";
    computeHOGs( pos_image_size, pos_lst, gradient_lst, false );
    size_t positive_count = gradient_lst.size();
    labels.assign( positive_count, +1 );
    std::clog << "...[done] ( positive images count : " << positive_count << " )" << std::endl;
    // ----------------------------------------- positive image HOG DONE

    // ----------------------------------------- negative image HOG START
    std::clog << "Histogram of Gradients are being calculated for negative images...";
    computeHOGs( pos_image_size, neg_lst, gradient_lst, false );
    size_t negative_count = gradient_lst.size() - positive_count;
    labels.insert( labels.end(), negative_count, -1 );
    CV_Assert( positive_count < labels.size() ); // returns error if expression is false
    std::clog << "...[done] ( negative images count : " << negative_count << " )" << std::endl;
    // ----------------------------------------- negative image HOG DONE
    
    // ----------------------------------------- svm model train START
	cv::Mat train_data;
    std::clog << "convert_to_ml" << std::endl ;
    convert_to_ml( gradient_lst, train_data );
    std::clog << "Training SVM...";
    cv::Ptr< cv::ml::SVM > svm = cv::ml::SVM::create();
    /* Default values to train SVM */
    svm->setCoef0( 0.0 ); // Parameter coef0 of a kernel function. For SVM::POLY or SVM::SIGMOID. Default value is 0. 
    svm->setDegree( 3 ); // Parameter degree of a kernel function. For SVM::POLY. Default value is 0. 

    // Termination criteria of the iterative SVM training procedure which solves a partial 
    // case of constrained quadratic optimization problem. You can specify tolerance and/or 
    // the maximum number of iterations. Default value is TermCriteria( TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, FLT_EPSILON );
    svm->setTermCriteria( cv::TermCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 1000, 1e-3 ) );
    
    // Parameter γ of a kernel function. For SVM::POLY, SVM::RBF, SVM::SIGMOID or SVM::CHI2. Default value is 1.
    svm->setGamma( 0 );

    // Initialize with one of predefined kernels. See SVM::KernelTypes. 
    // https://docs.opencv.org/3.4/d1/d2d/classcv_1_1ml_1_1SVM.html#aad7f1aaccced3c33bb256640910a0e56
    svm->setKernel( cv::ml::SVM::LINEAR );

    // Parameter ν of a SVM optimization problem. For SVM::NU_SVC, SVM::ONE_CLASS or SVM::NU_SVR. Default value is 0.
    svm->setNu( 0.5 );

    // Parameter ϵ of a SVM optimization problem. For SVM::EPS_SVR. Default value is 0. 
    svm->setP( 0.1 ); // for EPSILON_SVR, epsilon in loss function?

    // Parameter C of a SVM optimization problem. For SVM::C_SVC, SVM::EPS_SVR or SVM::NU_SVR. Default value is 0.
    svm->setC( 0.01 ); // From paper, soft classifier

    // Type of a SVM formulation. See SVM::Types. Default value is SVM::C_SVC. 
    svm->setType( cv::ml::SVM::EPS_SVR ); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task

    // Trains the statistical model.
    svm->train( train_data, cv::ml::ROW_SAMPLE, labels );
    std::clog << "...[done]" << std::endl;
    // ----------------------------------------- svm model train DONE

    cv::HOGDescriptor hog;
    hog.winSize = pos_image_size;
    hog.setSVMDetector( get_svm_detector( svm ) );
    hog.save( obj_det_filename );

    return 0;
}