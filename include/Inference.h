#ifndef inference_h
#define inference_h

#include "opencv2/highgui/highgui.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

struct Lanes{
    /*
    * * description: This is an struct that will cover the manipulation of the lanes in the video 
    * Memeber variables:
        * recent_fit (std::vector<cv::Mat>) which will be containing last 5 frames processsed 
        * avg_fit (cv::Mat) : the average of the output of the recent_fit
        * deviation (int) : this value will determine if the frame is considered on or off track 
    */
    std::vector<cv::Mat> recent_fit;
    cv::Mat avg_fit;
    int deviation;

};

class Inference{
    /*
    * This class is the core of the inference of tflite in this project 
    * Methods:
        *  output_video_segment (public) : The method that will be called if semantic segmentaion of the 
        *                                  lanes and space between them is requested by the user 
        * 
        *  artificial_video(public) : The method that will be called if the semantic segmentation is not 
        *                             needed, only will issue the fact that whether if we are on track or not
        * 
        *  getMean(public) : This method will compute the mean of a given std::vector<cv::Mat>
        * 
        *  findMedian(private) : this method will compute the median of a given std::vector<int>
        * 
        *  is_on_track (private) : this method will decide if the processed frame is off/on track 
        * 
        * put_on_frame (private) : will put on/off track on the given (x, y)
    */
    public:
        Inference(bool get_mean);
        cv::Mat output_video_segment(cv::Mat& image, cv::Mat& resized_frame, cv::Mat& frame, bool get_mean);
        cv::Mat artificial_video(cv::Mat& image, cv::Mat& resized_frame, cv::Mat& frame, bool get_mean, int height);
        cv::Mat getMean(const std::vector<cv::Mat>& images);
    private:

        int findMedian(std::vector<int> a,
                  int n);

        int is_on_track(cv::Mat& image);
        void put_on_frame(cv::Mat& frame,bool ontrack, int x, int y);

        std::string mode ; // segment or artificial video 
        bool should_get_mean ; // should we get or should we not get mean 
        Lanes lane; 
};

#endif