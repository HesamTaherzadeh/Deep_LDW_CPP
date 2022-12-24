#include "../include/Inference.h"
#include <vector>
#include <cmath>

Inference::Inference(bool get_mean){
    /*
    * constructor of Inference
    * *@param get_mean-name (boolean value : should it get mean of the last 5 frames or not)
    */
    should_get_mean = get_mean; 
    lane = Lanes();
};

cv::Mat Inference::getMean(const std::vector<cv::Mat>& images)
{
    /*
    * This method will compute the mean of a given std::vector<cv::Mat>
    * *@param images (std::vector of images to be gotten mean of)
    * returns : a cv::Mat that is the result of taking the mean of vector of images
    */
    // Created a 0 initialized image to use as accumulator
    cv::Mat temp = cv::Mat::zeros(images[0].rows, images[0].cols, CV_64FC1);

    // Use a temp image to hold the conversion of each input image to CV_64FC3
    // This will be allocated just the first time, since all images have
    // the same size.
    for (int i = 0; i < images.size(); ++i)
    {
        cv::accumulate(images[i], temp);
    }

    // Convert back to CV_8UC3 type, applying the division to get the actual mean
    temp.convertTo(temp, CV_8UC1, 1. / images.size());
    return temp;
};

cv::Mat Inference::output_video_segment(cv::Mat& image, cv::Mat& resized_frame, cv::Mat& frame, bool get_mean){
     /*
    * The method that will be called if semantic segmentaion of the 
    * lanes and space between them is requested by the user
    * 
    * *@param image: the image that will be processed(output of deep architecture)
    * *@param resized_frame: the resized frame of the last image 
    * *@param get_mean : boolean to make sure if we have to get mean or simply output the for each frame
    * returns : a cv::Mat that is the result of taking the mean of vector of images
    */
    
    if (get_mean){ // if the get_mean is true, we would get the average of consisting frames in lane.recent_fit
        lane.recent_fit.push_back(image);
        if(lane.recent_fit.size() > 5){
            lane.recent_fit.erase(lane.recent_fit.begin());
        }
        lane.avg_fit = getMean(lane.recent_fit);
    }
    else{
        lane.avg_fit = image;
    }

    int median = is_on_track(image); // the median will be gotten 
    bool ontrack {abs(median - image.cols/2) < 10}; // !THIS IS A HYPERPARAMETER: TO BE AUTOMATED (deciding if the frame is on/off track)
    // ! SET DEFAULT WITH RESPECT TO DEEP LEARNING ARCHITECTURE
    cv::Mat b = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1); // blue channel 
    cv::Mat g = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1); // green channel 

    // ! THIS ALSO IS DEPENDENT TO FRAME SIZE OF OUTPUT FROM cv::Point(80, 80) TO cv::Point(median, 50) : should be automatized
    cv::line(g, cv::Point(80, 80), cv::Point(median, 50), 255); // the line will be drawn on green 

    std::vector<cv::Mat> channels; // the vectors of the images
    cv::Mat final_image; 
    channels.push_back(b);
    channels.push_back(g);
    channels.push_back(lane.avg_fit); // this will produce a semantic segmentation with red mask on image 
    cv::merge(channels, final_image); // merging the channels
    final_image.convertTo(final_image, CV_8UC3);// converting to BGR uint8
    cv::resize(final_image, resized_frame, cv::Size(frame.cols, frame.rows),0); // resizing the output
    cv::addWeighted(frame, 0.5, resized_frame, 1, 0.0, frame); // adding weighted the (0.5)mask and (1)initial image
    this->put_on_frame(frame, ontrack, frame.cols - frame.cols/2.5, frame.rows/10); // putting the on or off track on frame 
    return frame;
};


cv::Mat Inference::artificial_video(cv::Mat& image, cv::Mat& resized_frame, cv::Mat& frame, bool get_mean, int height){
    /*
    * The method that will be called if the semantic segmentation is not 
    * needed, only will issue the fact that whether if we are on track or not
    * *@param image: the image that will be processed(output of deep architecture)
    * *@param resized_frame: the resized frame of the last image 
    * *@param get_mean : boolean to make sure if we have to get mean or simply output the for each frame
    * *height(int) : the automatized height of the last function
    * returns : a cv::Mat of the artifical image written on/off track giving the status of each frame
    */

    int median = is_on_track(image); // the median will be gotten 
    bool ontrack {abs(median - height/2) < 10}; // !THIS IS A HYPERPARAMETER: TO BE AUTOMATED (deciding if the frame is on/off track)
    // ! SET DEFAULT WITH RESPECT TO DEEP LEARNING ARCHITECTURE

    cv::Mat g = cv::Mat::zeros(cv::Size(480, 480), CV_8UC1); // one channel that will produce a 480,480 frame that will simply say on/off track
    put_on_frame(g, ontrack, g.cols/4, g.rows/2); //putting on frame the result
    return g;
}

int Inference::is_on_track(cv::Mat& image){
    /*
    * This method will fine the non zero coordinate of the segmented output of deep learning architecture
    * and find the median of their x components, which will be the x coordinate of the middle of the between the lanes 
    * this will be essential to finding if the frame status is off/on track
    * *@param image : the processing frame 
    * returns : the median x component 
    */
    cv::Mat nonZeroCoordinates;
    cv::findNonZero(image, nonZeroCoordinates);
    size_t count {nonZeroCoordinates.total()};
    std::vector<int> y_comps;
    for (int i = 0; i <count ; i++ ) {
        y_comps.push_back(nonZeroCoordinates.at<cv::Point>(i).x );
    }
    int median = findMedian(y_comps, count); // finding the median of a vector of ints
    return median;
    
}

void Inference::put_on_frame(cv::Mat& frame, bool ontrack, int x, int y){
    /*
    * this will put the status of being on/off track on a given frame 
    * @param ontrack whether if the status is on/off track
    * @param x : x components of the text 
    * @param y : y components of the text 
    */
    std::string is_on_track; 
    if (ontrack){

            is_on_track = "ON TRACK";
            }
    else{
            is_on_track = "OFF TRACK";
    };

    cv::putText(frame, 
                is_on_track ,
                cv::Point(x, y), 
                cv::FONT_HERSHEY_DUPLEX,
                1.0,
                CV_RGB(0, 0, 255), 
                2);

}
int Inference::findMedian(std::vector<int> a,
                  int n)
{
    /*
    * finding the mean of a given std::vector
    * *@param a : the vector 
    * *@param n : count of vector members
    * returns : an integer (median) of the x-components of segmentation output
    */
  
    // If size of the arr[] is even
    if (n % 2 == 0) {
  
        // Applying nth_element
        // on n/2th index
        nth_element(a.begin(),
                    a.begin() + n / 2,
                    a.end());
  
        // Applying nth_element
        // on (n-1)/2 th index
        nth_element(a.begin(),
                    a.begin() + (n - 1) / 2,
                    a.end());
  
        // Find the average of value at
        // index N/2 and (N-1)/2
        return (double)(a[(n - 1) / 2]
                        + a[n / 2])
               / 2.0;
    }
  
    // If size of the arr[] is odd
    else {
  
        // Applying nth_element
        // on n/2
        nth_element(a.begin(),
                    a.begin() + n / 2,
                    a.end());
  
        // Value at index (N/2)th
        // is the median
        return (double)a[n / 2];
    }
}