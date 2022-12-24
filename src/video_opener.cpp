#include "../include/video_opener.h"
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <ctime>


VideoStream::VideoStream(bool isvideo, tflite::Interpreter* interpreter_arg, int width, int height){
    is_it_video = isvideo;
    interpreter = std::move(interpreter_arg);
    this->width = width;
    this->height = height;
};

// void VideoStream::object_setter(cv::Mat (*function_pointer)(void*, cv::Mat), void* object, cv::Mat* frame, cv::Mat new_frame){
//     new_frame = function_pointer(object, *frame);
// }

void VideoStream::showcamera(const char* address, cv::Mat (*func)(cv::Mat& frame, tflite::Interpreter* interpreter_arg, int width, int height), int wait_key_no)
{
    cv::VideoCapture captured_video(address);
    captured_video.set(cv::CAP_PROP_BUFFERSIZE, 3);
    captured_video.set(cv::CAP_PROP_FPS, 60);
    clock_t prev_frame_time, compute_time;
    if(!captured_video.isOpened())
    {
        std::cout << "The stream either has ended or faced a problem" << std::endl;
    }

    while (1)
    {      
        cv::Mat frame;
        captured_video >> frame;
        if(frame.empty())
            break;
        cv::Mat frame_, frame_2;
        cv::resize(frame, frame_, cv::Size(), 0.5, 0.5);
        prev_frame_time = clock();
        // object_setter(func, this, &frame_, frame_2)
        frame = func(frame_, interpreter, width, height);
        compute_time = clock() - prev_frame_time;
        int fps = CLOCKS_PER_SEC / compute_time;
        cv::putText(frame, 
            "FPS : " + std::to_string(fps),
            cv::Point(10, frame.rows / 15), 
            cv::FONT_HERSHEY_DUPLEX,
            1.0,
            CV_RGB(0, 0, 255), //font color
            2);    
        cv::imshow("Frame", frame);
        char c=(char)cv::waitKey(wait_key_no);
        if(c==27)
            break;   
    };
    captured_video.release();
    cv::destroyAllWindows();
};