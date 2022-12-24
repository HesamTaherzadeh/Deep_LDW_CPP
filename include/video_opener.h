#ifndef video_h
#define video_h

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string>
#include <opencv2/videoio.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

class VideoStream
{
    public:
        VideoStream() = default;
        VideoStream(bool isvideo, tflite::Interpreter* interpreter_arg, int width, int height);
        void showcamera(const char* address, cv::Mat (*func)(cv::Mat& frame, tflite::Interpreter* interpreter_arg, int width, int height), int wait_key_no);
    private:
        bool is_it_video {false};
        int height {};
        int width {};
        tflite::Interpreter* interpreter;
};


#endif