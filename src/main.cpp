#include <iostream>
#include <cstdio>
#include <opencv2/videoio.hpp>
#include "opencv2/opencv.hpp"
#include "../include/video_opener.h"
#include <stdlib.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <list>
#include <vector>
#include <chrono>
#include "../include/Inference.h"

/*
* global parameters : case_of_output is the artifical_video and segmentation 
* get_mean : should we get mean of the last 5 frames or not 
*/
std::string case_of_output;
bool get_mean = false;

void GetImageTFLite(float* out, cv::Mat &src, int width, int height)
{
    /*
    * This function will turn a cv::Mat frame from stream and turns it into a float pointer which is suitable to be given to 
    * DL architecture 
    * @param out : the output float pointer of the method 
    * @param src : source frame given function 
    * @param width : the width that will be suited to be entered in DL network
    * @param height : the height that will be suited to be entered in DL network
    */
    int i;
    float f;
    uint8_t *in;
    static cv::Mat image;
    int Len;

    // copy image to input as input tensor
    cv::resize(src, image, cv::Size(width, height),0); // resizing to get a certain width and height 

    in=image.data; // getting the grey values
    Len=image.rows*image.cols*image.channels(); // inputing all the grey values inside a float* to be entered into DL network
    for(i=0;i<Len;i++){
        f     =in[i];
        out[i]=(f);
    }
}

cv::Mat predict(cv::Mat& frame, tflite::Interpreter* interpret, const int width, const int height){
    /*
    * The actual predict function of the project which will invoke the model and get the inference 
    * should be given to the VideoStream::showcamera to be iteratively executed 
    * @param frame : input frame from the video stream 
    * @param interpret : the pointer tflite::Interpreter that will have to be invoked everytime and gotten the result of 
    * @param width, height : the width and heights of the input frame
    */
    cv::Mat resized_frame, resized_float_frame;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC1); // creating a frame to be output of DL network
    GetImageTFLite(interpret->typed_tensor<float>(interpret->inputs()[0]), frame, width, height); // preprocessing the DL arch
    interpret -> Invoke(); // invoking the architecture 
    float* data = interpret->tensor(interpret->outputs()[0])->data.f; // getting the result with same DS with input vector
    for(int i=0;i < height;i++){
        for(int j=0; j < width;j++){
            float* input_pixel = data + (i * width) + (j); // looping to reshape and cast the output result to uchar
            image.at<uchar>(i, j) = (uchar) (abs(*(input_pixel)) * 255.0f);
        }
    }
    Inference inference(true); // calling a new Inference object insisting on getting mean
    if (case_of_output == "segment"){ // user specified output type 
        frame = inference.output_video_segment(image , resized_frame, frame, get_mean);
    }
    else if (case_of_output == "artificial video"){
        frame =  inference.artificial_video(image , resized_frame , frame, get_mean, width);
    }

    return frame; // returning frame to go be displayed
    
};

int main(int argc, char* argv[]) {

     /*-------------------------------------------- LDW :C++ inference-----
        |  Function 
        |
        |  Purpose:  
        |
        |  argyments:
        |   first argument : The model to the .tflite model(binary of the DL model)
            second argument : The path to the video for testing 
            third argument: 1 or 2 ==>
                            1 : artificial video
                            2 : segment 
            fourth argument : cv::waitKey arg

            Example :
                $ ./ldw ../models/unet8 ../Dataset/solidWhiteRight.mp4 1 1
         *-------------------------------------------------------------------*/
    
    std::string path_model , path_video;
    path_model = argv[1];
    path_video = argv[2];
    const char* path_as_array_model = path_model.c_str();
    const char* path_as_array_video = path_video.c_str();
    auto model = tflite::FlatBufferModel::BuildFromFile(path_as_array_model); 
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->SetAllowFp16PrecisionForFp32(true);
    interpreter->SetNumThreads(4);

    
    // Resize input tensors, if desired.
    interpreter->AllocateTensors();
    if(interpreter->AllocateTensors() != kTfLiteOk){
        std::cout << "Failed to allocate tensors\n" << std::endl;
    };
    long arg = std::strtol(argv[3], NULL, 10);
    long wait_key = std::strtol(argv[4], NULL, 10);

    switch(arg){
        case(1):
            case_of_output = "segment";
            break;
        case(2):
            case_of_output = "artificial video";
    }
    // Model_manip tflite_ldw();
    // float* output = interpreter->typed_output_tensor<float>(0);
    VideoStream video(true, interpreter.get(), 160, 80);
    video.showcamera(path_as_array_video, &predict, wait_key);

    return 0;
}