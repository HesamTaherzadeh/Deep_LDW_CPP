- # Deep lane departue warning c++ inference

  A Lane departure warning system is designed to notify the driver the stray of the vehicle from its normal path, this way, crashes due to drifting from lines will be minimized.This project has been written in C++ and python and also has been tested in Jetson Nano 2GB and Raspberrty pi 3b and has 3 different outputs regarding the warning.Training process has been done on a Google colab server with gpu support and then the model is exported as .h5 file. This file is further fed to a python script to be transformed to a .tflite binary file and this discretized file will be inferenced in this project
  
- # Perquisites 
  - gcc & g++ (9.4.0)
  - Opencv tested both on (4.5.5 / 4.6.1)
  - Tensorflow lite (2.6.0 / built from source)
  - Flatbuffers (downloaded and built from Tensorflow lite download directory)
  - Raspberry pi or Jetson embedded boards
  - Eigen 3.3

- # Outputs 
  This project will provide 3 different outputs:
  * **Segmented initial video stream**:
    This mode will process the input stream, segment the area between the lanes and will provide a line from  bottom middle of the stream which will be representing the middle of the stream, and the deviation of this line will determine the status of the vehicle (ON TRACK / OFF TRACK)
    
      ![image](https://user-images.githubusercontent.com/89359094/209444036-e7ec2959-890a-4118-8e22-4ac79f7f7bc0.png)
  * **Artificial video**:
    This method will as well return an image, but an image only outputing the status of the vehicle since this will eliminate the post process part, the algotithm will be relatively faster
    
    ![image](https://user-images.githubusercontent.com/89359094/209444264-cadce2ce-beea-4c11-888e-d9bed38cc045.png)
    
   * **Terminal output**:
     This mode will only output the state to the terminal and two messages will be printed as below:
      - Warning you are departing your path 
      - The vehicle is in its right track
    

- # benchmarks
    This model has been benchmarked on two different embedded ARM cpus( none of the dependencies were gpu-accerlated )
    | CPU | A57(Jetson nano)  | A53(Raspberry pi 3b) |
    | ------- | --- | --- |
    | FPS | 8-9 | 3-4 |


- # Profiling each functions (on Jetson nano)
    To find the bottlenecks of this algorithms, all of the actions were profiled and results are mentioned below in ms
    |    			Function 		   	|       			Time 			elapsed(ms)  			 		      	|
    |:---:	|:---:	|
    |    			Preprocess_for_tflite 		   	|    			1.1 		   	|
    |      			Invoking 			the interpretor 		    	|    			128 		   	|
    |       			For 			loop to convert back float to uchar  			 		      	|    			0.4 		   	|
    |       			Post 			processing  			 		      	|    			4.3 		   	|
    |      			Average 			n images 		    	|    			0.3 		   	|
    |      			Resizing 			back the image to initial frame size 		    	|    			1.4 		   	|
    |       			Convert 			to 8 bit ucahr  			 		      	|    			0.01 		   	|
    |      			Adding 			weighted 		    	|    			1.6 		   	|
    |       			Whole 			predict  			 		      	|    			133 		   	|
 - # Deep learning model 
   This model has been written and trained using Keras, and then further discretized to use only float16 to be accerlated using Tensorflow lite.
   few models has been tested and also benchmarked :

  ![image](https://user-images.githubusercontent.com/89359094/209445194-28fc7f25-70b2-4c15-9bed-d6270c6e32a2.png)

   

<p align="right">(<a href="#readme-top">back to top</a>)</p>

We can determine the target FPS for anything ADAS related, assuming the simplest form of velocity distance relationship, having the distance that the algorithm needs to be updated by the formula and table below.

![image](https://user-images.githubusercontent.com/89359094/209442998-5a18fb12-b989-448f-8789-fd316ebd528c.png)

![image](https://user-images.githubusercontent.com/89359094/209443130-df26cd00-5dc8-4fe8-98af-aeab715f80c2.png)
