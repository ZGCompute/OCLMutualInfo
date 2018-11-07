// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// File:   OCLMutualInfo.hpp
// Author: Zachary Greenberg
// Summary: Class OCLMutualInfo provides GPU support for calculating
//          entropy-based mutual information between two input images
//          using the OpenCL and OpenCV API.  

#ifndef OCLMUTUALINFO_HPP_
#define OCLMUTUALINFO_HPP_

#include <iomanip>
#include <iostream>
#include <stdio.h>

using namespace std;

#include "OCLEnv.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#ifdef USE_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#endif  // USE_OPENCL

#define NUM_KERN 3

class OCLMutualInfo
{
public:
    OCLMutualInfo(OCLEnv *ocl);
    ~OCLMutualInfo();

    // Run the mutual-information kernel on detected
    // OpenCL device (i.e., any available GPU )
    // Place the result in host memory using the provideded pointer
    void Run(cv::Mat img1, cv::Mat img2, int imgSize, double entropyData[]);

private:
    OCLEnv *ocl;

    size_t localSize;

#ifdef USE_OPENCL

    // OpenCL kernels to compile 
    cl_kernel clHistKern, clJointHistKern, clEntropyKern;

  
    // Container for all OpenCL kernels to compile
    // imgSize - number of pixels on a side
    // clKern - cl kernel object for the desired kernel 
    struct KernData
    {
        int imgSize;
        cl_kernel clKern;
    };

    // Cooresponding data member for kernels container
    KernData kernels[3];

    // Copy the data from a given cv::Mat into a working
    // buffer of cl_mem, to be passed to our kernel 
    // Arguments:
    // img - input cv::Mat image data to copy to buffer
    // clImg - cl_mem pointer to copy dat to
    // Returns:
    // cl_int - 1 for success, 0 otherwise 
    cl_int CopyImage(cv::Mat &img, cl_mem *clImg);

    // Calls OCLMutualInfo::Prep to compile the desired kernel
    // Arguments:
    // imgSize - the number of pixels on one side of the input image
    // Returns: compiled cl_kernel to run 
    cl_kernel MakeKern(int imgSize);
  
    // Compile an OpenCL kernel from a provided string buffer
    // Arguments:
    // compileStrBuf - string pointer to hold the kernel to compile
    // imgSize - number of pixels on one side of the image
    // Returns:
    // char pointer of the compiled OpenCL kernel
    const char *Prep(char *compileStrBuf, int imgSize);

    // Create an OpenCL 2D image object to write to the
    // image texture memory on the GPU device
    // Arguments:
    // clImg - epmpty pointer to place the created OCL image object in
    // img - OpenCV image data in a 2D cv::Mat
    // Returns:
    // cl_int - 1 for success, 0 otherwise
    cl_int CreateImg(cl_mem *clImg, cv::Mat img);

    // Called by OCLMutualInfo->Run()
    // Method calculates a histogram for a given input image
    // using texture memory on the available GPU.
    // Arguments:
    // clImg -- pointer to a 2D image object in texture memory
    // clHist -- pointer to the resultant histogram data
    //           describing 256 possible gray level values present
    // imgSize -- the number of pixels on a side of the square image
    // clKern -- compiled histogram kernel to launch
    // Returns:
    // cl_int -- 1 if success, 0 otherwise
    cl_int RunHistKern( cl_mem clImg, cl_mem &clHist, int imgSize);   


    // Called by OCLMutualInfo->Run()
    // Method calculates a joint histogram for two input images
    // using texture memory on the available GPU.
    // Arguments:
    // clImg1 -- pointer to a 2D image object in texture memory
    // clImg2 -- pointer to second 2D image object in texture memory
    // clJointHist -- pointer to the resultant histogram data
    //                describing 256 possible gray level values present
    // imgSize -- the number of pixels on a side of the square image
    // clKern -- compiled histogram kernel to launch
    // Returns:
    // cl_int -- 1 if success, 0 otherwise
    cl_int RunJointHistKern(cl_mem clImg1, cl_mem clImg2,
			    cl_mem &clJointHist, int imgSize);

    // Called by OCLMutualInfo->Run()
    // Method calculates entropy for two input images based on
    // individual and joint histograms. Then uses the result to
    // calculate the final mutual-information metric.
    // Arguments:
    // clHist1 -- pointer to the resultant histogram data for img1
    // clHist2 -- pointer to the resultant histogram data for img2
    // clJointHist -- pointer to the resultant joint histogram
    // clEntropy[] -- array holding 3 entropy values for each
    //                histogram, followed by the result of mutual-info 
    // imgSize -- the number of pixels on a side of the square image
    // clKern -- compiled histogram kernel to launch
    // Returns:
    // cl_int -- 1 if success, 0 otherwise
    cl_int RunEntropyKern(cl_mem clHist1, cl_mem clHist2,
			  cl_mem clJointHist, cl_mem &clEntropy,
			  int imgSize);
    
#endif // USE_OPENCL
};
#endif // OCLMutualInfo_HPP_
