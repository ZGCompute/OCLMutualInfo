// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// File:    MutualInfo.cpp
// Author:  Zachary Greenberg
// Summary: Module MutualInfo controls calculating
//          entropy-based mutual information between two input images
//          using the OpenCL and OpenCV API.

#include <iostream>
#include <iomanip>
#include <stdio.h>

#include "../headers/MutualInfo.hpp"

using namespace std;

int main(int argc, char** argv){

  // Welcome
  cout << "Welcome to Mutual-information!" << endl;

  // Enclose so m is destroyed (&clean up) before exit
  {
    // Initialize Mutuale Info
    MutualInfo m(argv);

    // Run the full mutual-information stack
    m.Execute();
  }

  return 0;

}

MutualInfo::MutualInfo(char** argv)
{

  // Initialize options for files to read in 
  this->opt_img1_fname = new Option();
  this->opt_img1_fname->answer = argv[1];
  this->opt_img2_fname = new Option();
  this->opt_img2_fname->answer = argv[2];

  // Initialize options for OpenCL
  this->opt_opencl_dev = new Option();
  this->opt_opencl_dev->answer = argv[3];
  this->opt_opencl_dev->verbose = 0;
  
  // Create the OpenCL environment
  ocl = new OCLEnv();
  cout << "::: Creating OCLEnv :::" << endl;
  cout << "::: Suggested OpenCL device # " << this->opt_opencl_dev->answer << " ::: " << endl;
  ocl->opt_opencl_dev = new Option();
  ocl->opt_opencl_dev->answer = this->opt_opencl_dev->answer;
  ocl->opt_opencl_dev->verbose = this->opt_opencl_dev->verbose;

}

MutualInfo::~MutualInfo()
{

  delete ocl;
  delete mutualInfo;
  delete oclMIM;
  delete opt_opencl_dev;
  delete opt_img1_fname;
  delete opt_img2_fname;

}

void MutualInfo::Execute()
{

  // Initialize based on user defined options
  cout << "::: Initializing OpenCL from options :::" << endl;
  ocl->InitFromOptions();
  cout << "::: Finished initializing OpenCL environment :::" << endl;

  // Initialize OCLMutualInfo stack
  oclMIM = new OCLMutualInfo( ocl );
  cout << "::: Finished initializing OCL MIM environment :::" << endl;

  // Initialize a pointer for the result
  mutualInfo = new double[4];
  
  // Load in the two images with OpenCV
  cv::Mat img1 = cv::imread(this->opt_img1_fname->answer, CV_LOAD_IMAGE_GRAYSCALE);
  cv::Mat img2 = cv::imread(this->opt_img2_fname->answer, CV_LOAD_IMAGE_GRAYSCALE);
  cout << "::: Finished reading image data into memory :::" << endl;
  
  // Get the image size
  assert( img1.rows == img1.cols );
  assert( img1.rows == img2.rows );
  const int imageSize = img1.rows;
  
  // Call the OpenCL stack
  cout << "::: Entering OpenCL MIM! :::" << endl;
  oclMIM->Run( img1, img2, imageSize, mutualInfo );

  // Display the result
  cout << "::: The mutual-information for image1 and image2 is: " << setprecision(15) << mutualInfo[3] << " ::: " << endl;
  
}
