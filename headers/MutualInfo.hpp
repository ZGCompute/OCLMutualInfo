// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// File: MutualInfo.hpp
// Author:  Zachary Greenberg
// Summary: This module controls mutual-information computations run to compare two image

#ifndef MUTUALINFO_HPP_
#define MUTUALINFO_HPP_

#include "OCLEnv.hpp"
#include "OCLMutualInfo.hpp"

#include <iomanip>
#include <iostream>
#include <stdio.h>

// This module controls mutual information
// computations run to compare two images
class MutualInfo
{
public:
  // Construct the MutualInfo module that consists of
  // internal vars, OpenCL environment, and stack.
  //
  // @see OCLEnv
  // @see OCLMutualInfo
  // @see Timer
  MutualInfo(char** argv);
  virtual ~MutualInfo();

#ifdef USE_OPENCL
  // The option for specifying the OpenCL device to use
  Option *opt_opencl_dev;
#endif //USE_OPENCL

  // Option for file names to read in
  Option *opt_img1_fname;
  Option *opt_img2_fname;
  
  // Internal var for end result. The first 3 elements
  // Store the result of entropy calculated for img1, img2,
  // and joint entropy respectively.
  // The 4th and final value is mutual-information
  double* mutualInfo;
  
  // Run the stack to do all required processing
  void Execute();
  
private:
#ifdef USE_OPENCL
  // OpenCL environment
  OCLEnv *ocl;

  // Mutual information stack
  OCLMutualInfo *oclMIM;
#endif //USE_OPENCL
  
};

#endif // MUTUALINFO_HPP
