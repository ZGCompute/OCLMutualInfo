// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// File:    OCLMutualInfo.cpp
// Author:  Zachary Greenberg
// Summary: Class OCLMutualInfo provides GPU support for calculating
//          entropy-based mutual information between two input images
//          using the OpenCL and OpenCV API.

#include "../headers/OCLMutualInfo.hpp"

OCLMutualInfo::OCLMutualInfo(OCLEnv *ocl)
{
    this->ocl = ocl;

    localSize = 64;

#ifdef USE_OPENCL
    for (int i = 0; i < NUM_KERN; i++)
        kernels[i].clKern = NULL;
#endif // USE_OPENCL
}

OCLMutualInfo::~OCLMutualInfo()
{
#ifdef USE_OPENCL
    for (int i = 0; i < NUM_KERN; i++)
        if (kernels[i].clKern)
            clReleaseKernel(kernels[i].clKern);
#endif // USE_OPENCL
}

#ifdef USE_OPENCL

const char *OCLMutualInfo::Prep(char *compileStrBuf, int imgSize)
{
    int locChipSize = exp2(ceil(log2(imgSize*imgSize)));

    // Compile some strings that you need for JIT kernel compilation.
    snprintf(compileStrBuf, 1024,
        "-Werror -cl-fast-relaxed-math "
        "-D LOC_SIZE=%d -D imgSize=%d",
        (int)locChipSize, imgSize);
        
        
    return
    
"#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable\n"

"const sampler_t samp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;\n"

"// Kernel function prototypes "
"__kernel void histogram( unsigned int* d_bins,\n"
"                         __read_only image2d_t img,\n"
"			  const int bin_count,\n"
"			  const int imgSize );\n"
      
"__kernel void joint_histogram( unsigned int* d_bins,"
"                               __read_only image2d_t img1,\n"
"                               __read_only image2d_t img2,\n"      
"			        const int bin_count, \n"
"			        const int imgSize );\n"

"__kernel void calculate_entropy( unsigned int* hist_1,\n"
"                                 unsigned int* hist_2,\n"
"                                 unsigned int* h_joint,\n"
"                                 double* entropy,\n"
"			          const int imgSize );\n"
      
"// Kernel function implementations\n "            
"__kernel void histogram( unsigned int* d_bins,\n"
"                         __read_only image2d_t img,\n"
"			  const int bin_count,\n"
"			  const int imgSize ) {\n"

" // calculate thread id to work with\n"
" int tid = (int)get_local_id(0);\n"
" int local_size = get_local_size(0);\n"
" int wgid = get_group_id(0);\n"

" // Linear address of local thread\n"
" int gid = tid + local_size * wgid;\n"

" // Position of pixel data in texture memory \n"    
" int2 pos = {gid % imgSize, gid / imgSize};\n"
      
" // Check for out of bounds\n"
" if ( pos.x >= imgSize || pos.y >= imgSize ) {\n"
"   return;\n"
" }"

" // Calculate bin\n"
" int4 val = read_imagei( img, samp, (int2)(pos.x, pos.y))\n"      
" int bin = val[0];\n"

" // increment the histogram for the value\n"
" atom_add( &d_bins[bin], 1);\n"

"}\n"

"__kernel void joint_histogram( unsigned int* d_bins,\n"
"                               __read_only image2d_t img1,\n"
"                               __read_only image2d_t img2,\n"      
"			        const int bin_count, \n"
"			        const int imgSize ) {\n"

" // calculate thread id to work with\n"
" int tid = (int)get_local_id(0);\n"
" int local_size = get_local_size(0);\n"
" int wgid = get_group_id(0);\n"
" int gid = tid + wgid * local_size;\n"    

" int2 pos = {gid % imgSize, gid / imgSize};\n"
      
" // Check for out of bounds\n"
" if ( pos.x >= imgSize || pos.y >= imgSize ) {\n"
"   return;\n"
" }\n"

" // Fetch value from texture memory and calculate bin\n"
" int4 val1 = read_imagei( img1, samp, (int2)(pos.x, pos.y));\n"
" int4 val2 = read_imagei( img2, samp, (int2)(pos.x, pos.y));\n"    

" int bin1 = val1[0];\n"
" int bin2 = val2[0];\n"
    
" // increment the histogram for the value\n"
" atomic_add( &d_bins[bin1], 1);\n"
" atomic_add( &d_bins[bin2], 1);\n"
				
"}\n"

"__kernel void calculate_entropy( unsigned int* hist_1,\n"
"                                 unsigned int* hist_2,\n"
"                                 unsigned int* h_joint,\n"
"                                 double* entropy,\n"
"			          const int imgSize ) {\n"

" // calculate thread id to work with\n"
" int tid = (int)get_local_id(0);\n"
" int local_size = get_local_size(0);\n"
" int wgid = get_group_id(0);\n"
" int gid = tid + wgid * local_size;\n"
    
"  // Shannon entropy\n"
"  // If it's all black, abort\n"
"  int total = imgSize*imgSize;\n"
"  if (hist_1[0] == total || hist_2[0] == total)\n"
"    return;\n"

"  float invTotal = 1.0/(float)total;\n"
"  if (gid < (256)) {\n"

"    // Probability\n"
"    float prob1 = hist_1[gid]*invTotal;\n"    
"    float prob2 = hist_2[gid]*invTotal;\n"
"    float joint_prob = h_joint[gid]*invTotal;\n"    

"    // Make sure all threads finish calculating prob\n"
"    // Equivelent to cudaSyncThreads()\n"
"    barrier(CLK_LOCAL_MEM_FENCE);\n"				
				
"    // Store Entropy for 1st, 2nd, and\n"
"    // joint histogram in entropy[0-2]\n"
"    if (prob1 > 0.0000001)\n"
"      atom_add(&entropy[0], (-prob1*log2(prob1)));\n"
"    if (prob2 > 0.0000001)\n"
"      atom_add(&entropy[1], (-prob2*log2(prob2)));\n"
"    if (join_prob > 0.0000001)\n"
"      atom_add(&entropy[2], (-joint_prob*log2(joint_prob)));\n"
    
"    barrier(CLK_LOCAL_MEM_FENCE);\n"   
        
"  }\n"
    
"  // Store the final result of the \n"
"  // mutual-info calc in the last element:\n"
"  // -- entropy[3]\n"    
"  entropy[3] = (entropy[0] + entropy[1] - entropy[2]) / sqrt(entropy[0] * entropy[1]):\n" 

"}";
				
}
      
cl_int OCLMutualInfo::CopyImage(cv::Mat &img, cl_mem *clImg)
{
    cl_int err = CL_SUCCESS;
    int rows = img.rows, cols = img.cols;

    // Define the format that we're using for the image
    cl_image_format imgFmt;
    imgFmt.image_channel_data_type = CL_SIGNED_INT16;
    imgFmt.image_channel_order = CL_R;

    cl_image_desc imgDesc;
    imgDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
    imgDesc.image_width = cols;
    imgDesc.image_height = rows;
    imgDesc.image_row_pitch = 0;
    imgDesc.image_slice_pitch = 0;
    imgDesc.num_mip_levels = 0;
    imgDesc.num_samples = 0;
    imgDesc.buffer = NULL;

    // Create the 2D image
    (*clImg) = clCreateImage(
        ocl->Context(), CL_MEM_READ_ONLY, &imgFmt, &imgDesc, NULL, &err);
    HandleErr(err);

    const size_t origin[3] = {0,0,0};
    const size_t region[3] = {cols, rows, 1};

    // Create working buffer
    size_t dataSz = rows * cols * sizeof(unsigned char);
    cl_mem clImgData;
    unsigned char *imgData;
    ocl->AllocPinnedMem(&clImgData, (void **)&imgData, dataSz);

    // Copy image data to pinned mem & remove `-9999`
    // values that screw up image calculations.
    unsigned char *ptr = img.ptr<unsigned char>();
    int step = (int)(img.step / img.elemSize());
    for (int y = 0; y < rows; y++)
    {
        for (int x = 0; x < cols; x++)
        {
            int v = ptr[x];
            imgData[y*cols+x] = (v < 0) ? 0 : v;
        }
        ptr += step;
    }

    // Copy data to image texture on the device
    cl_event ev;
    err = clEnqueueWriteImage(ocl->Queue(), (*clImg), CL_TRUE, origin, region,
                              sizeof(unsigned char) * cols, 0, imgData, 0, NULL, &ev);
    HandleErr(err);

    // Release working buffer
    HandleErr(err = clEnqueueUnmapMemObject(ocl->Queue(), clImgData, imgData, 0, NULL, NULL));
    HandleErr(err = clReleaseMemObject(clImgData));

    // Print debug
    ocl->PrintEventTiming(ev, "Write Image");
    HandleErr(err = clReleaseEvent(ev));

    return CL_SUCCESS;
}

cl_kernel OCLMutualInfo::MakeKern(int imgSize)
{
    // Find existing kernel
    int i = 0;
    for (; i < NUM_KERN; i++) {
        if (kernels[i].clKern == NULL)
            break;

        // Return matching kernel
        if (imgSize == kernels[i].imgSize)
        {
            return kernels[i].clKern;
        }
    }

    // Set next set of values
    kernels[i].imgSize = imgSize;

    // Compile kernel
    char *compileStrBuf = (char *) malloc(1024 * sizeof(char));
    const char *kernelStr = Prep(compileStrBuf, imgSize);

    const char *kernNames[3] = {"histogram", "joint_histogram", "calculate_entropy"};
    cl_kernel *kerns[1] = {&kernels[i].clKern};
    ocl->SetupKern(compileStrBuf, kernelStr, kernNames, kerns);

    free(compileStrBuf);

    return kernels[i].clKern;
}

#endif // USE_OPENCL


cl_int OCLMutualInfo::RunHistKern(cl_mem clImg, cl_mem &clHist, int imgSize)
{
    // Check for needed ocl objects 
    assert(clImg);
    assert(clHistKern);
    assert(!clHist);
    cl_int err = CL_SUCCESS;
    
    // create a working buffer for the histogram
    const int num_bins = 257;
    clHist = clCreateBuffer(ocl->Context(), CL_MEM_READ_WRITE,
			    num_bins*sizeof(unsigned int), NULL, &err);
    if (!clHist || err)
    {
      printf("clCreateBuffer() failed. (%d)\n", err);
      return EXIT_FAILURE;
    }


    // Set kernel args
    HandleErr(err = clSetKernelArg(clHistKern, 0, sizeof(cl_mem), &clHist));
    HandleErr(err = clSetKernelArg(clHistKern, 1, sizeof(cl_mem), &clImg));
    HandleErr(err = clSetKernelArg(clHistKern, 2, sizeof(cl_mem), &num_bins));
    HandleErr(err = clSetKernelArg(clHistKern, 3, sizeof(cl_mem), &imgSize));

    // Run the calculation by enqueuing it and forcing the
    // command queue to complete the task.
    cl_event ev;
    const size_t nThreads = (imgSize * imgSize);
    HandleErr(err = clEnqueueNDRangeKernel(ocl->Queue(), clHistKern, 1, NULL,
                                           &nThreads, &localSize, 0, NULL, &ev));
    HandleErr(err = clFinish(ocl->Queue()));

    ocl->PrintEventTiming(ev, "Histogram Kernel");
    HandleErr(err = clReleaseEvent(ev));
    return CL_SUCCESS;
}

cl_int OCLMutualInfo::RunJointHistKern(cl_mem clImg1, cl_mem clImg2,
				       cl_mem &clJointHist, int imgSize)
{
    // Check for needed ocl objects 
    assert(clJointHistKern);
    assert(clImg1);
    assert(clImg2);
    assert(!clJointHist);
    cl_int err = CL_SUCCESS;
    
    // create a working buffer for the histogram
    const int num_bins = 257;
    clJointHist = clCreateBuffer(ocl->Context(), CL_MEM_READ_WRITE,
				 num_bins*sizeof(unsigned int), NULL, &err);
    if (!clJointHist || err)
    {
      printf("clCreateBuffer() failed. (%d)\n", err);
      return EXIT_FAILURE;
    }

    // Set kernel args
    HandleErr(err = clSetKernelArg(clJointHistKern, 0, sizeof(cl_mem), &clJointHist));
    HandleErr(err = clSetKernelArg(clJointHistKern, 1, sizeof(cl_mem), &clImg1));
    HandleErr(err = clSetKernelArg(clJointHistKern, 2, sizeof(cl_mem), &clImg2));
    HandleErr(err = clSetKernelArg(clJointHistKern, 3, sizeof(cl_mem), &num_bins));
    HandleErr(err = clSetKernelArg(clJointHistKern, 4, sizeof(cl_mem), &imgSize));

    // Run the calculation by enqueuing it and forcing the
    // command queue to complete the task.
    cl_event ev;
    const size_t nThreads = (imgSize * imgSize) * 2;
    HandleErr(err = clEnqueueNDRangeKernel(ocl->Queue(), clJointHistKern, 1, NULL,
                                           &nThreads, &localSize, 0, NULL, &ev));
    HandleErr(err = clFinish(ocl->Queue()));

    ocl->PrintEventTiming(ev, "Joint Histogram Kernel");
    HandleErr(err = clReleaseEvent(ev));
    return CL_SUCCESS;
}

cl_int OCLMutualInfo::RunEntropyKern(cl_mem clHist1, cl_mem clHist2, cl_mem clJointHist,
				       cl_mem &clEntropy, int imgSize)
{
    // Check for needed ocl objects
    assert(clEntropyKern);
    assert(clHist1);
    assert(clHist2);
    assert(clJointHist);
    assert(!clEntropy);
    cl_int err = CL_SUCCESS;
    
    // create a working buffer for entropy and mutual info result
    clEntropy = clCreateBuffer(ocl->Context(), CL_MEM_READ_WRITE, 4 * sizeof(double), NULL, &err);
    if (!clEntropy || err)
    {
      printf("clCreateBuffer() failed. (%d)\n", err);
      return EXIT_FAILURE;
    }

    // Set kernel args
    HandleErr(err = clSetKernelArg(clEntropyKern, 0, sizeof(cl_mem), &clHist1));
    HandleErr(err = clSetKernelArg(clEntropyKern, 1, sizeof(cl_mem), &clHist2));
    HandleErr(err = clSetKernelArg(clEntropyKern, 2, sizeof(cl_mem), &clJointHist));
    HandleErr(err = clSetKernelArg(clEntropyKern, 3, sizeof(cl_mem), &clEntropy));
    HandleErr(err = clSetKernelArg(clEntropyKern, 4, sizeof(cl_int), &imgSize));

    // Run the calculation by enqueuing it and forcing the
    // command queue to complete the task.
    cl_event ev;
    const size_t nThreads = (imgSize * imgSize);
    HandleErr(err = clEnqueueNDRangeKernel(ocl->Queue(), clEntropyKern, 1, NULL,
                                           &nThreads, &localSize, 0, NULL, &ev));
    HandleErr(err = clFinish(ocl->Queue()));

    ocl->PrintEventTiming(ev, "Entropy Kernel");
    HandleErr(err = clReleaseEvent(ev));
    return CL_SUCCESS;
}

void OCLMutualInfo::Run(cv::Mat img1, cv::Mat img2,
                        int imgSize, double entropy[])
{
    // Confirm dims and data type 
    if (img1.empty() || img2.empty())
        return;
    assert(img1.channels() == 1 && img2.channels() == 1);
    assert(img1.depth() == CV_16S || img1.depth() == CV_16U);
    assert(img2.depth() == CV_16S || img2.depth() == CV_16U);
    
#ifdef USE_OPENCL
    cl_int err = CL_SUCCESS;
    
    // Compile kernel
    cout << "::: Compiling kenrel string ::: " << endl;
    char *compileStrBuf = (char *) malloc(2048 * sizeof(char));
    const char *kernelStr = Prep(compileStrBuf, imgSize);

    // Compile the kernel if needed
    const char *kernNames[4] = {"histogram",
				"joint_histogram",
				"calculate_entropy", NULL};
    cl_kernel *kerns[3] = {&clHistKern,
			   &clJointHistKern,
			   &clEntropyKern};
    
    ocl->SetupKern(compileStrBuf, kernelStr, kernNames, kerns);
    cout << "::: Setting up kernels :::" << endl;
    free(compileStrBuf);

    // Data structures that we'll need
    cl_mem clImg1 = NULL;
    cl_mem clImg2 = NULL;
    cl_mem clHist1 = NULL;
    cl_mem clHist2 = NULL;
    cl_mem clJointHist = NULL;
    cl_mem clEntropy = NULL;
    
    // Copy input to device
    cout << "::: Copying images :::" << endl;
    CopyImage(img1, &clImg1);
    CopyImage(img2, &clImg2);

    // Do the dirty work
    cout << "::: Running hist kern :::" << endl;
    RunHistKern(clImg1, clHist1, imgSize);
    RunHistKern(clImg2, clHist2, imgSize);
    RunJointHistKern(clImg1, clImg2, clJointHist, imgSize);
    RunEntropyKern(clHist1, clHist2, clJointHist, clEntropy, imgSize);
    cout << "::: Finished running kernels :::" << endl;
    
    // Create pinned-mem buffer to read back mutualInfo result from device
    size_t entropyDataSz = sizeof(double) * 4;
    cl_mem clEntropyData;
    double* entropyData;
    ocl->AllocPinnedMem(&clEntropyData, (void **)&entropyData, entropyDataSz);

    // Copy 3 entropy vals and mutualInfo from device into pinned mem
    err = clEnqueueReadBuffer(
        ocl->Queue(), clEntropy, CL_FALSE, 0, entropyDataSz, entropyData, 0, NULL, NULL);

    // Check for success
    if (!entropyData || err)
    {
      printf("clEnqueueReadBuffer() failed for final result. (%d)\n", err);
    }

    // Copy data from pinned mem buffer
    for (int i = 0; i < 4; i++) {
      entropy[i] = 0;
      entropy[i] = entropyData[i];
      cout << "::: Entropy[" << i << "] = " << entropy[i] << " :::" << endl; 
    }

    // Release working pinned mem buffer
    err = clEnqueueUnmapMemObject(ocl->Queue(), clEntropyData, entropyData, 0, NULL, NULL);

    // Check for success
    if (err != CL_SUCCESS)
    {
      printf("clEnqueueUnmapMemObject() failed for final result. (%d)\n", err);
    }
    
    // Free transient data
    clReleaseMemObject(clImg1);
    clReleaseMemObject(clImg2);
    clReleaseMemObject(clHist1);
    clReleaseMemObject(clHist2);
    clReleaseMemObject(clJointHist);
    clReleaseMemObject(clEntropy);
    clReleaseMemObject(clEntropyData);
#endif // USE_OPENCL
}
