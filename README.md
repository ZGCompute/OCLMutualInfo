# OpenCL/OpenCV based image comparison using mutual-information #

This repository contains the code to run an OpenCL/OpenCV based image comparison. The program is primarily used as an example use-case to calculate mutual-information as a similarity metric between two square images.

## Development Environment

This library uses a OpenCV/OpenCL for the development environment and requires OpenCL SDK/drivers for your GPU device installed on the host computer.

To start the build process, move into the root directory of the repo type:

```
cd /OCLMutualInfo
```

Then run the script to build ocl.mutual_information itself using:
```
sh ./build_core.sh
```

```
Try running the executable on two images included in /test_data using:

./ocl.mutual_information test_data/benoit.jpg test_data/arnaud.jpg 1

Here 1 is input as the device # to use. If on your machine, the ID of the GPU is different, use the desired ID instead

## Organization of the repo

The source code for the mutual-information algorithm is kept in 'sources', 'headers' and 'modules'
