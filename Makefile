BASEDIR = $(~/Desktop/clDev/OCLMutualInfo)

INCLUDES = -I$(BASEDIR)/headers -I$(BASEDIR)/sources -I$(BASEDIR)/modules\
           -I/opt/intel/opencl-1.2-sdk-6.2.0.1761/include/

OCV_LIB = -lopencv_core -lopencv_highgui 

LIB = -lOpenCL $(OCV_LIB)



FLAGS = $(INCLUDES) -g -std=c++11 -DUSE_OPENCL

CPP_FILES = $(wildcard sources/*.cpp)

MOD_FILES = $(wildcard modules/*.cpp)

OBJ_FILES = $(patsubst sources/%.cpp,build/%.o,$(CPP_FILES)) $(patsubst modules/%.cpp,build/%.o,$(MOD_FILES))


MUTUALINFO_OBJ_FILES = build/MutualInfo.o build/OCLMutualInfo.o build/OCLEnv.o build/Timer.o 

all: $(OBJ_FILES) ocl.mutual_information 

clean:
	rm -f build/*.o 

build/%.o: sources/%.cpp
	g++ -c $^ $(FLAGS) $(DEFINES) $(LIB) -o $@

build/%.o: modules/%.cpp
	g++ -c $^ $(FLAGS) $(DEFINES) $(LIB) -o $@

ocl.mutual_information: $(MUTUALINFO_OBJ_FILES)
	                libtool --mode=link g++ $(FLAGS) $(LIB) -o ocl.mutual_information $(MUTUALINFO_OBJ_FILES)

install: ocl.mutual_information 
	 cp ocl.mutual_information $(BASEDIR)/bin/ocl.mutual_information

