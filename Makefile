OBJS = parse.cpp.o test.cpp.o kernel.cu.o
TARGET = cudock

######################################################
CXX := g++
CC  := gcc
NVCC:= nvcc
LINK:= g++ -fPIC

LIB_CUDA:= -L/opt/cuda-toolkit/lib64 -lcudart
INC_CUDA:= -I/opt/cuda-toolkit/include

NVCCFLAGS = -arch sm_20 --ptxas-options=-v
CCFLAGS = 
CXXFLAGS =

######################################################
.SUFFIXES: .c .cpp .cu .o .h

%.c.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@ -$(INC_CUDA)

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(INC_CUDA)

%.cpp.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@ $(INC_CUDA)

$(TARGET): $(OBJS) Makefile
	$(LINK) -o $(TARGET) $(OBJS) $(LIB_CUDA)

clean:
	rm $(OBJS)
	rm $(TARGET)
