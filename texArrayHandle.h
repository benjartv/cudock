#ifndef __TEXARRAYHANDLE__
#define __TEXARRAYHANDLE__

#include <stdio.h>
#include <cuda_runtime.h>

template <class T>
class tex3DHandle
{
private:
	cudaArray * cuArray;
	cudaChannelFormatDesc channelDesc;
	cudaExtent range;
public:

	//to use this, use 
	//	tex3DHandle texH(w,h,d);
	//	texH.copyData(h_data);
	//	init_texture(texH,INDEX);
	//for full initialization and data transfer
	//
	
	tex3DHandle()
	{
		channelDesc=cudaCreateChannelDesc<T>();
		range=make_cudaExtent(0,0,0);
		cuArray=NULL;
	}

	tex3DHandle(int w, int h, int d)
	{
		channelDesc= cudaCreateChannelDesc<T>();
		range=make_cudaExtent(w,h,d);
		cudaMalloc3DArray(&(this->cuArray),&(this->channelDesc),range);
	}

	void copyData(void * cpuPtr)
	{
		cudaMemcpy3DParms copy={0};
		copy.srcPtr=make_cudaPitchedPtr(cpuPtr,range.width*sizeof(T),range.width,range.height);
		copy.dstArray=cuArray;
		copy.extent=range;
		copy.kind=cudaMemcpyHostToDevice;
		cudaMemcpy3D(&copy);
	}

	tex3DHandle & operator=(const tex3DHandle& other)
	{
		if (this==&other)
		{
			return *this;
		}
		if (cuArray!=NULL)
		{
			cudaFreeArray(cuArray);
		}
		channelDesc=other.channelDesc;
		range=other.range;
		cudaMalloc3DArray(&(this->cuArray),&(this->channelDesc),range);
	
		return *this;
	}

	void print()
	{
		printf("Texture Handle @%o\n",this);
		printf("[%d %d %d]\n",range.width,range.height, range.depth);
	}

	inline cudaArray * getPtr()
	{
		return cuArray;
	}

	inline cudaChannelFormatDesc * getChannel()
	{
		return &channelDesc;
	}

	~tex3DHandle()
	{
		cudaFreeArray(cuArray);
	}

};


template <class T>
class tex1DHandle
{
private:
	T * cuArray;
	cudaChannelFormatDesc channelDesc;
	int size;
public:

	//to use this, use 
	//	tex3DHandle texH(w,h,d);
	//	texH.copyData(h_data);
	//	init_texture(texH,INDEX);
	//for full initialization and data transfer
	//
	
	tex1DHandle()
	{
		channelDesc=cudaCreateChannelDesc<T>();
		size=0;
		cuArray=NULL;
	}

	tex1DHandle(int len)
	{
		channelDesc= cudaCreateChannelDesc<T>();
		size=len;
		cudaMalloc((void**)&cuArray,sizeof(T)*len);
	}

	void copyData(void * cpuPtr)
	{
		cudaMemcpy(cuArray,cpuPtr,sizeof(T)*size,cudaMemcpyHostToDevice);
	}

	tex1DHandle & operator=(const tex1DHandle& other)
	{
		if (this==&other)
		{
			return *this;
		}
		if (cuArray!=NULL)
		{
			cudaFreeArray(cuArray);
		}
		channelDesc=other.channelDesc;
		size=other.size;
		cudaMalloc((void**)&cuArray,sizeof(T)*size);
	
		return *this;
	}

	void print()
	{
		printf("1D Texture Handle [%d] @%o\n",size,this);
	}

	inline T* getPtr()
	{
		return cuArray;
	}

	inline cudaChannelFormatDesc * getChannel()
	{
		return &channelDesc;
	}

	~tex1DHandle()
	{
		cudaFree(cuArray);
	}

};


#endif
