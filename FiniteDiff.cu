
#include "cuda_runtime.h"
#include "assert.h"
#include "cuda.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
//#include "/usr/local/mpich/include/mpi.h"
#include <mpi.h>

#include "IniteDiff.h"

using namespace std;

#define CUDA_CALL(x) {const cudaError_t a=(x);\
		if(a!=cudaSuccess){printf("\nCUDA ERROR: %s (err_num=%d) \n",\
			cudaGetErrorString(a),a);cudaDeviceReset();assert(0);}}


__global__ void Warmup()
{

}

void GetDeviceName() 
{ 
	//Warmup<<<1,1>>>();
    int count= 0;
    cudaGetDeviceCount(&count);
    cudaDeviceProp prop;
    if (count== 0)
    {
        cout<< "There is no device."<< endl;
    }
    for(int i= 0;i< count;++i)
    {
    	cudaGetDeviceProperties(&prop,i) ;
    	cout << "Device name  is :" << prop.name<< endl;
    } 
}


__device__ float SingleFai(float *fai, unsigned int i,unsigned int j,size_t pitch) {
	float *a = (float*)((char*)fai + (i - 1)*pitch);
	float *b = (float*)((char*)fai + (i + 1)*pitch);
	float *c = (float*)((char*)fai + i*pitch);
	return ((a[j] + b[j] + c[j - 1] + c[j + 1]) / 4);
}



__global__ void FaiIter(float *fai,float *fai_n,size_t pitch,int M, int N, int flag) {
	//unsigned int i = blockDim.y*blockIdx.y + threadIdx.y;
	//unsigned int j = blockDim.x*blockIdx.x + threadIdx.x;
	for (int i = blockDim.y*blockIdx.y + threadIdx.y; i < M; i += blockDim.y*gridDim.y) {
		float *fai_row_n = (float*)((char*)fai_n + i*pitch);
		for (int j = blockDim.x*blockIdx.x + threadIdx.x; j < N; j += blockDim.x*gridDim.x) {
			if(flag==1){
				if (i > 1 && i < M - 1 && j > 0 && j < N - 1)
				fai_row_n[j] = SingleFai(fai, i, j, pitch);
			}
			else if(flag==2){
				if (i > 0 && i < M - 2 && j > 0 && j < N - 1)
				fai_row_n[j] = SingleFai(fai, i, j, pitch);
			}
			else{
				if (i > 0 && i < M - 1 && j > 0 && j < N - 1)
				fai_row_n[j] = SingleFai(fai, i, j, pitch);
			}
		}
	}
}


//the number of grids every node get is M*N, which equals to local_M*local_N in main.cpp
void GpuCalculate(float *fai, int M, int N, int my_rank, int comm_sz)
{

	float *fai_dev,*fai_dev_n, *temp;
	float *send_up,*send_down,*recv_up,*recv_down; 
	size_t pitch;
	unsigned int n=0;
	MPI_Status status;

	int flag = 0;
	if(my_rank==0)
		flag=1;             
	else if(my_rank==comm_sz-1)
		flag=2;     

	//cout<<fai[1]<<' '<<fai[N]<<' '<<fai[(M-1)*N]<<endl;

	send_up=fai+N;
	send_down=fai+(M-2)*N;
	recv_up=fai;
	recv_down=fai+(M-1)*N;

	//cout<<my_rank<<" :recv_down[3] "<<recv_down[3]<<endl;

	CUDA_CALL(cudaMallocPitch((void**)&fai_dev, &pitch, N * sizeof(float), M));
	CUDA_CALL(cudaMallocPitch((void**)&fai_dev_n, &pitch, N * sizeof(float), M));

	CUDA_CALL(cudaMemcpy2D(fai_dev, pitch, fai, sizeof(float)*N, sizeof(float)*N, M, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy2D(fai_dev_n, pitch, fai, sizeof(float)*N, sizeof(float)*N, M, cudaMemcpyHostToDevice));

	int N_G = pitch/sizeof(float);

	const dim3 blockDim(32, 16,1);
	const dim3 gridDim(8, 8,1);

	MPI_Barrier(MPI_COMM_WORLD);

	while(n<5000){
		FaiIter <<<gridDim, blockDim >>> (fai_dev, fai_dev_n, pitch, M, N, flag);

		temp = fai_dev;
		fai_dev = fai_dev_n;
		fai_dev_n = temp;



		//CUDA_CALL(cudaMemcpy2D(fai_dev, pitch, fai_dev_n, pitch, sizeof(float)*N, M, cudaMemcpyDeviceToDevice));
		CUDA_CALL(cudaMemcpy2D(send_up, sizeof(float)*N, fai_dev+N_G, pitch, sizeof(float)*N, 1, cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy2D(send_down, sizeof(float)*N, fai_dev+(M-2)*N_G, pitch, sizeof(float)*N, 1, cudaMemcpyDeviceToHost));
		//CUDA_CALL(cudaMemcpy2D(fai, sizeof(float)*N,fai_dev,pitch,sizeof(float)*N,M,cudaMemcpyDeviceToHost));

		//can mpi send or recieve data directly on gpu?
		MPI_Barrier(MPI_COMM_WORLD);

		if(flag==1){
			MPI_Send(send_down, N, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
			MPI_Recv(recv_down,	N, MPI_FLOAT, 1, 1, MPI_COMM_WORLD, &status);
			CUDA_CALL(cudaMemcpy2D(fai_dev+(M-1)*N_G, pitch, recv_down, sizeof(float)*N, sizeof(float)*N, 1, cudaMemcpyHostToDevice));
		}		
		else if(flag==2){
			MPI_Send(send_up, N, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD);
			MPI_Recv(recv_up, N, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);
			CUDA_CALL(cudaMemcpy2D(fai_dev, pitch, recv_up, sizeof(float)*N, sizeof(float)*N, 1, cudaMemcpyHostToDevice));
		}
		else{
			MPI_Send(send_up, N, MPI_FLOAT, my_rank-1, 1, MPI_COMM_WORLD);
			MPI_Recv(recv_down, N, MPI_FLOAT, my_rank+1, 1, MPI_COMM_WORLD, &status);
			MPI_Send(send_down, N, MPI_FLOAT, my_rank+1, 0, MPI_COMM_WORLD);
			MPI_Recv(recv_up, N, MPI_FLOAT, my_rank-1, 0, MPI_COMM_WORLD, &status);
			CUDA_CALL(cudaMemcpy2D(fai_dev, pitch, recv_up, sizeof(float)*N, sizeof(float)*N, 1, cudaMemcpyHostToDevice));
			CUDA_CALL(cudaMemcpy2D(fai_dev+(M-1)*N_G, pitch, recv_down, sizeof(float)*N, sizeof(float)*N, 1, cudaMemcpyHostToDevice));

			//CUDA_CALL(cudaMemcpy2D(fai_dev, pitch, fai, sizeof(float)*N, sizeof(float)*N, M, cudaMemcpyHostToDevice));
		}
		
		n++;
		MPI_Barrier(MPI_COMM_WORLD);
	}

	CUDA_CALL(cudaMemcpy2D(fai, sizeof(float)*N,fai_dev,pitch,sizeof(float)*N,M,cudaMemcpyDeviceToHost));
	cudaFree(fai_dev);
	cudaFree(fai_dev_n);
	return ;
}

