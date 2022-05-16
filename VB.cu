// This code implements the example shown in the Variational Bayes page on Wikipedia. Link: https://en.wikipedia.org/wiki/Variational_Bayesian_methods
// Here two kernels named nor and gam were launched in parallel with CUDA stream.
// Normal samples were drawn from a normal distribution and stored in vector ex_a.
//These samples were used to estimate the mean and standard deviation by using Variational Bayes algorithm.
//Run this code by using nvcc VB.cu
// The output is the mean and standard deviation calculated in each iteration.

#include <iostream>
#include <cuda.h>
#include <curand_kernel.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

__global__ void nor(double* dlambdan, double lambda0, int dim, int gene, double mu_n, double x_sq, double* dbn, double mu0,
                    double b0,double xb, double* dmean, curandState_t *d_states){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curandState_t state = d_states[idx]; 
    dbn[idx] = b0 + 0.5*((lambda0+(dim*gene))*(1./dlambdan[idx] + (mu_n*mu_n))-2.0*(lambda0*mu0 + xb)*mu_n+(x_sq)+(lambda0*mu0*mu0));
    double u2 = curand_normal(&state); //one normal sample
    dmean[idx] = ((1./dlambdan[idx]) * u2) + mu_n;
    
}

//Marsaglia and Tsang sampler (Gamma sampler)
__global__ void gam(double* dlambdan, double lambda0, double dim, double gene, double an, double* dbn, double mu_n,
                    double* dvar, int count2, curandState_t *d_states){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    curandState_t state = d_states[idx]; 
    dlambdan[idx] = (lambda0+(dim*gene))*(an/dbn[idx]);

    double d22 = (an) -1.0/3.;
    double c2 = 1.0/sqrt(9*d22);
    do{
        
        
        double u13 = curand_uniform(&state); 
        double u23 = curand_normal(&state); 
        double v2 = pow((1. + c2*u23), 3);
        int j2 = v2 > 0 && log(u13) < 0.5*pow(u23, 2)+d22 - d22*v2+d22*log(v2);
        count2 = j2;
        dvar[idx] = d22*v2/dbn[idx];//samples
        
        
    }while(count2 == 0);
    
}
__global__ void setup(curandState_t *d_states, int j)
{
    int id = threadIdx.x+ blockIdx.x * blockDim.x;
    
    curand_init(j, id, 0, &d_states[id]);
}

int main(){
    int gene = 30;
    int dim = 20;
    
    std::vector<double> ex_a(dim*gene);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(10.0,2.0);

    for(int i = 0; i < dim*gene; i++){ex_a[i] = distribution(generator);} 
    //for(int i = 0; i < dim*gene; i++) std::cout << ex_a[i] <<std::endl;
    
    double a = 0.3; double lambda0 = 1; double mu0 = 2; double b0 = 1.0; 

    thrust::device_vector<double> d1(gene*dim); //device vector
    thrust::device_vector<double> d2(gene*dim); //device vector

    d1 = ex_a; //copying host vector to device vector
    double xb = thrust::reduce(d1.begin(), d1.end()); //adding all the samples
    thrust::transform(d1.begin(), d1.end(), d2.begin(), thrust::square<double>()); //squaring all the samples and storing on device vector d2
    double x_sq = thrust::reduce(d2.begin(), d2.end()); //adding all the elements in device vector d2
    

    double an = a + ((dim*gene)+1)/2.0; double* bn = new double[1]; bn[0] = 0.2;

    double xbar = xb/(dim*gene);
    double mu_n = (lambda0*mu0 + dim*gene*xbar)/(lambda0 + (dim*gene));
    
    double* lambdan = new double[1]; lambdan[0] = 0.02;
    double* dlambdan; 
    cudaMalloc(&dlambdan, sizeof(double)*1);//device vector memory allocation
    double* dbn; 
    cudaMalloc(&dbn, sizeof(double)*1);  //device vector memory allocation
    double* dmean;
    cudaMalloc(&dmean, sizeof(double)*1); //device vector memory allocation
    double* meancopy = new double[1];

    double* varcopy = new double[1];
    double* dvar = new double[1];
    cudaMalloc(&dvar, sizeof(double)*1);

    cudaMemcpy(dbn, bn, sizeof(double)*1, cudaMemcpyHostToDevice); //copying from host to device
    cudaMemcpy(dlambdan, lambdan, sizeof(double)*1, cudaMemcpyHostToDevice); //copying from host to device
    
    //Random number
    curandState_t* d_states;
    cudaMalloc(&d_states, sizeof(curandState_t)*1);
  
    //Streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1); cudaStreamCreate(&stream2); 
    int count2 = 0; 
    
    for(int i =0; i < 1000; i++){
        setup<<<1,1>>>(d_states,i);//setting up random numbers

        nor<<<1,1,0,stream1>>>(dlambdan, lambda0, dim, gene, mu_n, x_sq, dbn, mu0,b0,xb, dmean, d_states); //Normal distribution sampler in stream1

        gam<<<1,1,0,stream2>>>(dlambdan, lambda0, dim, gene, an, dbn, mu_n, dvar, count2, d_states); //Gamma distribution sampler in stream2

        cudaMemcpy(bn, dbn, sizeof(double)*1, cudaMemcpyDeviceToHost); //copying from device to host
        //cudaMemcpy(lambdan, dlambdan, sizeof(double)*1, cudaMemcpyDeviceToHost);
        cudaMemcpy(meancopy, dmean, sizeof(double)*1, cudaMemcpyDeviceToHost); //copying from device to host
        cudaMemcpy(varcopy, dvar, sizeof(double)*1, cudaMemcpyDeviceToHost); //copying from device to host

        
        std::cout<<"mean is "<< meancopy[0] <<" and s.d. is " << sqrt(1/varcopy[0]) << std::endl;

    }
    
    cudaFree(dbn);
    cudaFree(lambdan);
    cudaFree(dmean);
    cudaFree(dvar);
}
