#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>

#include "pointop.h"
#include "checks.h"




template <typename scalar_t>
__device__ __forceinline__ scalar_t activation(scalar_t x) {
    return x < 0.0f ? 0.0f : x; 
}



// only works for channel size:
// first: is it a competetive implementation? benchmark this first;
// if yes, implemt masked version; does that provide benefit?
// if yes, yaay. if no, cry.


// assuems C,H,W format
// assumes number of threads == 
// each thread loops across input field;
template <typename scalar_t, int activation_fn=0, int FIELD_SIZE=3, int C_PER_BLOCK=4, int H_PER_BLOCK=3, int W_PER_BLOCK=3, int WARP_SIZE=32>
__global__ void activation_increment_kernel(
    scalar_t *__restrict__  X,
    scalar_t const *__restrict__ in_incr,
    scalar_t * __restrict__ out_incr,  // expect a zero tenor, out
    dim const X_dim
){

    // int const warp_idx = threadIdx.x/WARP_SIZE;
    int const lane_idx = threadIdx.x%WARP_SIZE;
    int const block_idx = blockIdx.x;

    int const H_up = divup(X_dim.H, H_PER_BLOCK);
    int const W_up = divup(X_dim.W, W_PER_BLOCK);
    int const HW_up = H_up*W_up;

    int const c_in_start = block_idx/HW_up;
    int const c_in_end = min(X_dim.C, c_in_start + C_PER_BLOCK);

    int const w = W_PER_BLOCK*(block_idx%H_up) + lane_idx%H_PER_BLOCK;
    int const h = H_PER_BLOCK*(block_idx/H_up) + lane_idx/H_PER_BLOCK;

    // out of bounds
    if(lane_idx >= H_PER_BLOCK*W_PER_BLOCK || h > X_dim.H || w > X_dim.W)
        return;

    int const px_offs = h*X_dim.H + w;

    // for(int i = 0; i < )
    for(int c = c_in_start; c < c_in_end ; c += 1){
        int x_id = c*X_dim.H*X_dim.W + px_offs;
        scalar_t* reserve = &X[x_id];
        scalar_t const * incr = &in_incr[x_id];
        scalar_t const full = *reserve + *incr;
        out_incr[x_id] = activation(full) - activation(*reserve);
        *reserve = full;
    }
}


template<typename scalar_t>
void activation_increment_cuda(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
){

    auto Xdim_sizes = X.sizes();
    auto X_dim = dim(Xdim_sizes);

    int const H_PER_BLOCK=3;
    int const W_PER_BLOCK=3;
    int const C_PER_BLOCK=3;

    // per block function: 3*3*C_PER_BLOCK
    int const H_up = divup(X_dim.H, H_PER_BLOCK);
    int const W_up = divup(X_dim.W, W_PER_BLOCK);
    int const C_up = divup(X_dim.C, C_PER_BLOCK);

    int const blocks = H_up*C_up*W_up;
    int const threads = 32;

    activation_increment_kernel<scalar_t><<<blocks, threads>>>(
        X.data_ptr<scalar_t>(), 
        in_incr.data_ptr<scalar_t>(), 
        out_incr.data_ptr<scalar_t>(), 
        X_dim
    );

    CUDA_CHECK_ERRORS();

}


void activation_increment_cuda_wrapper(
    torch::Tensor &X,
    torch::Tensor const &in_incr,
    torch::Tensor &out_incr  // expect a zero tensor
){
    activation_increment_cuda<float>(
        X,
        in_incr,
        out_incr  // expect a zero tensor
    );
}

