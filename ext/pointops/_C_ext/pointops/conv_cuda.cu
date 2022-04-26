#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "ops.h"
#include "utils.h"
#include "checks.h"



struct filter_dim{
    __device__ int const filter_in, filter_out;
    __device__ filter_dim(int fin, int fout): filter_in(fin), filter_out(fout)
    {
    }
};

// TODO: idk if it's kx,ky or ky,kx; +i is just a whimmed out like that
// assumption verified: conv filter is of the order (out, in,  ky, kx  )
template<int f_h, int f_w>
__device__ int get_filter_index(filter_dim const f_dims, int const c_in, int const c_out, int const i){
    return c_out*f_dims.filter_in*f_h*f_w + c_in*f_h*f_w + i;
}


// padding: only 'same' padding
// stride: 1 and 2 to be supported; only 1 works for now but very extensible
// implement convolution as a scatter opreation
template<int WARP_SIZE=32, int C_PER_BLOCK_PER_WARP=4, int H_PER_BLOCK=3, int W_PER_BLOCK=3>
__device__ __forceinline__ void calc_field_indices(int &c_in_start, int& c_in_end, int &c_out_start, int &c_out_end, int &h, int &w, dim const &in_dim, dim const &out_dim) {

    // block independent: change this later;
    c_in_start = 0;
    c_in_end = in_dim.C;

    w = W_PER_BLOCK*(blockIdx.y) + threadIdx.y;
    h = H_PER_BLOCK*(blockIdx.z) + threadIdx.z;

    // int lane_idx = threadIdx.x%WARP_SIZE;
    c_out_start = blockIdx.x*C_PER_BLOCK_PER_WARP*WARP_SIZE + threadIdx.x;
    c_out_end = c_out_start;
}


// one variant of convolution
// each thread only loops through the input block; computes one output block in same padding mode;
// this can be furthe rparallelized; but let's see how it performs


// H_PER_BLOCK, they cannot go through that execution path 
// C_PER_BLOCK

template <typename scalar_t, int WARP_SIZE=32, int C_OUT_PER_BLOCK_PER_WARP=3, int H_OUT_PER_BLOCK=2, int W_OUT_PER_BLOCK=2>
__global__ void conv3x3(
    scalar_t const *__restrict__ conv_filter,
    scalar_t const *__restrict__  x,
    scalar_t * __restrict__ out,
    dim const in_dim,
    dim const out_dim
){

    // int const warp_idx = threadIdx.x/WARP_SIZE;
    int c_in_start, c_in_end, c_out_start, c_out_end, h, w;
    calc_field_indices<WARP_SIZE, C_OUT_PER_BLOCK_PER_WARP, H_OUT_PER_BLOCK, W_OUT_PER_BLOCK>(c_in_start, c_in_end, c_out_start, c_out_end, h, w, in_dim, out_dim);


    // checks TODO: complete other checks
    if(h >= out_dim.H || w >= out_dim.W)
        return;

    filter_dim conv_filter_dims(in_dim.C, out_dim.C);

    for(int c_out = c_out_start; c_out < out_dim.C; c_out += gridDim.x*C_OUT_PER_BLOCK_PER_WARP*WARP_SIZE){
        int x_id = c_out*out_dim.H*out_dim.W + h*out_dim.W + w;
        scalar_t *out_ptr = &out[x_id];       // assumes initialized to 0


        // loop through c_ins
        for(int c_in = c_in_start; c_in < c_in_end ; c_in += 1){
            bool skip = false;
            // skip conv mask logic goes here, given h,w, and c_in :(
            // mask_ptr = mask + c_in*in_dim.H*in_dim*W
            if(!skip){
                scalar_t const *in_ptr = x + c_in*in_dim.H*in_dim.W;
                for(int i = 0; i < 9; i++){
                    *out_ptr += conv_filter[get_filter_index<3,3>(conv_filter_dims, c_in, c_out, i)] *
                     in_ptr[ (h + i/3)*in_dim.W +  w + (i%3) ];
                }

            }
        }
    }
}


template <typename scalar_t, int WARP_SIZE=32, int C_OUT_PER_BLOCK_PER_WARP=3, int H_OUT_PER_BLOCK=3, int W_OUT_PER_BLOCK=3>
void conv3x3_increment_cuda(
    torch::Tensor const &in_incr,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
){
    dim out_dim(out_incr.sizes()), in_dim(in_incr.sizes());

    int const W_up = divup(out_dim.W, W_OUT_PER_BLOCK);
    int const H_up = divup(out_dim.H, H_OUT_PER_BLOCK);
    int const C_up = divup(out_dim.C, C_OUT_PER_BLOCK_PER_WARP*WARP_SIZE);

    dim3 const blocks(C_up, W_up, H_up);
    dim3 const threads(WARP_SIZE*C_OUT_PER_BLOCK_PER_WARP, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK);
    conv3x3<scalar_t, WARP_SIZE, C_OUT_PER_BLOCK_PER_WARP, H_OUT_PER_BLOCK, W_OUT_PER_BLOCK><<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(), 
        in_incr.data_ptr<scalar_t>(), 
        out_incr.data_ptr<scalar_t>(), 
        in_dim,
        out_dim
    );

    CUDA_CHECK_ERRORS();
}


void conv3x3_increment_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
){

    conv3x3_increment_cuda<float, 32, 16, 3, 3>(
        in_incr,
        filter,
        out_incr  // expect a zero tensor
    );
}



