#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "ops.h"
#include "utils.h"
#include "checks.h"



// TODO: idk if it's kx,ky or ky,kx; +i is just a whimmed out like that
template<int f_h, int f_w>
__device__ __forceinline__ int const get_filter_index(int const f_in, int const f_out, 
    int const c_in, int const c_out, int const i){
    return c_out*f_in*f_h*f_w + c_in*f_h*f_w + i;
}


// padding: only 'same' padding
// stride: 1 and 2 to be supported; only 1 works for now but very extensible
// implement convolution as a scatter opreation
template<int WARP_SIZE=32, int C_PER_BLOCK_PER_WARP=4, int H_PER_BLOCK=3, int W_PER_BLOCK=3>
__device__ __forceinline__ void calc_field_indices(
    int &c_in_start, int& c_in_end, int &c_out_start, int &c_out_end, int &h, int &w, 
    int const in_C, int const in_H, int const in_W, int const out_C, int const out_H, int const out_W ) {

    // block independent: change this later;
    c_in_start = 0;
    c_in_end = in_C;

    w = W_PER_BLOCK*(blockIdx.y) + threadIdx.y;
    h = H_PER_BLOCK*(blockIdx.z) + threadIdx.z;

    // int lane_idx = threadIdx.x%WARP_SIZE;
    c_out_start = blockIdx.x*C_PER_BLOCK_PER_WARP*WARP_SIZE + threadIdx.x;
    c_out_end = c_out_start;
}


// one variant of convolution: input channelwise looped convolution;
// each thread only loops through the input block; computes one output block in same padding mode;
// this can be furthe rparallelized; but let's see how it performs

// H_PER_BLOCK, they cannot go through that execution path 
// C_PER_BLOCK
template <typename scalar_t, int WARP_SIZE=32, int C_OUT_PER_BLOCK_PER_WARP=3, int H_OUT_PER_BLOCK=2, int W_OUT_PER_BLOCK=2>
__global__ void conv3x3(
    scalar_t const *__restrict__ conv_filter,
    scalar_t const *__restrict__  x,
    scalar_t * __restrict__ out,
    int const in_C, int const in_H, int const in_W,
    int const out_C, int const out_H, int const out_W
){
    int c_in_start, c_in_end, c_out_start, c_out_end, h, w;
    calc_field_indices<WARP_SIZE, C_OUT_PER_BLOCK_PER_WARP, H_OUT_PER_BLOCK, W_OUT_PER_BLOCK>(
        c_in_start, c_in_end, c_out_start, c_out_end, h, w, 
        in_C,  in_H, in_W, out_C, out_H, out_W);

    // checks TODO: complete other checks
    if(h >= out_H || w >= out_W)
        return;
    // printf("cout_skip: %d\n", gridDim.x*C_OUT_PER_BLOCK_PER_WARP*WARP_SIZE);
    for(int c_out = c_out_start; c_out < out_C; c_out += gridDim.x*C_OUT_PER_BLOCK_PER_WARP*WARP_SIZE){
        int x_id = c_out*out_H*out_W + h*out_W + w;
        scalar_t *out_ptr = &out[x_id];       // assumes initialized to 0

        // loop through c_ins
        for(int c_in = c_in_start; c_in < c_in_end ; c_in += 1){
            bool skip = false;
            // skip conv mask logic goes here, given h,w, and c_in :(
            // mask_ptr = mask + c_in*in_dim.H*in_dim*W
            if(!skip){
                scalar_t const *in_ptr = x + c_in*in_H*in_W;
                for(int i = 0; i < 9; i++){
                    int h_in = (h-1 + i/3);
                    int w_in = w-1 + (i%3);
                    if (h_in >= 0 && h_in < in_H && w_in >= 0 && w_in < in_W)
                        *out_ptr += conv_filter[get_filter_index<3,3>(in_C, out_C, c_in, c_out, i)] *
                             in_ptr[ h_in*in_W +  w_in ];
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
    dim in_dim(in_incr.sizes()), out_dim(out_incr.sizes());

    int const W_up = divup(out_dim.W, W_OUT_PER_BLOCK);
    int const H_up = divup(out_dim.H, H_OUT_PER_BLOCK);
    int const C_up = divup(out_dim.C, C_OUT_PER_BLOCK_PER_WARP*WARP_SIZE);

    dim3 const blocks(C_up, W_up, H_up);
    dim3 const threads(WARP_SIZE*C_OUT_PER_BLOCK_PER_WARP, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK);

    // printf("kernel configuration: {%d %d %d}, {%d %d %d}\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    conv3x3<scalar_t, WARP_SIZE, C_OUT_PER_BLOCK_PER_WARP, H_OUT_PER_BLOCK, W_OUT_PER_BLOCK><<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(), 
        in_incr.data_ptr<scalar_t>(), 
        out_incr.data_ptr<scalar_t>(), 
        in_dim.C, in_dim.H, in_dim.W,
        out_dim.C, out_dim.H, out_dim.W
    );

    CUDA_CHECK_ERRORS();
}


void conv3x3_increment_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
){

    conv3x3_increment_cuda<float, 32, 2, 6, 6>(
        in_incr,
        filter,
        out_incr  // expect a zero tensor
    );
}



