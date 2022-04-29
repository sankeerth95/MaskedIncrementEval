#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "checks.h"


namespace Utils{
    template <typename T>
    constexpr T constexpr_min(const T a, const T b) {
        return a > b ? b : a;
    }
    template <typename T>
    constexpr T constexpr_max(const T a, const T b) {
        return a < b ? b : a;
    }
};


// compete with this impelemntation
template<int pixelsPerBlockX, int pixelsPerBlockY, int OUT_CHANNELS_PER_BLOCK, bool FULL_DEPTH>
__device__ __forceinline__ void calc_tile_indices(
    int& tile_start_out_y, int& tile_start_out_x, int& tile_start_in_y, int& tile_start_in_x, int& tile_start_z, int& batch, const int out_C) {

    tile_start_out_y = blockIdx.y * pixelsPerBlockY;
    tile_start_out_x = blockIdx.x * pixelsPerBlockX;
    tile_start_in_y = tile_start_out_y  - 1; // minus padding which is 1
    tile_start_in_x = tile_start_out_x  - 1; // minus padding which is 1

    const int blocksPerBatch = divup(out_C, OUT_CHANNELS_PER_BLOCK);
    tile_start_z = (blockIdx.z % blocksPerBatch) * OUT_CHANNELS_PER_BLOCK;
    batch = blockIdx.z / blocksPerBatch;
}


template<typename scalar_t=float, int WARP_SIZE=32, int pixelsPerBlockX=6, int pixelsPerBlockY=6, int OUT_CHANNELS_PER_BLOCK=256, int BLOCK_SIZE=OUT_CHANNELS_PER_BLOCK>
__global__ void conv_3x3_ext(
    const scalar_t* __restrict__ filter, //channel at the end;
    const bool*__restrict__ mask,
    const scalar_t* __restrict__ input,  // NHWC
    scalar_t* __restrict__ output,       // NHWC
    int const in_C, int const in_H, int const in_W,
    int const out_C, int const out_H, int const out_W
) {

    // true for full depth
    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, true>(
        tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, out_C
    );

    scalar_t* batch_out = output + (batch * out_H * out_W * out_C);
    const scalar_t* batch_in = input + (batch * in_H * in_W * in_C);

    const int w_in = pixelsPerBlockX + 2;
    const int h_in = pixelsPerBlockY + 2;
    const int n_in_px_aligned = divup(w_in * h_in, 4) * 4;


    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;


    union SMEM {
        // SparseSMEM sparse;
        bool mask_s_in[n_in_px_aligned][WARP_SIZE]; //bitwise operations wover WARP_SIZE: implement later
        scalar_t dense_s_in[n_in_px_aligned][WARP_SIZE];
    };
    __shared__ SMEM smem;

    // TODO fix dilation and striding for sparse version
    for (int out_c_off = tile_start_z; out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK; out_c_off += BLOCK_SIZE) {
        const int out_c = out_c_off + threadIdx.x; // parallelization across channels
        scalar_t t_out[pixelsPerBlockX * pixelsPerBlockY];
        #pragma unroll
        for (int i = 0; i < pixelsPerBlockX * pixelsPerBlockY; ++i) {
            t_out[i] = 0.0f;
        }

        for (int in_c_off = 0; in_c_off < in_C; in_c_off += WARP_SIZE) {
            __syncthreads();
            for (int px_idx = warp_idx; px_idx < w_in * h_in; px_idx += n_warps) {
                const int in_y = px_idx / w_in; 
                const int in_x = px_idx % w_in;
                const int in_c = in_c_off + lane_idx;
                const bool valid = in_c < in_C;

                if (valid) {
                    const int in_y_im = in_y + tile_start_in_y;
                    const int in_x_im = in_x + tile_start_in_x; 
                    smem.dense_s_in[in_y * w_in + in_x][lane_idx] = batch_in[in_y_im * in_W * in_C + in_x_im * in_C + in_c];
                    // smem.mask_s_in[in_y * w_in + in_x][lane_idx] = mask[in_y_im * in_W * in_C + in_x_im * in_C + in_c];
                } else {
                    smem.dense_s_in[in_y * w_in + in_x][lane_idx] = 0.0f;
                    // smem.mask_s_in[in_y * w_in + in_x][lane_idx] = true;
                }
            }
            // int const lane_idx = threadIdx.x % WARP_SIZE;
            // int const warp_idx = threadIdx.x/WARP_SIZE;
            __syncthreads();


            // sequential af. : can skip a part of this with the sequential mask. But before we actually do that; have alook oncemore
            for(int in_c = 0; in_c < WARP_SIZE && in_c + in_c_off < in_C; ++in_c) {

                if (out_c < out_C) {


                    //gather operation

                    const scalar_t *in_c_filter = &filter[(in_c_off+in_c) * 9 * out_C + out_c];

                    #pragma unroll
                    for(int out_y = 0; out_y < pixelsPerBlockY; out_y++ ){
                        #pragma unroll
                        for(int out_x = 0; out_x < pixelsPerBlockY; out_x++){

                            // check mask here

                            scalar_t vals[9];
                            #pragma unroll
                            for(int f_y = 0; f_y < 3; ++f_y) {
                                #pragma unroll
                                for(int f_x = 0; f_x < 3; ++f_x) { 
                                    vals[f_y*3 + f_x] = smem.dense_s_in[f_y*w_in + f_x][in_c];
                                }
                            }



                            scalar_t *out_val = &t_out[out_y* pixelsPerBlockX + out_x];
                            *out_val = 0.0f;

                            #pragma unroll
                            for(int in_y = 0; in_y < 3; in_y++){
                                #pragma unroll
                                for(int in_x = 0; in_x < 3; in_x++){

                                    // smem.dense()
                                    *out_val += vals[3*in_y + in_x]*in_c_filter[3*in_y + in_x];

                                }

                            }

                        }
                    }


                    // scatter b0ss

                    // const scalar_t *in_c_filter = &filter[(in_c_off+in_c) * 9 * out_C + out_c];
                    // scalar_t t_f[9];
                    // #pragma unroll
                    // for(int f_y = 0; f_y < 3; ++f_y) {
                    //     #pragma unroll
                    //     for(int f_x = 0; f_x < 3; ++f_x) { 
                    //         t_f[f_y*3 + f_x] = in_c_filter[((2-f_y) * 3 + 2 - f_x) * out_C];
                    //     }
                    // }



                    // #pragma unroll
                    // for (int in_y = -1; in_y < h_in -1; ++in_y) {
                    //     #pragma unroll
                    //     for (int in_x = -1; in_x < w_in -1; ++in_x) {

                    //         // const bool maskval = smem.mask_s_in[(in_y + 1) * w_in + (in_x + 1)][in_c];
                    //         // if(maskval)
                    //         //     continue;tf*smem.dense_s_in[in_y * w_in + (in_x)][in_c];
                    //         const scalar_t val = smem.dense_s_in[(in_y + 1) * w_in + (in_x + 1)][in_c];



                    //         const int min_f_y = -in_y;
                    //         const int min_f_x = -in_x;
                    //         const int max_f_y = h_in - in_y - 3;
                    //         const int max_f_x = w_in - in_x - 3;

                    //         #pragma unroll
                    //         for (int f_y = Utils::constexpr_max(-1 , min_f_y); f_y <= Utils::constexpr_min(1, max_f_y); f_y += 1) {
                    //             #pragma unroll
                    //             for (int f_x = Utils::constexpr_max(-1 , min_f_x); f_x <= Utils::constexpr_min(1, max_f_x); f_x += 1) {
                    //                 t_out[(in_y+f_y) * pixelsPerBlockX + (in_x+f_x)] += val * t_f[(f_y+1)*3 + f_x+1]; // scatter operation b0ss?
                    //             }
                    //         }
                    //     }
                    // }
                }
            }
        }

        if (out_c < out_C) {
            #pragma unroll
            for (int out_y = 0; out_y < pixelsPerBlockY; ++out_y) {
                const int out_y_im = out_y + tile_start_out_y;
                #pragma unroll
                for (int out_x = 0; out_x < pixelsPerBlockX; ++out_x) {
                    const int out_x_im = out_x + tile_start_out_x;
                    const bool valid = out_y_im < out_H && out_x_im < out_W;
                    if (valid) {
                        batch_out[(out_y_im * out_W + out_x_im) * out_C + out_c] = t_out[out_y*pixelsPerBlockX + out_x];
                    }
                }
            }
        }
    }
    
}



template <typename scalar_t, int WARP_SIZE=32, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
void conv3x3_increment_cuda_ext(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
){
    int const out_C = out_incr.sizes()[1], out_H=out_incr.sizes()[2], out_W=out_incr.sizes()[3];
    int const in_C = in_incr.sizes()[1], in_H=in_incr.sizes()[2], in_W=in_incr.sizes()[3];

    int constexpr threads = 256;
    int const W_up = divup(out_W, W_OUT_PER_BLOCK);
    int const H_up = divup(out_H, H_OUT_PER_BLOCK);
    int const C_up = divup(out_C, threads);
    dim3 const blocks(W_up, H_up, C_up);
    int constexpr out_channels_per_block = 32;
    // printf("kernel configuration: {%d %d %d}, {%d %d %d}\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    conv_3x3_ext<scalar_t, WARP_SIZE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<bool>(),
        in_incr.data_ptr<scalar_t>(),
        out_incr.data_ptr<scalar_t>(),
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );

    CUDA_CHECK_ERRORS();
}


void conv3x3_increment_ext_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // empty tensor;
){

    conv3x3_increment_cuda_ext<float, 32, 4, 8>(
        in_incr,
        mask,
        filter,
        out_incr  // expect a zero tensor
    );

}

