#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "checks.h"


// compete with this impelemntation
template<int KERNEL_SIZE=3, int pixelsPerBlockX, int pixelsPerBlockY, int OUT_CHANNELS_PER_BLOCK, bool FULL_DEPTH>
__device__ __forceinline__ void calc_tile_indices(
    int& tile_start_out_y, int& tile_start_out_x, int& tile_start_in_y, int& tile_start_in_x, int& tile_start_z, int& batch, const int out_C) {

    constexpr int PAD_LENGTH = KERNEL_SIZE/2;
    tile_start_out_y = blockIdx.y * pixelsPerBlockY;
    tile_start_out_x = blockIdx.x * pixelsPerBlockX;
    tile_start_in_y = tile_start_out_y  - PAD_LENGTH; // minus padding which is 1
    tile_start_in_x = tile_start_out_x  - PAD_LENGTH; // minus padding which is 1

    const int blocksPerBatch = divup(out_C, OUT_CHANNELS_PER_BLOCK);
    tile_start_z = (blockIdx.z % blocksPerBatch) * OUT_CHANNELS_PER_BLOCK;
    batch = blockIdx.z / blocksPerBatch;
}


template<typename scalar_t=float, int KERNEL_SIZE=3, int WARP_SIZE=32, int pixelsPerBlockX=6, int pixelsPerBlockY=6>
__device__ void gather_conv(scalar_t const *in_c_filter, scalar_t *t_out, scalar_t const dense_s_in[][WARP_SIZE], bool const smem_mask[][WARP_SIZE], 
                            int out_c, int out_C, int in_c, int in_C, int h_in, int w_in){

    int constexpr PAD_LENGTH = KERNEL_SIZE/2;

    #pragma unroll
    for(int out_y = 0; out_y < pixelsPerBlockY; out_y++ ){
        #pragma unroll
        for(int out_x = 0; out_x < pixelsPerBlockY; out_x++){
            // there's an unchecked pragma
            // smem.mask_s_in
            // if( smem_mask[WARP_SIZE*((out_y + 1)*w_in + out_x + 1) + in_c] )
            //     continue;

            scalar_t vals[KERNEL_SIZE*KERNEL_SIZE];
            #pragma unroll
            for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
                #pragma unroll
                for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
                    // this memory access load itself brings 2X speed to 1X speed :/
                    vals[f_y*KERNEL_SIZE + f_x] = dense_s_in[(out_y + f_y)*w_in + out_x + f_x][in_c];
                }
            }

            scalar_t *out_val = &t_out[out_y* pixelsPerBlockX + out_x];
            *out_val = 0.0f;

            #pragma unroll
            for(int in_y = 0; in_y < KERNEL_SIZE; in_y++){
                #pragma unroll
                for(int in_x = 0; in_x < KERNEL_SIZE; in_x++){
                   // smem.dense()
                    *out_val += vals[KERNEL_SIZE*in_y + in_x]*in_c_filter[KERNEL_SIZE*in_y + in_x];
                }

            }

        }
    }
}



template<typename scalar_t=float, int KERNEL_SIZE=3, int WARP_SIZE=32, int pixelsPerBlockX=6, int pixelsPerBlockY=6>
__device__ void scatter_conv(scalar_t const *in_c_filter, scalar_t *t_out, scalar_t const dense_s_in[][WARP_SIZE], bool const smem_mask[][WARP_SIZE], int out_c, int out_C, int in_c, int in_C ,int h_in, int w_in){

    scalar_t t_f[KERNEL_SIZE*KERNEL_SIZE];
    constexpr int PAD_LENGTH = KERNEL_SIZE/2;

    #pragma unroll
    for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
        #pragma unroll
        for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
            t_f[f_y*KERNEL_SIZE + f_x] = in_c_filter[((KERNEL_SIZE - 1 -f_y) * KERNEL_SIZE + KERNEL_SIZE - 1 - f_x) * out_C];
        }
    }


    #pragma unroll
    for (int in_y = -PAD_LENGTH; in_y < h_in -PAD_LENGTH; ++in_y) {
        #pragma unroll
        for (int in_x = -PAD_LENGTH; in_x < w_in -PAD_LENGTH; ++in_x) {

            // const bool maskval = smem.mask_s_in[(in_y + 1) * w_in + (in_x + 1)][in_c];
            // if(maskval)
            //     continue;tf*smem.dense_s_in[in_y * w_in + (in_x)][in_c];
            const scalar_t val = dense_s_in[(in_y + PAD_LENGTH) * w_in + in_x + PAD_LENGTH][in_c];

            const int min_f_y = -in_y;
            const int min_f_x = -in_x;
            const int max_f_y = h_in - in_y - 3;
            const int max_f_x = w_in - in_x - 3;


            #pragma unroll
            for (int f_y = Utils::constexpr_max(-PAD_LENGTH , min_f_y); f_y <= Utils::constexpr_min(PAD_LENGTH, max_f_y); f_y += 1) {
                #pragma unroll
                for (int f_x = Utils::constexpr_max(-PAD_LENGTH , min_f_x); f_x <= Utils::constexpr_min(PAD_LENGTH, max_f_x); f_x += 1) {
                    t_out[(in_y+f_y) * pixelsPerBlockX + (in_x+f_x)] += val * t_f[(f_y+PAD_LENGTH)*KERNEL_SIZE + f_x+1]; // scatter operation b0ss?
                }
            }
        }
    }
}




template<typename scalar_t=float, int KERNEL_SIZE=3, int WARP_SIZE=32, int pixelsPerBlockX=6, int pixelsPerBlockY=6, int OUT_CHANNELS_PER_BLOCK=256, int BLOCK_SIZE=OUT_CHANNELS_PER_BLOCK>
__global__ void conv_kxk_ext(
    const scalar_t* __restrict__ filter, //channel at the end;
    const bool*__restrict__ mask,
    const scalar_t* __restrict__ input,  // NHWC
    scalar_t* __restrict__ output,       // NHWC
    int const in_C, int const in_H, int const in_W,
    int const out_C, int const out_H, int const out_W
) {
    constexpr int PAD_LENGTH = KERNEL_SIZE/2;

    // true for full depth
    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<KERNEL_SIZE, pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, true>(
        tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, out_C
    );

    scalar_t* batch_out = output + (batch * out_H * out_W * out_C);
    const scalar_t* batch_in = input + (batch * in_H * in_W * in_C);

    const int w_in = pixelsPerBlockX + 2*PAD_LENGTH;
    const int h_in = pixelsPerBlockY + 2*PAD_LENGTH;

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
                    smem.mask_s_in[in_y * w_in + in_x][lane_idx] = mask[in_y_im * in_W * in_C + in_x_im * in_C + in_c];
                } else {
                    smem.dense_s_in[in_y * w_in + in_x][lane_idx] = 0.0f;
                    smem.mask_s_in[in_y * w_in + in_x][lane_idx] = true;
                }
            }
            __syncthreads();

            for(int in_c = 0; in_c < WARP_SIZE && in_c + in_c_off < in_C; ++in_c) {

                // do not evaluate for the whole block: 
                if(mask[tile_start_in_y * in_W * in_C + tile_start_in_x * in_C + in_c])
                    continue;

                if (out_c < out_C) {
                    scatter_conv<float, KERNEL_SIZE, WARP_SIZE, pixelsPerBlockX, pixelsPerBlockY>(
                        &filter[(in_c_off+in_c) * KERNEL_SIZE*KERNEL_SIZE * out_C + out_c], 
                        t_out, smem.dense_s_in, smem.mask_s_in, out_c, out_C, in_c, in_C, h_in, w_in);

                    // gather_conv<float, KERNEL_SIZE, WARP_SIZE, pixelsPerBlockX, pixelsPerBlockY>(
                    //     &filter[(in_c_off+in_c) * 9 * out_C + out_c], 
                    //     t_out, smem.dense_s_in, smem.mask_s_in, out_c, out_C, in_c, in_C, h_in, w_in);

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
    conv_kxk_ext<scalar_t, 3, WARP_SIZE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<bool>(),
        in_incr.data_ptr<scalar_t>(),
        out_incr.data_ptr<scalar_t>(),
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );

    CUDA_CHECK_ERRORS();
}






void convkxk_increment_ext_cuda_wrapper(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // empty tensor;
){

    // only calls in 3x3 kernes for now
    conv3x3_increment_cuda_ext<float, 32, 4, 6>(
        in_incr,
        mask,
        filter,
        out_incr  // expect a zero tensor
    );
}


