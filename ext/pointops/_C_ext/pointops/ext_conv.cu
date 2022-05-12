#include <cuda.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "checks.h"


// compete with this implemntation
template<int KERNEL_SIZE=3, int STRIDE, int pixelsPerBlockX, int pixelsPerBlockY, int OUT_CHANNELS_PER_BLOCK>
__device__ __forceinline__ void calc_tile_indices( 
    int& tile_start_out_y, int& tile_start_out_x, int& tile_start_in_y, int& tile_start_in_x, int& tile_start_z, int& batch, const int out_C) {

    constexpr int PAD_LENGTH = (KERNEL_SIZE-1)/2;
    tile_start_out_y = blockIdx.y * pixelsPerBlockY;
    tile_start_out_x = blockIdx.x * pixelsPerBlockX;
    tile_start_in_y = tile_start_out_y*STRIDE - PAD_LENGTH;
    tile_start_in_x = tile_start_out_x*STRIDE - PAD_LENGTH;

    const int blocksPerBatch = divup(out_C, OUT_CHANNELS_PER_BLOCK);
    tile_start_z = (blockIdx.z % blocksPerBatch) * OUT_CHANNELS_PER_BLOCK;
    batch = blockIdx.z / blocksPerBatch;
}


template<typename scalar_t=float, int KERNEL_SIZE=3, int STRIDE=1, int WARP_SIZE=32, int pixelsPerBlockX=6, int pixelsPerBlockY=6>
__device__ void gather_conv(scalar_t const *in_c_filter, scalar_t *t_out, scalar_t const dense_s_in[][WARP_SIZE], 
                            int out_c, int out_C, int in_c, int in_C, int h_in, int w_in){

    int constexpr PAD_LENGTH = (KERNEL_SIZE-1)/2;

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



template<typename scalar_t=float, int KERNEL_SIZE=3, int STRIDE=1, int WARP_SIZE=32, int pixelsPerBlockX=6, int pixelsPerBlockY=6>
__device__ void scatter_conv(scalar_t const *in_c_filter, scalar_t *t_out, scalar_t const dense_s_in[][WARP_SIZE], 
                            int out_c, int out_C, int in_c_lane, int in_C ,int h_in, int w_in){

    scalar_t t_f[KERNEL_SIZE*KERNEL_SIZE];
    constexpr int PAD_LENGTH = (KERNEL_SIZE-1)/2;

    #pragma unroll
    for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
        #pragma unroll
        for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) {
            t_f[f_y*KERNEL_SIZE + f_x] = in_c_filter[((KERNEL_SIZE-1-f_y)*KERNEL_SIZE + KERNEL_SIZE-1-f_x) * out_C];
            // t_f[f_y*KERNEL_SIZE + f_x] = in_c_filter[(f_y*KERNEL_SIZE + f_x) * out_C];
        }
    }

    #pragma unroll
    for (int in_y = -PAD_LENGTH; in_y < h_in -PAD_LENGTH; ++in_y) {
        #pragma unroll
        for (int in_x = -PAD_LENGTH; in_x < w_in -PAD_LENGTH; ++in_x) {

            scalar_t const val = dense_s_in[(in_y+PAD_LENGTH)*w_in + in_x+PAD_LENGTH][in_c_lane];

            int const min_f_y = - in_y;
            int const min_f_x = - in_x;
            int const max_f_y = h_in - in_y - KERNEL_SIZE;
            int const max_f_x = w_in - in_x - KERNEL_SIZE;
            int const stride_off_y = (((PAD_LENGTH-in_y) % STRIDE) + STRIDE) % STRIDE;
            int const stride_off_x = (((PAD_LENGTH-in_x) % STRIDE) + STRIDE) % STRIDE;

            #pragma unroll
            for (int f_y = Utils::constexpr_max(-PAD_LENGTH + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(PAD_LENGTH, max_f_y); f_y += STRIDE) {
                #pragma unroll
                for (int f_x = Utils::constexpr_max(-PAD_LENGTH + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(PAD_LENGTH, max_f_x); f_x += STRIDE) {
                    t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + (in_x+f_x)/STRIDE] += val * t_f[(f_y+PAD_LENGTH)*KERNEL_SIZE + f_x + PAD_LENGTH ]; 
                    // t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + (in_x+f_x)/STRIDE] += val * t_f[(f_y+PAD_LENGTH)*KERNEL_SIZE + f_x + PAD_LENGTH ]; 
                }
            }
        }
    }
}


template<typename scalar_t=float, int KERNEL_SIZE=3, int STRIDE=1, int WARP_SIZE=32, int pixelsPerBlockX=6, int pixelsPerBlockY=6, int OUT_CHANNELS_PER_BLOCK=256, int BLOCK_SIZE=OUT_CHANNELS_PER_BLOCK>
__global__ void conv_kxk_ext(
    const scalar_t* __restrict__ filter, //channel at the end;
    const bool * __restrict__ mask,
    const scalar_t* __restrict__ input,  // NHWC
    scalar_t* __restrict__ output,       // NHWC
    int const in_C, int const in_H, int const in_W,
    int const out_C, int const out_H, int const out_W
) {
    constexpr int PAD_LENGTH = (KERNEL_SIZE-1)/2;

    // true for full depth
    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<KERNEL_SIZE, STRIDE, pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK>(
        tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, out_C);

    scalar_t* batch_out = output + (batch * out_H * out_W * out_C);
    const scalar_t* batch_in = input + (batch * in_H * in_W * in_C);

    const int w_in = pixelsPerBlockX*STRIDE - STRIDE + 1 + 2*PAD_LENGTH;
    const int h_in = pixelsPerBlockY*STRIDE - STRIDE + 1 + 2*PAD_LENGTH;

    const int n_in_px_aligned = divup(w_in * h_in, 4) * 4;

    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

    union SMEM {
        // SparseSMEM sparse;
        // bool mask_s_in[WARP_SIZE]; //bitwise operations over WARP_SIZE: implement later
        scalar_t dense_s_in[n_in_px_aligned][WARP_SIZE];
    };
    __shared__ SMEM smem;

    // TODO fix dilation and striding for sparse version
    for (int out_c_off = tile_start_z; out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK; out_c_off += BLOCK_SIZE) {

        const int out_c = out_c_off + threadIdx.x;
        scalar_t t_out[pixelsPerBlockX * pixelsPerBlockY];
        #pragma unroll
        for (int i = 0; i < pixelsPerBlockX * pixelsPerBlockY; ++i) {
            t_out[i] = 0.0f;
        }

        for (int in_c_off = 0; in_c_off < in_C; in_c_off += WARP_SIZE) {

            __syncthreads();
            int const in_c = in_c_off + lane_idx;
            bool const valid_c = in_c < in_C;// && tile_start_in_y < in_H && tile_start_in_y >= 0 && tile_start_in_x < in_W && tile_start_in_x >= 0;
            // __shared__ bool mask_s_in[WARP_SIZE];
            // mask_s_in[lane_idx] = valid_c ? mask[tile_start_in_y * in_W * in_C + tile_start_in_x * in_C + in_c] : false;
            for (int px_idx = warp_idx; px_idx < w_in * h_in; px_idx += n_warps) {
                int const in_y = px_idx / w_in;
                int const in_x = px_idx % w_in;
                int const in_y_im = in_y + tile_start_in_y;
                int const in_x_im = in_x + tile_start_in_x;

                int const valid = valid_c && in_y_im < in_H && in_x_im < in_W && in_y_im >= 0 && in_x_im >= 0;
                if (valid) {
                    smem.dense_s_in[in_y * w_in + in_x][lane_idx] = batch_in[in_y_im*in_W*in_C + in_x_im*in_C + in_c];
                } else {
                    smem.dense_s_in[in_y * w_in + in_x][lane_idx] = 0.0f;
                }
            }
            __syncthreads();

            for(int in_c = 0; (in_c < WARP_SIZE) && (in_c + in_c_off < in_C); ++in_c) {

                if (out_c < out_C /*&& mask_s_in[in_c]*/) {

                    scatter_conv<float, KERNEL_SIZE, STRIDE, WARP_SIZE, pixelsPerBlockX, pixelsPerBlockY>(
                        &filter[(in_c_off+in_c) * KERNEL_SIZE*KERNEL_SIZE * out_C + out_c], t_out, smem.dense_s_in, 
                        out_c, out_C, in_c, in_C, h_in, w_in);

                    // gather_conv<float, KERNEL_SIZE, STRIDE, WARP_SIZE, pixelsPerBlockX, pixelsPerBlockY>(
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



template <typename scalar_t, int STRIDE=1, int WARP_SIZE=32, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
static void conv3x3_increment_cuda_ext(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
){
    int const out_C = out_incr.sizes()[1], out_H=out_incr.sizes()[2], out_W=out_incr.sizes()[3];
    int const in_C = in_incr.sizes()[1], in_H=in_incr.sizes()[2], in_W=in_incr.sizes()[3];

    int constexpr threads = 256;  //block size
    int const W_up = divup(out_W, W_OUT_PER_BLOCK);
    int const H_up = divup(out_H, H_OUT_PER_BLOCK);
    int const C_up = divup(out_C, threads);
    dim3 const blocks(W_up, H_up, C_up);
    int constexpr out_channels_per_block = 256; // has to be less than or equal to threads!!!!!!!!!
    // printf("kernel configuration: {%d %d %d}, {%d %d %d}\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    conv_kxk_ext<scalar_t, 3, STRIDE, WARP_SIZE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<bool>(),
        in_incr.data_ptr<scalar_t>(),
        out_incr.data_ptr<scalar_t>(),
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );

    CUDA_CHECK_ERRORS();
}




template <typename scalar_t, int STRIDE=1, int WARP_SIZE=32, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
static void conv5x5_increment_cuda_ext(
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
    conv_kxk_ext<scalar_t, 5, STRIDE, WARP_SIZE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<bool>(),
        in_incr.data_ptr<scalar_t>(),
        out_incr.data_ptr<scalar_t>(),
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );

    CUDA_CHECK_ERRORS();
}



template <typename scalar_t, int STRIDE=1, int WARP_SIZE=32, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
static void conv1x1_increment_cuda_ext(
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
    conv_kxk_ext<scalar_t, 1, STRIDE, WARP_SIZE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<bool>(),
        in_incr.data_ptr<scalar_t>(),
        out_incr.data_ptr<scalar_t>(),
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );

    CUDA_CHECK_ERRORS();
}



template <typename scalar_t, int STRIDE=1, int WARP_SIZE=32, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
static void conv7x7_increment_cuda_ext(
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
    int constexpr out_channels_per_block = 256;
    // printf("kernel configuration: {%d %d %d}, {%d %d %d}\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    conv_kxk_ext<scalar_t, 7, STRIDE, WARP_SIZE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
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
    torch::Tensor &out_incr,  // empty tensor;
    int k,
    int stride
){
    if(k == 3){
        if(stride == 1 ){
            conv3x3_increment_cuda_ext<float, 1, 32, 5, 5>(
                in_incr,
                mask,
                filter,
                out_incr
            );
        } else if (stride == 2){
            conv3x3_increment_cuda_ext<float, 2, 32, 6, 6>(
                in_incr,
                mask,
                filter,
                out_incr
            );
        } else{
            throw std::logic_error("not implemented stride size");
        }
    } else if(k == 5){
        if(stride == 1 ){
            conv5x5_increment_cuda_ext<float, 1, 32, 6, 6>(
                in_incr,
                mask,
                filter,
                out_incr  // expect a zero tensor
            );
        } else if (stride == 2){
            conv5x5_increment_cuda_ext<float, 2, 32, 6, 6>(
                in_incr,
                mask,
                filter,
                out_incr  // expect a zero tensor
            );
        } else{
            throw std::logic_error("not implemented stride size");
        }
    } else if(k == 1 ) {
        // stride, warp size, pX, pY
        conv1x1_increment_cuda_ext<float, 1, 32, 6, 6>(
            in_incr,
            mask,
            filter,
            out_incr  // expect a zero tensor
        );
    } else if (k == 7){
        // stride == 1
        if(stride == 1){
            conv7x7_increment_cuda_ext<float, 1, 32, 5, 5>(
                in_incr,
                mask,
                filter,
                out_incr  // expect a zero tensor
            );
        }
        else if (stride == 2){
            conv7x7_increment_cuda_ext<float, 2, 32, 5, 5>(
                in_incr,
                mask,
                filter,
                out_incr  // expect a zero tensor
            );
        } else {
            throw std::logic_error("7x7 convolution not implemented for this stride .");
        }
    } else {
        throw std::logic_error("Convolution kxk not implemented for this k.");
    }

}


