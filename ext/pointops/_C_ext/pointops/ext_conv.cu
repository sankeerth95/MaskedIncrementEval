#include <torch/extension.h>
#include "utils.h"
#include "checks.h"


#define FULL_DEPTH false
#define WARP_SIZE 32 

template<int pixelsPerBlockX, int pixelsPerBlockY, int OUT_CHANNELS_PER_BLOCK, int STRIDE, int PADDING>
__device__ __forceinline__ void calc_tile_indices(int &tile_start_out_y, int &tile_start_out_x, int &tile_start_in_y, int &tile_start_in_x, int &tile_start_z, int &batch, int const out_C) {

    tile_start_out_y = blockIdx.y * pixelsPerBlockY;
    tile_start_out_x = blockIdx.x * pixelsPerBlockX;
    tile_start_in_y = tile_start_out_y * STRIDE - PADDING;
    tile_start_in_x = tile_start_out_x * STRIDE - PADDING;

    if (FULL_DEPTH) {
        tile_start_z = 0;
        batch = blockIdx.z;
    } else {
        const int blocksPerBatch = divup(out_C, OUT_CHANNELS_PER_BLOCK);
        tile_start_z = (blockIdx.z % blocksPerBatch) * OUT_CHANNELS_PER_BLOCK;
        batch = blockIdx.z / blocksPerBatch;
    }
}


template<typename scalar_t = float, int KERNEL_SIZE=3, int PADDING=1, int STRIDE=1, int pixelsPerBlockX=3, int pixelsPerBlockY=3, int OUT_CHANNELS_PER_BLOCK=32, int BLOCK_SIZE=256>
__global__ void conv_kxk_ext(
    const scalar_t * __restrict__ filter,
    const int * __restrict__  mask,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int const in_C, int const in_H, int const in_W,
    int const out_C, int const  out_H, int const  out_W
        // filter.data_ptr<scalar_t>(),
        // mask.data_ptr<int>(),
        // in_incr.data_ptr<scalar_t>(),
        // out_incr.data_ptr<scalar_t>(),
        // in_C, in_H, in_W,
        // out_C, out_H, out_W
) {

    int tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch;
    calc_tile_indices<pixelsPerBlockX, pixelsPerBlockY, OUT_CHANNELS_PER_BLOCK, STRIDE, PADDING>(tile_start_out_y, tile_start_out_x, tile_start_in_y, tile_start_in_x, tile_start_z, batch, out_C);

    scalar_t* batch_out = output + (batch * out_H * out_W * out_C);
    const scalar_t* batch_in = input + (batch * out_H * out_W * out_C);
    // const uint32_t* batch_mask = mask == nullptr ? nullptr : mask + (batch * out_H * out_W);
    
    const int n_pixels_out = pixelsPerBlockX * pixelsPerBlockY;
    const int K_HALF = (KERNEL_SIZE-1) / 2;
    const int w_in = pixelsPerBlockX + (pixelsPerBlockX-1) * (STRIDE-1) + 2 * K_HALF;
    const int h_in = pixelsPerBlockY + (pixelsPerBlockY-1) * (STRIDE-1) + 2 * K_HALF;
    const int n_in_px = w_in * h_in;
    const int in_row_vals = in_W*in_C;

    const int lane_idx = threadIdx.x % WARP_SIZE;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int sub_warp_idx = lane_idx / 8;
    const int sub_warp_lane_idx = lane_idx % 8;
    const int n_warps = BLOCK_SIZE / WARP_SIZE;

    __shared__ scalar_t s_in[n_in_px][WARP_SIZE];
    __shared__ uint32_t s_mask[n_in_px];
    // uint64_t t_mask = 0LLU;


    // TODO add sparse mode
    for (int out_c_off = tile_start_z; out_c_off < out_C && (FULL_DEPTH || out_c_off < tile_start_z + OUT_CHANNELS_PER_BLOCK); out_c_off += BLOCK_SIZE) {
        const int out_c = out_c_off + threadIdx.x;
        scalar_t t_out[n_pixels_out]; // for each thread
        // const scalar_t t_bias = bias == nullptr || out_c >= out_C ? 0.0f : bias[out_c];
        for (int i = 0; i < n_pixels_out; ++i) {
            s_mask[i] = 1u;
        }

        #pragma unroll
        for (int i = 0; i < n_pixels_out; ++i) {
            t_out[i] = 0.f;//t_bias;
        }

        for (int in_c_off = 0; in_c_off < in_C; in_c_off += WARP_SIZE) {
            __syncthreads();
            // only used vector instructions when input is aligned
            if (in_C % 4 == 0) { 
                for (int px_idx = warp_idx * 4 + sub_warp_idx; px_idx < n_in_px; px_idx += n_warps*4) {
                    const int in_y = px_idx / w_in; 
                    const int in_x = px_idx % w_in;
                    const int in_c = in_c_off + sub_warp_lane_idx * 4;

                    const int in_y_im = in_y + tile_start_in_y;
                    const int in_x_im = in_x + tile_start_in_x; 
                    const bool valid = in_c<in_C && in_y_im>=0 && in_x_im>=0 && in_y_im<in_H && in_x_im<in_W;// && is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);
                   
                    if (valid) {
                        const float4 val = reinterpret_cast<const float4*>(batch_in)[(in_y_im * in_row_vals + in_x_im * in_C + in_c) / 4];
                        reinterpret_cast<float4*>(&s_in[in_y * w_in + in_x])[sub_warp_lane_idx] = val;
                    } else {
                        s_in[in_y * w_in + in_x][sub_warp_lane_idx*4] = 0.0f;
                        s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 1] = 0.0f;
                        s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 2] = 0.0f;
                        s_in[in_y * w_in + in_x][sub_warp_lane_idx*4 + 3] = 0.0f;
                    }
                }
            } else if (in_C == 3) {
                for (int val_idx = threadIdx.x; val_idx < n_in_px * 3; val_idx += BLOCK_SIZE) {
                    const int px_idx = val_idx / 3;
                    const int in_c = val_idx % 3;
                    const int in_y = px_idx / w_in; 
                    const int in_x = px_idx % w_in;

                    const int in_y_im = in_y + tile_start_in_y;
                    const int in_x_im = in_x + tile_start_in_x; 
                    const bool valid = in_y_im>=0 && in_x_im>=0 && in_y_im<in_H && in_x_im<in_W;// && is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);

                    if (valid) {
                        s_in[in_y * w_in + in_x][in_c] = batch_in[in_y_im * in_row_vals + in_x_im * in_C + in_c];
                    } else {
                        s_in[in_y * w_in + in_x][in_c] = 0.0f;
                    }
                }
            } 
            else {
                for (int px_idx = warp_idx; px_idx < n_in_px; px_idx += n_warps) {
                    const int in_y = px_idx / w_in; 
                    const int in_x = px_idx % w_in;
                    const int in_c = in_c_off + lane_idx;

                    const int in_y_im = in_y + tile_start_in_y;
                    const int in_x_im = in_x + tile_start_in_x;
                    const bool valid = in_c<in_C && in_y_im>=0 && in_x_im>=0 && in_y_im<in_H && in_x_im<in_W;// && is_px_mask_set<w_in, n_in_px>(in_x, in_y, t_mask, s_mask);

                    if (valid) {
                        // if(in_y_im * in_row_vals + in_x_im * in_C + in_c >= in_H*in_C*in_W  || in_y_im * in_row_vals + in_x_im * in_C + in_c  < 0){
                        //     printf("the illeal access  is: %d, %d, %d, %d, %d, %d\n", in_y_im * in_row_vals + in_x_im * in_C + in_c, in_y_im , in_row_vals , in_x_im , in_C , in_c );
                        //     printf("in_y, tile_start_in_y, in_y_im: %d, %d, %d\n", in_y,  tile_start_in_y, in_y_im  );
                        //     // exit(1);
                        // }
                        s_in[in_y * w_in + in_x][lane_idx] = batch_in[in_y_im * in_row_vals + in_x_im * in_C + in_c];
                    } else {
                        s_in[in_y * w_in + in_x][lane_idx] = 0.0f;
                    }
                    
                }
            }
            __syncthreads();
            

            if (out_c < out_C) {
                for(int in_c = 0; in_c < 32 && in_c + in_c_off < in_C; ++in_c) {
                    const scalar_t *in_c_filter = &filter[(in_c_off+in_c) * KERNEL_SIZE*KERNEL_SIZE * out_C + out_c];
                    scalar_t t_f[KERNEL_SIZE*KERNEL_SIZE];
                    #pragma unroll
                    for(int f_y = 0; f_y < KERNEL_SIZE; ++f_y) {
                        #pragma unroll
                        for(int f_x = 0; f_x < KERNEL_SIZE; ++f_x) { 
                            t_f[f_y*KERNEL_SIZE + f_x] = in_c_filter[((KERNEL_SIZE-1-f_y) * KERNEL_SIZE + (KERNEL_SIZE-1) - f_x) * out_C];
                        }
                    }

                    // loop split this thing to tile and skip convolution
                    #pragma unroll
                    for (int in_y = -K_HALF; in_y < h_in - K_HALF; ++in_y) {
                        #pragma unroll
                        for (int in_x = -K_HALF; in_x < w_in - K_HALF; ++in_x) {
                            const scalar_t val = s_in[(in_y+K_HALF) * w_in + (in_x+K_HALF)][in_c];

                            // TODO try to skip pixels where mask is not set -> might be worth it in this 7x7 kernel
                            const int min_f_y = -in_y;
                            const int min_f_x = -in_x;
                            const int max_f_y = h_in - in_y - KERNEL_SIZE;
                            const int max_f_x = w_in - in_x - KERNEL_SIZE;                        
                            const int stride_off_y = (((-in_y + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                            const int stride_off_x = (((-in_x + K_HALF) % STRIDE) + STRIDE) % STRIDE;
                            #pragma unroll
                            for (int f_y = Utils::constexpr_max(-K_HALF + stride_off_y, min_f_y); f_y <= Utils::constexpr_min(K_HALF, max_f_y); f_y += STRIDE) {
                                #pragma unroll
                                for (int f_x = Utils::constexpr_max(-K_HALF + stride_off_x, min_f_x); f_x <= Utils::constexpr_min(K_HALF, max_f_x); f_x += STRIDE) {

                                    // printf("%d, %d\n", f_x, f_y);

                                    t_out[((in_y+f_y)/STRIDE) * pixelsPerBlockX + ((in_x+f_x)/STRIDE)] += val * t_f[(f_y+K_HALF)*KERNEL_SIZE + f_x+K_HALF];
                                }
                            }
                        }
                    }
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



template <typename scalar_t, int STRIDE=1, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
static void conv3x3_increment_cuda_ext(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
){
    int const out_C = out_incr.sizes()[1], out_H=out_incr.sizes()[2], out_W=out_incr.sizes()[3];
    int const in_C = in_incr.sizes()[1], in_H=in_incr.sizes()[2], in_W=in_incr.sizes()[3];

    int constexpr threads = 256;  //block size
    int constexpr out_channels_per_block = 256; 

    int const W_up = divup(out_W, W_OUT_PER_BLOCK);
    int const H_up = divup(out_H, H_OUT_PER_BLOCK);
    int const C_up = divup(out_C, out_channels_per_block);
    dim3 const blocks(W_up, H_up, C_up);
    conv_kxk_ext<scalar_t, 3, 1, STRIDE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<int>(),
        in_incr.data_ptr<scalar_t>(),
        out_incr.data_ptr<scalar_t>(),
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );
    // cudaDeviceSynchronize();
    // CUDA_CHECK_ERRORS();   
}


template <typename scalar_t, int STRIDE=1, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
static void conv5x5_increment_cuda_ext(
    torch::Tensor const &in_incr,
    torch::Tensor const &mask,
    torch::Tensor const &filter,
    torch::Tensor &out_incr  // expect a zero tensor
){
    int const out_C = out_incr.sizes()[1], out_H=out_incr.sizes()[2], out_W=out_incr.sizes()[3];
    int const in_C = in_incr.sizes()[1], in_H=in_incr.sizes()[2], in_W=in_incr.sizes()[3];

    int constexpr threads = 256;
    int constexpr out_channels_per_block = 256;
    int const W_up = divup(out_W, W_OUT_PER_BLOCK);
    int const H_up = divup(out_H, H_OUT_PER_BLOCK);
    int const C_up = divup(out_C, out_channels_per_block);
    dim3 const blocks(W_up, H_up, C_up);
    // printf("kernel configuration: {%d %d %d}, {%d %d %d}\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
    conv_kxk_ext<scalar_t, 5,2, STRIDE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<int>(),
        in_incr.data_ptr<scalar_t>(),
        out_incr.data_ptr<scalar_t>(),
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );

    CUDA_CHECK_ERRORS();
}



template <typename scalar_t, int STRIDE=1, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
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
    conv_kxk_ext<scalar_t, 1,2, STRIDE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<int>(),
        in_incr.data_ptr<scalar_t>(),
        out_incr.data_ptr<scalar_t>(),
        in_C, in_H, in_W,
        out_C, out_H, out_W
    );

    CUDA_CHECK_ERRORS();
}



template <typename scalar_t, int STRIDE=1, int H_OUT_PER_BLOCK=6, int W_OUT_PER_BLOCK=6>
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
    conv_kxk_ext<scalar_t, 7, 3, STRIDE, W_OUT_PER_BLOCK, H_OUT_PER_BLOCK, out_channels_per_block, threads> <<<blocks, threads>>>(
        filter.data_ptr<scalar_t>(),
        mask.data_ptr<int>(),
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
            conv3x3_increment_cuda_ext<float, 1, 8, 8>(
                in_incr,
                mask,
                filter,
                out_incr
            );
        } else if (stride == 2){
            conv3x3_increment_cuda_ext<float, 2, 3, 3>(
                in_incr,
                mask,
                filter,
                out_incr
            );
        } else {
            throw std::logic_error("not implemented stride size");
        }
    } else if(k == 5){
        if(stride == 1 ){
            conv5x5_increment_cuda_ext<float, 1, 5, 5>(
                in_incr,
                mask,
                filter,
                out_incr  // expect a zero tensor
            );
        } else if (stride == 2){
            conv5x5_increment_cuda_ext<float, 2, 5, 5>(
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
        conv1x1_increment_cuda_ext<float, 1, 8, 8>(
            in_incr,
            mask,
            filter,
            out_incr  // expect a zero tensor
        );
    } else if (k == 7){
        // stride == 1
        if(stride == 1){
            conv7x7_increment_cuda_ext<float, 1, 5, 5>(
                in_incr,
                mask,
                filter,
                out_incr  // expect a zero tensor
            );
        }
        else if (stride == 2){
            conv7x7_increment_cuda_ext<float, 2, 5, 5>(
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



