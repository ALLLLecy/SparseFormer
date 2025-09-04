/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

#include <cstdio>
#include <algorithm>
#include <cstring>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THCAtomics.cuh>

#define CUDA_KERNEL_LOOP(i, n)                          \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;   \
      i < (n);                                          \
      i += blockDim.x * gridDim.x)

const int CUDA_NUM_THREADS = 1024;
inline int GET_BLOCKS(const int N, const int num_threads)
{
  return (N + num_threads - 1) / num_threads;
}


template <typename scalar_t>
__device__ scalar_t ms_deform_attn_im2col_linear(const scalar_t* &bottom_data,
                                                 const int &length, const int &nheads, const int &channels,
                                                 const scalar_t &x, const int &m, const int &c)
{
  const int x_low = floor(x);
  const int x_high = x_low + 1;

  const scalar_t lx = x - x_low;
  const scalar_t hx = 1 - lx;

  const int x_stride = nheads * channels;
  const int x_low_ptr_offset = x_low * x_stride;
  const int x_high_ptr_offset = x_low_ptr_offset + x_stride;
  const int base_ptr = m * channels + c;

  scalar_t v1 = 0;
  if (x_low >= 0 && x_low < length)
  {
    const int ptr1 = x_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
  }
  scalar_t v2 = 0;
  if (x_high >= 0 && x_high < length)
  {
    const int ptr2 = x_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
  }

  const scalar_t val = (hx * v1 + lx * v2);
  return val;
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_linear(const scalar_t* &bottom_data,
                                             const int &length, const int &nheads, const int &channels,
                                             const scalar_t &x, const int &m, const int &c,
                                             const scalar_t &top_grad,
                                             const scalar_t &attn_weight,
                                             scalar_t* &grad_value,
                                             scalar_t* grad_sampling_loc,
                                             scalar_t* grad_attn_weight)
{
  // 计算插值的低、高位置
  const int x_low = floor(x);
  const int x_high = x_low + 1;

  // 插值权重
  const scalar_t lx = x - x_low;  // 高位置的权重
  const scalar_t hx = 1 - lx;     // 低位置的权重
  // 步长计算
  const int x_stride = nheads * channels;
  const int x_low_ptr_offset = x_low * x_stride;
  const int x_high_ptr_offset = x_low_ptr_offset + x_stride;
  const int base_ptr = m * channels + c;

  // 反向传播计算
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_x_weight = 0;

  scalar_t v1 = 0;
  if (x_low >= 0 && x_low < length)
  {
    const int ptr1 = x_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_x_weight -= v1; // 低位置对 x 的梯度贡献
    atomicAdd(grad_value + ptr1, hx * top_grad_value);
  }


  scalar_t v2 = 0;
  if (x_high >= 0 && x_high < length)
  {
    const int ptr2 = x_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_x_weight += v2; // 高位置对 x 的梯度贡献
    atomicAdd(grad_value + ptr2, lx * top_grad_value);
  }

  // 计算梯度并更新
  const scalar_t val = (hx * v1 + lx * v2);
  *grad_attn_weight = top_grad * val;                 // 更新权重的梯度
  *grad_sampling_loc = length * grad_x_weight * top_grad_value; // 更新采样点位置的梯度
}


template <typename scalar_t>
__device__ void ms_deform_attn_col2im_linear_gm(const scalar_t* &bottom_data,
                                                const int &length, const int &nheads, const int &channels,
                                                const scalar_t &x, const int &m, const int &c,
                                                const scalar_t &top_grad,
                                                const scalar_t &attn_weight,
                                                scalar_t* &grad_value,
                                                scalar_t* grad_sampling_loc,
                                                scalar_t* grad_attn_weight)
{
  // 计算插值位置的低值和高值
  const int x_low = floor(x);
  const int x_high = x_low + 1;

  // 插值权重
  const scalar_t lx = x - x_low;  // 高位置的权重
  const scalar_t hx = 1 - lx;     // 低位置的权重

  // 数据指针的偏移量计算
  const int x_stride = nheads * channels;
  const int x_low_ptr_offset = x_low * x_stride;
  const int x_high_ptr_offset = x_low_ptr_offset + x_stride;
  const int base_ptr = m * channels + c;

  // 初始化梯度计算变量
  const scalar_t top_grad_value = top_grad * attn_weight;
  scalar_t grad_x_weight = 0;

  // 对低位置的梯度贡献
  scalar_t v1 = 0;
  if (x_low >= 0 && x_low < length)
  {
    const int ptr1 = x_low_ptr_offset + base_ptr;
    v1 = bottom_data[ptr1];
    grad_x_weight -= v1; // 对采样位置 x 的梯度贡献
    atomicAdd(grad_value + ptr1, hx * top_grad_value); // 累加到梯度值
  }

  // 对高位置的梯度贡献
  scalar_t v2 = 0;
  if (x_high >= 0 && x_high < length)
  {
    const int ptr2 = x_high_ptr_offset + base_ptr;
    v2 = bottom_data[ptr2];
    grad_x_weight += v2; // 对采样位置 x 的梯度贡献
    atomicAdd(grad_value + ptr2, lx * top_grad_value); // 累加到梯度值
  }

  // 计算对 attention weight 和采样位置的梯度
  const scalar_t val = (hx * v1 + lx * v2);
  atomicAdd(grad_attn_weight, top_grad * val);                 // 对 attention 权重的梯度
  atomicAdd(grad_sampling_loc, length * grad_x_weight * top_grad_value); // 对采样位置的梯度
}


template <typename scalar_t>
__global__ void ms_deformable_im2col_gpu_kernel(const int n,
                                                   const scalar_t *data_value,
                                                   const int64_t *data_spatial_shapes,
                                                   const int64_t *data_level_start_index,
                                                   const scalar_t *data_sampling_loc,
                                                   const scalar_t *data_attn_weight,
                                                   const int batch_size,
                                                   const int spatial_size,
                                                   const int num_heads,
                                                   const int channels,
                                                   const int num_levels,
                                                   const int num_query,
                                                   const int num_point,
                                                   scalar_t *data_col)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    scalar_t *data_col_ptr = data_col + index;
    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_x_ptr = sampling_index * num_levels * num_point;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;
    scalar_t col = 0;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_length = data_spatial_shapes[l_col]; // 一维长度
      const scalar_t *data_value_ptr = data_value + (data_value_ptr_init_offset + level_start_id * qid_stride);
      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_x = data_sampling_loc[data_loc_x_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t x_im = loc_x * spatial_length - 1;
        if (x_im > -1 && x_im < spatial_length)
        {
          col += ms_deform_attn_im2col_linear(data_value_ptr, spatial_length, num_heads, channels, x_im, m_col, c_col) * weight;
        }

        data_weight_ptr += 1;
        data_loc_x_ptr += 1;
      }
    }
    *data_col_ptr = col;
  }
}

template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index,
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size,
                                                const int spatial_size,
                                                const int num_heads,
                                                const int channels,
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;

    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_x_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_length = data_spatial_shapes[l_col];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_x = data_sampling_loc[data_loc_x_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t x_im = loc_x * spatial_length - 1;
        *(cache_grad_sampling_loc + threadIdx.x) = 0;
        *(cache_grad_attn_weight + threadIdx.x) = 0;

        if (x_im > -1 && x_im < spatial_length)
        {
          ms_deform_attn_col2im_linear(
            data_value_ptr, spatial_length, num_heads, channels, x_im, m_col, c_col,
            top_grad, weight, grad_value_ptr,
            cache_grad_sampling_loc+threadIdx.x, cache_grad_attn_weight+threadIdx.x);
        }

        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_x = cache_grad_sampling_loc[0];
          scalar_t _grad_a = cache_grad_attn_weight[0];
          for (unsigned int tid = 1; tid < blockSize; ++tid)
          {
            _grad_x += cache_grad_sampling_loc[tid];
            _grad_a += cache_grad_attn_weight[tid];
          }

          *grad_sampling_loc = _grad_x;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_x_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}



template <typename scalar_t, unsigned int blockSize>
__global__ void ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index,
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size,
                                                const int spatial_size,
                                                const int num_heads,
                                                const int channels,
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    __shared__ scalar_t cache_grad_sampling_loc[blockSize];
    __shared__ scalar_t cache_grad_attn_weight[blockSize];
    unsigned int tid = threadIdx.x;

    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_x_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_length = data_spatial_shapes[l_col];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_x = data_sampling_loc[data_loc_x_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t x_im = loc_x * spatial_length - 1;
        *(cache_grad_sampling_loc+threadIdx.x) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;

        if (x_im > -1 && x_im < spatial_length)
        {
          ms_deform_attn_col2im_linear(
            data_value_ptr, spatial_length, num_heads, channels, x_im, m_col, c_col,
            top_grad, weight, grad_value_ptr,
            cache_grad_sampling_loc+threadIdx.x, cache_grad_attn_weight+threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s=blockSize/2; s>0; s>>=1)
        {
          if (tid < s)
          {
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[tid] += cache_grad_sampling_loc[tid + s];
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_x_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v1(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index,
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size,
                                                const int spatial_size,
                                                const int num_heads,
                                                const int channels,
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + blockDim.x;

    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_x_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_length = data_spatial_shapes[l_col];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_x = data_sampling_loc[data_loc_x_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t x_im = loc_x * spatial_length - 1;
        *(cache_grad_sampling_loc+threadIdx.x)=0;
        *(cache_grad_attn_weight+threadIdx.x)=0;

        if (x_im > -1 && x_im < spatial_length)
        {
          ms_deform_attn_col2im_linear(
            data_value_ptr, spatial_length, num_heads, channels, x_im, m_col, c_col,
            top_grad, weight, grad_value_ptr,
            cache_grad_sampling_loc+threadIdx.x, cache_grad_attn_weight+threadIdx.x);
        }

        __syncthreads();
        if (tid == 0)
        {
          scalar_t _grad_x = cache_grad_sampling_loc[0];
          scalar_t _grad_a = cache_grad_attn_weight[0];
          for (unsigned int tid = 1; tid < blockDim.x; ++tid)
          {
            _grad_x += cache_grad_sampling_loc[tid];
            _grad_a += cache_grad_attn_weight[tid];
          }

          *grad_sampling_loc = _grad_x;
          *grad_attn_weight = _grad_a;
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_x_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index,
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size,
                                                const int spatial_size,
                                                const int num_heads,
                                                const int channels,
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_x_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col = 0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_length = data_spatial_shapes[l_col];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col = 0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_x = data_sampling_loc[data_loc_x_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t x_im = loc_x * spatial_length - 1;
        *(cache_grad_sampling_loc+threadIdx.x) = 0;
        *(cache_grad_attn_weight+threadIdx.x)=0;

        if (x_im > -1 && x_im < spatial_length)
        {
          ms_deform_attn_col2im_linear(
            data_value_ptr, spatial_length, num_heads, channels, x_im, m_col, c_col,
            top_grad, weight, grad_value_ptr,
            cache_grad_sampling_loc+threadIdx.x, cache_grad_attn_weight+threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[tid] += cache_grad_sampling_loc[tid + s];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[tid] += cache_grad_sampling_loc[tid + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          *grad_sampling_loc = cache_grad_sampling_loc[0];
          *grad_attn_weight = cache_grad_attn_weight[0];
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_x_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}

template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index,
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size,
                                                const int spatial_size,
                                                const int num_heads,
                                                const int channels,
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    extern __shared__ int _s[];
    scalar_t* cache_grad_sampling_loc = (scalar_t*)_s;
    scalar_t* cache_grad_attn_weight = cache_grad_sampling_loc + blockDim.x;
    unsigned int tid = threadIdx.x;
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_x_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_length = data_spatial_shapes[l_col];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_x = data_sampling_loc[data_loc_x_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t x_im = loc_x * spatial_length - 1;
        *(cache_grad_sampling_loc+threadIdx.x)=0;
        *(cache_grad_attn_weight+threadIdx.x)=0;

        if (x_im > -1 && x_im < spatial_length)
        {
          ms_deform_attn_col2im_linear(
            data_value_ptr, spatial_length, num_heads, channels, x_im, m_col, c_col,
            top_grad, weight, grad_value_ptr,
            cache_grad_sampling_loc+threadIdx.x, cache_grad_attn_weight+threadIdx.x);
        }

        __syncthreads();

        for (unsigned int s=blockDim.x/2, spre=blockDim.x; s>0; s>>=1, spre>>=1)
        {
          if (tid < s) {
            cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + s];
            cache_grad_sampling_loc[tid] += cache_grad_sampling_loc[tid + s];
            if (tid + (s << 1) < spre)
            {
              cache_grad_attn_weight[tid] += cache_grad_attn_weight[tid + (s << 1)];
              cache_grad_sampling_loc[tid] += cache_grad_sampling_loc[tid + (s << 1)];
            }
          }
          __syncthreads();
        }

        if (tid == 0)
        {
          atomicAdd(grad_sampling_loc, cache_grad_sampling_loc[0]);
          atomicAdd(grad_attn_weight, cache_grad_attn_weight[0]);
        }
        __syncthreads();

        data_weight_ptr += 1;
        data_loc_x_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
__global__ void ms_deformable_col2im_gpu_kernel_gm(const int n,
                                                const scalar_t *grad_col,
                                                const scalar_t *data_value,
                                                const int64_t *data_spatial_shapes,
                                                const int64_t *data_level_start_index,
                                                const scalar_t *data_sampling_loc,
                                                const scalar_t *data_attn_weight,
                                                const int batch_size,
                                                const int spatial_size,
                                                const int num_heads,
                                                const int channels,
                                                const int num_levels,
                                                const int num_query,
                                                const int num_point,
                                                scalar_t *grad_value,
                                                scalar_t *grad_sampling_loc,
                                                scalar_t *grad_attn_weight)
{
  CUDA_KERNEL_LOOP(index, n)
  {
    int _temp = index;
    const int c_col = _temp % channels;
    _temp /= channels;
    const int sampling_index = _temp;
    const int m_col = _temp % num_heads;
    _temp /= num_heads;
    const int q_col = _temp % num_query;
    _temp /= num_query;
    const int b_col = _temp;

    const scalar_t top_grad = grad_col[index];

    int data_weight_ptr = sampling_index * num_levels * num_point;
    int data_loc_x_ptr = data_weight_ptr;
    const int grad_sampling_ptr = data_weight_ptr;
    grad_sampling_loc += grad_sampling_ptr;
    grad_attn_weight += grad_sampling_ptr;
    const int grad_weight_stride = 1;
    const int grad_loc_stride = 1;
    const int qid_stride = num_heads * channels;
    const int data_value_ptr_init_offset = b_col * spatial_size * qid_stride;

    for (int l_col=0; l_col < num_levels; ++l_col)
    {
      const int level_start_id = data_level_start_index[l_col];
      const int spatial_length = data_spatial_shapes[l_col];
      const int value_ptr_offset = data_value_ptr_init_offset + level_start_id * qid_stride;
      const scalar_t *data_value_ptr = data_value + value_ptr_offset;
      scalar_t *grad_value_ptr = grad_value + value_ptr_offset;

      for (int p_col=0; p_col < num_point; ++p_col)
      {
        const scalar_t loc_x = data_sampling_loc[data_loc_x_ptr];
        const scalar_t weight = data_attn_weight[data_weight_ptr];

        const scalar_t x_im = loc_x * spatial_length - 1;
        if (x_im > -1 && x_im < spatial_length)
        {
          ms_deform_attn_col2im_linear_gm(
            data_value_ptr, spatial_length, num_heads, channels, x_im, m_col, c_col,
            top_grad, weight, grad_value_ptr,
            grad_sampling_loc, grad_attn_weight);
        }
        data_weight_ptr += 1;
        data_loc_x_ptr += 1;
        grad_attn_weight += grad_weight_stride;
        grad_sampling_loc += grad_loc_stride;
      }
    }
  }
}


template <typename scalar_t>
void ms_deformable_im2col_cuda(cudaStream_t stream,
                              const scalar_t* data_value,
                              const int64_t* data_spatial_shapes,
                              const int64_t* data_level_start_index,
                              const scalar_t* data_sampling_loc,
                              const scalar_t* data_attn_weight,
                              const int batch_size,
                              const int spatial_size,
                              const int num_heads,
                              const int channels,
                              const int num_levels,
                              const int num_query,
                              const int num_point,
                              scalar_t* data_col)
{
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  const int num_threads = CUDA_NUM_THREADS;
  ms_deformable_im2col_gpu_kernel<scalar_t>
      <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
          0, stream>>>(
      num_kernels, data_value, data_spatial_shapes, data_level_start_index, data_sampling_loc, data_attn_weight,
      batch_size, spatial_size, num_heads, channels, num_levels, num_query, num_point, data_col);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_im2col_cuda: %s\n", cudaGetErrorString(err));
  }

}

template <typename scalar_t>
void ms_deformable_col2im_cuda(cudaStream_t stream,
                              const scalar_t* grad_col,
                              const scalar_t* data_value,
                              const int64_t * data_spatial_shapes,
                              const int64_t * data_level_start_index,
                              const scalar_t * data_sampling_loc,
                              const scalar_t * data_attn_weight,
                              const int batch_size,
                              const int spatial_size,
                              const int num_heads,
                              const int channels,
                              const int num_levels,
                              const int num_query,
                              const int num_point,
                              scalar_t* grad_value,
                              scalar_t* grad_sampling_loc,
                              scalar_t* grad_attn_weight)
{
  const int num_threads = (channels > CUDA_NUM_THREADS)?CUDA_NUM_THREADS:channels;
  const int num_kernels = batch_size * num_query * num_heads * channels;
  const int num_actual_kernels = batch_size * num_query * num_heads * channels;
  if (channels > 1024)
  {
    if ((channels & 1023) == 0)
    {
      ms_deformable_col2im_gpu_kernel_shm_reduce_v2_multi_blocks<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*2*sizeof(scalar_t), stream>>>(
                        num_kernels,
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
    }
    else
    {
      ms_deformable_col2im_gpu_kernel_gm<scalar_t>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
    }
  }
  else{
    switch(channels)
    {
      case 1:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 1>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 2:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 2>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 4:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 4>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 8:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 8>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 16:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 16>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 32:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v1<scalar_t, 32>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 64:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 64>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 128:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 128>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 256:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 256>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 512:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 512>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      case 1024:
        ms_deformable_col2im_gpu_kernel_shm_blocksize_aware_reduce_v2<scalar_t, 1024>
        <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
            0, stream>>>(
                      num_kernels,
                      grad_col,
                      data_value,
                      data_spatial_shapes,
                      data_level_start_index,
                      data_sampling_loc,
                      data_attn_weight,
                      batch_size,
                      spatial_size,
                      num_heads,
                      channels,
                      num_levels,
                      num_query,
                      num_point,
                      grad_value,
                      grad_sampling_loc,
                      grad_attn_weight);
        break;
      default:
        if (channels < 64)
        {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v1<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*2*sizeof(scalar_t), stream>>>(
                        num_kernels,
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
        else
        {
          ms_deformable_col2im_gpu_kernel_shm_reduce_v2<scalar_t>
          <<<GET_BLOCKS(num_actual_kernels, num_threads), num_threads,
              num_threads*2*sizeof(scalar_t), stream>>>(
                        num_kernels,
                        grad_col,
                        data_value,
                        data_spatial_shapes,
                        data_level_start_index,
                        data_sampling_loc,
                        data_attn_weight,
                        batch_size,
                        spatial_size,
                        num_heads,
                        channels,
                        num_levels,
                        num_query,
                        num_point,
                        grad_value,
                        grad_sampling_loc,
                        grad_attn_weight);
        }
    }
  }
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    printf("error in ms_deformable_col2im_cuda: %s\n", cudaGetErrorString(err));
  }

}