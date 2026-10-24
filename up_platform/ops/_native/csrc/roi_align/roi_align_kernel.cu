#include "roi_align/roi_align.h"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

using at::Tensor;

#define CUDA_1D_KERNEL_LOOP(i, n)                              \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
         i += blockDim.x * gridDim.x)

    template <typename scalar_t>
    __device__ scalar_t bilinear_interpolate(const scalar_t* bottom_data, const int height, const int width,
                                          scalar_t y, scalar_t x, const int index /* index for debug only*/) {
            // deal with cases that inverse elements are out of feature map boundary
            if (y < -1.0 || y > height || x < -1.0 || x > width) {
                // empty
                return 0;
            }

            if (y <= 0) {
                y = 0;
            }
            if (x <= 0) {
                x = 0;
            }
            
            int y_low = (int)y;
            int x_low = (int)x;
            int y_high;
            int x_high;
            
            if (y_low >= height - 1) {
                y_high = y_low = height - 1;
                y = (scalar_t)y_low;
            } else {
                y_high = y_low + 1;
            }
            
            if (x_low >= width - 1) {
                x_high = x_low = width - 1;
                x = (scalar_t)x_low;
            } else {
                x_high = x_low + 1;
            }
            
            scalar_t ly = y - y_low;
            scalar_t lx = x - x_low;
            scalar_t hy = 1. -ly, hx = 1. - lx;
            // do bilinear interpolation
            scalar_t v1 = bottom_data[y_low * width + x_low];
            scalar_t v2 = bottom_data[y_low * width + x_high];
            scalar_t v3 = bottom_data[y_high * width + x_low];
            scalar_t v4 = bottom_data[y_high * width + x_high];
            scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

            scalar_t val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

            return val;
        }

    template <typename scalar_t>
    __global__ void ROIAlignAvgForward(const bool aligned, const int nthreads, const scalar_t* bottom_data, const scalar_t spatial_scale, const int height, const int width,
                                    const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
                                    const scalar_t* bottom_rois, scalar_t* top_data) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            int pw = index % aligned_width;
            int ph = (index / aligned_width) % aligned_height;
            int c  = (index / aligned_width / aligned_height) % channels;
            int n  = index / aligned_width / aligned_height / channels;

            const scalar_t* offset_bottom_rois = bottom_rois + n * 5;
            int roi_batch_ind = offset_bottom_rois[0];

            // Do not using rounding; this implementation detail is critical
            scalar_t offset = aligned ? (scalar_t)0.5 : (scalar_t)0.0;
            scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale - offset;
            scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale - offset;
            scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale - offset;
            scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale - offset;

            // Force malformed ROIs to be 1x1
            scalar_t roi_width = roi_end_w - roi_start_w;
            scalar_t roi_height = roi_end_h - roi_start_h;
            if (!aligned) { // for backward-compatibility only
                roi_width = max(roi_width, (scalar_t)1.);
                roi_height = max(roi_height, (scalar_t)1.);
            }
            // scalar_t roi_width = fmaxf(roi_end_w - roi_start_w, 1.f);
            // scalar_t roi_height = fmaxf(roi_end_h - roi_start_h, 1.f);
            scalar_t bin_size_h = roi_height / aligned_height;
            scalar_t bin_size_w = roi_width / aligned_width;

            const scalar_t* offset_bottom_data =
                bottom_data + (roi_batch_ind * channels + c) * height * width;

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_height / aligned_height); // e.g., = 2
            int roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);

            // We do average (integral) pooling inside a bin
            const scalar_t count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

            scalar_t output_val = 0.;
            for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
            {
                const scalar_t y = roi_start_h + ph * bin_size_h +
                    (iy + .5f) * bin_size_h / roi_bin_grid_h;  // e.g., 0.5, 1.5
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const scalar_t x = roi_start_w + pw * bin_size_w +
                    (ix + .5f) * bin_size_w / roi_bin_grid_w;

                    scalar_t val = bilinear_interpolate(
                        offset_bottom_data, height, width, y, x, index);
                    output_val += val;
                }
            }
            output_val /= count;

            top_data[index] = output_val;
        }
    }

    int ROIAlignAvgForwardLaucher(const bool aligned, Tensor bottom_data, const float spatial_scale, const int num_rois, const int height, const int width,
                               const int channels, const int aligned_height, const int aligned_width,  const int sampling_ratio,
                               Tensor bottom_rois, Tensor top_data) {
        const int kThreadsPerBlock = bottom_data.scalar_type() == at::kDouble ? 256 : 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;
        const c10::cuda::CUDAGuard guard(bottom_data.device());
        const auto stream = at::cuda::getCurrentCUDAStream();

        AT_DISPATCH_FLOATING_TYPES_AND_HALF(bottom_data.scalar_type(), "roi_align_forward_cuda", ([&] {
            ROIAlignAvgForward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
              aligned, output_size, 
              bottom_data.data_ptr<scalar_t>(), 
              (scalar_t)spatial_scale, height, width, 
              channels, aligned_height, aligned_width, sampling_ratio, 
              bottom_rois.data_ptr<scalar_t>(), 
              top_data.data_ptr<scalar_t>() );
        }));
        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }

    template <typename scalar_t>
    __device__ void bilinear_interpolate_gradient(const int height, const int width, scalar_t y, scalar_t x,
                                                  scalar_t& w1, scalar_t& w2, scalar_t& w3, scalar_t& w4,
                                                  int& x_low, int& x_high, int& y_low, int& y_high,
                                                  const int index /* index for debug only*/) {
        // deal with cases that inverse elements are out of feature map boundary
        if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            w1 = w2 = w3 = w4 = 0.;
            x_low = x_high = y_low = y_high = -1;
            return;
        }

        if (y <= 0) {
            y = 0;
        }
        if (x <= 0) {
            x = 0;
        }

        y_low = (int)y;
        x_low = (int)x;

        if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (scalar_t)y_low;
        } else {
            y_high = y_low + 1;
        }

        if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (scalar_t)x_low;
        } else {
            x_high = x_low + 1;
        }

        scalar_t ly = y - y_low;
        scalar_t lx = x - x_low;
        scalar_t hy = 1. - ly, hx = 1. - lx;

        w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        return;
    }

    template <typename scalar_t>
    __global__ void ROIAlignAvgBackward(const bool aligned, const int nthreads, const scalar_t* top_diff, const scalar_t spatial_scale, const int height, const int width,
                                     const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
                                     scalar_t* bottom_diff, const scalar_t* bottom_rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            // (n, c, ph, pw) is an element in the aligned output
            int pw = index % aligned_width;
            int ph = (index / aligned_width) % aligned_height;
            int c  = (index / aligned_width / aligned_height) % channels;
            int n  = index / aligned_width / aligned_height / channels;

            const scalar_t* offset_bottom_rois = bottom_rois + n * 5;
            int roi_batch_ind = offset_bottom_rois[0];

            // Do not using rounding; this implementation detail is critical
            scalar_t offset = aligned ? (scalar_t)0.5 : (scalar_t)0.0;
            scalar_t roi_start_w = offset_bottom_rois[1] * spatial_scale - offset;
            scalar_t roi_start_h = offset_bottom_rois[2] * spatial_scale - offset;
            scalar_t roi_end_w = offset_bottom_rois[3] * spatial_scale - offset;
            scalar_t roi_end_h = offset_bottom_rois[4] * spatial_scale - offset;

            // Force malformed ROIs to be 1x1
            scalar_t roi_width = roi_end_w - roi_start_w;
            scalar_t roi_height = roi_end_h - roi_start_h;
            if (!aligned) { // for backward-compatibility only
                roi_width = fmaxf(roi_width, (scalar_t)1.);
                roi_height = fmaxf(roi_height, (scalar_t)1.);
            }
            scalar_t bin_size_h = roi_height / aligned_height;
            scalar_t bin_size_w = roi_width / aligned_width;

            scalar_t* offset_bottom_diff =
                bottom_diff + (roi_batch_ind * channels + c) * height * width;

            int top_offset = (n * channels + c) * aligned_height * aligned_width;
            const scalar_t* offset_top_diff = top_diff + top_offset;
            const scalar_t top_diff_this_bin = offset_top_diff[ph * aligned_width + pw];

            // We use roi_bin_grid to sample the grid and mimic integral
            int roi_bin_grid_h = (sampling_ratio > 0)
                ? sampling_ratio
                : ceil(roi_height / aligned_height); // e.g., = 2
            int roi_bin_grid_w =
                (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / aligned_width);

            // We do average (integral) pooling inside a bin
            const scalar_t count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

            for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
            {
                const scalar_t y = roi_start_h + ph * bin_size_h +
                    (iy + .5f) * bin_size_h / roi_bin_grid_h; // e.g., 0.5, 1.5
                for (int ix = 0; ix < roi_bin_grid_w; ix++) {
                    const scalar_t x = roi_start_w + pw * bin_size_w +
                        (ix + .5f) * bin_size_w / roi_bin_grid_w;

                    scalar_t w1, w2, w3, w4;
                    int x_low, x_high, y_low, y_high;

                    bilinear_interpolate_gradient(
                        height, width, y, x, w1, w2, w3, w4,
                        x_low, x_high, y_low, y_high, index);

                    scalar_t g1 = top_diff_this_bin * w1 / count;
                    scalar_t g2 = top_diff_this_bin * w2 / count;
                    scalar_t g3 = top_diff_this_bin * w3 / count;
                    scalar_t g4 = top_diff_this_bin * w4 / count;

                    if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
                        atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
                        atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
                        atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
                        atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
                        // gpu_atomic_add<scalar_t>(g1, offset_bottom_diff + y_low * width + x_low);
                        // gpu_atomic_add<scalar_t>(g2, offset_bottom_diff + y_low * width + x_high);
                        // gpu_atomic_add<scalar_t>(g3, offset_bottom_diff + y_high * width + x_low);
                        // gpu_atomic_add<scalar_t>(g4, offset_bottom_diff + y_high * width + x_high);
                    } // if
                } // ix
            } // iy
        } // CUDA_1D_KERNEL_LOOP
    } // RoIAlignBackward

    // Each input pixel has one writer. RoIs, bins, samples and the four bilinear
    // contributions are accumulated in a fixed order, using the original dtype.
    template <typename scalar_t>
    __global__ void ROIAlignAvgBackwardDeterministic(
        const bool aligned, const int nthreads, const scalar_t* top_diff,
        const scalar_t spatial_scale, const int num_rois, const int height,
        const int width, const int channels, const int pooled_h, const int pooled_w,
        const int sampling_ratio, scalar_t* bottom_diff, const scalar_t* rois) {
        CUDA_1D_KERNEL_LOOP(index, nthreads) {
            const int px = index % width, py = (index / width) % height;
            const int channel = (index / width / height) % channels;
            const int batch = index / width / height / channels;
            scalar_t total = bottom_diff[index];
            for (int n = 0; n < num_rois; ++n) {
                const scalar_t* roi = rois + n * 5;
                if (static_cast<int>(roi[0]) != batch) continue;
                const scalar_t offset = aligned ? scalar_t(.5) : scalar_t(0);
                const scalar_t start_x = roi[1] * spatial_scale - offset;
                const scalar_t start_y = roi[2] * spatial_scale - offset;
                scalar_t roi_w = roi[3] * spatial_scale - offset - start_x;
                scalar_t roi_h = roi[4] * spatial_scale - offset - start_y;
                if (!aligned) {
                    roi_w = fmaxf(roi_w, scalar_t(1));
                    roi_h = fmaxf(roi_h, scalar_t(1));
                }
                const scalar_t bin_h = roi_h / pooled_h, bin_w = roi_w / pooled_w;
                const int grid_h = sampling_ratio > 0 ? sampling_ratio : ceil(roi_h / pooled_h);
                const int grid_w = sampling_ratio > 0 ? sampling_ratio : ceil(roi_w / pooled_w);
                if (grid_h <= 0 || grid_w <= 0) continue;
                const scalar_t count = max(grid_h * grid_w, 1);
                int first_h = 0, last_h = pooled_h - 1, first_w = 0, last_w = pooled_w - 1;
                // A bilinear sample can affect only its adjacent pixels. Padding
                // the bin bounds retains exact-boundary samples after rounding.
                if (bin_h > scalar_t(0)) {
                    first_h = int(fminf(float(pooled_h), fmaxf(0.f, floorf(float((scalar_t(py - 1) - start_y) / bin_h)) - 1)));
                    last_h = int(fmaxf(-1.f, fminf(float(last_h), floorf(float((scalar_t(py + 1) - start_y) / bin_h)) + 1)));
                }
                if (bin_w > scalar_t(0)) {
                    first_w = int(fminf(float(pooled_w), fmaxf(0.f, floorf(float((scalar_t(px - 1) - start_x) / bin_w)) - 1)));
                    last_w = int(fmaxf(-1.f, fminf(float(last_w), floorf(float((scalar_t(px + 1) - start_x) / bin_w)) + 1)));
                }
                for (int ph = first_h; ph <= last_h; ++ph) {
                    for (int pw = first_w; pw <= last_w; ++pw) {
                        const scalar_t grad = top_diff[((n * channels + channel) * pooled_h + ph) * pooled_w + pw];
                        for (int iy = 0; iy < grid_h; ++iy) {
                            const scalar_t y = start_y + ph * bin_h + (iy + .5f) * bin_h / grid_h;
                            for (int ix = 0; ix < grid_w; ++ix) {
                                const scalar_t x = start_x + pw * bin_w + (ix + .5f) * bin_w / grid_w;
                                scalar_t w1, w2, w3, w4;
                                int xl, xh, yl, yh;
                                bilinear_interpolate_gradient(height, width, y, x, w1, w2, w3, w4, xl, xh, yl, yh, index);
                                if (xl < 0 || xh < 0 || yl < 0 || yh < 0) continue;
                                if (px == xl && py == yl) total += grad * w1 / count;
                                if (px == xh && py == yl) total += grad * w2 / count;
                                if (px == xl && py == yh) total += grad * w3 / count;
                                if (px == xh && py == yh) total += grad * w4 / count;
                            }
                        }
                    }
                }
            }
            bottom_diff[index] = total;
        }
    }

    int ROIAlignAvgBackwardLaucher(const bool aligned, Tensor top_diff, const float spatial_scale, const int batch_size, const int num_rois, const int height, const int width,
                                const int channels, const int aligned_height, const int aligned_width, const int sampling_ratio,
                                Tensor bottom_rois, Tensor bottom_diff) {
        const int kThreadsPerBlock = top_diff.scalar_type() == at::kDouble ? 256 : 1024;
        const int output_size = num_rois * aligned_height * aligned_width * channels;
        cudaError_t err;
        const c10::cuda::CUDAGuard guard(top_diff.device());
        const auto stream = at::cuda::getCurrentCUDAStream();
        if (at::globalContext().deterministicAlgorithms()) {
            const int input_size = bottom_diff.numel();
            if (!input_size) return 1;
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_diff.scalar_type(), "roi_align_backward_deterministic", ([&] {
                ROIAlignAvgBackwardDeterministic<scalar_t><<<(input_size + 255) / 256, 256, 0, stream>>>(
                    aligned, input_size, top_diff.data_ptr<scalar_t>(), scalar_t(spatial_scale),
                    num_rois, height, width, channels, aligned_height, aligned_width, sampling_ratio,
                    bottom_diff.data_ptr<scalar_t>(), bottom_rois.data_ptr<scalar_t>());
            }));
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return 1;
        }
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(top_diff.scalar_type(), "roi_align_forward_cuda", ([&] {
            ROIAlignAvgBackward<scalar_t><<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
              aligned, output_size, 
              top_diff.data_ptr<scalar_t>(), (scalar_t)spatial_scale, height, width, channels,
              aligned_height, aligned_width,  sampling_ratio, 
              bottom_diff.data_ptr<scalar_t>(), 
              bottom_rois.data_ptr<scalar_t>()
              );
        }));
        err = cudaGetLastError();
        if(cudaSuccess != err) {
            fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
            exit( -1 );
        }

        return 1;
    }
