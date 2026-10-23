// Corrected KITTI evaluation geometry. This intentionally fixes upstream edge cases.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

namespace {
struct Point { double x, y; };

__device__ double distance(Point p, int axis, double sign, double bound) {
    return bound - sign * (axis == 0 ? p.x : p.y);
}

// Clip against a closed half-plane. Duplicate edge vertices have zero area;
// no angle sorting or strict segment-intersection test is needed.
__device__ int clip(const Point* in, int count, Point* out,
                    int axis, double sign, double bound) {
    if (!count) return 0;
    int size = 0;
    Point previous = in[count - 1];
    double dp = distance(previous, axis, sign, bound);
    for (int i = 0; i < count; ++i) {
        Point current = in[i];
        double dc = distance(current, axis, sign, bound);
        if ((dp >= 0) != (dc >= 0)) {
            double t = dp / (dp - dc);
            out[size++] = {previous.x + t * (current.x - previous.x),
                           previous.y + t * (current.y - previous.y)};
        }
        if (dc >= 0) out[size++] = current;
        previous = current;
        dp = dc;
    }
    return size;
}

__device__ double intersection(const float* a, const float* b) {
    // Work in a's local frame to avoid cancellation at large world coordinates.
    // Double intermediates are confined to this non-differentiable GPU metric.
    double ca = cos(static_cast<double>(a[4])), sa = sin(static_cast<double>(a[4]));
    double dx = static_cast<double>(b[0]) - a[0], dy = static_cast<double>(b[1]) - a[1];
    double cx = ca * dx - sa * dy, cy = sa * dx + ca * dy;
    double angle = static_cast<double>(b[4]) - a[4];
    double c = cos(angle), s = sin(angle);
    double w = static_cast<double>(b[2]) / 2, h = static_cast<double>(b[3]) / 2;
    double x[4] = {-w, -w, w, w}, y[4] = {-h, h, h, -h};
    Point p[16], q[16];
    for (int i = 0; i < 4; ++i)
        p[i] = {cx + c * x[i] + s * y[i], cy - s * x[i] + c * y[i]};
    int count = clip(p, 4, q, 0, 1, static_cast<double>(a[2]) / 2);
    count = clip(q, count, p, 0, -1, static_cast<double>(a[2]) / 2);
    count = clip(p, count, q, 1, 1, static_cast<double>(a[3]) / 2);
    count = clip(q, count, p, 1, -1, static_cast<double>(a[3]) / 2);
    double twice_area = 0;
    for (int i = 1; i + 1 < count; ++i)
        twice_area += (p[i].x - p[0].x) * (p[i + 1].y - p[0].y)
                    - (p[i].y - p[0].y) * (p[i + 1].x - p[0].x);
    return fabs(twice_area) / 2;
}

__global__ void evaluate(const float* boxes, const float* queries, float* out,
                         int64_t n, int64_t k, int criterion) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= n * k) return;
    // Preserve UP's unusual denominator convention: 0=query area, 1=box area.
    const float *a = queries + 5 * (index % k), *b = boxes + 5 * (index / k);
    double area1 = static_cast<double>(a[2]) * a[3], area2 = static_cast<double>(b[2]) * b[3];
    if (a[2] <= 0 || a[3] <= 0 || b[2] <= 0 || b[3] <= 0) {
        out[index] = 0;
        return;
    }
    double overlap = fmin(intersection(a, b), fmin(area1, area2));
    out[index] = criterion == -1 ? overlap / (area1 + area2 - overlap) :
        criterion == 0 ? overlap / area1 : criterion == 1 ? overlap / area2 : overlap;
}
}

torch::Tensor kitti_rotate_iou(torch::Tensor boxes, torch::Tensor queries, int64_t criterion) {
    TORCH_CHECK(boxes.is_cuda() && queries.is_cuda(), "KITTI IoU requires accelerator tensors");
    TORCH_CHECK(boxes.device() == queries.device(), "KITTI inputs must share a device");
    TORCH_CHECK(boxes.scalar_type() == torch::kFloat32 && queries.scalar_type() == torch::kFloat32, "KITTI IoU requires float32");
    TORCH_CHECK(boxes.is_contiguous() && queries.is_contiguous(), "KITTI inputs must be contiguous");
    TORCH_CHECK(boxes.dim() == 2 && queries.dim() == 2 && boxes.size(1) == 5 && queries.size(1) == 5, "Expected [N,5] and [K,5]");
    c10::cuda::CUDAGuard guard(boxes.device());
    auto out = torch::zeros({boxes.size(0), queries.size(0)}, boxes.options());
    if (out.numel()) {
        evaluate<<<(out.numel() + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            boxes.data_ptr<float>(), queries.data_ptr<float>(), out.data_ptr<float>(), boxes.size(0), queries.size(0), criterion);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
