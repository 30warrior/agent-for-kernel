// Handwritten transport/compilation smoke example, not LLM-generated output.
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <arm_neon.h>

at::Tensor add(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "CPU tensors required");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat, "FP32 required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Contiguous inputs required");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1), "Expected two matrices of equal width");
    TORCH_CHECK(b.size(0) == 1 || b.size(0) == a.size(0), "Unsupported broadcast");
    auto out = at::empty_like(a);
    const auto rows = a.size(0), cols = a.size(1);
    for (int64_t row = 0; row < rows; ++row) {
        const float* x = a.data_ptr<float>() + row * cols;
        const float* y = b.data_ptr<float>() + (b.size(0) == 1 ? 0 : row * cols);
        float* z = out.data_ptr<float>() + row * cols;
        int64_t col = 0;
        for (; col + 4 <= cols; col += 4)
            vst1q_f32(z + col, vaddq_f32(vld1q_f32(x + col), vld1q_f32(y + col)));
        for (; col < cols; ++col) z[col] = x[col] + y[col];
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add", &add); }
