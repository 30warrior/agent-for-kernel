// Handwritten transport/compilation smoke example, not LLM-generated output.
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
#include <arm_sve.h>

at::Tensor add(const at::Tensor& a, const at::Tensor& b) {
    TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "CPU tensors required");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat, "FP32 required");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Contiguous inputs required");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(1), "Expected two matrices of equal width");
    TORCH_CHECK(b.size(0) == 1 || b.size(0) == a.size(0), "Unsupported broadcast");
    auto out = at::empty_like(a);
    const uint64_t cols = a.size(1);
    for (int64_t row = 0; row < a.size(0); ++row) {
        const float* x = a.data_ptr<float>() + row * cols;
        const float* y = b.data_ptr<float>() + (b.size(0) == 1 ? 0 : row * cols);
        float* z = out.data_ptr<float>() + row * cols;
        for (uint64_t col = 0; col < cols; col += svcntw()) {
            svbool_t pg = svwhilelt_b32(col, cols);
            svst1_f32(pg, z + col, svadd_f32_x(pg, svld1_f32(pg, x + col), svld1_f32(pg, y + col)));
        }
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add", &add); }
