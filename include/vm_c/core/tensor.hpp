#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>
#include <functional>

#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
#include "vm_c/official/ggml_weight.hpp"
#endif

struct CUstream_st;
typedef CUstream_st* cudaStream_t;

namespace vm_c {

enum class DataType {
    FLOAT32 = 0,
    FLOAT16 = 1,
    BFLOAT16 = 2,
    FLOAT8_E4M3 = 3,
    FLOAT8_E5M2 = 4,
    INT8 = 5,
    INT32 = 6,
    INT64 = 7,
    UINT8 = 8,
    BOOL = 9,
    UINT32 = 10,
    IQ4_XS = 11,  // IQ4_XS 量化块格式，每个块 136B 代表 256 个元素
};

inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32:    return 4;
        case DataType::FLOAT16:    return 2;
        case DataType::BFLOAT16:   return 2;
        case DataType::FLOAT8_E4M3: return 1;
        case DataType::FLOAT8_E5M2: return 1;
        case DataType::INT8:       return 1;
        case DataType::INT32:      return 4;
        case DataType::INT64:      return 8;
        case DataType::UINT8:      return 1;
        case DataType::BOOL:       return 1;
        case DataType::UINT32:     return 4;
        case DataType::IQ4_XS:     return 136; // sizeof(BlockIQ4XS)
    }
    return 0;
}

using ScalarType = DataType;

struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    Shape(std::initializer_list<int64_t> l) : dims(l) {}
    Shape(std::vector<int64_t> d) : dims(std::move(d)) {}

    int64_t operator[](size_t i) const { return dims[i]; }
    size_t ndim() const { return dims.size(); }
    int64_t numel() const {
        int64_t n = 1;
        for (auto d : dims) n *= d;
        return n;
    }
};

class Tensor {
public:
    Tensor() : data_(nullptr), dtype_(DataType::FLOAT32), device_(-1), is_uva_(false) {}

    Tensor(void* data, Shape shape, DataType dtype, int device = -1)
        : data_(data), shape_(std::move(shape)), dtype_(dtype), device_(device), is_uva_(false) {}

    void* data() { return data_; }
    const void* data() const { return data_; }

    template<typename T>
    T* data_as() { return static_cast<T*>(data_); }

    template<typename T>
    const T* data_as() const { return static_cast<const T*>(data_); }

    const Shape& shape() const { return shape_; }
    DataType dtype() const { return dtype_; }
    int device() const { return device_; }
    int64_t numel() const { return shape_.numel(); }
    size_t nbytes() const { return shape_.numel() * dtype_size(dtype_); }

    bool is_cuda() const { return device_ >= 0; }
    bool is_cpu() const { return device_ < 0; }
    bool is_uva() const { return is_uva_; }
    void set_uva(bool v) { is_uva_ = v; }

    void set_data(void* data) { data_ = data; }
    void set_shape(Shape shape) { shape_ = std::move(shape); }
    void set_dtype(DataType dtype) { dtype_ = dtype; }
    void set_device(int device) { device_ = device; }

    void* data_ptr() { return data_; }
    const void* data_ptr() const { return data_; }

protected:
    void* data_;
    Shape shape_;
    DataType dtype_;
    int device_;
    bool is_uva_;
};

using TensorPtr = std::shared_ptr<Tensor>;

struct GpuTensor : public Tensor {
    GpuTensor() = default;
    GpuTensor(Shape shape, DataType dtype, int device = 0);
    ~GpuTensor();

    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;
    GpuTensor(GpuTensor&& o) noexcept;
    GpuTensor& operator=(GpuTensor&& o) noexcept;

    void* gpu_ptr() { return data(); }

#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
    void adopt_ggml_weight(std::shared_ptr<official::GgmlWeightStorage> storage,
                           Shape shape, DataType dtype, int device);
    bool has_ggml_weight() const { return static_cast<bool>(ggml_weight_); }
    official::GgmlWeightRef ggml_weight_ref() const;
#endif

    static std::shared_ptr<GpuTensor> create(Shape shape, DataType dtype, int device = 0);

#if defined(VM_C_USE_OFFICIAL_GGML_MOE) && VM_C_USE_OFFICIAL_GGML_MOE
private:
    std::shared_ptr<official::GgmlWeightStorage> ggml_weight_;
#endif
};

struct CpuTensor : public Tensor {
    CpuTensor() = default;
    CpuTensor(Shape shape, DataType dtype);
    ~CpuTensor();

    CpuTensor(const CpuTensor&) = delete;
    CpuTensor& operator=(const CpuTensor&) = delete;
    CpuTensor(CpuTensor&& o) noexcept;
    CpuTensor& operator=(CpuTensor&& o) noexcept;

    void* cpu_ptr() { return data(); }
    const void* cpu_ptr() const { return data(); }

    static std::shared_ptr<CpuTensor> create(Shape shape, DataType dtype);
};

}
