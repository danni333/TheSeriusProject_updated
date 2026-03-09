#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/serius_check.h"
#include "../hal/device.h"


namespace serius {
namespace core {

// Forward declaration
namespace memory {
class MemoryPool;
}

class Tensor
{
   public:
    // --- Constructors ---

    /// Create a tensor with the given shape, zero-initialized
    explicit Tensor(const std::vector<size_t> &shape, hal::DType dtype = hal::DType::Float32);

    /// Create a tensor with the given shape, using memory pool
    Tensor(const std::vector<size_t> &shape, memory::MemoryPool *pool, hal::DType dtype = hal::DType::Float32);

    /// Create a view on external data (does NOT own the data)
    Tensor(void *data_ptr, const std::vector<size_t> &shape);

    /// Create a view on external data with custom strides
    Tensor(void *data_ptr, const std::vector<size_t> &shape, const std::vector<size_t> &strides);

    /// Copy constructor (deep copy)
    Tensor(const Tensor &other);

    /// Move constructor
    Tensor(Tensor &&other) noexcept;

    /// Copy assignment (deep copy)
    Tensor &operator=(const Tensor &other);

    /// Move assignment
    Tensor &operator=(Tensor &&other) noexcept;

    /// Destructor
    ~Tensor();

    // --- Element Access ---

    /// Access element by flat index (Float32 only)
    float &operator[](size_t idx);
    const float &operator[](size_t idx) const;

    /// Access element by multi-dimensional indices (Float32 only)
    float &at(const std::vector<size_t> &indices);
    const float &at(const std::vector<size_t> &indices) const;

    /// 2D convenience accessor (Float32 only)
    float &at(size_t row, size_t col);
    const float &at(size_t row, size_t col) const;

    /// 3D convenience accessor (Float32 only)
    float &at(size_t dim0, size_t dim1, size_t dim2);
    const float &at(size_t dim0, size_t dim1, size_t dim2) const;

    /// 4D convenience accessor (Float32 only)
    float &at(size_t dim0, size_t dim1, size_t dim2, size_t dim3);
    const float &at(size_t dim0, size_t dim1, size_t dim2, size_t dim3) const;

    /// Value-based accessors for any DType (converts on-the-fly)
    float item(size_t idx) const;
    void set_item(size_t idx, float val);

    float item(const std::vector<size_t> &indices) const;
    void set_item(const std::vector<size_t> &indices, float val);

    float item(size_t row, size_t col) const { return item(std::vector<size_t>{row, col}); }
    float item(size_t dim0, size_t dim1, size_t dim2) const { return item(std::vector<size_t>{dim0, dim1, dim2}); }
    float item(size_t dim0, size_t dim1, size_t dim2, size_t dim3) const
    {
        return item(std::vector<size_t>{dim0, dim1, dim2, dim3});
    }

    void set_item(size_t row, size_t col, float val) { set_item(std::vector<size_t>{row, col}, val); }
    void set_item(size_t dim0, size_t dim1, size_t dim2, float val)
    {
        set_item(std::vector<size_t>{dim0, dim1, dim2}, val);
    }
    void set_item(size_t dim0, size_t dim1, size_t dim2, size_t dim3, float val)
    {
        set_item(std::vector<size_t>{dim0, dim1, dim2, dim3}, val);
    }

    /// Raw data pointer
    float *data()
    {
        if (dtype_ != hal::DType::Float32) throw std::runtime_error("Tensor::data() used on non-float32 tensor");
        return static_cast<float *>(data_);
    }
    const float *data() const
    {
        if (dtype_ != hal::DType::Float32) throw std::runtime_error("Tensor::data() const used on non-float32 tensor");
        return static_cast<const float *>(data_);
    }

    void *raw_data() { return data_; }
    const void *raw_data() const { return data_; }

    // --- Shape Operations ---
    // ... rest of the public members ...

    /// Make the tensor contiguous in memory
    Tensor contiguous() const;

    /// Broadcast tensor to a new shape
    Tensor expand(const std::vector<size_t> &new_shape) const;

    /// Reshape the tensor (in-place, no data copy if contiguous)
    Tensor reshape(const std::vector<size_t> &new_shape) const;

    /// Slice along a dimension
    Tensor slice(size_t dim, size_t start, size_t end) const;

    /// Transpose (swaps last two dimensions by default)
    Tensor transpose() const;

    /// Transpose specific dimensions
    Tensor transpose(size_t dim0, size_t dim1) const;

    /// Permute dimensions
    Tensor permute(const std::vector<size_t> &dims) const;

    /// Sum along a dimension
    Tensor sum(size_t dim) const;

    /// Deep copy
    Tensor copy() const;

    // --- Fill & Initialize ---

    /// Fill all elements with a value
    void fill(float value);

    /// Fill with zeros (in-place)
    void zeros();

    /// Fill with ones (in-place)
    void ones();

    /// Create a new tensor filled with zeros
    static Tensor zeros(const std::vector<size_t> &shape, memory::MemoryPool *pool = nullptr);

    /// Create a new tensor filled with ones
    static Tensor ones(const std::vector<size_t> &shape, memory::MemoryPool *pool = nullptr);

    /// Fill with random values in [0, 1)
    void rand();

    // --- Operators ---

    Tensor &operator+=(const Tensor &other);
    Tensor &operator-=(const Tensor &other);
    Tensor operator+(const Tensor &other) const;
    Tensor operator-(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;
    Tensor operator/(const Tensor &other) const;

    Tensor operator+(float scalar) const;
    Tensor operator-(float scalar) const;
    Tensor operator*(float scalar) const;
    Tensor operator/(float scalar) const;

    // --- Properties ---

    const std::vector<size_t> &shape() const { return shape_; }
    const std::vector<size_t> &strides() const { return strides_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    bool owns_data() const { return owns_data_; }
    bool is_contiguous() const;
    hal::DType dtype() const { return dtype_; }
    size_t element_size() const;

    /// Human-readable string
    std::string to_string() const;

    enum class DataLocation : std::uint8_t { CPU, GPU, SYNCED };

    const float *data_gpu() const { return static_cast<const float *>(data_gpu_); }
    float *data_gpu() { return static_cast<float *>(data_gpu_); }

    bool is_gpu() const { return data_gpu_ != nullptr; }

    void set_device(hal::DeviceType type)
    {
        if (type == hal::DeviceType::CUDA) {
            syncToDevice();
        } else {
            syncToHost();
        }
    }

    DataLocation location() const { return location_; }
    void mark_dirty(DataLocation loc) { location_ = loc; }

    /// Synchronize data to the compute device (GPU if available)
    void syncToDevice() const;

    /// Synchronize data back to the host (CPU)
    void syncToHost() const;

   private:
    void *data_;                     // Host pointer
    void *data_gpu_;                 // Device pointer
    mutable DataLocation location_;  // Where is the most recent data?

    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    hal::DType dtype_;
    bool owns_data_;
    memory::MemoryPool *pool_;  // nullptr if heap-allocated

    /// Calculate strides from shape (row-major, C-order)
    static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);

    /// Compute linear offset from multi-dimensional indices
    size_t compute_offset(const std::vector<size_t> &indices) const;
};

}  // namespace core
}  // namespace serius
