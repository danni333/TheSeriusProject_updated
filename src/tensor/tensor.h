#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace serius {

// Forward declaration
namespace memory {
class MemoryPool;
}

class Tensor
{
   public:
    // --- Constructors ---

    /// Create a tensor with the given shape, zero-initialized
    explicit Tensor(const std::vector<size_t> &shape);

    /// Create a tensor with the given shape, using memory pool
    Tensor(const std::vector<size_t> &shape, memory::MemoryPool *pool);

    /// Create a view on external data (does NOT own the data)
    Tensor(float *data_ptr, const std::vector<size_t> &shape);

    /// Create a view on external data with custom strides
    Tensor(float *data_ptr, const std::vector<size_t> &shape, const std::vector<size_t> &strides);

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

    /// Access element by flat index
    float &operator[](size_t idx);
    const float &operator[](size_t idx) const;

    /// Access element by multi-dimensional indices
    float &at(const std::vector<size_t> &indices);
    const float &at(const std::vector<size_t> &indices) const;

    /// 2D convenience accessor
    float &at(size_t row, size_t col);
    const float &at(size_t row, size_t col) const;

    /// 3D convenience accessor
    float &at(size_t dim0, size_t dim1, size_t dim2);
    const float &at(size_t dim0, size_t dim1, size_t dim2) const;

    /// Raw data pointer
    float *data() { return data_; }
    const float *data() const { return data_; }

    // --- Shape Operations ---

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
    Tensor operator+(const Tensor &other) const;
    Tensor operator*(const Tensor &other) const;

    // --- Properties ---

    const std::vector<size_t> &shape() const { return shape_; }
    const std::vector<size_t> &strides() const { return strides_; }
    size_t size() const { return size_; }
    size_t ndim() const { return shape_.size(); }
    bool owns_data() const { return owns_data_; }
    bool is_contiguous() const;

    /// Human-readable string
    std::string to_string() const;

   private:
    float *data_;
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    size_t size_;
    bool owns_data_;
    memory::MemoryPool *pool_;  // nullptr if heap-allocated

    /// Calculate strides from shape (row-major, C-order)
    static std::vector<size_t> compute_strides(const std::vector<size_t> &shape);

    /// Compute linear offset from multi-dimensional indices
    size_t compute_offset(const std::vector<size_t> &indices) const;
};

}  // namespace serius
