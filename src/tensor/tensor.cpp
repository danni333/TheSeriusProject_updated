#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>  // For std::accumulate
#include <random>

#include "../hal/device.h"
#include "../memory/memory_pool.h"

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace serius {

// Define a named constant for memory alignment
constexpr size_t MEMORY_ALIGNMENT = 64;

// --- Static helpers ---

std::vector<size_t> Tensor::compute_strides(const std::vector<size_t> &shape)
{
    std::vector<size_t> strides(shape.size());
    if (shape.empty()) {
        return strides;
    }

    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

size_t Tensor::compute_offset(const std::vector<size_t> &indices) const
{
    size_t offset = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        offset += (indices[i] * strides_[i]);
    }
    return offset;
}

// --- Constructors ---

Tensor::Tensor(const std::vector<size_t> &shape)
    : data_(nullptr),
      data_gpu_(nullptr),
      location_(DataLocation::CPU),
      shape_(shape),
      strides_(compute_strides(shape)),
      size_(shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>())),
      owns_data_(true),
      pool_(nullptr)  // This constructor does not use a pool directly
{
    if (size_ > 0) {
        // Use backend allocator directly if no pool
        hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
        data_ = static_cast<float *>(backend->malloc(size_ * sizeof(float)));
        std::memset(data_, 0, size_ * sizeof(float));
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, memory::MemoryPool *pool)
    : data_(nullptr),  // Initialize data_ to nullptr
      data_gpu_(nullptr),
      location_(DataLocation::CPU),
      shape_(shape),
      strides_(compute_strides(shape)),
      size_(shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>())),
      owns_data_(true),
      pool_(pool)
{
    if (size_ > 0) {
        if (pool_ != nullptr) {
            data_ = static_cast<float *>(pool_->allocate(size_ * sizeof(float)));
        } else {
            hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
            data_ = static_cast<float *>(backend->malloc(size_ * sizeof(float)));
        }
        std::memset(data_, 0, size_ * sizeof(float));
    }
}

Tensor::Tensor(float *data_ptr, const std::vector<size_t> &shape)
    : data_(data_ptr),
      data_gpu_(nullptr),
      location_(DataLocation::CPU),
      shape_(shape),
      strides_(compute_strides(shape)),
      size_(shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>())),
      owns_data_(false),
      pool_(nullptr)
{
}

Tensor::Tensor(float *data_ptr, const std::vector<size_t> &shape, const std::vector<size_t> &strides)
    : data_(data_ptr),
      data_gpu_(nullptr),
      location_(DataLocation::CPU),
      shape_(shape),
      strides_(strides),
      size_(shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>())),
      owns_data_(false),
      pool_(nullptr)
{
}

// --- Copy/Move ---

Tensor::Tensor(const Tensor &other)
    : data_(nullptr),  // Initialize data_ to nullptr
      data_gpu_(nullptr),
      location_(other.location_),  // Copy location from other
      shape_(other.shape_),
      strides_(other.strides_),  // Copy strides directly, as they are already computed for 'other'
      size_(other.size_),
      owns_data_(true),
      pool_(nullptr)  // Copy constructor does not inherit pool ownership
{
    if (size_ > 0) {
        hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
        data_ = static_cast<float *>(backend->malloc(size_ * sizeof(float)));

        // Copy CPU data
        if (data_ != nullptr && other.data_ != nullptr) {
            // Ensure other's CPU data is up-to-date if it was on GPU
            other.syncToHost();
            if (other.is_contiguous()) {
                std::memcpy(data_, other.data_, size_ * sizeof(float));
            } else {
                // Element-by-element copy for non-contiguous tensors
                for (size_t i = 0; i < size_; ++i) {
                    // Decompose flat index into multi-dim indices using OTHER's layout
                    std::vector<size_t> indices(shape_.size());
                    size_t remainder = i;
                    for (size_t d = 0; d < shape_.size(); ++d) {
                        size_t stride_val = 1;
                        for (size_t dd = d + 1; dd < shape_.size(); ++dd) {
                            stride_val *= shape_[dd];
                        }
                        indices[d] = remainder / stride_val;
                        remainder %= stride_val;
                    }
                    data_[i] = other.at(indices);
                }
            }
        }

        // Copy GPU data if present in 'other'
        if (other.data_gpu_ != nullptr) {
            hal::IBackend *backend = hal::get_gpu_backend();
            if (backend != nullptr && backend->is_available()) {
                data_gpu_ = backend->malloc(size_ * sizeof(float));
                if (data_gpu_ != nullptr) {
                    backend->memcpy_d2d(data_gpu_, other.data_gpu_, size_ * sizeof(float));
                }
            }
        }
    }
}

Tensor::Tensor(Tensor &&other) noexcept
    : data_(other.data_),
      data_gpu_(other.data_gpu_),
      location_(other.location_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      size_(other.size_),
      owns_data_(other.owns_data_),
      pool_(other.pool_)
{
    other.data_ = nullptr;
    other.data_gpu_ = nullptr;
    other.location_ = DataLocation::CPU;
    other.size_ = 0;
    other.owns_data_ = false;
    other.pool_ = nullptr;
}

Tensor &Tensor::operator=(const Tensor &other)
{
    if (this != &other) {
        Tensor tmp(other);
        *this = std::move(tmp);
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    if (this != &other) {
        // Free existing resources
        if (owns_data_) {
            if (data_ != nullptr && pool_ == nullptr) {
                hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
                backend->free(data_);
            }
            if (data_gpu_ != nullptr) {
                hal::IBackend *backend = hal::get_gpu_backend();
                if (backend != nullptr) {
                    backend->free(data_gpu_);
                }
            }
        }

        data_ = other.data_;
        data_gpu_ = other.data_gpu_;
        location_ = other.location_;
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        size_ = other.size_;
        owns_data_ = other.owns_data_;
        pool_ = other.pool_;

        other.data_ = nullptr;
        other.data_gpu_ = nullptr;
        other.location_ = DataLocation::CPU;
        other.size_ = 0;
        other.owns_data_ = false;
        other.pool_ = nullptr;
    }
    return *this;
}

Tensor::~Tensor()
{
    if (owns_data_) {
        // Free CPU memory using backend if no pool was used
        if (data_ != nullptr && pool_ == nullptr) {
            hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
            backend->free(data_);
            data_ = nullptr;
        } else if (data_ != nullptr && pool_ != nullptr) {
            // Memory pool resets en-masse, no per-tensor deallocate required usually
            data_ = nullptr;
        }
    }
    if (data_gpu_ != nullptr) {
        hal::IBackend *backend = hal::get_gpu_backend();
        if (backend != nullptr) {
            backend->free(data_gpu_);
            data_gpu_ = nullptr;
        }
    }
}

// --- Element Access ---

float &Tensor::operator[](size_t idx)
{
    return data_[idx];
}

const float &Tensor::operator[](size_t idx) const
{
    return data_[idx];
}

float &Tensor::at(const std::vector<size_t> &indices)
{
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }
    return data_[compute_offset(indices)];
}

const float &Tensor::at(const std::vector<size_t> &indices) const
{
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }
    return data_[compute_offset(indices)];
}

float &Tensor::at(size_t row, size_t col)
{
    return data_[(row * strides_[0]) + (col * strides_[1])];
}

const float &Tensor::at(size_t row, size_t col) const
{
    return data_[(row * strides_[0]) + (col * strides_[1])];
}

float &Tensor::at(size_t dim0, size_t dim1, size_t dim2)
{
    return data_[(dim0 * strides_[0]) + (dim1 * strides_[1]) + (dim2 * strides_[2])];
}

const float &Tensor::at(size_t dim0, size_t dim1, size_t dim2) const
{
    return data_[(dim0 * strides_[0]) + (dim1 * strides_[1]) + (dim2 * strides_[2])];
}

// --- Shape Operations ---

Tensor Tensor::reshape(const std::vector<size_t> &new_shape) const
{
    size_t new_size = new_shape.empty()
                          ? 0
                          : std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size != size_) {
        throw std::invalid_argument("Cannot reshape: total size mismatch");
    }
    if (!is_contiguous()) {
        throw std::invalid_argument("Cannot reshape non-contiguous tensor (make a copy first)");
    }
    // Return a view with new shape
    return Tensor(data_, new_shape);
}

Tensor Tensor::slice(size_t dim, size_t start, size_t end) const
{
    if (dim >= shape_.size()) {
        throw std::out_of_range("Slice dimension out of range");
    }
    if (start >= end || end > shape_[dim]) {
        throw std::out_of_range("Slice indices out of range");
    }

    std::vector<size_t> new_shape = shape_;
    new_shape[dim] = end - start;

    // Offset the data pointer by start * stride along the sliced dimension
    float *new_data = data_ + start * strides_[dim];

    return Tensor(new_data, new_shape, strides_);
}

Tensor Tensor::transpose() const
{
    if (ndim() < 2) {
        throw std::invalid_argument("Transpose requires at least 2 dimensions");
    }
    return transpose(ndim() - 2, ndim() - 1);
}

Tensor Tensor::transpose(size_t dim0, size_t dim1) const
{
    if (dim0 >= ndim() || dim1 >= ndim()) {
        throw std::out_of_range("Transpose dimensions out of range");
    }
    std::vector<size_t> dims(ndim());
    std::iota(dims.begin(), dims.end(), 0);
    std::swap(dims[dim0], dims[dim1]);
    return permute(dims);
}

Tensor Tensor::permute(const std::vector<size_t> &dims) const
{
    if (dims.size() != ndim()) {
        throw std::invalid_argument("Permute dimensions count mismatch");
    }

    std::vector<size_t> new_shape(ndim());
    std::vector<size_t> new_strides(ndim());
    std::vector<bool> seen(ndim(), false);

    for (size_t i = 0; i < ndim(); ++i) {
        if (dims[i] >= ndim() || seen[dims[i]]) {
            throw std::invalid_argument("Invalid permute dimensions");
        }
        seen[dims[i]] = true;
        new_shape[i] = shape_[dims[i]];
        new_strides[i] = strides_[dims[i]];
    }

    return Tensor(data_, new_shape, new_strides);
}

Tensor Tensor::copy() const
{
    return Tensor(*this);  // Uses copy constructor
}

Tensor Tensor::sum(size_t dim) const
{
    if (dim >= shape_.size()) {
        throw std::out_of_range("Dimension out of range for sum");
    }

    std::vector<size_t> new_shape;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i != dim) {
            new_shape.push_back(shape_[i]);
        }
    }
    if (new_shape.empty()) {
        new_shape = {1};  // Return (1,) for scalar sum of 1D
    }

    Tensor res(new_shape);
    res.zeros();

    std::vector<size_t> indices(shape_.size(), 0);
    for (size_t i = 0; i < size_; ++i) {
        std::vector<size_t> res_indices;
        for (size_t d = 0; d < shape_.size(); ++d) {
            if (d != dim) {
                res_indices.push_back(indices[d]);
            }
        }
        if (res_indices.empty()) {
            res[0] += this->at(indices);
        } else {
            res.at(res_indices) += this->at(indices);
        }

        // Increment indices
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape_[d]) {
                break;
            }
            indices[d] = 0;
        }
    }
    return res;
}

// --- Fill & Initialize ---

void Tensor::fill(float value)
{
    if (is_contiguous()) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = value;
        }
    } else {
        // Non-contiguous fill
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            data_[compute_offset(indices)] = value;
            // Increment indices
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }
}

void Tensor::zeros()
{
    fill(0.0F);
}

void Tensor::ones()
{
    fill(1.0F);
}

Tensor Tensor::zeros(const std::vector<size_t> &shape, memory::MemoryPool *pool)
{
    // Constructor zero-initializes memory already.
    return pool != nullptr ? Tensor(shape, pool) : Tensor(shape);
}

Tensor Tensor::ones(const std::vector<size_t> &shape, memory::MemoryPool *pool)
{
    Tensor t = pool != nullptr ? Tensor(shape, pool) : Tensor(shape);
    t.ones();
    return t;
}

void Tensor::rand()
{
    const uint32_t SEED = 42;
    static std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (size_t i = 0; i < size_; ++i) {
        data_[i] = dist(gen);
    }
}

// --- Properties ---

bool Tensor::is_contiguous() const
{
    auto expected = compute_strides(shape_);
    return strides_ == expected;
}

std::string Tensor::to_string() const
{
    std::ostringstream oss;
    oss << "Tensor(shape=[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape_[i];
    }
    oss << "], size=" << size_;
    oss << ", contiguous=" << (is_contiguous() ? "true" : "false");
    oss << ", owns_data=" << (owns_data_ ? "true" : "false");
    oss << ")";
    return oss.str();
}

// --- Operators ---

Tensor &Tensor::operator+=(const Tensor &other)
{
    if (shape_ == other.shape_ && is_contiguous() && other.is_contiguous()) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] += other.data_[i];
        }
    } else if (shape_ == other.shape_) {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            this->at(indices) += other.at(indices);
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    } else if (ndim() > other.ndim() && shape_.back() == other.shape_.back() && other.ndim() == 1) {
        // Broadcasting: (..., D) += (D)
        size_t d = shape_.back();
        size_t n = size_ / d;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                data_[i * d + j] += other.at({j});
            }
        }
    } else {
        throw std::invalid_argument("Shapes must match or be broadcastable for operator+=");
    }
    return *this;
}

Tensor Tensor::operator+(const Tensor &other) const
{
    if (shape_ == other.shape_) {
        Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_) : Tensor(shape_);
        if (is_contiguous() && other.is_contiguous()) {
            for (size_t i = 0; i < size_; ++i) {
                result.data_[i] = data_[i] + other.data_[i];
            }
        } else {
            std::vector<size_t> indices(shape_.size(), 0);
            for (size_t i = 0; i < size_; ++i) {
                result.at(indices) = this->at(indices) + other.at(indices);
                for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < shape_[d]) break;
                    indices[d] = 0;
                }
            }
        }
        return result;
    }

    if (ndim() > other.ndim() && shape_.back() == other.shape_.back() && other.ndim() == 1) {
        // Broadcasting: (..., D) + (D,)
        Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_) : Tensor(shape_);
        size_t d = shape_.back();
        size_t n = size_ / d;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                result.data_[i * d + j] = data_[i * d + j] + other.at({j});
            }
        }
        return result;
    }

    throw std::invalid_argument("Shapes must match or be broadcastable for operator+");
}

Tensor Tensor::operator*(const Tensor &other) const
{
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shapes must match for operator*");
    }
    Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_) : Tensor(shape_);
    if (is_contiguous() && other.is_contiguous()) {
        for (size_t i = 0; i < size_; ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
    } else {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            result.at(indices) = this->at(indices) * other.at(indices);
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }
    return result;
}

// --- Device Synchronization ---

// --- Device Synchronization ---

void Tensor::syncToDevice() const
{
    if (location_ == DataLocation::GPU || location_ == DataLocation::SYNCED) {
        return;  // Already on GPU
    }

    hal::IBackend *backend = hal::get_gpu_backend();
    if (backend == nullptr || !backend->is_available()) {
        return;  // No GPU available
    }

    if (data_gpu_ == nullptr) {
        // We cast const away here to modify internal state safely
        const_cast<Tensor *>(this)->data_gpu_ = backend->malloc(size_ * sizeof(float));
    }

    if (data_ != nullptr && data_gpu_ != nullptr) {
        backend->memcpy_h2d(data_gpu_, data_, size_ * sizeof(float));
        const_cast<Tensor *>(this)->location_ = DataLocation::SYNCED;
    }
}

void Tensor::syncToHost() const
{
    if (location_ == DataLocation::CPU || location_ == DataLocation::SYNCED) {
        return;  // Already on CPU
    }

    hal::IBackend *backend = hal::get_gpu_backend();
    if (backend == nullptr) {
        return;
    }

    if (data_ != nullptr && data_gpu_ != nullptr) {
        backend->memcpy_d2h(data_, data_gpu_, size_ * sizeof(float));
        const_cast<Tensor *>(this)->location_ = DataLocation::SYNCED;
    }
}

}  // namespace serius
