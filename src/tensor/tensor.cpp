#include "tensor.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <numeric>  // For std::accumulate
#include <random>

#include "../hal/device.h"
#include "../hal/fused_kernels.h"
#include "../memory/memory_pool.h"
#include "../quant/conversions.h"

#ifdef _MSC_VER
#include <malloc.h>
#endif

namespace serius {
namespace core {

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

Tensor::Tensor(const std::vector<size_t> &shape, hal::DType dtype)
    : data_(nullptr),
      data_gpu_(nullptr),
      location_(DataLocation::CPU),
      shape_(shape),
      strides_(compute_strides(shape)),
      size_(shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>())),
      dtype_(dtype),
      owns_data_(true),
      pool_(nullptr),
      is_gpu_pool_(false),
      is_pinned_(false)
{
    if (size_ > 0) {
        // Use backend allocator directly if no pool
        hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
        size_t bytes = size_ * element_size();
        data_ = backend->malloc(bytes);
        std::memset(data_, 0, bytes);
    }
}

Tensor::Tensor(const std::vector<size_t> &shape, memory::MemoryPool *pool, hal::DType dtype)
    : data_(nullptr),
      data_gpu_(nullptr),
      location_(DataLocation::CPU),
      shape_(shape),
      strides_(compute_strides(shape)),
      size_(shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>())),
      dtype_(dtype),
      owns_data_(true),
      pool_(pool),
      is_gpu_pool_(false),
      is_pinned_(false)
{
    if (size_ > 0) {
        size_t bytes = size_ * element_size();
        if (pool_ != nullptr) {
            data_ = pool_->allocate(bytes);
        } else {
            hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
            data_ = backend->malloc(bytes);
        }
        std::memset(data_, 0, bytes);
    }
}

Tensor::Tensor(void *data_ptr, const std::vector<size_t> &shape)
    : data_(data_ptr),
      data_gpu_(nullptr),
      location_(DataLocation::CPU),
      shape_(shape),
      strides_(compute_strides(shape)),
      size_(shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>())),
      dtype_(hal::DType::Float32),
      owns_data_(false),
      pool_(nullptr),
      is_gpu_pool_(false),
      is_pinned_(false)
{
}

Tensor::Tensor(void *data_ptr, const std::vector<size_t> &shape, const std::vector<size_t> &strides)
    : data_(data_ptr),
      data_gpu_(nullptr),
      location_(DataLocation::CPU),
      shape_(shape),
      strides_(strides),
      dtype_(hal::DType::Float32),
      owns_data_(false),
      pool_(nullptr),
      is_gpu_pool_(false),
      is_pinned_(false)
{
    size_ = 1;
    for (auto dim : shape) size_ *= dim;
}

// --- Copy/Move ---

Tensor::Tensor(const Tensor &other)
    : data_(nullptr),
      data_gpu_(nullptr),
      location_(other.location_),
      shape_(other.shape_),
      strides_(compute_strides(other.shape_)),
      size_(other.size_),
      dtype_(other.dtype_),
      owns_data_(true),
      pool_(nullptr),
      is_gpu_pool_(false),
      is_pinned_(false)
{
    if (size_ > 0) {
        hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
        size_t bytes = size_ * element_size();
        data_ = backend->malloc(bytes);

        // Copy CPU data
        if (data_ != nullptr && other.data_ != nullptr) {
            // Ensure other's CPU data is up-to-date if it was on GPU
            other.syncToHost();
            if (other.is_contiguous()) {
                std::memcpy(data_, other.data_, bytes);
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
                    this->set_item(i, other.item(indices));
                }
            }
        }

        // Copy GPU data if present in 'other'
        if (other.data_gpu_ != nullptr) {
            hal::IBackend *gpu_backend = hal::get_gpu_backend();
            if (gpu_backend != nullptr && gpu_backend->is_available()) {
                bytes = size_ * element_size();
                data_gpu_ = gpu_backend->malloc(bytes);
                if (data_gpu_ != nullptr) {
                    gpu_backend->memcpy_d2d(data_gpu_, other.data_gpu_, bytes);
                }
            }
        }
    }
    // Critical: copy location from other. If we only copied host data, we are dirty.
    // If other was on GPU, and we didn't copy GPU data, we should be marked CPU.
    if (other.data_gpu_ != nullptr && this->data_gpu_ == nullptr) {
        location_ = DataLocation::CPU;
    } else {
        location_ = other.location_;
    }
}

Tensor::Tensor(Tensor &&other) noexcept
    : data_(other.data_),
      data_gpu_(other.data_gpu_),
      location_(other.location_),
      shape_(std::move(other.shape_)),
      strides_(std::move(other.strides_)),
      size_(other.size_),
      dtype_(other.dtype_),
      owns_data_(other.owns_data_),
      pool_(other.pool_),
      is_gpu_pool_(other.is_gpu_pool_),
      is_pinned_(other.is_pinned_)
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
    if (this == &other) {
        return *this;
    }

    if (!other.owns_data_) {
        // Source is a non-owning view. We must perform a DEEP COPY to become an owner.
        // This prevents dangling pointers if the original owner of 'other' is destroyed.
        // We use a temporary to safely handle cases where 'other' is a view of '*this'.
        Tensor tmp(other); 
        *this = std::move(tmp); // Recurse: now 'other' (as 'tmp') owns_data_ is true
        return *this;
    }

    // Standard move assignment for owning tensors:
    // 1. Free existing managed resources
    free_data();

    // 2. Transfer ownership/state from 'other'
    data_ = other.data_;
    data_gpu_ = other.data_gpu_;
    location_ = other.location_;
    shape_ = std::move(other.shape_);
    strides_ = std::move(other.strides_);
    size_ = other.size_;
    dtype_ = other.dtype_;
    owns_data_ = true; // 'other' was an owner, so we are now
    pool_ = other.pool_;
    is_gpu_pool_ = other.is_gpu_pool_;
    is_pinned_ = other.is_pinned_;

    // 3. Reset 'other' so its destructor doesn't free the memory we just took
    other.data_ = nullptr;
    other.data_gpu_ = nullptr;
    other.location_ = DataLocation::CPU;
    other.size_ = 0;
    other.owns_data_ = false;
    other.pool_ = nullptr;
    other.is_gpu_pool_ = false;
    other.is_pinned_ = false;

    return *this;
}

void Tensor::free_data()
{
    if (owns_data_) {
        // Free CPU memory using backend if no pool was used
        if (data_ != nullptr && pool_ == nullptr) {
            hal::IBackend *backend = hal::get_backend(hal::DeviceType::CPU);
            if (backend) {
                if (is_pinned_) {
                    backend->free_pinned(data_);
                } else {
                    backend->free(data_);
                }
            }
            data_ = nullptr;
        } else if (data_ != nullptr && pool_ != nullptr) {
            // Memory pool resets en-masse, no per-tensor deallocate required usually
            data_ = nullptr;
        }

        if (data_gpu_ != nullptr) {
            hal::IBackend *backend = hal::get_gpu_backend();
            if (backend != nullptr) {
                backend->free(data_gpu_);
                data_gpu_ = nullptr;
            }
        }
    }
    // For non-owners, we just detach
    data_ = nullptr;
    data_gpu_ = nullptr;
    owns_data_ = false;
}

Tensor::~Tensor()
{
    free_data();
}

// --- Element Access ---

float &Tensor::operator[](size_t idx)
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("operator[] requires Float32");
    return static_cast<float *>(data_)[idx];
}

const float &Tensor::operator[](size_t idx) const
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("operator[] requires Float32");
    return static_cast<const float *>(data_)[idx];
}

float &Tensor::at(const std::vector<size_t> &indices)
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("at() requires Float32");
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }
    return static_cast<float *>(data_)[compute_offset(indices)];
}

const float &Tensor::at(const std::vector<size_t> &indices) const
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("at() requires Float32");
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Number of indices doesn't match tensor dimensions");
    }
    return static_cast<const float *>(data_)[compute_offset(indices)];
}

float &Tensor::at(size_t row, size_t col)
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("at() requires Float32");
    return static_cast<float *>(data_)[(row * strides_[0]) + (col * strides_[1])];
}

const float &Tensor::at(size_t row, size_t col) const
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("at() requires Float32");
    return static_cast<const float *>(data_)[(row * strides_[0]) + (col * strides_[1])];
}

float &Tensor::at(size_t dim0, size_t dim1, size_t dim2)
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("at() requires Float32");
    if (shape_.size() != 3) throw std::runtime_error("Tensor is not 3D");
    return static_cast<float *>(data_)[(dim0 * strides_[0]) + (dim1 * strides_[1]) + (dim2 * strides_[2])];
}

const float &Tensor::at(size_t dim0, size_t dim1, size_t dim2) const
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("at() requires Float32");
    if (shape_.size() != 3) throw std::runtime_error("Tensor is not 3D");
    return static_cast<const float *>(data_)[(dim0 * strides_[0]) + (dim1 * strides_[1]) + (dim2 * strides_[2])];
}

float &Tensor::at(size_t dim0, size_t dim1, size_t dim2, size_t dim3)
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("at() requires Float32");
    if (shape_.size() != 4) throw std::runtime_error("Tensor is not 4D");
    return static_cast<float *>(
        data_)[(dim0 * strides_[0]) + (dim1 * strides_[1]) + (dim2 * strides_[2]) + (dim3 * strides_[3])];
}

const float &Tensor::at(size_t dim0, size_t dim1, size_t dim2, size_t dim3) const
{
    if (dtype_ != hal::DType::Float32) throw std::runtime_error("at() requires Float32");
    if (shape_.size() != 4) throw std::runtime_error("Tensor is not 4D");
    return static_cast<const float *>(
        data_)[(dim0 * strides_[0]) + (dim1 * strides_[1]) + (dim2 * strides_[2]) + (dim3 * strides_[3])];
}

float Tensor::item(size_t idx) const
{
    switch (dtype_) {
        case hal::DType::Float32:
            return static_cast<const float *>(data_)[idx];
        case hal::DType::Float16:
            return quant::half_to_float(static_cast<const uint16_t *>(data_)[idx]);
        case hal::DType::BFloat16:
            return quant::bf16_to_float(static_cast<const uint16_t *>(data_)[idx]);
        default:
            throw std::runtime_error("Unsupported DType in item()");
    }
}

void Tensor::set_item(size_t idx, float val)
{
    switch (dtype_) {
        case hal::DType::Float32:
            static_cast<float *>(data_)[idx] = val;
            break;
        case hal::DType::Float16:
            static_cast<uint16_t *>(data_)[idx] = quant::float_to_half(val);
            break;
        case hal::DType::BFloat16:
            static_cast<uint16_t *>(data_)[idx] = quant::float_to_bf16(val);
            break;
        default:
            throw std::runtime_error("Unsupported DType in set_item()");
    }
}

float Tensor::item(const std::vector<size_t> &indices) const
{
    return item(compute_offset(indices));
}

void Tensor::set_item(const std::vector<size_t> &indices, float val)
{
    set_item(compute_offset(indices), val);
}

// --- Shape Operations ---

Tensor Tensor::contiguous() const
{
    if (is_contiguous()) {
        if (owns_data_) {
            Tensor view(data_, shape_);
            view.dtype_ = dtype_;
            view.data_gpu_ = data_gpu_;
            view.location_ = location_;
            view.is_pinned_ = is_pinned_;
            return view;
        }
    }

    Tensor result(shape_, pool_, dtype_);

    // Check if we can do GPU-native contiguous copy
    if (location_ == DataLocation::GPU || location_ == DataLocation::SYNCED) {
        hal::IBackend *gpu_backend = hal::get_gpu_backend();
        if (gpu_backend != nullptr && gpu_backend->is_available()) {
            // Allocate GPU memory for result
            result.syncToDevice(); 
            
            // Call our new GPU-native copy layout kernel
            hal::copy_layout(result.data_gpu_, this->data_gpu_, shape_.data(), strides_.data(), shape_.size(), dtype_);
            
            result.location_ = DataLocation::GPU;
            return result;
        }
    }

    // Fallback to CPU-side copy
    if (location_ == DataLocation::GPU) {
        syncToHost();
    }

    // Decompose and copy element by element on CPU
    std::vector<size_t> indices(shape_.size(), 0);
    for (size_t i = 0; i < size_; ++i) {
        result.set_item(i, this->item(indices));
        for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape_[d]) break;
            indices[d] = 0;
        }
    }

    if (location_ == DataLocation::GPU || location_ == DataLocation::SYNCED) {
        result.syncToDevice();
    }

    return result;
}

Tensor Tensor::expand(const std::vector<size_t> &new_shape) const
{
    if (new_shape.size() < shape_.size()) {
        throw std::invalid_argument("Cannot expand to fewer dimensions");
    }

    std::vector<size_t> new_strides(new_shape.size(), 0);
    int offset = static_cast<int>(new_shape.size() - shape_.size());

    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (i < static_cast<size_t>(offset)) {
            // New prepended dimensions are broadcasted (stride 0)
            new_strides[i] = 0;
        } else {
            size_t orig_idx = i - offset;
            if (shape_[orig_idx] == 1 && new_shape[i] > 1) {
                // Expanding dimension 1 to N -> stride 0
                new_strides[i] = 0;
            } else if (shape_[orig_idx] == new_shape[i]) {
                // Keep original stride
                new_strides[i] = strides_[orig_idx];
            } else {
                throw std::invalid_argument("Incompatible dimensions for expand");
            }
        }
    }

    Tensor res(data_, new_shape, new_strides);
    res.dtype_ = dtype_;
    return res;
}

Tensor Tensor::reshape(const std::vector<size_t> &new_shape) const
{
    size_t new_size = new_shape.empty()
                          ? 0
                          : std::accumulate(new_shape.begin(), new_shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_size != size_) {
        throw std::invalid_argument("Cannot reshape: total size mismatch");
    }
    if (!is_contiguous()) {
        std::string shape_str = "[";
        for (auto s : shape_) shape_str += std::to_string(s) + ",";
        shape_str += "]";
        std::string strides_str = "[";
        for (auto s : strides_) strides_str += std::to_string(s) + ",";
        strides_str += "]";
        throw std::invalid_argument("Cannot reshape non-contiguous tensor (make a copy first). Shape: " + shape_str + " Strides: " + strides_str);
    }
    // Return a view with new shape
    Tensor res = Tensor(data_, new_shape);
    res.dtype_ = dtype_;
    return res;
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

    // Offset the data pointer by (start * stride) elements
    size_t element_offset = start * strides_[dim];
    void *new_data = static_cast<uint8_t *>(data_) + element_offset * element_size();

    Tensor res = Tensor(new_data, new_shape, strides_);
    res.dtype_ = dtype_;
    if (this->data_gpu_) {
        res.data_gpu_ = static_cast<uint8_t *>(this->data_gpu_) + element_offset * element_size();
        res.location_ = this->location_;
    }
    return res;
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

    Tensor res = Tensor(data_, new_shape, new_strides);
    res.dtype_ = dtype_;
    if (this->data_gpu_) {
        res.data_gpu_ = this->data_gpu_;
        res.location_ = this->location_;
    }
    return res;
}

Tensor Tensor::copy() const
{
    return Tensor(*this);  // Uses copy constructor
}

Tensor Tensor::cast(hal::DType target_dtype) const
{
    if (target_dtype == dtype_) {
        return copy();  // No conversion needed
    }

    Tensor result(shape_, target_dtype);
    if (size_ == 0) {
        return result;
    }

    // Ensure CPU data is up to date
    syncToHost();

    if (dtype_ == hal::DType::Float32 && target_dtype == hal::DType::Float16) {
        quant::convert_float_to_half(static_cast<const float *>(data_), static_cast<uint16_t *>(result.raw_data()),
                                     size_);
    } else if (dtype_ == hal::DType::Float32 && target_dtype == hal::DType::BFloat16) {
        quant::convert_float_to_bf16(static_cast<const float *>(data_), static_cast<uint16_t *>(result.raw_data()),
                                     size_);
    } else if (dtype_ == hal::DType::Float16 && target_dtype == hal::DType::Float32) {
        quant::convert_half_to_float(static_cast<const uint16_t *>(data_), static_cast<float *>(result.raw_data()),
                                     size_);
    } else if (dtype_ == hal::DType::BFloat16 && target_dtype == hal::DType::Float32) {
        quant::convert_bf16_to_float(static_cast<const uint16_t *>(data_), static_cast<float *>(result.raw_data()),
                                     size_);
    } else if (dtype_ == hal::DType::Float16 && target_dtype == hal::DType::BFloat16) {
        // FP16 → FP32 → BF16 (two-step conversion)
        std::vector<float> tmp(size_);
        quant::convert_half_to_float(static_cast<const uint16_t *>(data_), tmp.data(), size_);
        quant::convert_float_to_bf16(tmp.data(), static_cast<uint16_t *>(result.raw_data()), size_);
    } else if (dtype_ == hal::DType::BFloat16 && target_dtype == hal::DType::Float16) {
        // BF16 → FP32 → FP16 (two-step conversion)
        std::vector<float> tmp(size_);
        quant::convert_bf16_to_float(static_cast<const uint16_t *>(data_), tmp.data(), size_);
        quant::convert_float_to_half(tmp.data(), static_cast<uint16_t *>(result.raw_data()), size_);
    } else {
        throw std::runtime_error("Unsupported dtype conversion");
    }

    return result;
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

    Tensor res(new_shape, nullptr, dtype_);
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
            res.set_item(0, res.item(0) + this->item(indices));
        } else {
            res.set_item(res_indices, res.item(res_indices) + this->item(indices));
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
    if (location_ == DataLocation::GPU) {
        syncToHost();
    }

    if (dtype_ == hal::DType::Float32) {
        std::fill(static_cast<float *>(data_), static_cast<float *>(data_) + size_, value);
    } else {
        uint16_t h_val = (dtype_ == hal::DType::Float16) ? quant::float_to_half(value) : quant::float_to_bf16(value);
        uint16_t *h_ptr = static_cast<uint16_t *>(data_);
        std::fill(h_ptr, h_ptr + size_, h_val);
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
    return pool != nullptr ? Tensor(shape, pool, hal::DType::Float32) : Tensor(shape, hal::DType::Float32);
}

Tensor Tensor::ones(const std::vector<size_t> &shape, memory::MemoryPool *pool)
{
    Tensor t = pool != nullptr ? Tensor(shape, pool, hal::DType::Float32) : Tensor(shape, hal::DType::Float32);
    t.ones();
    return t;
}

void Tensor::rand()
{
    syncToHost();
    const uint32_t SEED = 42;
    static std::mt19937 gen(SEED);
    std::uniform_real_distribution<float> dist(0.0F, 1.0F);
    for (size_t i = 0; i < size_; ++i) {
        this->set_item(i, dist(gen));
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

size_t Tensor::element_size() const
{
    switch (dtype_) {
        case hal::DType::Float32:
            return 4;
        case hal::DType::Float16:
        case hal::DType::BFloat16:
            return 2;
        case hal::DType::Int8:
            return 1;
        case hal::DType::Int4:
            return 1;  // Packed
        default:
            return 4;
    }
}

// --- Operators ---

Tensor &Tensor::operator+=(const Tensor &other)
{
    if (shape_ == other.shape_ && is_contiguous() && other.is_contiguous() && dtype_ == hal::DType::Float32) {
        float *t_ptr = static_cast<float *>(data_);
        float *o_ptr = static_cast<float *>(other.data_);
        for (size_t i = 0; i < size_; ++i) {
            t_ptr[i] += o_ptr[i];
        }
    } else if (shape_ == other.shape_) {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            this->set_item(indices, this->item(indices) + other.item(indices));
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
                this->set_item(i * d + j, this->item(i * d + j) + other.item(j));
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
        Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
        if (is_contiguous() && other.is_contiguous() && dtype_ == hal::DType::Float32) {
            float *r_ptr = static_cast<float *>(result.data_);
            float *t_ptr = static_cast<float *>(data_);
            float *o_ptr = static_cast<float *>(other.data_);
            for (size_t i = 0; i < size_; ++i) {
                r_ptr[i] = t_ptr[i] + o_ptr[i];
            }
        } else {
            std::vector<size_t> indices(shape_.size(), 0);
            for (size_t i = 0; i < size_; ++i) {
                result.set_item(indices, this->item(indices) + other.item(indices));
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
        Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
        size_t d = shape_.back();
        size_t n = size_ / d;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                result.set_item(i * d + j, this->item(i * d + j) + other.item(j));
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
    Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
    if (is_contiguous() && other.is_contiguous() && dtype_ == hal::DType::Float32) {
        float *r_ptr = static_cast<float *>(result.data_);
        float *t_ptr = static_cast<float *>(data_);
        float *o_ptr = static_cast<float *>(other.data_);
        for (size_t i = 0; i < size_; ++i) {
            r_ptr[i] = t_ptr[i] * o_ptr[i];
        }
    } else {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            result.set_item(indices, this->item(indices) * other.item(indices));
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }
    return result;
}

Tensor &Tensor::operator-=(const Tensor &other)
{
    if (shape_ == other.shape_ && is_contiguous() && other.is_contiguous() && dtype_ == hal::DType::Float32) {
        float *t_ptr = static_cast<float *>(data_);
        float *o_ptr = static_cast<float *>(other.data_);
        for (size_t i = 0; i < size_; ++i) {
            t_ptr[i] -= o_ptr[i];
        }
    } else if (shape_ == other.shape_) {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            this->set_item(indices, this->item(indices) - other.item(indices));
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    } else if (ndim() > other.ndim() && shape_.back() == other.shape_.back() && other.ndim() == 1) {
        size_t d = shape_.back();
        size_t n = size_ / d;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                this->set_item(i * d + j, this->item(i * d + j) - other.item(j));
            }
        }
    } else {
        throw std::invalid_argument("Shapes must match or be broadcastable for operator-=");
    }
    return *this;
}

Tensor Tensor::operator-(const Tensor &other) const
{
    if (shape_ == other.shape_) {
        Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
        if (is_contiguous() && other.is_contiguous() && dtype_ == hal::DType::Float32) {
            float *r_ptr = static_cast<float *>(result.data_);
            float *t_ptr = static_cast<float *>(data_);
            float *o_ptr = static_cast<float *>(other.data_);
            for (size_t i = 0; i < size_; ++i) {
                r_ptr[i] = t_ptr[i] - o_ptr[i];
            }
        } else {
            std::vector<size_t> indices(shape_.size(), 0);
            for (size_t i = 0; i < size_; ++i) {
                result.set_item(indices, this->item(indices) - other.item(indices));
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
        Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
        size_t d = shape_.back();
        size_t n = size_ / d;
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < d; ++j) {
                result.set_item(i * d + j, this->item(i * d + j) - other.item(j));
            }
        }
        return result;
    }

    throw std::invalid_argument("Shapes must match or be broadcastable for operator-");
}

Tensor Tensor::operator/(const Tensor &other) const
{
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Shapes must match for operator/");
    }
    Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
    if (is_contiguous() && other.is_contiguous() && dtype_ == hal::DType::Float32) {
        float *r_ptr = static_cast<float *>(result.data_);
        float *t_ptr = static_cast<float *>(data_);
        float *o_ptr = static_cast<float *>(other.data_);
        for (size_t i = 0; i < size_; ++i) {
            r_ptr[i] = t_ptr[i] / (o_ptr[i] == 0.0f ? 1e-7f : o_ptr[i]);
        }
    } else {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            float denom = other.item(indices);
            result.set_item(indices, this->item(indices) / (denom == 0.0f ? 1e-7f : denom));
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }
    return result;
}

Tensor Tensor::operator+(float scalar) const
{
    Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
    if (is_contiguous() && dtype_ == hal::DType::Float32) {
        float *r_ptr = static_cast<float *>(result.data_);
        float *t_ptr = static_cast<float *>(data_);
        for (size_t i = 0; i < size_; ++i) {
            r_ptr[i] = t_ptr[i] + scalar;
        }
    } else {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            result.set_item(indices, this->item(indices) + scalar);
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }
    return result;
}

Tensor Tensor::operator-(float scalar) const
{
    Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
    if (is_contiguous() && dtype_ == hal::DType::Float32) {
        float *r_ptr = static_cast<float *>(result.data_);
        float *t_ptr = static_cast<float *>(data_);
        for (size_t i = 0; i < size_; ++i) {
            r_ptr[i] = t_ptr[i] - scalar;
        }
    } else {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            result.set_item(indices, this->item(indices) - scalar);
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }
    return result;
}

Tensor Tensor::operator*(float scalar) const
{
    Tensor result = this->pool_ != nullptr ? Tensor(shape_, pool_, dtype_) : Tensor(shape_, dtype_);
    if (is_contiguous() && dtype_ == hal::DType::Float32) {
        float *r_ptr = static_cast<float *>(result.data_);
        float *t_ptr = static_cast<float *>(data_);
        for (size_t i = 0; i < size_; ++i) {
            r_ptr[i] = t_ptr[i] * scalar;
        }
    } else {
        std::vector<size_t> indices(shape_.size(), 0);
        for (size_t i = 0; i < size_; ++i) {
            result.set_item(indices, this->item(indices) * scalar);
            for (int d = static_cast<int>(shape_.size()) - 1; d >= 0; --d) {
                indices[d]++;
                if (indices[d] < shape_[d]) break;
                indices[d] = 0;
            }
        }
    }
    return result;
}

Tensor Tensor::operator/(float scalar) const
{
    float inv_scalar = 1.0f / (scalar == 0.0f ? 1e-7f : scalar);
    return *this * inv_scalar;
}

// --- Device Synchronization ---

// --- Device Synchronization ---

void Tensor::syncToDevice() const
{
    if (size_ == 0) return;  // Nothing to sync

    hal::IBackend *backend = hal::get_gpu_backend();
    if (backend == nullptr || !backend->is_available()) {
        return;  // No GPU available
    }

    if (data_gpu_ == nullptr) {
        const_cast<Tensor *>(this)->data_gpu_ = backend->malloc(size_ * element_size());
        if (data_gpu_ == nullptr) throw std::runtime_error("CUDA malloc failed in syncToDevice");
    }

    if (location_ == DataLocation::CPU) {
        if (data_ != nullptr && data_gpu_ != nullptr) {
            backend->memcpy_h2d(data_gpu_, data_, size_ * element_size());
            const_cast<Tensor *>(this)->location_ = DataLocation::SYNCED;
        }
    } else if (location_ == DataLocation::GPU) {
        // Already dirty on GPU, nothing to do
    } else if (location_ == DataLocation::SYNCED) {
        // Already synced, nothing to do
    }
}

void Tensor::syncToHost() const
{
    if (size_ == 0 || data_gpu_ == nullptr) return;  // Nothing to sync

    if (location_ == DataLocation::CPU || location_ == DataLocation::SYNCED) {
        return;  // Already up-to-date on CPU
    }

    hal::IBackend *gpu_backend = hal::get_gpu_backend();
    if (gpu_backend == nullptr || !gpu_backend->is_available()) {
        return;
    }

    // Ensure host memory is allocated before copying back
    if (data_ == nullptr) {
        hal::IBackend *cpu_backend = hal::get_backend(hal::DeviceType::CPU);
        const_cast<Tensor *>(this)->data_ = cpu_backend->malloc(size_ * element_size());
        if (data_ == nullptr) throw std::runtime_error("CPU malloc failed in syncToHost");
    }

    if (data_ != nullptr && data_gpu_ != nullptr) {
        gpu_backend->memcpy_d2h(data_, data_gpu_, size_ * element_size());
        const_cast<Tensor *>(this)->location_ = DataLocation::SYNCED;
    }
}

}  // namespace core
}  // namespace serius
