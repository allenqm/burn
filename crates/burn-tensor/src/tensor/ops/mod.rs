mod activation;
mod alias;
mod binary;
mod bool_tensor;
mod int_tensor;
mod modules;
mod qtensor;
mod tensor;
mod transaction;

pub use activation::*;
pub use alias::*;
pub use binary::*;
pub use bool_tensor::*;
pub use int_tensor::*;
pub use modules::*;
pub use qtensor::*;
pub use tensor::*;
pub use transaction::*;

use super::DType;

/// Error type
#[derive(Debug)]
pub enum ExternalMemoryError {
    UnsupportedMemoryType,
    AllocationFailed,
    InvalidDescriptor,
}

/// A description of GPU memory managed by a process external
/// to burn, for a variety of sources
// TODO(aqm): There shouldn't be backend specific code in this crate...
// Where would be a more fitting place to put this?
#[derive(Debug)]
pub enum ExternalMemoryDescriptor {
    /// For Cuda
    CudaPtr {
        ptr: cudarc::driver::sys::CUdeviceptr,
        size_bytes: usize,
        shape: Vec<usize>,
        elem_size: usize,
        dtype: DType,
    },
    // Future: Metal, Vulkan, etc.
}
