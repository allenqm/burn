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
    /// An external source that we do not yet support
    UnsupportedMemoryType,
    /// Creation of the Cubecl handle failed
    AllocationFailed,
    /// The External Memory Descriptor Cannot be Parsed
    InvalidDescriptor,
}

/// A description of GPU memory populated by another process.  Should contain
/// all the information (pointer type, size, shape, etc.) needed to copy or
/// transfer ownership.
// TODO(aqm): Does not make sense to have cudarc in here; the runtime should be
// agnostic to implementations.
// Perhaps a genertic struct ExternalMemoryDescriptor<T> over the pointer type
// would be better
#[derive(Debug)]
pub enum ExternalMemoryDescriptor {
    /// For Cuda
    CudaPtr {
        /// Cudarc's pointer type
        ptr: cudarc::driver::sys::CUdeviceptr,
        /// Size of the memory in bytes
        size_bytes: usize,
        /// The intended shape of the Tensor that will be prodced from this memory
        shape: Vec<usize>,
        /// The size of the individual element type
        elem_size: usize,
        /// The intended DType of the tensor
        // TODO(aqm): elem size should be determined by this?
        dtype: DType,
    },
    // Future: Metal, Vulkan, etc.
}
