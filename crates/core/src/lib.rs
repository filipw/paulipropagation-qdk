//! Pauli propagation.
//!
//! An observable is held as a sparse sum of Pauli strings and evolved backwards through a
//! circuit (Heisenberg picture): every gate `U` maps each Pauli string `P` to `U† P U`.
//! Clifford gates map one string to one string; a Pauli rotation maps a string that
//! anticommutes with its generator to two strings. Truncating the sum by Pauli weight or
//! by coefficient magnitude after every gate keeps it tractable. A state held as a Pauli
//! sum is evolved forwards the same way (Schrödinger picture).
//!
//! The semantics (rotation convention, gate order, when truncation happens, defaults)
//! follow PauliPropagation.jl, <https://github.com/SparqleSim/PauliPropagation.jl>.

mod algebra;
mod capped;
mod coeff;
mod engine;
mod gate;
mod gradient;
mod montecarlo;
mod pauli;
mod propagate;
pub mod reference;
mod sum;
mod surrogate;

pub use capped::propagate_capped;
pub use coeff::Coefficient;
pub use gate::Gate;
pub use gradient::{gradient, State};
pub use montecarlo::{
    estimate_mse, propagate_resampling, resample, sample_paths, sample_sum, PathLimits, Resampling,
    Rng,
};
pub use num_complex::Complex64;
pub use pauli::{PauliOp, PauliString};
pub use propagate::{
    propagate, propagate_in_zones, propagate_with, CustomTruncation, Error, Picture, Truncation,
    WeightDamping, Zones,
};
pub use sum::PauliSum;
pub use surrogate::Surrogate;
