//! Coefficients of a Pauli sum.

use num_complex::Complex64;
use std::fmt::Debug;

/// A number that can be the coefficient of a Pauli string: a real or a complex number.
///
/// Gates act on coefficients only through real factors and sums, so an observable with
/// complex coefficients propagates like its real and imaginary parts.
pub trait Coefficient: Copy + Debug + PartialEq + Send + Sync + 'static {
    const ZERO: Self;

    fn from_real(value: f64) -> Self;

    /// The product with a real number.
    #[must_use]
    fn scale(self, factor: f64) -> Self;

    #[must_use]
    fn plus(self, other: Self) -> Self;

    /// The product of two coefficients, without conjugation.
    #[must_use]
    fn times(self, other: Self) -> Self;

    fn abs(self) -> f64;

    fn norm_sqr(self) -> f64;

    /// Whether the absolute value is below `threshold`.
    fn is_below(self, threshold: f64) -> bool;
}

impl Coefficient for f64 {
    const ZERO: Self = 0.0;

    #[inline]
    fn from_real(value: f64) -> Self {
        value
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        self * factor
    }

    #[inline]
    fn plus(self, other: Self) -> Self {
        self + other
    }

    #[inline]
    fn times(self, other: Self) -> Self {
        self * other
    }

    #[inline]
    fn abs(self) -> f64 {
        f64::abs(self)
    }

    #[inline]
    fn norm_sqr(self) -> f64 {
        self * self
    }

    #[inline]
    fn is_below(self, threshold: f64) -> bool {
        f64::abs(self) < threshold
    }
}

impl Coefficient for Complex64 {
    const ZERO: Self = Complex64::new(0.0, 0.0);

    #[inline]
    fn from_real(value: f64) -> Self {
        Complex64::new(value, 0.0)
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        Complex64::new(self.re * factor, self.im * factor)
    }

    #[inline]
    fn plus(self, other: Self) -> Self {
        self + other
    }

    #[inline]
    fn times(self, other: Self) -> Self {
        self * other
    }

    #[inline]
    fn abs(self) -> f64 {
        self.norm()
    }

    #[inline]
    fn norm_sqr(self) -> f64 {
        Complex64::norm_sqr(&self)
    }

    /// Compares the squares, as PauliPropagation.jl does for complex coefficients.
    #[inline]
    fn is_below(self, threshold: f64) -> bool {
        Complex64::norm_sqr(&self) < threshold * threshold
    }
}

/// What the propagation keeps per Pauli string: a coefficient, and for some truncations a
/// record of the rotations that produced it.
pub(crate) trait Path: Copy + Send + Sync {
    type Coeff: Coefficient;

    fn start(coeff: Self::Coeff) -> Self;

    fn coeff(&self) -> Self::Coeff;

    /// The product with a real number that is not a branch of a rotation.
    fn scale(self, factor: f64) -> Self;

    /// The branch of a rotation that keeps the Pauli string, with the factor `cos`.
    fn cos_branch(self, cos: f64) -> Self;

    /// The branch of a rotation that changes the Pauli string, with the factor `sin * sign`.
    fn sin_branch(self, sin: f64, sign: f64) -> Self;

    /// Two paths that end on the same Pauli string.
    fn merge(self, other: Self) -> Self;

    /// The number of rotations that branched on the way, sine and cosine branches alike.
    fn freq(&self) -> u32;

    /// The number of sine branches on the way.
    fn sins(&self) -> u32;
}

impl<C: Coefficient> Path for C {
    type Coeff = C;

    #[inline]
    fn start(coeff: C) -> Self {
        coeff
    }

    #[inline]
    fn coeff(&self) -> C {
        *self
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        Coefficient::scale(self, factor)
    }

    #[inline]
    fn cos_branch(self, cos: f64) -> Self {
        Coefficient::scale(self, cos)
    }

    #[inline]
    fn sin_branch(self, sin: f64, sign: f64) -> Self {
        Coefficient::scale(Coefficient::scale(self, sin), sign)
    }

    #[inline]
    fn merge(self, other: Self) -> Self {
        self.plus(other)
    }

    #[inline]
    fn freq(&self) -> u32 {
        0
    }

    #[inline]
    fn sins(&self) -> u32 {
        0
    }
}

/// A coefficient with the counters of PauliPropagation.jl's `PauliFreqTracker`. Where paths
/// merge, the coefficients add and each counter takes the smaller value.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Tracked<C> {
    coeff: C,
    freq: u32,
    sins: u32,
}

impl<C: Coefficient> Path for Tracked<C> {
    type Coeff = C;

    #[inline]
    fn start(coeff: C) -> Self {
        Self {
            coeff,
            freq: 0,
            sins: 0,
        }
    }

    #[inline]
    fn coeff(&self) -> C {
        self.coeff
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        Self {
            coeff: self.coeff.scale(factor),
            ..self
        }
    }

    #[inline]
    fn cos_branch(self, cos: f64) -> Self {
        Self {
            coeff: self.coeff.scale(cos),
            freq: self.freq + 1,
            sins: self.sins,
        }
    }

    #[inline]
    fn sin_branch(self, sin: f64, sign: f64) -> Self {
        Self {
            coeff: self.coeff.scale(sin).scale(sign),
            freq: self.freq + 1,
            sins: self.sins,
        }
    }

    #[inline]
    fn merge(self, other: Self) -> Self {
        Self {
            coeff: self.coeff.plus(other.coeff),
            freq: self.freq.min(other.freq),
            sins: self.sins.min(other.sins),
        }
    }

    #[inline]
    fn freq(&self) -> u32 {
        self.freq
    }

    #[inline]
    fn sins(&self) -> u32 {
        self.sins
    }
}
