//! Bit-packed Pauli strings.

/// A single-qubit Pauli operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum PauliOp {
    I,
    X,
    Y,
    Z,
}

impl PauliOp {
    /// `(x, z)` bits of the operator, with `Y = (1, 1)`.
    #[must_use]
    pub fn bits(self) -> (bool, bool) {
        match self {
            PauliOp::I => (false, false),
            PauliOp::X => (true, false),
            PauliOp::Y => (true, true),
            PauliOp::Z => (false, true),
        }
    }

    #[must_use]
    pub fn from_bits(x: bool, z: bool) -> Self {
        match (x, z) {
            (false, false) => PauliOp::I,
            (true, false) => PauliOp::X,
            (true, true) => PauliOp::Y,
            (false, true) => PauliOp::Z,
        }
    }

    #[must_use]
    pub fn from_char(c: char) -> Option<Self> {
        match c {
            'I' | 'i' => Some(PauliOp::I),
            'X' | 'x' => Some(PauliOp::X),
            'Y' | 'y' => Some(PauliOp::Y),
            'Z' | 'z' => Some(PauliOp::Z),
            _ => None,
        }
    }

    #[must_use]
    pub fn to_char(self) -> char {
        match self {
            PauliOp::I => 'I',
            PauliOp::X => 'X',
            PauliOp::Y => 'Y',
            PauliOp::Z => 'Z',
        }
    }
}

/// A Pauli string on up to `64 * W` qubits, without a phase.
///
/// Qubit `q` holds the operator with X-bit `x[q]` and Z-bit `z[q]`; `Y` sets both.
/// The string stands for the Hermitian tensor product of its operators, so `Y` is the
/// Pauli Y matrix itself and not `XZ`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PauliString<const W: usize> {
    x: [u64; W],
    z: [u64; W],
}

impl<const W: usize> Default for PauliString<W> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<const W: usize> PauliString<W> {
    /// Number of qubits a string of this width can hold.
    pub const CAPACITY: usize = 64 * W;

    #[must_use]
    pub fn identity() -> Self {
        Self {
            x: [0; W],
            z: [0; W],
        }
    }

    /// Builds a string from `(qubit, operator)` pairs. A qubit given twice keeps the last operator.
    ///
    /// # Panics
    /// If a qubit is outside the capacity.
    #[must_use]
    pub fn from_ops(ops: &[(usize, PauliOp)]) -> Self {
        let mut p = Self::identity();
        for &(q, op) in ops {
            p.set(q, op);
        }
        p
    }

    #[inline]
    #[must_use]
    pub fn x_bit(&self, q: usize) -> bool {
        (self.x[q >> 6] >> (q & 63)) & 1 == 1
    }

    #[inline]
    #[must_use]
    pub fn z_bit(&self, q: usize) -> bool {
        (self.z[q >> 6] >> (q & 63)) & 1 == 1
    }

    #[inline]
    pub fn flip_x(&mut self, q: usize) {
        self.x[q >> 6] ^= 1 << (q & 63);
    }

    #[inline]
    pub fn flip_z(&mut self, q: usize) {
        self.z[q >> 6] ^= 1 << (q & 63);
    }

    #[inline]
    #[must_use]
    pub fn get(&self, q: usize) -> PauliOp {
        PauliOp::from_bits(self.x_bit(q), self.z_bit(q))
    }

    #[inline]
    pub fn set(&mut self, q: usize, op: PauliOp) {
        let (x, z) = op.bits();
        if self.x_bit(q) != x {
            self.flip_x(q);
        }
        if self.z_bit(q) != z {
            self.flip_z(q);
        }
    }

    /// Number of non-identity operators.
    #[inline]
    #[must_use]
    pub fn weight(&self) -> u32 {
        let mut w = 0;
        for i in 0..W {
            w += (self.x[i] | self.z[i]).count_ones();
        }
        w
    }

    /// Whether the string holds an X or a Y, in which case its expectation value in a
    /// computational basis state is zero.
    #[inline]
    #[must_use]
    pub fn has_x_or_y(&self) -> bool {
        self.x.iter().any(|&w| w != 0)
    }

    /// Whether the two strings commute.
    #[inline]
    #[must_use]
    pub fn commutes(&self, other: &Self) -> bool {
        let mut parity = 0;
        for i in 0..W {
            parity ^= ((self.x[i] & other.z[i]) ^ (self.z[i] & other.x[i])).count_ones();
        }
        parity & 1 == 0
    }

    /// The product `self * rhs = i^k * result`, returned as `(result, k)` with `k` in `0..4`.
    #[inline]
    #[must_use]
    pub fn mul(&self, rhs: &Self) -> (Self, u8) {
        let mut out = Self::identity();
        let mut plus = 0u32;
        let mut minus = 0u32;
        for i in 0..W {
            let (x1, z1, x2, z2) = (self.x[i], self.z[i], rhs.x[i], rhs.z[i]);
            // XY = iZ, YZ = iX, ZX = iY
            plus +=
                ((x1 & !z1 & x2 & z2) | (x1 & z1 & !x2 & z2) | (!x1 & z1 & x2 & !z2)).count_ones();
            // YX = -iZ, ZY = -iX, XZ = -iY
            minus +=
                ((x1 & z1 & x2 & !z2) | (!x1 & z1 & x2 & z2) | (x1 & !z1 & !x2 & z2)).count_ones();
            out.x[i] = x1 ^ x2;
            out.z[i] = z1 ^ z2;
        }
        (out, (plus.wrapping_sub(minus) & 3) as u8)
    }

    /// The highest qubit with a non-identity operator.
    #[must_use]
    pub fn max_qubit(&self) -> Option<usize> {
        (0..W).rev().find_map(|i| {
            let w = self.x[i] | self.z[i];
            (w != 0).then(|| 64 * i + 63 - w.leading_zeros() as usize)
        })
    }

    /// The operators on qubits `0..num_qubits` as a string of `I`, `X`, `Y`, `Z`, qubit 0 first.
    #[must_use]
    pub fn label(&self, num_qubits: usize) -> String {
        (0..num_qubits).map(|q| self.get(q).to_char()).collect()
    }
}
