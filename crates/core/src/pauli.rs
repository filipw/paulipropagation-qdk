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

    /// Whether the string holds a Y or a Z, in which case its expectation value in a state of
    /// X eigenstates is zero.
    #[inline]
    #[must_use]
    pub fn has_y_or_z(&self) -> bool {
        self.z.iter().any(|&w| w != 0)
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

    /// The string whose X-bits and Z-bits are those in which the two strings differ. Up to a
    /// phase this is their product.
    #[inline]
    #[must_use]
    pub fn xor(&self, other: &Self) -> Self {
        let mut out = *self;
        for i in 0..W {
            out.x[i] ^= other.x[i];
            out.z[i] ^= other.z[i];
        }
        out
    }

    /// Whether the string comes before `other` in the order of Pauli strings. Unlike a
    /// comparison that stops at the first word that differs, this takes the same steps for
    /// all strings, which is quicker where the first such word cannot be foreseen.
    #[inline(always)]
    pub(crate) fn is_before(&self, other: &Self) -> bool {
        let mut before = false;
        // From the word that counts least to the one that counts most.
        for i in (0..W).rev() {
            before = (self.z[i] < other.z[i]) | ((self.z[i] == other.z[i]) & before);
        }
        for i in (0..W).rev() {
            before = (self.x[i] < other.x[i]) | ((self.x[i] == other.x[i]) & before);
        }
        before
    }

    /// The string with the X-bits and Z-bits that either of the two strings has. For the
    /// strings of a sum this adds up to the X-bits and Z-bits that any of them has.
    #[inline]
    pub(crate) fn union(&self, other: &Self) -> Self {
        let mut out = *self;
        for i in 0..W {
            out.x[i] |= other.x[i];
            out.z[i] |= other.z[i];
        }
        out
    }

    /// Whether a string that has no X-bits and Z-bits but those of `self` can anticommute
    /// with `other`.
    #[inline]
    pub(crate) fn can_anticommute(&self, other: &Self) -> bool {
        let mut crossing = 0;
        for i in 0..W {
            crossing |= (self.x[i] & other.z[i]) | (self.z[i] & other.x[i]);
        }
        crossing != 0
    }

    /// Whether the two strings have the same bits wherever `mask` has a bit set.
    #[inline]
    pub(crate) fn agrees(&self, other: &Self, mask: &Self) -> bool {
        let mut differing = 0;
        for i in 0..W {
            differing |= (self.x[i] ^ other.x[i]) & mask.x[i];
            differing |= (self.z[i] ^ other.z[i]) & mask.z[i];
        }
        differing == 0
    }

    /// The words of the string by how much they count when two strings are compared: strings
    /// compare by their X words from the first to the last and then by their Z words, so limb
    /// `0`, which counts least, is the last Z word.
    #[inline]
    fn limb_mut(&mut self, limb: usize) -> &mut u64 {
        if limb < W {
            &mut self.z[W - 1 - limb]
        } else {
            &mut self.x[2 * W - 1 - limb]
        }
    }

    /// The set bits of the string as runs of bits that are neighbours in the order in which
    /// strings are compared, from the run that counts most to the one that counts least.
    /// Each run comes with all the bits that count more than it.
    pub(crate) fn bit_runs(&self) -> Vec<(Self, Self)> {
        let mut copy = *self;
        let mut runs = Vec::new();
        for limb in (0..2 * W).rev() {
            let mut rest = *copy.limb_mut(limb);
            while rest != 0 {
                let top = 63 - rest.leading_zeros();
                let length = (rest << (63 - top)).leading_ones();
                let bottom = top + 1 - length;
                let bits = (u64::MAX >> (64 - length)) << bottom;

                let mut run = Self::identity();
                *run.limb_mut(limb) = bits;
                let mut above = Self::identity();
                *above.limb_mut(limb) = if top == 63 { 0 } else { u64::MAX << (top + 1) };
                for higher in limb + 1..2 * W {
                    *above.limb_mut(higher) = u64::MAX;
                }
                runs.push((run, above));
                rest &= !bits;
            }
        }
        runs
    }

    /// The highest qubit with a non-identity operator.
    #[must_use]
    pub fn max_qubit(&self) -> Option<usize> {
        (0..W).rev().find_map(|i| {
            let w = self.x[i] | self.z[i];
            (w != 0).then(|| 64 * i + 63 - w.leading_zeros() as usize)
        })
    }

    /// Compares two strings by the operators on the highest qubit where they differ, with
    /// `I < X < Y < Z`. This is the order of the integers that PauliPropagation.jl holds
    /// Pauli strings in.
    #[must_use]
    pub fn cmp_from_highest_qubit(&self, other: &Self) -> std::cmp::Ordering {
        for i in (0..W).rev() {
            let differing = (self.x[i] ^ other.x[i]) | (self.z[i] ^ other.z[i]);
            if differing != 0 {
                let q = 64 * i + 63 - differing.leading_zeros() as usize;
                return self.get(q).cmp(&other.get(q));
            }
        }
        std::cmp::Ordering::Equal
    }

    /// The qubits with a non-identity operator, in ascending order, and their operators.
    #[must_use]
    pub fn ops(&self) -> Vec<(usize, PauliOp)> {
        let mut ops = Vec::with_capacity(self.weight() as usize);
        for i in 0..W {
            let mut word = self.x[i] | self.z[i];
            while word != 0 {
                let q = 64 * i + word.trailing_zeros() as usize;
                ops.push((q, self.get(q)));
                word &= word - 1;
            }
        }
        ops
    }

    /// The operators on qubits `0..num_qubits` as a string of `I`, `X`, `Y`, `Z`, qubit 0 first.
    #[must_use]
    pub fn label(&self, num_qubits: usize) -> String {
        (0..num_qubits).map(|q| self.get(q).to_char()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    /// A string with bits in every word, more of them equal between two strings than chance
    /// would give, so that comparisons are decided in any word.
    fn random_string<const W: usize>(rng: &mut StdRng) -> PauliString<W> {
        let mut string = PauliString::identity();
        for i in 0..W {
            let mut word = || match rng.gen_range(0..4) {
                0 => 0,
                1 => u64::MAX,
                2 => 1 << rng.gen_range(0..64),
                _ => rng.gen(),
            };
            string.x[i] = word();
            string.z[i] = word();
        }
        string
    }

    /// The bits of a string from the one that counts most in a comparison to the one that
    /// counts least.
    fn bits<const W: usize>(string: &PauliString<W>) -> Vec<bool> {
        let words = string.x.iter().chain(&string.z);
        words
            .flat_map(|word| (0..64).rev().map(move |bit| (word >> bit) & 1 == 1))
            .collect()
    }

    fn from_bits<const W: usize>(bits: &[bool]) -> PauliString<W> {
        let mut string = PauliString::identity();
        for (place, _) in bits.iter().enumerate().filter(|(_, &bit)| bit) {
            let (word, bit) = (place / 64, 63 - place % 64);
            if word < W {
                string.x[word] |= 1 << bit;
            } else {
                string.z[word - W] |= 1 << bit;
            }
        }
        string
    }

    #[test]
    fn strings_compare_as_their_bits() {
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..2000 {
            let (a, b) = (random_string::<3>(&mut rng), random_string::<3>(&mut rng));
            assert_eq!(a.cmp(&b), bits(&a).cmp(&bits(&b)));
            assert_eq!(a.is_before(&b), a < b);
            assert_eq!(b.is_before(&a), b < a);
            assert!(!a.is_before(&a));
        }
    }

    #[test]
    fn bit_runs_are_the_runs_of_set_bits_with_the_bits_above_them() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut most_runs = 0;
        for _ in 0..500 {
            let string = random_string::<2>(&mut rng);
            let set = bits(&string);
            let runs = string.bit_runs();
            most_runs = most_runs.max(runs.len());

            // Together the runs are the set bits, each bit in one run.
            let mut covered = vec![false; set.len()];
            let mut last_end = 0;
            for (run, above) in &runs {
                let run = bits(run);
                let first = run.iter().position(|&bit| bit).expect("a run has a bit");
                let end = run.iter().rposition(|&bit| bit).unwrap() + 1;
                assert!(run[first..end].iter().all(|&bit| bit), "a run has no gap");
                assert!(
                    first >= last_end,
                    "the runs go from the bits that count most"
                );
                last_end = end;
                covered[first..end].fill(true);
                // Above a run are all the bits that count more, set or not.
                let expected: Vec<bool> = (0..set.len()).map(|place| place < first).collect();
                assert_eq!(bits(above), expected);
            }
            assert_eq!(covered, set);
        }
        assert!(most_runs > 20, "the strings have too few runs to mean much");
    }

    #[test]
    fn bit_runs_of_whole_words() {
        let all: PauliString<2> = from_bits(&[true; 256]);
        let runs = all.bit_runs();
        assert_eq!(runs.len(), 4);
        assert_eq!(runs[0].1, PauliString::identity());
        assert!(PauliString::<2>::identity().bit_runs().is_empty());
    }

    #[test]
    fn agreement_is_on_the_bits_of_the_mask() {
        let mut rng = StdRng::seed_from_u64(3);
        let (mut agreed, mut differed) = (0, 0);
        for _ in 0..2000 {
            let a = random_string::<2>(&mut rng);
            let mask = random_string::<2>(&mut rng);
            // A string that differs from `a` in a few bits.
            let mut b = a;
            for _ in 0..rng.gen_range(0..3) {
                if rng.gen() {
                    b.flip_x(rng.gen_range(0..128));
                } else {
                    b.flip_z(rng.gen_range(0..128));
                }
            }
            let places = bits(&a).into_iter().zip(bits(&b)).zip(bits(&mask));
            let expected = places.into_iter().all(|((a, b), mask)| !mask || a == b);
            assert_eq!(a.agrees(&b, &mask), expected);
            if expected {
                agreed += 1;
            } else {
                differed += 1;
            }
        }
        assert!(
            agreed > 200 && differed > 200,
            "{agreed} agreed, {differed} differed"
        );
    }

    #[test]
    fn xor_and_union_work_bit_by_bit() {
        let mut rng = StdRng::seed_from_u64(4);
        for _ in 0..500 {
            let (a, b) = (random_string::<2>(&mut rng), random_string::<2>(&mut rng));
            let pairs = || bits(&a).into_iter().zip(bits(&b));
            let xor: Vec<bool> = pairs().map(|(a, b)| a != b).collect();
            let union: Vec<bool> = pairs().map(|(a, b)| a || b).collect();
            assert_eq!(bits(&a.xor(&b)), xor);
            assert_eq!(bits(&a.union(&b)), union);
            assert_eq!(a.xor(&b), a.mul(&b).0);
        }
    }

    #[test]
    fn a_string_within_the_bits_of_another_anticommutes_only_if_it_can() {
        let mut rng = StdRng::seed_from_u64(5);
        let (mut can, mut cannot) = (0, 0);
        for _ in 0..3000 {
            // Strings on a few qubits, so that often no operators meet.
            let sparse = |rng: &mut StdRng| {
                let mut string = PauliString::<2>::identity();
                for _ in 0..rng.gen_range(1..5) {
                    let op = [PauliOp::X, PauliOp::Y, PauliOp::Z][rng.gen_range(0..3)];
                    string.set(rng.gen_range(0..12) * 10, op);
                }
                string
            };
            let (reach, other, inside) = (sparse(&mut rng), sparse(&mut rng), sparse(&mut rng));
            // A string with no bits but those of `reach`.
            let mut within = PauliString::<2>::identity();
            for i in 0..2 {
                within.x[i] = inside.x[i] & reach.x[i];
                within.z[i] = inside.z[i] & reach.z[i];
            }
            if reach.can_anticommute(&other) {
                can += 1;
            } else {
                cannot += 1;
                assert!(within.commutes(&other));
                assert!(reach.commutes(&other));
            }
            // If the string itself anticommutes, it can.
            if !reach.commutes(&other) {
                assert!(reach.can_anticommute(&other));
            }
        }
        assert!(can > 300 && cannot > 300, "{can} can, {cannot} cannot");
    }
}
