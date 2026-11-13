//! A propagation that keeps the sum at or below a given number of terms.

use crate::{
    propagate_in_zones, Coefficient, CustomTruncation, Error, Gate, PauliSum, Picture, Truncation,
    Zones,
};

/// A run of fewer gates than this is applied on the thread of the caller, because dividing
/// a sum among threads costs as much as several gates.
const FEWEST_DIVIDED: usize = 16;

/// How many times as many terms a sum can have after the gate as before it, at most.
fn growth(gate: &Gate) -> usize {
    match gate {
        Gate::H(_)
        | Gate::X(_)
        | Gate::Y(_)
        | Gate::Z(_)
        | Gate::S(_)
        | Gate::Sdg(_)
        | Gate::Sx(_)
        | Gate::Sxdg(_)
        | Gate::Cx(..)
        | Gate::Cy(..)
        | Gate::Cz(..)
        | Gate::Swap(..)
        | Gate::PauliChannel { .. } => 1,
        // One more Pauli string from every term.
        Gate::Rotation { .. } | Gate::AmplitudeDamping { .. } => 2,
        // Every Pauli string on the qubits of the map from every term.
        Gate::TransferMap { columns, .. } => columns.len(),
    }
}

impl<const W: usize, C: Coefficient> PauliSum<W, C> {
    /// Keeps the `count` terms whose coefficients are the largest in absolute value and
    /// drops the others. Of terms that are equally large, those with the lower Pauli
    /// strings are kept.
    pub fn keep_largest(&mut self, count: usize) {
        if self.len() <= count {
            return;
        }
        let (strings, coeffs) = self.take_ordered();
        let sizes: Vec<f64> = coeffs.iter().map(|coeff| coeff.abs()).collect();
        // The places of the terms, which are in the order of the Pauli strings.
        let mut places: Vec<usize> = (0..sizes.len()).collect();
        places.select_nth_unstable_by(count, |&a, &b| {
            sizes[b].total_cmp(&sizes[a]).then(a.cmp(&b))
        });
        places.truncate(count);
        places.sort_unstable();
        let kept = places.iter().map(|&place| strings[place]).collect();
        let kept_coeffs = places.iter().map(|&place| coeffs[place]).collect();
        self.set_ordered(kept, kept_coeffs);
    }
}

/// Propagates like [`crate::propagate_with`], and whenever a gate leaves the sum with more
/// than `max_terms` terms, keeps the `max_terms` largest of them, see
/// [`PauliSum::keep_largest`].
///
/// The sum then never holds more than `max_terms` terms after a gate, whatever the circuit
/// and the other truncations. `max_freq` and `max_sins` are not supported. While the sum is
/// so small that the next gates cannot take it beyond `max_terms`, they are applied
/// together and on all threads; close to `max_terms` the gates are applied one at a time on
/// the thread of the caller.
pub fn propagate_capped<const W: usize, C: Coefficient>(
    circuit: &[Gate],
    sum: &mut PauliSum<W, C>,
    truncation: &Truncation,
    picture: Picture,
    custom: Option<&CustomTruncation<'_, W, C>>,
    max_terms: usize,
) -> Result<(), Error> {
    if max_terms == 0 {
        return Err(Error::Invalid(
            "the largest number of terms must be positive",
        ));
    }
    if truncation.max_freq.is_some() || truncation.max_sins.is_some() {
        return Err(Error::Invalid(
            "a largest number of terms cannot be combined with max_freq or max_sins",
        ));
    }
    for gate in circuit {
        gate.validate(sum.num_qubits())?;
    }

    // The gates that have not been applied, which are taken from the end for an observable
    // and from the start for a state.
    let mut remaining = circuit;
    while !remaining.is_empty() {
        let next = |taken: usize| match picture {
            Picture::Heisenberg => remaining.len().checked_sub(taken + 1),
            Picture::Schrodinger => Some(taken).filter(|&index| index < remaining.len()),
        };
        // As many gates as cannot take the sum beyond the largest number of terms.
        let mut room = max_terms / sum.len().max(1);
        let mut safe = 0;
        while let Some(index) = next(safe) {
            let factor = growth(&remaining[index]);
            if factor > room {
                break;
            }
            room /= factor;
            safe += 1;
        }

        let taken = safe.max(1);
        let (gates, rest) = match picture {
            Picture::Heisenberg => {
                let (rest, gates) = remaining.split_at(remaining.len() - taken);
                (gates, rest)
            }
            Picture::Schrodinger => remaining.split_at(taken),
        };
        let zones = if taken < FEWEST_DIVIDED {
            Zones::ONE
        } else {
            Zones::default()
        };
        propagate_in_zones(gates, sum, truncation, picture, custom, &zones)?;
        if safe == 0 {
            sum.keep_largest(max_terms);
        }
        remaining = rest;
    }
    Ok(())
}
