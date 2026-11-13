//! Checks the bookkeeping of a Pauli sum against a map from Pauli strings to coefficients.
//!
//! A sum keeps its terms in the order of their Pauli strings and collects terms that are
//! added out of order on the side. What it reports must not depend on where a term is kept.

use pauliprop::{Complex64, PauliOp, PauliString, PauliSum};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use std::collections::BTreeMap;

const OPS: [PauliOp; 4] = [PauliOp::I, PauliOp::X, PauliOp::Y, PauliOp::Z];

/// A string on `n` qubits out of a pool that is small enough for strings to come up twice.
fn random_string(rng: &mut StdRng, n: usize, pool: u64) -> PauliString<2> {
    let mut number = rng.gen_range(0..pool);
    let mut string = PauliString::identity();
    for qubit in (0..n).rev() {
        string.set(qubit, OPS[(number % 4) as usize]);
        number /= 4;
    }
    string
}

fn assert_matches(sum: &PauliSum<2>, model: &BTreeMap<PauliString<2>, f64>, what: &str) {
    assert_eq!(sum.len(), model.len(), "{what}: number of terms");
    assert_eq!(sum.is_empty(), model.is_empty(), "{what}");
    let expected: Vec<(PauliString<2>, f64)> = model.iter().map(|(p, c)| (*p, *c)).collect();
    assert!(sum.sorted_terms() == expected, "{what}: terms in order");
    let mut listed: Vec<(PauliString<2>, f64)> = sum.iter().map(|(p, c)| (*p, c)).collect();
    listed.sort_unstable_by_key(|term| term.0);
    assert!(listed == expected, "{what}: terms as listed");
    for (string, coeff) in model {
        assert_eq!(sum.coeff(string), *coeff, "{what}: coefficient");
    }
}

#[test]
fn terms_added_in_any_order_are_all_there() {
    let mut rng = StdRng::seed_from_u64(1);
    // Few and many terms, so that the sum puts the added terms in order several times.
    for (round, count) in [0, 1, 7, 300, 5000, 20_000].into_iter().enumerate() {
        let n = 70;
        let mut sum: PauliSum<2> = PauliSum::new(n).unwrap();
        let mut model = BTreeMap::new();
        let mut repeated = 0;
        for step in 0..count {
            let string = random_string(&mut rng, n, 3 * count as u64 + 1);
            let coeff: f64 = rng.gen_range(-1.0..1.0);
            sum.add_string(string, coeff);
            let entry = model.entry(string).or_insert(0.0);
            repeated += usize::from(*entry != 0.0);
            *entry += coeff;
            if step % 997 == 0 {
                assert_matches(&sum, &model, &format!("round {round}, step {step}"));
            }
        }
        assert_matches(&sum, &model, &format!("round {round}"));
        assert!(
            count < 300 || repeated > count / 20,
            "only {repeated} strings came twice"
        );

        // A string that is not in the sum has the coefficient zero.
        let absent = random_string(&mut rng, n, u64::MAX);
        if !model.contains_key(&absent) {
            assert_eq!(sum.coeff(&absent), 0.0);
        }
    }
}

#[test]
fn terms_added_in_ascending_or_descending_order_are_all_there() {
    let mut rng = StdRng::seed_from_u64(2);
    let n = 70;
    let mut strings: Vec<PauliString<2>> = (0..3000)
        .map(|_| random_string(&mut rng, n, 1 << 40))
        .collect();
    strings.sort_unstable();
    strings.dedup();
    for descending in [false, true] {
        let mut sum: PauliSum<2> = PauliSum::new(n).unwrap();
        let mut model = BTreeMap::new();
        let mut order = strings.clone();
        if descending {
            order.reverse();
        }
        for (index, string) in order.iter().enumerate() {
            sum.add_string(*string, index as f64 + 1.0);
            model.insert(*string, index as f64 + 1.0);
        }
        // Adding to a string that is in the sum changes its coefficient, wherever it is.
        for string in order.choose_multiple(&mut rng, 200) {
            sum.add_string(*string, 0.5);
            *model.get_mut(string).unwrap() += 0.5;
        }
        assert_matches(
            &sum,
            &model,
            if descending {
                "descending"
            } else {
                "ascending"
            },
        );
    }
}

#[test]
fn sums_and_products_do_not_depend_on_the_order_of_adding() {
    let mut rng = StdRng::seed_from_u64(3);
    let n = 9;
    let mut terms: Vec<(PauliString<2>, f64)> = Vec::new();
    let mut others: Vec<(PauliString<2>, f64)> = Vec::new();
    // Strings of I and Z and of I and X among them, for the overlaps to add up.
    for index in 0..4000u64 {
        let mut string = random_string(&mut rng, n, 1 << 18);
        if index % 3 == 0 {
            let op = if index % 2 == 0 {
                PauliOp::Z
            } else {
                PauliOp::X
            };
            for qubit in 0..n {
                if string.get(qubit) != PauliOp::I {
                    string.set(qubit, op);
                }
            }
        }
        terms.push((string, rng.gen_range(-1.0..1.0)));
        if index % 2 == 0 {
            others.push((string, rng.gen_range(-1.0..1.0)));
        }
    }

    let build = |terms: &[(PauliString<2>, f64)]| {
        let mut sum: PauliSum<2> = PauliSum::new(n).unwrap();
        for (string, coeff) in terms {
            sum.add_string(*string, *coeff);
        }
        sum
    };
    // In ascending order every term is kept in order at once. Coefficients of a string that
    // comes twice are added up in the order of adding, so that order is kept here.
    let mut ascending = terms.clone();
    ascending.sort_by_key(|term| term.0);
    let mut descending = terms.clone();
    descending.sort_by_key(|term| std::cmp::Reverse(term.0));

    let reference = build(&ascending);
    let other = build(&others);
    assert!(reference.overlap_with_zero() != 0.0 && reference.overlap_with_plus() != 0.0);
    for (what, order) in [("as drawn", &terms), ("descending", &descending)] {
        let sum = build(order);
        let (ours, theirs) = (sum.sorted_terms(), reference.sorted_terms());
        assert_eq!(ours.len(), theirs.len(), "{what}");
        for (ours, theirs) in ours.iter().zip(&theirs) {
            assert_eq!(ours, theirs, "{what}");
        }
        assert_eq!(
            sum.overlap_with_zero(),
            reference.overlap_with_zero(),
            "{what}"
        );
        assert_eq!(
            sum.overlap_with_plus(),
            reference.overlap_with_plus(),
            "{what}"
        );
        assert_eq!(
            sum.overlap_with_computational(&[0, 3, 4]).unwrap(),
            reference.overlap_with_computational(&[0, 3, 4]).unwrap(),
            "{what}"
        );
        assert_eq!(
            sum.overlap_with_max_mixed(),
            reference.overlap_with_max_mixed(),
            "{what}"
        );
        assert_eq!(sum.norm_sqr(), reference.norm_sqr(), "{what}");
        assert_eq!(sum.max_abs_coeff(), reference.max_abs_coeff(), "{what}");
        assert_eq!(
            sum.scalar_product(&other),
            reference.scalar_product(&other),
            "{what}"
        );
        assert_eq!(
            other.scalar_product(&sum),
            other.scalar_product(&reference),
            "{what}"
        );
    }

    // The scalar product is the sum of the products of the coefficients of the strings
    // that are in both.
    let theirs: BTreeMap<PauliString<2>, f64> = other.sorted_terms().into_iter().collect();
    let expected: f64 = reference
        .sorted_terms()
        .iter()
        .filter_map(|(string, coeff)| theirs.get(string).map(|other| coeff * other))
        .sum();
    assert!(expected != 0.0);
    assert_eq!(reference.scalar_product(&other), expected);
}

#[test]
fn complex_coefficients_add_up() {
    let mut sum: PauliSum<2, Complex64> = PauliSum::new(3).unwrap();
    let z = [(1, PauliOp::Z)];
    let x = [(0, PauliOp::X)];
    sum.add(&z, Complex64::new(1.0, 2.0)).unwrap();
    sum.add(&x, Complex64::new(0.5, 0.0)).unwrap();
    sum.add(&z, Complex64::new(0.25, -1.0)).unwrap();
    assert_eq!(sum.len(), 2);
    assert_eq!(
        sum.coeff(&PauliString::from_ops(&z)),
        Complex64::new(1.25, 1.0)
    );
    assert_eq!(sum.overlap_with_zero(), Complex64::new(1.25, 1.0));
    assert_eq!(sum.overlap_with_plus(), Complex64::new(0.5, 0.0));
}
