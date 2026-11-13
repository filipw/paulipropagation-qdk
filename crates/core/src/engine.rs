//! Propagation on sorted arrays of terms.
//!
//! The terms of the sum are kept in the order of their Pauli strings. A gate that makes two
//! terms of one, such as a rotation, changes the coefficients of the terms in place and
//! collects the terms it creates. A created term is the term it comes from with some bits of
//! its Pauli string flipped, the same bits for all of them. That makes the created terms
//! cheap to put in order: they only have to be regrouped by the flipped bits. A merge of the
//! two ordered sequences then adds up the terms with the same Pauli string and drops the
//! truncated ones in the same pass. This is the vector backend of PauliPropagation.jl.
//!
//! To use several threads the terms are spread over zones. A zone is picked by whether the
//! Pauli string of a term commutes with each of a few fixed Pauli strings, the marks.
//! Flipping the same bits in all strings of a zone moves all of them to one other zone, so
//! under such a gate the zones pair up, and each pair is worked on without looking at the
//! rest.
//!
//! The work on the zones is handed out in rounds, one per gate as a rule, see [`Crew`].
//!
//! Every coefficient is computed by the same operations in the same order whatever the
//! zones are, so the result does not depend on them or on the number of threads.

use crate::coeff::Path;
use crate::propagate::{Rules, Zones};
use crate::{Coefficient, Gate, PauliOp, PauliString, Picture};
use rayon::prelude::*;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::SeqCst};
use std::sync::{Condvar, Mutex, MutexGuard, RwLock};

/// With more runs of flipped bits than this, created terms are put in order by a general
/// sort, which then takes fewer passes.
const MAX_REGROUPINGS: usize = 8;

/// The largest number of zones, which leaves room to count the tasks of a round.
const MAX_ZONES: usize = 1 << 12;

/// The fewest zones that a sum is divided into, unless no more are allowed.
const FEWEST_ZONES: usize = 8;

/// Propagates the terms with the Pauli strings `keys`, which are in ascending order, and
/// returns the terms of the result in ascending order.
pub(crate) fn run<const W: usize, P: Path>(
    circuit: &[Gate],
    num_qubits: usize,
    keys: Vec<PauliString<W>>,
    paths: Vec<P>,
    rules: &Rules<'_, W, P::Coeff>,
    picture: Picture,
    zones: &Zones,
) -> (Vec<PauliString<W>>, Vec<P>) {
    // The gates in the order in which they are applied, each as it acts in the picture.
    let adjoints: Vec<Gate>;
    let gates: Vec<&Gate> = match picture {
        Picture::Heisenberg => circuit.iter().rev().collect(),
        Picture::Schrodinger => {
            adjoints = circuit.iter().map(Gate::adjoint).collect();
            adjoints.iter().collect()
        }
    };

    let most = 1 << zones.max.clamp(1, MAX_ZONES).ilog2();
    let identity = PauliString::identity();
    let reach = keys.iter().fold(identity, |reach, key| reach.union(key));
    let crew = Crew::new(most, Zone::new(keys, paths));
    let mut engine = Engine {
        crew: &crew,
        count: 1,
        marks: Vec::new(),
        rules,
        terms_per_zone: zones.terms,
        most,
        num_qubits,
        random: 0x9e37_79b9_7f4a_7c15,
        reach,
        checked: false,
        grown: true,
    };

    // The other threads of the pool are called in when the sum is first divided into
    // zones, which a small sum never is.
    let helpers = rayon::current_num_threads() - 1;
    let done = engine.run(&gates, helpers == 0);
    if done < gates.len() {
        rayon::scope(|scope| {
            let _dismiss = Dismiss(&crew);
            for _ in 0..helpers {
                scope.spawn(|_| crew.help());
            }
            engine.run(&gates[done..], true);
        });
    }
    let count = engine.count;
    crew.collapse(count)
}

/// Some of the terms of the sum, in the order of their Pauli strings.
///
/// The arrays other than `keys` and `paths` are room to work in. They keep their length from
/// one gate to the next, so that they rarely have to grow, and what they hold beyond the
/// part in use means nothing.
struct Zone<const W: usize, P> {
    keys: Vec<PauliString<W>>,
    paths: Vec<P>,
    /// The places of the terms that a gate changes, of which there are `changed`.
    places: Vec<usize>,
    changed: usize,
    /// The terms that a gate creates from the terms of this zone, of which there are `made`.
    made_keys: Vec<PauliString<W>>,
    made_paths: Vec<P>,
    made: usize,
    /// For a Clifford gate, which bits it flipped in each of the created terms.
    made_classes: Vec<u8>,
    /// Where the terms of the zone after the gate are written to.
    next_keys: Vec<PauliString<W>>,
    next_paths: Vec<P>,
    /// The largest absolute coefficient, when it was last asked for.
    largest: f64,
}

/// The rules for dropping terms with the smallest absolute coefficient that is kept filled
/// in, laid out so that the common rules cost a few comparisons per term.
struct Keep<'a, const W: usize, C> {
    rules: &'a Rules<'a, W, C>,
    /// Whether every term is kept.
    all: bool,
    /// Whether a term that the gate leaves as it was is known to be kept.
    settled: bool,
    min_abs: f64,
    /// The largest weight that is kept, if the weights have to be looked at.
    max_weight: Option<u32>,
    /// The limits of the counters, with the largest number for a rule that is not set.
    max_freq: u32,
    max_sins: u32,
    /// Whether there are rules beyond these.
    more: bool,
}

impl<const W: usize, C> Clone for Keep<'_, W, C> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const W: usize, C> Copy for Keep<'_, W, C> {}

impl<'a, const W: usize, C: Coefficient> Keep<'a, W, C> {
    /// Keeps the terms that the rules do not drop.
    fn new(rules: &'a Rules<'a, W, C>, min_abs: f64) -> Self {
        let truncation = rules.truncation;
        Self {
            rules,
            all: false,
            settled: false,
            min_abs,
            max_weight: truncation.max_weight,
            max_freq: truncation.max_freq.unwrap_or(u32::MAX),
            max_sins: truncation.max_sins.unwrap_or(u32::MAX),
            more: truncation.weight_damping.is_some() || rules.custom.is_some(),
        }
    }

    /// Keeps every term.
    fn all(rules: &'a Rules<'a, W, C>) -> Self {
        Self {
            all: true,
            ..Self::new(rules, 0.0)
        }
    }

    /// The same for a gate after which no term is heavier than the rules allow: it leaves
    /// the Pauli strings as they are or creates no string that is too heavy, and every term
    /// was kept by all the rules before.
    fn of_settled_weights(self) -> Self {
        Self {
            settled: true,
            max_weight: None,
            ..self
        }
    }

    #[inline(always)]
    fn keeps<P: Path<Coeff = C>>(&self, key: &PauliString<W>, path: &P) -> bool {
        let dropped = path.coeff().is_below(self.min_abs)
            | self.max_weight.is_some_and(|max| key.weight() > max)
            | (path.freq() > self.max_freq)
            | (path.sins() > self.max_sins);
        self.all || !(dropped || (self.more && self.rules.drops(key, path, self.min_abs)))
    }
}

/// Puts terms in the order of their Pauli strings that were in order before the same bits
/// were flipped in all of the strings. `runs` are the runs of the flipped bits, see
/// [`PauliString::bit_runs`].
///
/// Flipping a run of neighbouring bits reverses the order of the values these bits take,
/// among strings that agree on all the bits that count more. So for one run after the other,
/// from the one that counts most, the stretches of terms that agree on the bits above the run
/// are taken, and within each the blocks of terms that agree on the run are put back to
/// front.
fn regroup<const W: usize, P: Path>(
    keys: &mut [PauliString<W>],
    paths: &mut [P],
    runs: &[(PauliString<W>, PauliString<W>)],
) {
    if keys.len() < 2 {
        return;
    }
    if runs.len() > MAX_REGROUPINGS {
        let mut terms: Vec<(PauliString<W>, P)> =
            keys.iter().copied().zip(paths.iter().copied()).collect();
        terms.sort_unstable_by_key(|term| term.0);
        for (place, (key, path)) in terms.into_iter().enumerate() {
            keys[place] = key;
            paths[place] = path;
        }
        return;
    }

    for (run, above) in runs {
        let mut start = 0;
        while start < keys.len() {
            let first = keys[start];
            let mut end = start + 1;
            let mut blocks = false;
            while end < keys.len() && keys[end].agrees(&first, above) {
                blocks |= !keys[end].agrees(&first, run);
                end += 1;
            }
            // Turning the stretch around puts the blocks back to front, and turning each
            // block around again restores the order within it.
            if blocks {
                keys[start..end].reverse();
                paths[start..end].reverse();
                let mut block = start;
                while block < end {
                    let mut block_end = block + 1;
                    while block_end < end && keys[block_end].agrees(&keys[block], run) {
                        block_end += 1;
                    }
                    keys[block..block_end].reverse();
                    paths[block..block_end].reverse();
                    block = block_end;
                }
            }
            start = end;
        }
    }
}

/// What a Clifford gate does to a Pauli string. It depends on the operators on the qubits
/// of the gate only: they decide which bits of the string are flipped and whether the sign
/// changes.
struct Images<const W: usize> {
    qubits: Vec<usize>,
    /// For each Pauli string on the qubits, numbered as for a Pauli channel: which of
    /// `flips` the gate flips, and whether it changes the sign.
    class: [u8; 16],
    minus: [bool; 16],
    /// The different sets of bits that the gate flips, of which the first is no bit, and
    /// the runs of each, see [`PauliString::bit_runs`].
    flips: Vec<PauliString<W>>,
    runs: Vec<Vec<(PauliString<W>, PauliString<W>)>>,
}

impl<const W: usize> Images<W> {
    /// The images under a Clifford gate other than a Pauli gate.
    fn of(gate: &Gate) -> Self {
        let qubits = match *gate {
            Gate::Cx(a, b) | Gate::Cy(a, b) | Gate::Cz(a, b) | Gate::Swap(a, b) => vec![a, b],
            Gate::H(q) | Gate::S(q) | Gate::Sdg(q) | Gate::Sx(q) | Gate::Sxdg(q) => vec![q],
            _ => unreachable!("not a Clifford gate that changes Pauli strings"),
        };
        let mut images = Self {
            class: [0; 16],
            minus: [false; 16],
            flips: vec![PauliString::identity()],
            runs: vec![Vec::new()],
            qubits,
        };
        for index in 0..1 << (2 * images.qubits.len()) {
            let string = Gate::with_index(&images.qubits, PauliString::identity(), index);
            let mut image = string;
            images.minus[index] = gate.conjugate_clifford(&mut image);
            let flips = string.xor(&image);
            let known = images.flips.iter().position(|known| *known == flips);
            let class = known.unwrap_or_else(|| {
                images.flips.push(flips);
                images.runs.push(flips.bit_runs());
                images.flips.len() - 1
            });
            images.class[index] = class as u8;
        }
        images
    }
}

/// What a gate that flips the same bits in every Pauli string it changes does to a term.
trait Split<const W: usize, P>: Sync {
    /// The bits that the gate flips in the Pauli string of a created term.
    fn flips(&self) -> PauliString<W>;

    /// Whether the gate changes the term with the Pauli string `key`.
    fn changes(&self, key: &PauliString<W>) -> bool;

    /// For a term that the gate changes: the path that stays on the Pauli string, and the
    /// path of the term that is created with the bits flipped, if there is one.
    fn split(&self, key: &PauliString<W>, path: &P) -> (P, Option<P>);
}

/// `P -> U† P U` for `U = exp(-i theta / 2 G)`.
///
/// A string that commutes with `G` is unchanged. One that anticommutes becomes
/// `cos(theta) P + sin(theta) (i G P)`, and `i G P` is a Pauli string up to a sign.
struct Rotation<const W: usize> {
    generator: PauliString<W>,
    sin: f64,
    cos: f64,
}

impl<const W: usize, P: Path> Split<W, P> for Rotation<W> {
    fn flips(&self) -> PauliString<W> {
        self.generator
    }

    #[inline(always)]
    fn changes(&self, key: &PauliString<W>) -> bool {
        !key.commutes(&self.generator)
    }

    #[inline(always)]
    fn split(&self, key: &PauliString<W>, path: &P) -> (P, Option<P>) {
        // G P = i^k P' with k odd, so i G P = i^(k + 1) P' = -P' for k = 1 and +P' for k = 3.
        let (_, k) = self.generator.mul(key);
        debug_assert!(k == 1 || k == 3);
        let sign = if k == 3 { 1.0 } else { -1.0 };
        (
            path.cos_branch(self.cos),
            Some(path.sin_branch(self.sin, sign)),
        )
    }
}

/// The adjoint of amplitude damping on a qubit.
///
/// X and Y on the qubit are multiplied by `sqrt(1 - gamma)`. A string with Z there keeps
/// `1 - gamma` of its coefficient and adds `gamma` of it to the string with the identity there.
struct Damping {
    qubit: usize,
    gamma: f64,
}

impl<const W: usize, P: Path> Split<W, P> for Damping {
    fn flips(&self) -> PauliString<W> {
        PauliString::from_ops(&[(self.qubit, PauliOp::Z)])
    }

    #[inline(always)]
    fn changes(&self, key: &PauliString<W>) -> bool {
        key.x_bit(self.qubit) | key.z_bit(self.qubit)
    }

    #[inline(always)]
    fn split(&self, key: &PauliString<W>, path: &P) -> (P, Option<P>) {
        if key.x_bit(self.qubit) {
            (path.scale((1.0 - self.gamma).sqrt()), None)
        } else {
            (path.scale(1.0 - self.gamma), Some(path.scale(self.gamma)))
        }
    }
}

impl<const W: usize, P: Path> Zone<W, P> {
    fn new(keys: Vec<PauliString<W>>, paths: Vec<P>) -> Self {
        Self {
            keys,
            paths,
            places: Vec::new(),
            changed: 0,
            made_keys: Vec::new(),
            made_paths: Vec::new(),
            made: 0,
            made_classes: Vec::new(),
            next_keys: Vec::new(),
            next_paths: Vec::new(),
            largest: 0.0,
        }
    }

    /// Applies the gate to every term: changes the paths in place and collects the created
    /// terms, in the order of the terms they come from. A created term with a Pauli string
    /// heavier than `max_weight` is left out.
    ///
    /// Which terms a gate changes cannot be foreseen, so nothing here is decided term by
    /// term: every place is written down as the next one that changes and only counted if
    /// it does, and likewise every created term.
    fn split<S: Split<W, P>>(&mut self, gate: &S, max_weight: Option<u32>) {
        if self.places.len() < self.keys.len() {
            self.places.resize(self.keys.len(), 0);
        }
        let mut changed = 0;
        for (place, key) in self.keys.iter().enumerate() {
            self.places[changed] = place;
            changed += usize::from(gate.changes(key));
        }
        self.changed = changed;

        if self.made_keys.len() < changed {
            self.made_keys.resize(changed, PauliString::identity());
            self.made_paths.resize(changed, P::start(P::Coeff::ZERO));
        }
        let flips = gate.flips();
        let mut made = 0;
        for &place in &self.places[..changed] {
            let key = self.keys[place];
            let (kept, created) = gate.split(&key, &self.paths[place]);
            self.paths[place] = kept;
            if let Some(created) = created {
                let moved = key.xor(&flips);
                self.made_keys[made] = moved;
                self.made_paths[made] = created;
                made += usize::from(max_weight.is_none_or(|max| moved.weight() <= max));
            }
        }
        self.made = made;
    }

    /// Puts the created terms in the order of their Pauli strings. `runs` are the runs of
    /// the bits that were flipped in all of them.
    fn order_made(&mut self, runs: &[(PauliString<W>, PauliString<W>)]) {
        let made = self.made;
        regroup(
            &mut self.made_keys[..made],
            &mut self.made_paths[..made],
            runs,
        );
    }

    /// Adds the ordered terms `(new_keys, new_paths)` to the zone. Terms with the same Pauli
    /// string are merged into one, and only the terms that `keep` accepts stay.
    fn merge(&mut self, new_keys: &[PauliString<W>], new_paths: &[P], keep: Keep<'_, W, P::Coeff>) {
        if new_keys.is_empty() {
            if self.changed > 0 || !keep.settled {
                self.retain(keep);
            }
            return;
        }
        let (keys, paths) = (self.keys.as_slice(), self.paths.as_slice());
        let most = keys.len() + new_keys.len();
        if self.next_keys.len() < most {
            self.next_keys.resize(most, PauliString::identity());
            self.next_paths.resize(most, P::start(P::Coeff::ZERO));
        }
        let out_keys = &mut self.next_keys[..most];
        let out_paths = &mut self.next_paths[..most];

        // Whether a term is kept cannot be foreseen, so every term is written to the next
        // free place, and the place is given away only if the term is kept. Most terms are
        // old ones that come before the next new one, and these go by in a tight loop.
        let (mut old, mut count) = (0, 0);
        for (new_key, new_path) in new_keys.iter().zip(new_paths) {
            while old < keys.len() && keys[old].is_before(new_key) {
                out_keys[count] = keys[old];
                out_paths[count] = paths[old];
                count += usize::from(keep.keeps(&keys[old], &paths[old]));
                old += 1;
            }
            // The new term merges with the old one at this place if that has its Pauli
            // string.
            let same = old < keys.len() && keys[old] == *new_key;
            let merged = paths.get(old).unwrap_or(new_path).merge(*new_path);
            let path = if same { merged } else { *new_path };
            out_keys[count] = *new_key;
            out_paths[count] = path;
            count += usize::from(keep.keeps(new_key, &path));
            old += usize::from(same);
        }
        for (key, path) in keys[old..].iter().zip(&paths[old..]) {
            out_keys[count] = *key;
            out_paths[count] = *path;
            count += usize::from(keep.keeps(key, path));
        }
        self.next_keys.truncate(count);
        self.next_paths.truncate(count);
        mem::swap(&mut self.keys, &mut self.next_keys);
        mem::swap(&mut self.paths, &mut self.next_paths);
    }

    /// Adds the terms that a gate created from the terms of this zone to the zone itself.
    fn merge_made(&mut self, keep: Keep<'_, W, P::Coeff>) {
        let made_keys = mem::take(&mut self.made_keys);
        let made_paths = mem::take(&mut self.made_paths);
        self.merge(&made_keys[..self.made], &made_paths[..self.made], keep);
        self.made_keys = made_keys;
        self.made_paths = made_paths;
    }

    /// Drops the terms that `keep` does not accept.
    fn retain(&mut self, keep: Keep<'_, W, P::Coeff>) {
        if keep.all {
            return;
        }
        // Nothing is written before the first term that is dropped, which often is none.
        let terms = self.keys.iter().zip(&self.paths);
        let Some(first) = terms
            .into_iter()
            .position(|(key, path)| !keep.keeps(key, path))
        else {
            return;
        };
        let mut kept = first;
        for index in first + 1..self.keys.len() {
            let (key, path) = (self.keys[index], self.paths[index]);
            self.keys[kept] = key;
            self.paths[kept] = path;
            kept += usize::from(keep.keeps(&key, &path));
        }
        self.keys.truncate(kept);
        self.paths.truncate(kept);
    }

    /// Applies a Pauli channel, which multiplies every coefficient by a factor that depends
    /// on the operators on `qubits`, and drops the terms that `keep` does not accept.
    fn scale(&mut self, qubits: &[usize], factors: &[f64], keep: Keep<'_, W, P::Coeff>) {
        let mut kept = 0;
        for index in 0..self.keys.len() {
            let key = self.keys[index];
            let path = self.paths[index].scale(factors[Gate::channel_index(qubits, &key)]);
            self.keys[kept] = key;
            self.paths[kept] = path;
            kept += usize::from(keep.keeps(&key, &path));
        }
        self.keys.truncate(kept);
        self.paths.truncate(kept);
    }

    /// Applies a Pauli gate, which only changes signs, and drops the terms that `keep` does
    /// not accept.
    fn flip_signs(&mut self, gate: &Gate, keep: Keep<'_, W, P::Coeff>) {
        for (key, path) in self.keys.iter().zip(&mut self.paths) {
            let mut image = *key;
            if gate.conjugate_clifford(&mut image) {
                *path = path.scale(-1.0);
            }
        }
        self.retain(keep);
    }

    /// Replaces every term by its image under a Clifford gate. Of the terms with a new
    /// Pauli string those stay that `keep` accepts, and of the others those that
    /// `keep_others` accepts.
    ///
    /// The terms that get a new Pauli string are taken out of the sum and put in order
    /// again, each set of flipped bits by itself, and then merged with the others.
    fn conjugate(
        &mut self,
        images: &Images<W>,
        keep: Keep<'_, W, P::Coeff>,
        keep_others: Keep<'_, W, P::Coeff>,
    ) {
        let terms = self.keys.len();
        if self.made_classes.len() < terms {
            self.made_keys.resize(terms, PauliString::identity());
            self.made_paths.resize(terms, P::start(P::Coeff::ZERO));
            self.made_classes.resize(terms, 0);
        }
        // Every term is written down both as one that stays and as one that moves, and
        // counted as what it is.
        let (mut stay, mut made) = (0, 0);
        for place in 0..terms {
            let key = self.keys[place];
            let index = Gate::channel_index(&images.qubits, &key);
            let class = images.class[index];
            let sign = if images.minus[index] { -1.0 } else { 1.0 };
            let path = self.paths[place].scale(sign);
            let image = key.xor(&images.flips[usize::from(class)]);
            self.keys[stay] = key;
            self.paths[stay] = path;
            stay += usize::from(class == 0);
            self.made_keys[made] = image;
            self.made_paths[made] = path;
            self.made_classes[made] = class;
            made += usize::from(class != 0 && keep.keeps(&image, &path));
        }
        self.keys.truncate(stay);
        self.paths.truncate(stay);
        self.made = made;
        // Something has changed if a term has moved, even if it was dropped at once.
        self.changed = terms - stay;

        if images.flips.len() == 2 {
            // The same bits are flipped in all the terms that moved.
            self.order_made(&images.runs[1]);
        } else if made > 0 {
            self.order_made_by_class(images);
        }
        self.merge_made(keep_others);
    }

    /// Puts the terms that a Clifford gate moved in the order of their Pauli strings, where
    /// the gate flipped different bits in different terms.
    fn order_made_by_class(&mut self, images: &Images<W>) {
        let made = self.made;
        if self.next_keys.len() < made {
            self.next_keys.resize(made, PauliString::identity());
            self.next_paths.resize(made, P::start(P::Coeff::ZERO));
        }
        // Gathers the terms of each class in the order they are in, and regroups them.
        let mut ends = vec![0; images.flips.len()];
        for &class in &self.made_classes[..made] {
            ends[usize::from(class)] += 1;
        }
        let mut total = 0;
        for end in &mut ends {
            total += *end;
            *end = total;
        }
        let mut starts: Vec<usize> = [0].iter().chain(&ends).copied().collect();
        starts.pop();
        let mut free = starts.clone();
        for place in 0..made {
            let to = &mut free[usize::from(self.made_classes[place])];
            self.next_keys[*to] = self.made_keys[place];
            self.next_paths[*to] = self.made_paths[place];
            *to += 1;
        }
        for (class, runs) in images.runs.iter().enumerate() {
            let (start, end) = (starts[class], ends[class]);
            regroup(
                &mut self.next_keys[start..end],
                &mut self.next_paths[start..end],
                runs,
            );
        }
        // Takes the term that comes first among the next ones of all classes, time after
        // time. No two terms have the same Pauli string.
        for place in 0..made {
            let mut first: Option<usize> = None;
            for class in 0..starts.len() {
                if starts[class] < ends[class]
                    && first.is_none_or(|first| {
                        self.next_keys[starts[class]] < self.next_keys[starts[first]]
                    })
                {
                    first = Some(class);
                }
            }
            let from = &mut starts[first.expect("there is a term left for every place")];
            self.made_keys[place] = self.next_keys[*from];
            self.made_paths[place] = self.next_paths[*from];
            *from += 1;
        }
    }

    /// Moves the terms whose Pauli string anticommutes with `mark` to a new zone.
    fn divide(&mut self, mark: &PauliString<W>) -> Self {
        let mut moved = Self::new(Vec::new(), Vec::new());
        let mut kept = 0;
        for index in 0..self.keys.len() {
            if self.keys[index].commutes(mark) {
                self.keys[kept] = self.keys[index];
                self.paths[kept] = self.paths[index];
                kept += 1;
            } else {
                moved.keys.push(self.keys[index]);
                moved.paths.push(self.paths[index]);
            }
        }
        self.keys.truncate(kept);
        self.paths.truncate(kept);
        moved
    }

    fn max_abs_coeff(&self) -> f64 {
        let sizes = self.paths.iter().map(|path| path.coeff().abs());
        sizes.fold(0.0, f64::max)
    }
}

/// How a gate that makes two terms of one is applied to the zones.
struct Splitting<'a, const W: usize, C> {
    /// Flipping the bits moves every term of a zone to the zone whose number differs by
    /// this, which is the number of the zone that the flipped bits would be in as a Pauli
    /// string.
    shift: usize,
    /// The runs of the flipped bits, see [`PauliString::bit_runs`].
    runs: Vec<(PauliString<W>, PauliString<W>)>,
    /// The weight above which no term is created.
    max_weight: Option<u32>,
    keep: Keep<'a, W, C>,
}

impl<const W: usize, C: Coefficient> Splitting<'_, W, C> {
    /// The number of tasks: one per zone if the terms stay in their zone, and else one
    /// per pair of zones that send their created terms to each other.
    fn tasks(&self, zones: usize) -> usize {
        if self.shift == 0 {
            zones
        } else {
            zones / 2
        }
    }

    /// One of the two zones of the pair with the number `index`; the other one has the
    /// number that differs by the shift. The pairs are numbered by their zone in which the
    /// lowest bit of the shift is not set, which is the number of the pair with a zero put
    /// in at that bit.
    fn lower_zone_of_pair(&self, index: usize) -> usize {
        let bit = self.shift.trailing_zeros();
        ((index >> bit) << (bit + 1)) | (index & ((1 << bit) - 1))
    }

    fn task<P: Path<Coeff = C>, S: Split<W, P>>(
        &self,
        gate: &S,
        index: usize,
        zones: &[Mutex<Zone<W, P>>],
    ) {
        if self.shift == 0 {
            let mut zone = lock(&zones[index]);
            zone.split(gate, self.max_weight);
            zone.order_made(&self.runs);
            zone.merge_made(self.keep);
            return;
        }
        let lower = self.lower_zone_of_pair(index);
        let mut lower_zone = lock(&zones[lower]);
        let mut upper_zone = lock(&zones[lower ^ self.shift]);
        let (lower, upper) = (&mut *lower_zone, &mut *upper_zone);
        lower.split(gate, self.max_weight);
        upper.split(gate, self.max_weight);
        lower.order_made(&self.runs);
        upper.order_made(&self.runs);
        lower.merge(
            &upper.made_keys[..upper.made],
            &upper.made_paths[..upper.made],
            self.keep,
        );
        upper.merge(
            &lower.made_keys[..lower.made],
            &lower.made_paths[..lower.made],
            self.keep,
        );
    }
}

/// What a round does to the zones.
enum Job<'a, const W: usize, P: Path> {
    Rest,
    Rotation {
        gate: Rotation<W>,
        splitting: Splitting<'a, W, P::Coeff>,
    },
    Damping {
        gate: Damping,
        splitting: Splitting<'a, W, P::Coeff>,
    },
    /// A Pauli channel.
    Scale {
        qubits: &'a [usize],
        factors: &'a [f64],
        keep: Keep<'a, W, P::Coeff>,
    },
    /// A Pauli gate.
    FlipSigns {
        gate: &'a Gate,
        keep: Keep<'a, W, P::Coeff>,
    },
    /// Any other Clifford gate.
    Conjugate {
        images: Images<W>,
        keep: Keep<'a, W, P::Coeff>,
        keep_others: Keep<'a, W, P::Coeff>,
    },
    /// Notes the largest absolute coefficient of every zone.
    Measure,
    Retain {
        keep: Keep<'a, W, P::Coeff>,
    },
    /// Doubles the number of zones, which is `zones` before.
    Divide {
        mark: PauliString<W>,
        zones: usize,
    },
}

impl<const W: usize, P: Path> Job<'_, W, P> {
    fn task(&self, index: usize, zones: &[Mutex<Zone<W, P>>]) {
        match self {
            Job::Rest => {}
            Job::Rotation { gate, splitting } => splitting.task(gate, index, zones),
            Job::Damping { gate, splitting } => splitting.task(gate, index, zones),
            Job::Scale {
                qubits,
                factors,
                keep,
            } => lock(&zones[index]).scale(qubits, factors, *keep),
            Job::FlipSigns { gate, keep } => lock(&zones[index]).flip_signs(gate, *keep),
            Job::Conjugate {
                images,
                keep,
                keep_others,
            } => lock(&zones[index]).conjugate(images, *keep, *keep_others),
            Job::Measure => {
                let mut zone = lock(&zones[index]);
                zone.largest = zone.max_abs_coeff();
            }
            Job::Retain { keep } => lock(&zones[index]).retain(*keep),
            Job::Divide { mark, zones: count } => {
                let moved = lock(&zones[index]).divide(mark);
                *lock(&zones[index + count]) = moved;
            }
        }
    }
}

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().expect("no thread failed while it held a zone")
}

/// How long a thread looks out for the next round before it lets other threads run in
/// between, and how long before it goes to sleep. Most rounds follow the last one at once
/// and take less time than waking a thread does.
const LOOKS_BEFORE_YIELDING: u32 = 1 << 14;
const LOOKS_BEFORE_SLEEPING: u32 = LOOKS_BEFORE_YIELDING + (1 << 10);

/// The zones and the threads that work on them.
///
/// The thread that runs the propagation hands out the work in rounds. A round is a job and a
/// number of tasks, each of which applies the job to one zone or one pair of zones. Every
/// thread claims tasks until none is left, the thread that started the round as well, which
/// then waits for the last task to finish. No two tasks of a round share a zone.
struct Crew<'a, const W: usize, P: Path> {
    zones: Vec<Mutex<Zone<W, P>>>,
    /// The job of the round. It is replaced only between rounds.
    job: RwLock<Job<'a, W, P>>,
    /// The number of the round, the number of its tasks and the number of the next task to
    /// claim, in the bits from 32, from 16 and from 0. As one number they change together,
    /// so that a thread which is late for a round cannot claim a task of the next one.
    claim: AtomicU64,
    /// The tasks of the round that are not finished.
    unfinished: AtomicUsize,
    /// Whether a task failed.
    failed: AtomicBool,
    /// Whether the propagation is over.
    dismissed: AtomicBool,
    /// The threads that went to sleep until the next round, and what wakes them.
    sleepers: AtomicUsize,
    asleep: Mutex<()>,
    wake: Condvar,
}

/// Counts a task as finished when it ends, in whatever way.
struct Finish<'c, 'a, const W: usize, P: Path>(&'c Crew<'a, W, P>);

impl<const W: usize, P: Path> Drop for Finish<'_, '_, W, P> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.0.failed.store(true, SeqCst);
        }
        self.0.unfinished.fetch_sub(1, SeqCst);
    }
}

/// Sends the helping threads home when the propagation ends, in whatever way.
struct Dismiss<'c, 'a, const W: usize, P: Path>(&'c Crew<'a, W, P>);

impl<const W: usize, P: Path> Drop for Dismiss<'_, '_, W, P> {
    fn drop(&mut self) {
        self.0.dismissed.store(true, SeqCst);
        let _asleep = self.0.asleep.lock();
        self.0.wake.notify_all();
    }
}

impl<'a, const W: usize, P: Path> Crew<'a, W, P> {
    /// A crew with room for `most` zones, of which the first holds all the terms.
    fn new(most: usize, first: Zone<W, P>) -> Self {
        let mut zones = vec![Mutex::new(first)];
        zones.resize_with(most, || Mutex::new(Zone::new(Vec::new(), Vec::new())));
        Self {
            zones,
            job: RwLock::new(Job::Rest),
            claim: AtomicU64::new(0),
            unfinished: AtomicUsize::new(0),
            failed: AtomicBool::new(false),
            dismissed: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            asleep: Mutex::new(()),
            wake: Condvar::new(),
        }
    }

    fn zone(&self, number: usize) -> MutexGuard<'_, Zone<W, P>> {
        lock(&self.zones[number])
    }

    /// Runs a round of `tasks` tasks of `job` and returns when all of them are finished.
    /// Only the thread that runs the propagation calls this.
    fn round(&self, job: Job<'a, W, P>, tasks: usize) {
        *self.job.write().expect("no thread failed with the job") = job;
        if tasks == 1 {
            let job = self.job.read().expect("no thread failed with the job");
            job.task(0, &self.zones);
            return;
        }
        if tasks == 0 {
            return;
        }
        // The number of the round starts again from zero when it has used up its bits.
        let round = ((self.claim.load(SeqCst) >> 32) + 1) & u64::from(u32::MAX);
        self.unfinished.store(tasks, SeqCst);
        self.claim
            .store((round << 32) | ((tasks as u64) << 16), SeqCst);
        // A thread that is going to sleep either sees the new round before it does, or is
        // counted here, and then it sleeps by the time the lock is taken.
        if self.sleepers.load(SeqCst) > 0 {
            let _asleep = self.asleep.lock();
            self.wake.notify_all();
        }
        self.work(round);
        while self.unfinished.load(SeqCst) != 0 {
            std::hint::spin_loop();
        }
        assert!(!self.failed.load(SeqCst), "a thread failed in a task");
    }

    /// Claims and runs tasks of the round with the number `round` until none is left.
    fn work(&self, round: u64) {
        loop {
            let claim = self.claim.load(SeqCst);
            let (tasks, next) = ((claim >> 16) & 0xffff, claim & 0xffff);
            if claim >> 32 != round || next >= tasks {
                return;
            }
            let claimed = self
                .claim
                .compare_exchange(claim, claim + 1, SeqCst, SeqCst);
            if claimed.is_ok() {
                // The round cannot end before this task does, so the job stays.
                let _finish = Finish(self);
                let job = self.job.read().expect("no thread failed with the job");
                job.task(next as usize, &self.zones);
            }
        }
    }

    /// Takes part in every round until the crew is dismissed.
    fn help(&self) {
        let mut seen = 0;
        let mut looks = 0;
        loop {
            if self.dismissed.load(SeqCst) {
                return;
            }
            let round = self.claim.load(SeqCst) >> 32;
            if round != seen {
                seen = round;
                looks = 0;
                self.work(round);
            } else if looks < LOOKS_BEFORE_YIELDING {
                looks += 1;
                std::hint::spin_loop();
            } else if looks < LOOKS_BEFORE_SLEEPING {
                looks += 1;
                std::thread::yield_now();
            } else {
                looks = 0;
                self.sleep_until_after(seen);
            }
        }
    }

    /// Sleeps until there is a round after the one with the number `seen`.
    fn sleep_until_after(&self, seen: u64) {
        let Ok(mut asleep) = self.asleep.lock() else {
            return;
        };
        self.sleepers.fetch_add(1, SeqCst);
        while !self.dismissed.load(SeqCst) && self.claim.load(SeqCst) >> 32 == seen {
            asleep = match self.wake.wait(asleep) {
                Ok(asleep) => asleep,
                Err(_) => break,
            };
        }
        self.sleepers.fetch_sub(1, SeqCst);
    }

    /// All the terms of the first `count` zones in the order of their Pauli strings.
    ///
    /// The Pauli strings are cut into ranges that hold about equally many terms. The terms
    /// of a range are in one stretch of every zone, and the ranges are merged by different
    /// threads.
    fn collapse(self, count: usize) -> (Vec<PauliString<W>>, Vec<P>) {
        let zones = self.zones.into_iter().take(count);
        let mut parts: Vec<(Vec<PauliString<W>>, Vec<P>)> = zones
            .map(|zone| {
                zone.into_inner()
                    .expect("no thread failed while it held a zone")
            })
            .map(|zone| (zone.keys, zone.paths))
            .collect();
        if parts.len() == 1 {
            return parts.pop().expect("there is always a zone");
        }
        let terms: usize = parts.iter().map(|part| part.0.len()).sum();
        let ranges = (terms / TERMS_PER_RANGE).clamp(1, 8 * rayon::current_num_threads());

        // The ranges start at evenly spaced strings of a sample of all the zones.
        let mut sample: Vec<PauliString<W>> = Vec::new();
        for (keys, _) in &parts {
            sample.extend((0..ranges).filter_map(|index| keys.get(index * keys.len() / ranges)));
        }
        sample.sort_unstable();
        let starts: Vec<PauliString<W>> = (1..ranges)
            .map(|range| sample[range * sample.len() / ranges])
            .collect();
        let bounds: Vec<Vec<usize>> = parts
            .iter()
            .map(|(keys, _)| {
                let inner = starts
                    .iter()
                    .map(|start| keys.partition_point(|key| key < start));
                [0].into_iter().chain(inner).chain([keys.len()]).collect()
            })
            .collect();

        let merged: Vec<(Vec<PauliString<W>>, Vec<P>)> = (0..ranges)
            .into_par_iter()
            .map(|range| {
                let stretches = parts.iter().zip(&bounds).map(|((keys, paths), bounds)| {
                    let stretch = bounds[range]..bounds[range + 1];
                    (&keys[stretch.clone()], &paths[stretch])
                });
                merge_all(stretches.collect())
            })
            .collect();
        let mut keys = Vec::with_capacity(terms);
        let mut paths = Vec::with_capacity(terms);
        for (range_keys, range_paths) in merged {
            keys.extend_from_slice(&range_keys);
            paths.extend_from_slice(&range_paths);
        }
        (keys, paths)
    }
}

/// A range of Pauli strings with fewer terms than this is not worth a thread of its own.
const TERMS_PER_RANGE: usize = 1 << 14;

/// Ordered sequences of terms, of which no two have a Pauli string in common, as one.
fn merge_all<const W: usize, P: Path>(
    sequences: Vec<(&[PauliString<W>], &[P])>,
) -> (Vec<PauliString<W>>, Vec<P>) {
    let pair = |pair: &[(&[PauliString<W>], &[P])]| match pair {
        [first, second] => interleave(*first, *second),
        [only] => (only.0.to_vec(), only.1.to_vec()),
        _ => unreachable!("a chunk has one or two elements"),
    };
    let mut merged: Vec<(Vec<PauliString<W>>, Vec<P>)> = sequences.chunks(2).map(pair).collect();
    while merged.len() > 1 {
        let borrowed: Vec<(&[PauliString<W>], &[P])> = merged
            .iter()
            .map(|(keys, paths)| (keys.as_slice(), paths.as_slice()))
            .collect();
        merged = borrowed.chunks(2).map(pair).collect();
    }
    merged.pop().unwrap_or_default()
}

/// The propagation as the thread that runs it sees it.
struct Engine<'c, 'a, const W: usize, P: Path> {
    crew: &'c Crew<'a, W, P>,
    /// The number of zones in use. The zone of a term has bit `r` of its number set if the
    /// Pauli string of the term anticommutes with `marks[r]`.
    count: usize,
    marks: Vec<PauliString<W>>,
    rules: &'a Rules<'a, W, P::Coeff>,
    /// The number of zones doubles when there are more terms per zone than this.
    terms_per_zone: usize,
    /// The largest number of zones.
    most: usize,
    num_qubits: usize,
    /// The state of the generator that the marks are drawn from.
    random: u64,
    /// Every X-bit and Z-bit that the Pauli string of a term can have. The sum may have
    /// lost the terms that had some of them.
    reach: PauliString<W>,
    /// Whether every term has been kept by all the rules since it last changed.
    checked: bool,
    /// Whether the sum can have more terms than when the zones were last counted.
    grown: bool,
}

impl<'a, const W: usize, P: Path> Engine<'_, 'a, W, P> {
    /// Applies the gates one after the other and returns how many of them it applied. If
    /// it may not divide the sum into more zones, it stops before the gate at which the
    /// sum has grown enough to be divided.
    fn run(&mut self, gates: &[&'a Gate], may_divide: bool) -> usize {
        for (done, gate) in gates.iter().enumerate() {
            if self.grown && !self.grow(may_divide) {
                return done;
            }
            self.apply(gate);
        }
        gates.len()
    }

    /// Applies the action of `gate` on observables to every term, then truncates.
    fn apply(&mut self, gate: &'a Gate) {
        // A gate that changes no term leaves nothing to truncate if every term was kept by
        // the rules before, and if the rules give the same answer when asked again.
        let settled = self.settled();
        let reaches = |qubits: &[usize]| qubits.iter().any(|&q| self.reach.get(q) != PauliOp::I);
        match gate {
            Gate::Rotation { generator, theta } => {
                let generator = PauliString::from_ops(generator);
                if settled && !self.reach.can_anticommute(&generator) {
                    return;
                }
                let (sin, cos) = theta.sin_cos();
                let gate = Rotation {
                    generator,
                    sin,
                    cos,
                };
                let splitting = self.splitting(&generator);
                let tasks = splitting.tasks(self.count);
                self.crew.round(Job::Rotation { gate, splitting }, tasks);
                self.reach = self.reach.union(&generator);
            }
            Gate::AmplitudeDamping { qubit, gamma } => {
                if settled && !reaches(&[*qubit]) {
                    return;
                }
                let gate = Damping {
                    qubit: *qubit,
                    gamma: *gamma,
                };
                let splitting = self.splitting(&Split::<W, P>::flips(&gate));
                let tasks = splitting.tasks(self.count);
                self.crew.round(Job::Damping { gate, splitting }, tasks);
            }
            Gate::PauliChannel { qubits, factors } => {
                if settled && factors[0] == 1.0 && !reaches(qubits) {
                    return;
                }
                let keep = self.keep_after_split(settled);
                let job = Job::Scale {
                    qubits,
                    factors,
                    keep,
                };
                self.crew.round(job, self.count);
            }
            Gate::TransferMap { qubits, columns } => {
                self.transfer(qubits, columns);
                self.reach_all_of(qubits);
                self.checked = true;
                self.grown = true;
                return;
            }
            _ => {
                self.conjugate(gate);
                return;
            }
        }
        if self.rules.is_relative() {
            self.truncate();
        }
        self.checked = true;
        self.grown = true;
    }

    /// Whether every term is known to be kept by the rules as long as it does not change.
    /// A relative threshold moves with the other terms, and nothing is known about what a
    /// rule of the caller does.
    fn settled(&self) -> bool {
        self.checked && !self.rules.is_relative() && self.rules.custom.is_none()
    }

    /// Notes that a gate can leave any operator on `qubits`.
    fn reach_all_of(&mut self, qubits: &[usize]) {
        let all: Vec<(usize, PauliOp)> = qubits.iter().map(|&q| (q, PauliOp::Y)).collect();
        self.reach = self.reach.union(&PauliString::from_ops(&all));
    }

    /// The zone of the term with the Pauli string `key`.
    fn zone_of(&self, key: &PauliString<W>) -> usize {
        let marks = self.marks.iter().enumerate();
        marks.fold(0, |zone, (bit, mark)| {
            zone | (usize::from(!key.commutes(mark)) << bit)
        })
    }

    /// Which terms stay when a gate has changed coefficients but has left the Pauli
    /// strings as they were, or has created no string that is too heavy.
    ///
    /// The smallest coefficient that a relative threshold keeps is known only when the gate
    /// has been applied to all the terms, so then all of them stay until
    /// [`Engine::truncate`] is called.
    fn keep_after_split(&self, settled: bool) -> Keep<'a, W, P::Coeff> {
        let rules = self.rules;
        if rules.is_relative() {
            Keep::all(rules)
        } else if settled {
            Keep::new(rules, rules.truncation.min_abs_coeff).of_settled_weights()
        } else {
            Keep::new(rules, rules.truncation.min_abs_coeff)
        }
    }

    /// How to apply a gate that flips the bits `flips` in the Pauli strings it creates.
    fn splitting(&self, flips: &PauliString<W>) -> Splitting<'a, W, P::Coeff> {
        // Unless the threshold is relative, a created term that is too heavy is not
        // created: the term it would merge with has the same Pauli string and is dropped
        // as well.
        let max_weight = if self.rules.is_relative() {
            None
        } else {
            self.rules.truncation.max_weight
        };
        Splitting {
            shift: self.zone_of(flips),
            runs: flips.bit_runs(),
            max_weight,
            keep: self.keep_after_split(self.settled()),
        }
    }

    /// Drops the terms that a relative threshold selects, with the largest absolute
    /// coefficient taken from the sum as it is.
    fn truncate(&mut self) {
        let rules = self.rules;
        self.crew.round(Job::Measure, self.count);
        let min_abs = rules.threshold(|| {
            let largest = (0..self.count).map(|zone| self.crew.zone(zone).largest);
            largest.fold(0.0, f64::max)
        });
        let keep = Keep::new(rules, min_abs);
        self.crew.round(Job::Retain { keep }, self.count);
    }

    /// Applies a Clifford gate, which is a bijection of the Pauli strings up to signs. That
    /// leaves nothing to drop for a rule that reads only the coefficient, and the relative
    /// threshold is not applied.
    fn conjugate(&mut self, gate: &'a Gate) {
        let rules = self.rules;
        let keep = if rules.reads_string() {
            Keep::new(rules, rules.truncation.min_abs_coeff)
        } else {
            Keep::all(rules)
        };

        // A Pauli gate only changes signs.
        if matches!(gate, Gate::X(_) | Gate::Y(_) | Gate::Z(_)) {
            self.crew.round(Job::FlipSigns { gate, keep }, self.count);
            return;
        }

        // The gate can leave any operator on its qubits, unless every term has the identity
        // on all of them. Then it changes nothing, which leaves nothing to truncate if
        // every term was kept by the rules before.
        let images = Images::of(gate);
        if images
            .qubits
            .iter()
            .all(|&q| self.reach.get(q) == PauliOp::I)
        {
            if keep.all || self.settled() {
                return;
            }
        } else {
            self.reach_all_of(&images.qubits);
        }

        // Two strings anticommute exactly if their images do, so with the marks replaced by
        // their images every term stays in its zone.
        for mark in &mut self.marks {
            gate.conjugate_clifford(mark);
        }
        // A term that keeps its Pauli string keeps its coefficient up to the sign, so it
        // stays if it was kept by the rules before.
        let keep_others = if self.settled() {
            Keep::all(rules).of_settled_weights()
        } else {
            keep
        };
        let job = Job::Conjugate {
            images,
            keep,
            keep_others,
        };
        self.crew.round(job, self.count);
    }

    /// Replaces every term by its image under a transfer map on `qubits`, then truncates.
    ///
    /// Several terms can contribute to one Pauli string. Their contributions are merged in
    /// the order of the Pauli strings they come from.
    fn transfer(&mut self, qubits: &[usize], columns: &[Vec<(usize, f64)>]) {
        let rules = self.rules;
        let mut zones: Vec<MutexGuard<'_, Zone<W, P>>> =
            (0..self.count).map(|zone| self.crew.zone(zone)).collect();
        let mut images: Vec<(PauliString<W>, PauliString<W>, P)> = Vec::new();
        for zone in &mut zones {
            for (key, path) in zone.keys.iter().zip(&zone.paths) {
                for &(index, factor) in &columns[Gate::channel_index(qubits, key)] {
                    images.push((
                        Gate::with_index(qubits, *key, index),
                        *key,
                        path.scale(factor),
                    ));
                }
            }
            zone.keys.clear();
            zone.paths.clear();
        }
        images.sort_unstable_by_key(|&(image, source, _)| (image, source));

        let mut keys: Vec<PauliString<W>> = Vec::new();
        let mut paths: Vec<P> = Vec::new();
        for (image, _, path) in images {
            if keys.last() == Some(&image) {
                let last = paths.last_mut().expect("there is a path for every key");
                *last = last.merge(path);
            } else {
                keys.push(image);
                paths.push(path);
            }
        }

        let min_abs = rules.threshold(|| {
            let sizes = paths.iter().map(|path| path.coeff().abs());
            sizes.fold(0.0, f64::max)
        });
        for (key, path) in keys.iter().zip(&paths) {
            if !rules.drops(key, path, min_abs) {
                let zone = &mut zones[self.zone_of(key)];
                zone.keys.push(*key);
                zone.paths.push(*path);
            }
        }
    }

    /// Doubles the number of zones until the sum has no more terms per zone than asked for.
    /// Returns whether the zones are as many as they should be, which they are not if the
    /// sum has to be divided and `may_divide` is not set.
    fn grow(&mut self, may_divide: bool) -> bool {
        let terms: usize = (0..self.count)
            .map(|zone| self.crew.zone(zone).keys.len())
            .sum();
        let mut wanted = self.count;
        while 2 * wanted <= self.most && terms > wanted.saturating_mul(self.terms_per_zone) {
            wanted *= 2;
        }
        // Most gates pair the zones up, which leaves half as many tasks as zones, so a few
        // zones are not worth what it costs to hand out their work.
        if wanted > self.count && wanted >= FEWEST_ZONES.min(self.most) {
            if !may_divide {
                return false;
            }
            while self.count < wanted {
                let mark = self.draw_mark();
                let job = Job::Divide {
                    mark,
                    zones: self.count,
                };
                self.crew.round(job, self.count);
                self.marks.push(mark);
                self.count *= 2;
            }
        }
        self.grown = false;
        true
    }

    /// A Pauli string with an operator other than the identity on every qubit, each drawn
    /// from a fixed sequence of numbers that look random. About half of any varied set of
    /// Pauli strings anticommutes with it.
    fn draw_mark(&mut self) -> PauliString<W> {
        let mut mark = PauliString::identity();
        for qubit in 0..self.num_qubits {
            // The xorshift generator of Marsaglia, multiplied to mix the high bits.
            self.random ^= self.random >> 12;
            self.random ^= self.random << 25;
            self.random ^= self.random >> 27;
            let number = self.random.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 32;
            mark.set(
                qubit,
                [PauliOp::X, PauliOp::Y, PauliOp::Z][(number % 3) as usize],
            );
        }
        mark
    }
}

/// Two ordered sequences of terms with no Pauli string in common as one.
fn interleave<const W: usize, P: Path>(
    first: (&[PauliString<W>], &[P]),
    second: (&[PauliString<W>], &[P]),
) -> (Vec<PauliString<W>>, Vec<P>) {
    let length = first.0.len() + second.0.len();
    let mut keys = Vec::with_capacity(length);
    let mut paths = Vec::with_capacity(length);
    let (mut a, mut b) = (0, 0);
    while a < first.0.len() && b < second.0.len() {
        if first.0[a].is_before(&second.0[b]) {
            keys.push(first.0[a]);
            paths.push(first.1[a]);
            a += 1;
        } else {
            keys.push(second.0[b]);
            paths.push(second.1[b]);
            b += 1;
        }
    }
    keys.extend_from_slice(&first.0[a..]);
    paths.extend_from_slice(&first.1[a..]);
    keys.extend_from_slice(&second.0[b..]);
    paths.extend_from_slice(&second.1[b..]);
    (keys, paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

    /// A string with operators on a few of the qubits `places`, which lie in different words.
    fn random_string(rng: &mut StdRng, places: &[usize]) -> PauliString<3> {
        let mut string = PauliString::identity();
        for &qubit in places {
            let op = [PauliOp::I, PauliOp::I, PauliOp::X, PauliOp::Y, PauliOp::Z];
            string.set(qubit, *op.choose(rng).unwrap());
        }
        string
    }

    #[test]
    fn regrouping_orders_the_strings_after_bits_were_flipped() {
        let mut rng = StdRng::seed_from_u64(1);
        let places = [0, 1, 2, 3, 62, 63, 64, 65, 100, 130, 191];
        let (mut by_passes, mut by_sorting, mut moved) = (0, 0, 0);
        for round in 0..600 {
            // Strings in order, and some bits to flip in all of them.
            let mut before: Vec<PauliString<3>> = (0..rng.gen_range(0..300))
                .map(|_| random_string(&mut rng, &places))
                .collect();
            before.sort_unstable();
            before.dedup();
            let few = &places[..rng.gen_range(1..places.len())];
            let flips = if round % 2 == 0 {
                random_string(&mut rng, few)
            } else {
                random_string(&mut rng, &places)
            };
            let runs = flips.bit_runs();
            if runs.len() > MAX_REGROUPINGS {
                by_sorting += 1;
            } else {
                by_passes += 1;
            }

            let mut keys: Vec<PauliString<3>> = before.iter().map(|key| key.xor(&flips)).collect();
            // Each path remembers the place its string came from.
            let mut paths: Vec<f64> = (0..keys.len()).map(|place| place as f64).collect();
            moved += usize::from(keys.windows(2).any(|pair| pair[0] > pair[1]));
            regroup(&mut keys, &mut paths, &runs);

            assert!(
                keys.windows(2).all(|pair| pair[0] < pair[1]),
                "round {round}"
            );
            for (key, path) in keys.iter().zip(&paths) {
                assert_eq!(*key, before[*path as usize].xor(&flips), "round {round}");
            }
        }
        assert!(
            by_passes > 50 && by_sorting > 50,
            "{by_passes} by passes, {by_sorting} sorted"
        );
        assert!(moved > 300, "the order changed in only {moved} rounds");
    }

    #[test]
    fn the_images_under_a_clifford_gate_are_those_of_its_table() {
        let mut rng = StdRng::seed_from_u64(2);
        let places = [3, 63, 64, 70, 150];
        let (a, b) = (63, 150);
        let gates = [
            Gate::H(a),
            Gate::S(a),
            Gate::Sdg(a),
            Gate::Sx(a),
            Gate::Sxdg(a),
            Gate::Cx(a, b),
            Gate::Cx(b, a),
            Gate::Cy(a, b),
            Gate::Cy(b, a),
            Gate::Cz(a, b),
            Gate::Swap(a, b),
        ];
        for gate in &gates {
            let images = Images::<3>::of(gate);
            assert_eq!(images.flips[0], PauliString::identity());
            assert!(images.flips.len() > 1, "{gate:?} changes some string");
            assert_eq!(images.flips.len(), images.runs.len());
            let mut flipped = vec![false; images.flips.len()];
            for _ in 0..500 {
                let key = random_string(&mut rng, &places);
                let index = Gate::channel_index(&images.qubits, &key);
                let class = usize::from(images.class[index]);
                flipped[class] = true;

                let mut expected = key;
                let minus = gate.conjugate_clifford(&mut expected);
                assert_eq!(key.xor(&images.flips[class]), expected, "{gate:?}");
                assert_eq!(images.minus[index], minus, "{gate:?}");
                assert_eq!(images.runs[class], images.flips[class].bit_runs());
            }
            assert!(
                flipped.iter().all(|&seen| seen),
                "{gate:?}: a class was not tried"
            );
        }
    }

    #[test]
    fn the_pairs_of_zones_are_numbered_without_gaps() {
        // Every zone is in the pair of exactly one task, whatever the shift.
        let truncation = crate::Truncation::none();
        let rules: Rules<'_, 1, f64> = Rules {
            truncation: &truncation,
            custom: None,
        };
        for zones in [2usize, 8, 64] {
            for shift in 1..zones {
                let splitting = Splitting {
                    shift,
                    runs: Vec::new(),
                    max_weight: None,
                    keep: Keep::all(&rules),
                };
                assert_eq!(splitting.tasks(zones), zones / 2);
                let mut seen = vec![0; zones];
                for index in 0..splitting.tasks(zones) {
                    let lower = splitting.lower_zone_of_pair(index);
                    seen[lower] += 1;
                    seen[lower ^ shift] += 1;
                }
                assert!(
                    seen.iter().all(|&count| count == 1),
                    "{zones} zones, shift {shift}"
                );
            }
        }
    }
}
