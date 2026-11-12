"""The cases of ``test/test_paulialgebra_utils.jl`` of PauliPropagation.jl that have a
counterpart here: reading and writing the operators of a Pauli string by qubit, and changing
the type of the coefficients.
"""

import pytest

import paulipropagation_qdk as pp


def test_operators_are_read_and_written_by_qubit():
    total = pp.PauliSum(4)
    total.add("X", [1])
    assert total.terms() == [("IXII", 1.0)]
    assert total.coeff("X", [1]) == 1.0
    assert total.coeff("IX", [0, 1]) == 1.0
    assert total.coeff("IXI", [2, 1, 0]) == 1.0
    assert total.coeff("X", [0]) == 0.0

    # The order of the qubits is the order of the operators, whatever it is.
    for paulis, qubits in [("XZ", [1, 3]), ("ZX", [3, 1]), ("ZIX", [3, 0, 1])]:
        ordered = pp.PauliSum(4)
        ordered.add(paulis, qubits, 0.5)
        assert ordered.terms() == [("IXIZ", 0.5)]
        assert ordered.coeff("ZX", [3, 1]) == ordered.coeff("XZ", [1, 3]) == 0.5
        assert ordered.coeff("XZ", [3, 1]) == 0.0


def test_operators_far_apart_on_the_widest_register():
    total = pp.PauliSum(512)
    total.add("XZY", [2, 70, 511], 0.25)
    ((text, coeff),) = total.terms()
    assert coeff == 0.25
    assert len(text) == 512
    assert {q: char for q, char in enumerate(text) if char != "I"} == {2: "X", 70: "Z", 511: "Y"}
    assert total.coeff("YZX", [511, 70, 2]) == 0.25

    with pytest.raises(ValueError, match="513 qubits do not fit"):
        pp.PauliSum(513)


def test_strings_and_qubits_that_do_not_fit_are_rejected():
    total = pp.PauliSum(4)
    with pytest.raises(ValueError, match="not one of I, X, Y, Z"):
        total.add("A", [0])
    with pytest.raises(ValueError, match="2 Pauli operators were given for 1 qubits"):
        total.add("XY", [0])
    with pytest.raises(ValueError, match="out of range"):
        total.add("X", [4])
    with pytest.raises(ValueError):
        total.add("XY", [1, 1])
    assert len(total) == 0


def test_coefficient_types_convert():
    total = pp.PauliSum(2, dtype=complex)
    total.add("XY", [0, 1], 2 + 0j)
    total.add("Z", [0], 1 + 0j)
    expected = pp.PauliSum(2)
    expected.add("XY", [0, 1], 2.0)
    expected.add("Z", [0], 1.0)

    real = total.astype(float)
    assert real.dtype is float
    assert real.terms() == expected.terms()
    assert all(type(coeff) is float for _, coeff in real.terms())

    back = real.astype(complex)
    assert back.dtype is complex
    assert back.terms() == total.terms()
    assert all(type(coeff) is complex for _, coeff in back.terms())

    # A conversion to the same type is a copy.
    copy = real.astype(float)
    copy.add("Z", [1])
    assert len(copy) == len(real) + 1


def test_a_coefficient_with_an_imaginary_part_has_no_real_type():
    total = pp.PauliSum(2, dtype=complex)
    total.add("X", [0], 1.0)
    total.add("Y", [1], 0.5j)
    with pytest.raises(ValueError, match="IY has an imaginary part"):
        total.astype(float)
    with pytest.raises(ValueError, match="dtype must be float or complex"):
        total.astype(int)
