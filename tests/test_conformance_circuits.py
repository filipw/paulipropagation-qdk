"""The circuits and topologies of the conformance tests against those PauliPropagation.jl builds.

``tests/reference/tables/circuits.json`` holds what the Julia package returns; the builders
in ``reference_helpers`` have to return the same gates in the same order.
"""

import json

import reference_helpers as helpers

TABLES = json.loads((helpers.REFERENCE_DIR / "tables" / "circuits.json").read_text())


def test_topologies_match_the_julia_package():
    builders = {
        "bricklayer": helpers.bricklayer_topology,
        "staircase": helpers.staircase_topology,
        "rectangle": helpers.rectangle_topology,
    }
    assert {entry["kind"] for entry in TABLES["topologies"]} == set(builders)
    for entry in TABLES["topologies"]:
        pairs = builders[entry["kind"]](*entry["size"], periodic=entry["periodic"])
        assert pairs == [tuple(pair) for pair in entry["pairs"]], entry


def test_circuits_match_the_julia_package():
    builders = {
        "hardware_efficient": helpers.hardware_efficient_circuit,
        "efficient_su2": helpers.efficient_su2_circuit,
        "tfi_trotter": helpers.tfi_trotter_circuit,
        "tilted_tfi_trotter": helpers.tilted_tfi_trotter_circuit,
    }
    assert {entry["kind"] for entry in TABLES["circuits"]} == set(builders)
    for entry in TABLES["circuits"]:
        topology = entry["topology"] and [tuple(pair) for pair in entry["topology"]]
        gates = builders[entry["kind"]](entry["num_qubits"], entry["layers"], topology)
        expected = [(kind, tuple(qubits)) for kind, qubits in entry["gates"]]
        assert gates == expected, (entry["kind"], entry["num_qubits"], entry["layers"])
        assert (len(gates) == 0) == (entry["layers"] == 0)
