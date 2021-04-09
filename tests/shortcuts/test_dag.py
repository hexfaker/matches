import torch
from pytest import fixture

from matches.shortcuts.dag import ComputationGraph, graph_node


class TestCG(ComputationGraph):
    @graph_node
    def make_mask(self):
        return torch.randn((10, 10, 10), requires_grad=True)


@fixture
def dag() -> TestCG:
    return TestCG()


def test_cache(dag: TestCG):
    with dag.cache_scope():
        res1 = dag.make_mask()
        res2 = dag.make_mask()

    assert id(res1) == id(res2)


def test_no_grad(dag: TestCG):
    with dag.cache_scope():
        res1 = dag.make_mask()
        with torch.no_grad():
            res2 = dag.make_mask()

    assert res1.requires_grad
    assert not res2.requires_grad
