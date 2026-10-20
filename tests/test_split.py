import numpy as np
import src.train as train


def test_split_deterministic():
    # monkeypatch splits path to temp by calling get_split with small n
    tr1, va1, te1 = train.get_split(100, 42)
    tr2, va2, te2 = train.get_split(100, 42)
    assert np.array_equal(tr1, tr2)
    assert len(tr1) + len(va1) + len(te1) == 100
