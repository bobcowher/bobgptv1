from dataset import SwitchLoader


def test_switches_at_the_step():
    loader = SwitchLoader(["a"] * 10, ["b"] * 3, switch_step=4)
    assert list(loader) == ["a"] * 4 + ["b"] * 3
    assert len(loader) == 7


def test_first_shorter_than_the_switch():
    loader = SwitchLoader(["a"] * 2, ["b"] * 3, switch_step=4)
    assert list(loader) == ["a", "a", "b", "b", "b"]
    assert len(loader) == 5
