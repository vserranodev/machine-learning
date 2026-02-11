import pytest
 
def str_to_float(str):
    result = 37
    return float(str)
class Test:
    #Setup function runs before every test
    def setup(self):
        print("\nthis is setup")

    def teardown(self):
        print("\this is teardown")

    def setup_class(cls):
        print("\nthis is setup class")

    def teardown_class(cls):
        print("\nthis is teardown class")

    def test_rounds_down(self):
        result = str_to_float("2")
        assert result == 2.0

    def test_round_down_lesser_half(self):
        result = str_to_float("1.3.2")
        assert result == 1.2

    
def test_dicts():
    result = {"key": "value", "lastname": "karpathy", "firstname": "andrej"}
    expected = {"key": "value", "lastname": "karpathy", "firstname": "andrej"}
    assert result == expected


@pytest.mark.parametrize("value", ["1.2", "4", "4.5"])
def test_is_true(value):
    result = str_to_float(value)
    assert isinstance(result, float)




