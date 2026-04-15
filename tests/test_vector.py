from demetria import VectorField


class TestVector():
    def test_scalar_times_vector(self, trough, rotating_disk, unit_square):
        computed = trough * rotating_disk
        assert isinstance(computed, VectorField), \
            f"Expected a VectorField, got {type(computed)}"
