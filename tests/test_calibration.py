from puzzle_recognition.calibration import order_corners, parse_corners, parse_size


def test_parse_size():
    assert parse_size("2000x1000") == (2000, 1000)


def test_parse_corners():
    assert parse_corners("0,0;10,0;10,10;0,10") == [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]


def test_order_corners():
    ordered = order_corners([(10, 10), (0, 0), (10, 0), (0, 10)])
    assert ordered.tolist() == [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]

