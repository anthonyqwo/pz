from puzzle_recognition.piece_labeler import sort_pieces_spatial


def test_sort_pieces_spatial_relabels_top_to_bottom_left_to_right():
    pieces = [
        {"piece_id": "piece_x", "center": [300, 200]},
        {"piece_id": "piece_x", "center": [100, 20]},
        {"piece_id": "piece_x", "center": [200, 20]},
    ]

    sorted_pieces = sort_pieces_spatial(pieces, row_bucket=100)

    assert [piece["center"] for piece in sorted_pieces] == [[100, 20], [200, 20], [300, 200]]
    assert [piece["piece_id"] for piece in sorted_pieces] == ["piece_001", "piece_002", "piece_003"]

