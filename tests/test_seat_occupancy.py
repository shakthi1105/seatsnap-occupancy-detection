"""Lightweight tests for the supplied SeatSnap geometry and CSV logic.

Dependencies are stubbed so CI doesn't download YOLO models or OpenCV wheels.
"""
import csv
import importlib
import os
import sys
import tempfile
import types
import unittest

sys.modules.setdefault("ultralytics", types.SimpleNamespace(YOLO=object))
sys.modules.setdefault("cv2", types.SimpleNamespace())
seat = importlib.import_module("seat_occupancy")


class SeatGeometryTests(unittest.TestCase):
    def test_iou_identical_boxes(self):
        self.assertAlmostEqual(seat.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1)

    def test_iou_disjoint_boxes(self):
        self.assertEqual(seat.iou([0, 0, 2, 2], [3, 3, 5, 5]), 0)

    def test_iou_partial_overlap(self):
        self.assertAlmostEqual(seat.iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_seat_ids_sort_rows_then_columns(self):
        chairs = [[100, 100, 130, 130], [100, 0, 130, 30], [0, 0, 30, 30]]
        ordered = seat.assign_seat_ids(chairs)
        self.assertEqual(ordered, [(1, chairs[2]), (2, chairs[1]), (3, chairs[0])])

    def test_occupancy_matches_person_intersection(self):
        chairs = [(1, [0, 0, 10, 10]), (2, [20, 0, 30, 10])]
        persons = [[0, 0, 10, 10]]
        result = seat.compute_seat_status(chairs, persons)
        self.assertEqual([(x["id"], x["occupied"]) for x in result], [(1, True), (2, False)])

    def test_csv_export_schema_and_append(self):
        with tempfile.TemporaryDirectory() as folder:
            filename = os.path.join(folder, "seats.csv")
            statuses = [{"id": 1, "occupied": True}, {"id": 2, "occupied": False}]
            seat.generate_seat_level_csv(statuses, "example.jpg", output_file=filename)
            seat.generate_seat_level_csv(statuses[:1], "next.jpg", output_file=filename)
            with open(filename, newline="") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows, [
                ["image_name", "seat_id", "occupied"],
                ["example.jpg", "1", "yes"],
                ["example.jpg", "2", "no"],
                ["next.jpg", "1", "yes"],
            ])


if __name__ == "__main__":
    unittest.main()
