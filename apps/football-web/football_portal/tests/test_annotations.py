import copy
import re
import unittest
from unittest.mock import patch

from PIL import Image, ImageColor, ImageDraw

from football_portal.annotations import ROLE_COLORS, render_annotations


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (400, 240), (35, 90, 52))
        self.boxes = [{"cls": name, "conf": .876, "x1": 20 + index * 90, "y1": 80,
                       "x2": 40 + index * 90, "y2": 140}
                      for index, name in enumerate(("player", "goalkeeper", "referee"))]

    def test_role_colors_and_original_data_are_preserved(self):
        original, boxes = self.image.tobytes(), copy.deepcopy(self.boxes)
        result = render_annotations(self.image, self.boxes)
        self.assertEqual(self.image.tobytes(), original)
        self.assertEqual(self.boxes, boxes)
        self.assertEqual(result.size, self.image.size)
        for box in boxes:
            self.assertEqual(result.getpixel((box["x1"], 110)), ImageColor.getrgb(ROLE_COLORS[box["cls"]]))

    def test_photo_text_contains_only_confidence(self):
        texts = []
        original = ImageDraw.ImageDraw.text

        def capture(draw, xy, text, *args, **kwargs):
            texts.append(text)
            return original(draw, xy, text, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", new=capture):
            render_annotations(self.image, self.boxes)
        self.assertEqual(len(texts), 3)
        self.assertTrue(all(re.fullmatch(r"0\.\d{2}", text) for text in texts))

    def test_small_ball_has_a_yellow_ring_outside_its_true_box(self):
        ball = {"cls": "ball", "conf": .43, "x1": 300, "y1": 100, "x2": 304, "y2": 104}
        result = render_annotations(self.image, [ball])
        self.assertEqual(result.getpixel((302, 95)), ImageColor.getrgb(ROLE_COLORS["ball"]))
        self.assertEqual(ball["x2"] - ball["x1"], 4)

    def test_no_detection_keeps_the_picture_unchanged(self):
        self.assertEqual(render_annotations(self.image, []).tobytes(), self.image.tobytes())

    def test_ball_at_image_edge_is_rendered_without_out_of_range_errors(self):
        result = render_annotations(self.image, [{"cls": "ball", "conf": .9, "x1": 0, "y1": 0, "x2": 3, "y2": 3}])
        self.assertEqual(result.size, (400, 240))
