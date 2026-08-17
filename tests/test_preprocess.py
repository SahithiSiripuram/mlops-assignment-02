"""Unit tests for the data pre-processing functions (M3.1)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.data.preprocess import is_valid_image, resize_image, stratified_split
from tests.conftest import make_image


class TestIsValidImage:
    def test_accepts_a_readable_jpeg(self, tmp_path: Path) -> None:
        path = tmp_path / "cat.jpg"
        make_image().save(path)
        assert is_valid_image(path) is True

    def test_rejects_a_zero_byte_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jpg"
        path.write_bytes(b"")
        assert is_valid_image(path) is False

    def test_rejects_a_corrupt_file(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"this is definitely not a jpeg")
        assert is_valid_image(path) is False

    def test_rejects_a_non_image_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.txt"
        path.write_text("hello", encoding="utf-8")
        assert is_valid_image(path) is False

    def test_rejects_a_missing_file(self, tmp_path: Path) -> None:
        assert is_valid_image(tmp_path / "nope.jpg") is False


class TestResizeImage:
    def test_resizes_to_the_requested_square(self, tmp_path: Path) -> None:
        source = tmp_path / "source.jpg"
        make_image(size=400).save(source)

        target = resize_image(source, tmp_path / "out" / "target.jpg", size=224)

        assert target.exists()
        with Image.open(target) as img:
            assert img.size == (224, 224)
            assert img.mode == "RGB"

    def test_converts_greyscale_and_rgba_to_rgb(self, tmp_path: Path) -> None:
        for mode, name in [("L", "grey.png"), ("RGBA", "alpha.png")]:
            source = tmp_path / name
            Image.new(mode, (128, 96)).save(source)

            target = resize_image(source, tmp_path / f"out_{name}.jpg", size=64)

            with Image.open(target) as img:
                assert img.mode == "RGB"
                assert img.size == (64, 64)


class TestStratifiedSplit:
    @pytest.fixture()
    def items(self) -> list[Path]:
        return [Path(f"img_{i}.jpg") for i in range(100)]

    def test_splits_80_10_10(self, items: list[Path]) -> None:
        splits = stratified_split(items, 0.8, 0.1, 0.1, seed=42)
        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_every_item_lands_in_exactly_one_split(self, items: list[Path]) -> None:
        splits = stratified_split(items, 0.8, 0.1, 0.1, seed=42)
        combined = splits["train"] + splits["val"] + splits["test"]
        assert len(combined) == len(items)
        assert set(combined) == set(items)
        assert not set(splits["train"]) & set(splits["val"])
        assert not set(splits["val"]) & set(splits["test"])

    def test_is_deterministic_for_a_fixed_seed(self, items: list[Path]) -> None:
        first = stratified_split(items, 0.8, 0.1, 0.1, seed=7)
        second = stratified_split(items, 0.8, 0.1, 0.1, seed=7)
        assert first == second

    def test_different_seeds_shuffle_differently(self, items: list[Path]) -> None:
        first = stratified_split(items, 0.8, 0.1, 0.1, seed=1)
        second = stratified_split(items, 0.8, 0.1, 0.1, seed=2)
        assert first["test"] != second["test"]

    def test_rejects_ratios_that_do_not_sum_to_one(self, items: list[Path]) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            stratified_split(items, 0.7, 0.1, 0.1)

    def test_handles_an_odd_number_of_items(self) -> None:
        items = [Path(f"{i}.jpg") for i in range(7)]
        splits = stratified_split(items, 0.8, 0.1, 0.1, seed=42)
        assert sum(len(v) for v in splits.values()) == 7
        assert len(splits["train"]) >= 5
