# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for LazyDataset."""
import dataclasses
from typing import cast

from absl.testing import absltest
from absl.testing import parameterized
from grain._src.core import transforms
from grain._src.python import options
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations import filter as filter_lazy_dataset


@dataclasses.dataclass(frozen=True)
class FilterEvenElementsOnly(transforms.FilterTransform):

  def filter(self, element: int):
    return element % 2


class RangeLazyMapDatasetTest(absltest.TestCase):

  def test_len(self):
    ds = lazy_dataset.RangeLazyMapDataset(12)
    self.assertLen(ds, 12)
    ds = lazy_dataset.RangeLazyMapDataset(0, 12)
    self.assertLen(ds, 12)
    ds = lazy_dataset.RangeLazyMapDataset(2, 12)
    self.assertLen(ds, 10)
    ds = lazy_dataset.RangeLazyMapDataset(2, 12, 1)
    self.assertLen(ds, 10)
    ds = lazy_dataset.RangeLazyMapDataset(2, 12, 2)
    self.assertLen(ds, 5)
    ds = lazy_dataset.RangeLazyMapDataset(2, 13, 2)
    self.assertLen(ds, 6)

  def test_getitem(self):
    ds = lazy_dataset.RangeLazyMapDataset(12)
    for i in range(12):
      self.assertEqual(ds[i], i)
    for i in range(12):
      self.assertEqual(ds[i + 12], i)
    ds = lazy_dataset.RangeLazyMapDataset(2, 9, 2)
    self.assertEqual(ds[0], 2)
    self.assertEqual(ds[1], 4)
    self.assertEqual(ds[2], 6)
    self.assertEqual(ds[3], 8)
    self.assertEqual(ds[4], 2)
    self.assertEqual(ds[5], 4)

  def test_iter(self):
    ds = lazy_dataset.RangeLazyMapDataset(12)
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(12)]
    self.assertEqual(elements, list(range(12)))
    ds = lazy_dataset.RangeLazyMapDataset(2, 9, 2)
    ds_iter = iter(ds)
    elements = [next(ds_iter) for _ in range(4)]
    self.assertEqual(elements, [2, 4, 6, 8])


class PrefetchLazyIterDatasetTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.range_ds = lazy_dataset.RangeLazyMapDataset(20)
    self.filtered_range_ds = filter_lazy_dataset.FilterLazyMapDataset(
        self.range_ds, FilterEvenElementsOnly()
    )
    self.prefetch_lazy_iter_ds = lazy_dataset.PrefetchLazyIterDataset(
        self.range_ds, options.ReadOptions()
    )

  def test_dataset_and_iterator_types(self):
    self.assertIsInstance(
        self.prefetch_lazy_iter_ds, lazy_dataset.PrefetchLazyIterDataset
    )
    ds_iter = iter(self.prefetch_lazy_iter_ds)
    self.assertIsInstance(ds_iter, lazy_dataset.PrefetchLazyDatasetIterator)

  @parameterized.parameters(0, 1, 10)
  def test_prefetch_data_dense(self, prefetch_buffer_size: int):
    read_options = options.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size
    )
    prefetch_lazy_iter_ds = lazy_dataset.PrefetchLazyIterDataset(
        self.range_ds, read_options
    )
    self.assertEqual(prefetch_lazy_iter_ds._read_options, read_options)  # pylint: disable=protected-access
    ds_iter = iter(prefetch_lazy_iter_ds)
    actual = [next(ds_iter) for _ in range(20)]
    expected = list(range(20))
    self.assertSequenceEqual(actual, expected)

  @parameterized.parameters(0, 1, 10)
  def test_prefetch_data_sparse(self, prefetch_buffer_size: int):
    read_options = options.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size
    )
    prefetch_lazy_iter_ds = lazy_dataset.PrefetchLazyIterDataset(
        self.filtered_range_ds, read_options, allow_nones=True
    )
    self.assertEqual(prefetch_lazy_iter_ds._read_options, read_options)  # pylint: disable=protected-access
    ds_iter = iter(prefetch_lazy_iter_ds)
    actual = [next(ds_iter) for _ in range(20)]
    expected = [i if i % 2 == 1 else None for i in range(20)]
    self.assertSequenceEqual(actual, expected)

  def test_prefetch_iterates_one_epoch(self):
    ds_iter = iter(self.prefetch_lazy_iter_ds)
    _ = [next(ds_iter) for _ in range(20)]
    with self.assertRaises(StopIteration):
      next(ds_iter)

  def test_prefetch_does_not_buffer_unnecessary_elements(self):
    prefetch_buffer_size = 15
    read_options = options.ReadOptions(
        prefetch_buffer_size=prefetch_buffer_size
    )
    prefetch_lazy_iter_ds_large_buffer = lazy_dataset.PrefetchLazyIterDataset(
        self.range_ds, read_options
    )
    ds_iter = iter(prefetch_lazy_iter_ds_large_buffer)
    self.assertIsInstance(ds_iter, lazy_dataset.PrefetchLazyDatasetIterator)
    ds_iter = cast(lazy_dataset.PrefetchLazyDatasetIterator, ds_iter)
    self.assertIsNone(ds_iter._buffer)
    _ = next(ds_iter)
    self.assertLen(ds_iter._buffer, prefetch_buffer_size)
    _ = [next(ds_iter) for _ in range(14)]
    self.assertLen(
        ds_iter._buffer, len(self.range_ds) - prefetch_buffer_size
    )  # iterated through 15 elements so far
    _ = [next(ds_iter) for _ in range(5)]
    self.assertEmpty(ds_iter._buffer)  # iterated through all elements

  def test_checkpoint(self):
    ds_iter = iter(self.prefetch_lazy_iter_ds)

    max_steps = 20
    values_without_interruption = []
    checkpoints = []
    for _ in range(max_steps):
      checkpoints.append(ds_iter.get_state())  # pytype: disable=attribute-error
      values_without_interruption.append(next(ds_iter))

    for starting_step in [0, 1, 5, 12, 18]:
      ds_iter.set_state(checkpoints[starting_step])  # pytype: disable=attribute-error
      for i in range(starting_step, max_steps):
        value = next(ds_iter)
        self.assertEqual(value, values_without_interruption[i])


if __name__ == '__main__':
  absltest.main()
