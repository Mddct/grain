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
"""Tests for LazyDatasetGrainPool."""

import os
import signal

from absl.testing import absltest
import multiprocessing as mp
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset import lazy_dataset_grain_pool
import numpy as np


# Functions needs to be defined at the top level in order to be picklable.
class RangeLazyDatasetWorkerFunction(
    lazy_dataset_grain_pool.LazyDatasetWorkerFunction
):

  def get_lazy_dataset(self) -> lazy_dataset.LazyMapDataset:
    ds_src = lazy_dataset.RangeLazyMapDataset(start=0, stop=10, step=1)
    return ds_src


class LazyDatasetGrainPoolTest(absltest.TestCase):

  def test_grain_pool_produces_correct_single_elements(self):
    ctx = mp.get_context("spawn")
    num_processes = 1
    with lazy_dataset_grain_pool.LazyDatasetGrainPool(
        ctx=ctx,
        lazy_ds_worker_function=RangeLazyDatasetWorkerFunction(),
        num_processes=num_processes,
        batch_size=1,
    ) as grain_pool:
      output_elements = list(iter(grain_pool))

    expected_elements = [np.array([i]) for i in range(10)]
    self.assertTrue(
        all(
            [
                np.array_equal(expected_elements[i], output_elements[i])
                for i in range(10)
            ]
        )
    )
    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool._processes, num_processes)
    # Make sure all child processes exited successfully.
    for child_process in grain_pool._processes:
      self.assertEqual(child_process.exitcode, 0)

  def test_grain_pool_produces_correct_batched_elements(self):
    ctx = mp.get_context("spawn")
    num_processes = 1
    with lazy_dataset_grain_pool.LazyDatasetGrainPool(
        ctx=ctx,
        lazy_ds_worker_function=RangeLazyDatasetWorkerFunction(),
        num_processes=num_processes,
        batch_size=2,
    ) as grain_pool:
      output_elements = list(iter(grain_pool))

    expected_elements = [np.array([i, i + 1]) for i in range(0, 10, 2)]
    self.assertTrue(
        all(
            [
                np.array_equal(expected_elements[i], output_elements[i])
                for i in range(5)
            ]
        )
    )
    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool._processes, num_processes)
    # Make sure all child processes exited successfully.
    for child_process in grain_pool._processes:
      self.assertEqual(child_process.exitcode, 0)

  def test_grain_pool_enter_and_exit_context(self):
    ctx = mp.get_context("spawn")
    num_processes = 1
    lazy_dataset_worker_function = RangeLazyDatasetWorkerFunction()

    with lazy_dataset_grain_pool.LazyDatasetGrainPool(
        ctx=ctx,
        lazy_ds_worker_function=lazy_dataset_worker_function,
        num_processes=num_processes,
        batch_size=1,
    ) as grain_pool:
      output_elements = list(iter(grain_pool))

    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool._processes, num_processes)
    self.assertLen(
        output_elements,
        len(lazy_dataset.RangeLazyMapDataset(start=0, stop=10, step=1)),
    )

    # Make sure all child processes exited successfully.
    for child_process in grain_pool._processes:
      self.assertEqual(child_process.exitcode, 0)

  def test_grain_pool_terminate_child_processes(self):
    ctx = mp.get_context("spawn")
    num_processes = 4
    with lazy_dataset_grain_pool.LazyDatasetGrainPool(
        ctx=ctx,
        lazy_ds_worker_function=RangeLazyDatasetWorkerFunction(),
        num_processes=num_processes,
    ) as grain_pool:
      child_pid = grain_pool._processes[0].pid
      os.kill(child_pid, signal.SIGKILL)

    # Make sure num_processes processes were launched.
    self.assertLen(grain_pool._processes, num_processes)
    self.assertEqual(
        grain_pool._processes[0].exitcode, -1 * signal.SIGKILL.value
    )
    for child_process in grain_pool._processes[1:]:
      self.assertEqual(child_process.exitcode, 0)

  def test_grain_pool_object_deletion(self):
    ctx = mp.get_context("spawn")
    num_processes = 4
    # Users should generally use the with statement, here we test if GrainPool
    # was created without the "with statement", that object deletion would
    # have child processes gracefully exited.
    grain_pool = lazy_dataset_grain_pool.LazyDatasetGrainPool(
        ctx=ctx,
        lazy_ds_worker_function=RangeLazyDatasetWorkerFunction(),
        num_processes=num_processes,
    )
    child_processes = grain_pool._processes
    grain_pool.__del__()
    for child_process in child_processes:
      self.assertEqual(child_process.exitcode, 0)


if __name__ == "__main__":
  absltest.main()
