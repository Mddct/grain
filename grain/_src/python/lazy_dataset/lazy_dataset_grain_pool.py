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
"""GrainPool for LazyDataset, with batching in the parent process."""

import abc
import dataclasses
from multiprocessing import synchronize
import os
import socket
import struct
import traceback
import types
from typing import Sequence, TypeVar

from absl import logging
from grain._src.core import parallel
from grain._src.python import options as grain_options
from grain._src.python.lazy_dataset import lazy_dataset
from grain._src.python.lazy_dataset.transformations.slice import SliceLazyMapDataset
import numpy as np
import tree


_IN = TypeVar("_IN")
_OUT = TypeVar("_OUT")
T = TypeVar("T")


# Maximum number of threads for starting and stopping processes.
_PROCESS_MANAGEMENT_MAX_THREADS = 64
_PROCESS_JOIN_TIMEOUT = 10


# Hack to embed stringification of remote traceback in local traceback.
class RemoteTracebackError(Exception):

  def __init__(self, tb: types.TracebackType):
    self.tb = tb

  def __str__(self):
    return self.tb


class ExceptionWithTraceback:
  """Exception that can be pickled and sent over the queue."""

  def __init__(self, exception: Exception, tb: types.TracebackType):
    tb = traceback.format_exception(type(exception), exception, tb)
    tb = "".join(tb)
    self.exception = exception
    self.tb = '\n"""\n%s"""' % tb

  def __reduce__(self):
    return rebuild_exception, (self.exception, self.tb)


def rebuild_exception(exception: Exception, tb: types.TracebackType):
  """Rebuilds the exception at the received side."""
  exception.__cause__ = RemoteTracebackError(tb)
  return exception


# Classes for termination events and error handling


@dataclasses.dataclass
class _ProcessingComplete:
  """Indicates child process finished processing."""


@dataclasses.dataclass
class _ExceptionRaised:
  """Indicates child process has raised an exception."""

  exception_with_traceback: ExceptionWithTraceback


# Shared memory utils


def put_data_in_ramfile(
    data: np.ndarray | _ProcessingComplete | _ExceptionRaised,
) -> tuple[ram_file.RamFile, ram_pickle.MemoryRegion]:
  ram = ram_file.RamFile()
  region = ram_pickle.dump(ram, data, use_cloudpickle=True)
  return ram, region


def get_data_from_ramfile(
    ram: ram_file.RamFile, region: ram_pickle.MemoryRegion
) -> np.ndarray | _ProcessingComplete | _ExceptionRaised:
  data = ram_pickle.load(ram, region, inplace=True)
  return data


# Worker loop on child process(es)


class LazyDatasetWorkerFunction(abc.ABC):
  """Function to run on each child process."""

  @abc.abstractmethod
  def get_lazy_dataset(self) -> lazy_dataset.LazyMapDataset:
    ...

  def _build_child_lazy_dataset_iterator(
      self,
      process_index: int,
      process_count: int,
      read_options: grain_options.ReadOptions,
  ):
    """Builds a prefetch iterator for this child process.

    Args:
      process_index: index of this child process.
      process_count: number of child processes.
      read_options: Options to use for reading.

    Returns:
      Prefetch iterator for this child process.
    """
    lazy_ds = self.get_lazy_dataset()
    ds_shard_per_process = SliceLazyMapDataset(
        parent=lazy_ds, sl=slice(process_index, len(lazy_ds), process_count)
    )
    iter_ds = lazy_dataset.PrefetchLazyIterDataset(
        parent=ds_shard_per_process,
        read_options=read_options,
        allow_nones=True,
    )
    ds_iter = iter(iter_ds)
    return ds_iter

  def __call__(
      self,
      child_socket: socket.socket,
      process_index: int,
      process_count: int,
      read_options: grain_options.ReadOptions,
      termination_event: synchronize.Event,
  ):
    try:
      ds_iter = self._build_child_lazy_dataset_iterator(
          process_index,
          process_count,
          read_options,
      )

      while not termination_event.is_set():
        try:
          element = next(ds_iter)
          ram, region = put_data_in_ramfile(element)
          socket.send_fds(
              sock=child_socket,
              buffers=[struct.pack("2Q", region.begin, region.end)],
              fds=[ram.fd],
          )
        except StopIteration:
          ram, region = put_data_in_ramfile(_ProcessingComplete())
          socket.send_fds(
              sock=child_socket,
              buffers=[struct.pack("2Q", region.begin, region.end)],
              fds=[ram.fd],
          )
          logging.info(
              "Stop iteration reached. Setting termination event in process"
              " with process_idx: %i",
              process_index,
          )
          termination_event.set()
          break
    except Exception as e:  # pylint: disable=broad-except
      logging.exception(
          "Error occurred in child process with process_idx: %i", process_index
      )
      remote_error = ExceptionWithTraceback(e, e.__traceback__)

      # signaling to worker pool to stop
      exception_raised = _ExceptionRaised(exception_with_traceback=remote_error)
      ram, region = put_data_in_ramfile(exception_raised)
      socket.send_fds(
          sock=child_socket,
          buffers=[struct.pack("2Q", region.begin, region.end)],
          fds=[ram.fd],
      )

      logging.info(
          "Setting termination event in process with process_idx: %i",
          process_index,
      )
      termination_event.set()

    if termination_event.is_set():
      child_socket.shutdown(socket.SHUT_RDWR)
      child_socket.close()
      logging.info("Process %i exiting.", process_index)


# LazyDataset GrainPool


def _make_batch(values: Sequence[T]) -> T:
  return tree.map_structure(lambda *xs: np.stack(xs), values[0], *values[1:])


class LazyDatasetGrainPool:
  """Pool to parallelize processing of LazyDataset PyGrain pipelines among a set of processes."""

  def __init__(
      self,
      ctx,
      *,
      lazy_ds_worker_function: LazyDatasetWorkerFunction,
      num_processes: int = 1,
      worker_idx_to_start_reading: int = 0,
      batch_size: int = 1,
      read_options: grain_options.ReadOptions | None = None,
  ):
    """Initialize a Grain Pool.

    Args:
      ctx: Context for g3 multiprocessing.
      lazy_ds_worker_function: Function to apply to input elements.
      num_processes: Number of child processes.
      worker_idx_to_start_reading: index of worker to start reading output
        batches from (needed for checkpointing support).
      batch_size: Size of output batch.
      read_options: Options to use for reading.
    """
    if num_processes is None:
      self._num_processes = os.cpu_count()
      if self._num_processes is None:
        raise NotImplementedError("Cannot determine the number of CPUs.")
    else:
      self._num_processes = num_processes
    logging.info("Grain pool will use %i processes.", self._num_processes)

    if read_options is None:
      self._read_options = grain_options.ReadOptions(
          num_threads=20,
          prefetch_buffer_size=256,
      )
    else:
      self._read_options = read_options

    self._worker_output_sockets = []
    self._processes = []
    self._next_process_idx = worker_idx_to_start_reading
    self._num_steps_iterated_thru = 0
    self._batch_size = batch_size
    self._termination_event = ctx.Event()

    # Set up child processes.
    for process_idx in range(self._num_processes):
      parent_socket, child_socket = socket.socketpair()
      process = ctx.Process(
          target=lazy_ds_worker_function,
          args=(
              child_socket,
              process_idx,
              num_processes,
              self._read_options,
              self._termination_event,
          ),
      )
      self._worker_output_sockets.append(parent_socket)
      self._processes.append(process)

    # Start child processes.
    parallel.run_in_parallel(
        function=lambda child_process: child_process.start(),
        list_of_kwargs_to_function=[
            {"child_process": p} for p in self._processes
        ],
        num_workers=self._num_processes,
    )

  def __iter__(self):
    return self

  def __next__(self):
    stop_processing = False
    exception_raised = False
    exception_to_raise = None

    collated_elements = []
    while len(collated_elements) < self._batch_size:
      packet, fds, *_ = socket.recv_fds(
          sock=self._worker_output_sockets[self._next_process_idx],
          bufsize=struct.calcsize("2Q"),
          maxfds=1,
      )
      reg_begin, reg_end = struct.unpack("2Q", packet)
      example_ram = ram_file.RamFile(fd=fds[0])
      example_region = ram_pickle.MemoryRegion(begin=reg_begin, end=reg_end)
      element = get_data_from_ramfile(example_ram, example_region)

      if isinstance(element, _ProcessingComplete):  # terminate
        stop_processing = True
        break
      elif isinstance(element, _ExceptionRaised):
        exception_to_raise = element.exception_with_traceback
        exception_raised = True
        stop_processing = True
        break
      elif element is not None:
        collated_elements.append(element)

      # round robin fetch
      self._next_process_idx = (
          self._next_process_idx + 1
      ) % self._num_processes

    if exception_raised:
      raise exception_to_raise

    if stop_processing:
      raise StopIteration()

    self._num_steps_iterated_thru += 1
    return _make_batch(collated_elements)

  def __del__(self):
    self._shutdown()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    logging.info("Grain pool is exiting.")
    self._shutdown()

  def _shutdown(self) -> None:
    logging.info("Shutting down GrainPool.")
    self._termination_event.set()
    for process in self._processes:
      process.join()
      process.terminate()
