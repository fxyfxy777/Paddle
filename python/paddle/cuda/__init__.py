# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# paddle/cuda/__init__.py

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import paddle
from paddle import CUDAPlace, CustomPlace
from paddle.device import (
    PaddleStream as Stream,
    _device_to_paddle as _device_to_paddle,
    stream_guard as _PaddleStreamGuard,
)

if TYPE_CHECKING:
    from paddle.base import core

DeviceLike = Union[CUDAPlace, CustomPlace, int, str, None]


def is_available() -> bool:
    """
    Check whether CUDA is available in the current environment

    If Paddle is built with CUDA support and there is at least one CUDA device
    available, this function returns True. Otherwise, it returns False.

    Returns:
        bool: True if CUDA is available, False otherwise.

    Examples:
        .. code-block:: python

            >>> import paddle

            >>> if paddle.device.cuda.is_available():
            ...     print("CUDA is available")
            ... else:
            ...     print("CUDA is not available")
    """
    return paddle.device.cuda.device_count() >= 1


def synchronize(device: DeviceLike = None) -> None:
    """
    Wait for all streams on a given device to complete.

    This function blocks the calling thread until all the operations
    on the specified device have finished. It is useful for ensuring
    synchronization between CPU and GPU or across multiple devices.

    Args:
        device (CUDAPlace | CustomPlace | int | str | None, optional): The target device to synchronize.
            - None: Synchronize the current device.
            - int: Device index, e.g., ``2`` means ``gpu:2``.
            - str: Device string, e.g., ``'cuda:0'`` or ``'gpu:0'``.
            - CUDAPlace: A Paddle CUDA place object.
            - CustomPlace: A Paddle custom device place object.

    Returns:
        None

    Examples:
        .. code-block:: python

            >>> import paddle

            # synchronize the current device
            >>> paddle.device.cuda.synchronize()

            # synchronize device 0
            >>> paddle.device.cuda.synchronize(0)

            # synchronize device 'cuda:1'
            >>> paddle.device.cuda.synchronize('cuda:1')

            # synchronize with a Paddle CUDAPlace
            >>> place = paddle.CUDAPlace(0)
            >>> paddle.device.cuda.synchronize(place)
    """
    dev = _device_to_paddle(device)
    paddle.device.synchronize(dev)


def current_stream(device: DeviceLike = None) -> core.CUDAStream:
    """
    Return the current stream for the given device.

    Args:
        device (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, optional):
            The target device to query.

            - None: use the current device.
            - int: device index (e.g., 0 -> 'gpu:0').
            - str: device string (e.g., "cuda:0", "gpu:1").
            - CUDAPlace or CustomPlace: Paddle device objects.

    Returns:
        core.CUDAStream: The current CUDA stream associated with the given device.

    Examples:
        .. code-block:: python

            >>> import paddle

            # Get the current stream on the default CUDA device
            >>> s1 = paddle.device.cuda.current_stream()
            >>> print(s1)

            # Get the current stream on device cuda:0
            >>> s2 = paddle.device.cuda.current_stream("cuda:0")
            >>> print(s2)
    """
    dev = _device_to_paddle(device)
    return paddle.device.current_stream(dev)


def get_device_properties(device: DeviceLike = None):
    """
    Get the properties of a CUDA device.

    Args:
        device (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, optional):
            The target device to query.

            - None: use the current device.
            - int: device index (e.g., 0 -> 'gpu:0').
            - str: device string (e.g., "cuda:0", "gpu:1").
            - CUDAPlace or CustomPlace: Paddle device objects.

    Returns:
        DeviceProperties: An object containing the device properties, such as
        name, total memory, compute capability, and multiprocessor count.

    Examples:
        .. code-block:: python

            >>> import paddle

            # Get the properties of the current CUDA device
            >>> props = paddle.device.cuda.get_device_properties()
            >>> print(props)

            # Get the properties of device cuda:0
            >>> props0 = paddle.device.cuda.get_device_properties("cuda:0")
            >>> print(props0.name, props0.total_memory)
    """
    dev = _device_to_paddle(device)
    return paddle.device.cuda.get_device_properties(dev)


def get_device_name(device: DeviceLike = None) -> str:
    """
    Get the name of a device.

    Args:
        device (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, optional):
            The target device to query.

            - None: use the current device.
            - int: device index (e.g., 0 -> 'gpu:0').
            - str: device string (e.g., "cuda:0", "gpu:1").
            - CUDAPlace or CustomPlace: Paddle device objects.

    Returns:
        str: The name of the CUDA device.

    Examples:
        .. code-block:: python

            >>> import paddle

            # Get the name of the current CUDA device
            >>> name = paddle.device.cuda.get_device_name()
            >>> print(name)

            # Get the name of device cuda:0
            >>> name0 = paddle.device.cuda.get_device_name("cuda:0")
            >>> print(name0)
    """
    dev = _device_to_paddle(device)
    return paddle.device.cuda.get_device_name(device)


def get_device_capability(device: DeviceLike = None) -> tuple[int, int]:
    """
    Get the compute capability (major, minor) of a device.

    Args:
        device (int | str | paddle.CUDAPlace | paddle.CustomPlace | None, optional):
            The target device to query.

            - None: use the current device.
            - int: device index (e.g., 0 -> 'gpu:0').
            - str: device string (e.g., "cuda:0", "gpu:1").
            - CUDAPlace or CustomPlace: Paddle device objects.

    Returns:
        tuple[int, int]: A tuple ``(major, minor)`` representing the compute capability of the CUDA device.

    Examples:
        .. code-block:: python

            >>> import paddle

            # Get compute capability of the current CUDA device
            >>> capability = paddle.device.cuda.get_device_capability()
            >>> print(capability)  # e.g., (8, 0)

            # Get compute capability of device cuda:0
            >>> capability0 = paddle.device.cuda.get_device_capability("cuda:0")
            >>> print(capability0)
    """
    dev = _device_to_paddle(device)
    return paddle.device.cuda.get_device_capability(device)


class StreamContext(_PaddleStreamGuard):
    """
    Notes:
        This API only supports dynamic graph mode currently.
    A context manager that specifies the current stream context by the given stream.

    Args:
        stream(Stream, optional): the selected stream. If stream is None, just yield.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('cuda')
            >>> s = paddle.cuda.Stream()
            >>> data1 = paddle.ones(shape=[20])
            >>> data2 = paddle.ones(shape=[20])
            >>> data3 = data1 + data2
            >>> with paddle.cuda.StreamContext(s):
            ...     s.wait_stream(paddle.cuda.current_stream()) # type: ignore[attr-defined]
            ...     data4 = data1 + data3

    """

    def __init__(self, stream: paddle.device.Stream):
        super().__init__(stream)


def stream(stream_obj: paddle.device.Stream | None) -> StreamContext:
    '''

    Notes:
        This API only supports dynamic graph mode currently.
    A context manager that specifies the current stream context by the given stream.

    Args:
        stream(Stream, optional): the selected stream. If stream is None, just yield.

    Returns:
        None.

    Examples:
        .. code-block:: python

            >>> # doctest: +REQUIRES(env:CUSTOM_DEVICE)
            >>> import paddle

            >>> paddle.set_device('cuda')
            >>> s = paddle.cuda.Stream()
            >>> data1 = paddle.ones(shape=[20])
            >>> data2 = paddle.ones(shape=[20])
            >>> data3 = data1 + data2

            >>> with paddle.cuda.stream_guard(s):
            ...     s.wait_stream(paddle.cuda.current_stream())
            ...     data4 = data1 + data3
            >>> print(data4)

    '''
    return StreamContext(stream_obj)


def get_stream_from_external(
    data_ptr: int, device: DeviceLike = None
) -> Stream:
    """
    Wrap an externally allocated CUDA stream into a Paddle :class:`paddle.cuda.Stream` object.

    This function allows integrating CUDA streams allocated by other libraries
    into Paddle, enabling multi-library interoperability and data exchange.

    Note:
        - This function does not manage the lifetime of the external stream.
          It is the caller's responsibility to ensure the external stream remains valid
          while the returned Paddle stream is in use.
        - Providing an incorrect `device` may result in errors during kernel launches.

    Args:
        data_ptr (int): Integer representation of the external `cudaStream_t`.
        device (DeviceLike, optional): The device where the external stream was created.
            Can be a Paddle device string (e.g., "cuda:0"), an int index (e.g., 0),
            or a PaddlePlace (CUDAPlace). Default: None (current device).

    Returns:
        paddle.cuda.Stream: A Paddle Stream object that wraps the external CUDA stream.

    Examples:
        .. code-block:: python

            >>> import paddle
            >>> import ctypes

            >>> paddle.set_device("cuda:0")

            >>> # Assume an external library provides a stream pointer
            >>> external_stream_ptr = ctypes.c_void_p(12345678).value

            >>> # Wrap it into a Paddle Stream
            >>> stream = paddle.cuda.get_stream_from_external(external_stream_ptr, device=0)

            >>> # Execute operations in the context of this stream
            >>> with paddle.cuda.stream_guard(stream):
            ...     x = paddle.ones([10], dtype='float32')
            ...     y = x * 2
            >>> print(y)
    """

    device = _device_to_paddle(device)
    stream_ex = paddle.device.get_stream_from_external(data_ptr, device)

    return stream_ex


__all__ = [
    "is_available",
    "synchronize",
    "current_stream",
    "get_device_properties",
    "get_device_name",
    "get_device_capability",
    "stream",
    "Stream",
    "get_stream_from_external",
]
