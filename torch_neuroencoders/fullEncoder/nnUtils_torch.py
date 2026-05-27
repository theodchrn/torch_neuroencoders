"""PyTorch-compatible utility helpers for neuroencoders.

This is an incremental companion to the existing TensorFlow-based
`nnUtils.py`. It provides small, well-tested utilities that are safe to
use while converting the rest of the codebase to PyTorch.

Start small: tensor conversion, dtype helpers, device context and a
few mask/standardization helpers used throughout the code.
"""

from __future__ import annotations

import contextlib
import struct
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

logger = logging.getLogger(__name__)

_TF_EXAMPLE_CLASSES = None


def _get_tfrecord_example_classes():
    """Build a minimal Example protobuf schema compatible with TFRecords."""
    global _TF_EXAMPLE_CLASSES
    if _TF_EXAMPLE_CLASSES is not None:
        return _TF_EXAMPLE_CLASSES

    file_proto = descriptor_pb2.FileDescriptorProto()
    file_proto.name = "neuroencoders_example.proto"
    file_proto.package = "neuroencoders.tfrecord"
    file_proto.syntax = "proto3"

    def add_message(name):
        message = file_proto.message_type.add()
        message.name = name
        return message

    bytes_list = add_message("BytesList")
    field = bytes_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    float_list = add_message("FloatList")
    field = float_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_FLOAT
    field.options.packed = True

    int64_list = add_message("Int64List")
    field = int64_list.field.add()
    field.name = "value"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_INT64
    field.options.packed = True

    feature = add_message("Feature")
    oneof = feature.oneof_decl.add()
    oneof.name = "kind"

    field = feature.field.add()
    field.name = "bytes_list"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".neuroencoders.tfrecord.BytesList"
    field.oneof_index = 0

    field = feature.field.add()
    field.name = "float_list"
    field.number = 2
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".neuroencoders.tfrecord.FloatList"
    field.oneof_index = 0

    field = feature.field.add()
    field.name = "int64_list"
    field.number = 3
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".neuroencoders.tfrecord.Int64List"
    field.oneof_index = 0

    features = add_message("Features")
    entry = features.nested_type.add()
    entry.name = "FeatureEntry"
    entry.options.map_entry = True
    key_field = entry.field.add()
    key_field.name = "key"
    key_field.number = 1
    key_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    key_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_STRING
    value_field = entry.field.add()
    value_field.name = "value"
    value_field.number = 2
    value_field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    value_field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    value_field.type_name = ".neuroencoders.tfrecord.Feature"

    field = features.field.add()
    field.name = "feature"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".neuroencoders.tfrecord.Features.FeatureEntry"

    example = add_message("Example")
    field = example.field.add()
    field.name = "features"
    field.number = 1
    field.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field.type = descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE
    field.type_name = ".neuroencoders.tfrecord.Features"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(file_proto)

    def cls(name):
        return message_factory.GetMessageClass(
            pool.FindMessageTypeByName(f"neuroencoders.tfrecord.{name}")
        )

    _TF_EXAMPLE_CLASSES = {
        "BytesList": cls("BytesList"),
        "FloatList": cls("FloatList"),
        "Int64List": cls("Int64List"),
        "Feature": cls("Feature"),
        "Features": cls("Features"),
        "Example": cls("Example"),
    }
    return _TF_EXAMPLE_CLASSES


def _iter_tfrecord_payloads(data_path: str) -> Iterator[bytes]:
    """Yield raw TFRecord payload bytes from an uncompressed TFRecord file."""
    with open(data_path, "rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"Malformed TFRecord file: {data_path}")
            (record_length,) = struct.unpack("<Q", length_bytes)
            # Skip the masked CRC for the length field.
            handle.seek(4, 1)
            payload = handle.read(record_length)
            if len(payload) != record_length:
                raise ValueError(f"Truncated TFRecord payload in {data_path}")
            # Skip the masked CRC for the payload.
            handle.seek(4, 1)
            yield payload


def load_tfrecord_examples(
    data_path: str,
    description: Dict[str, str],
) -> Iterator[Dict[str, np.ndarray]]:
    """Load TFRecord examples without TensorFlow.

    The description maps feature names to one of ``int``, ``float``, or
    ``bytes``. The returned values mirror the common tfrecord_loader shape
    convention: scalars are returned as length-1 arrays.
    """
    classes = _get_tfrecord_example_classes()
    example_cls = classes["Example"]

    for payload in _iter_tfrecord_payloads(data_path):
        example = example_cls()
        example.ParseFromString(payload)
        record: Dict[str, np.ndarray] = {}
        feature_map = example.features.feature
        for key, dtype in description.items():
            if key not in feature_map:
                raise KeyError(f"Key {key} missing from TFRecord example")
            feature = feature_map[key]
            if feature.HasField("float_list"):
                values = np.asarray(feature.float_list.value, dtype=np.float32)
            elif feature.HasField("int64_list"):
                values = np.asarray(feature.int64_list.value, dtype=np.int64)
            elif feature.HasField("bytes_list"):
                values = np.asarray(feature.bytes_list.value, dtype=object)
            else:
                values = np.asarray([], dtype=np.float32)

            if dtype == "float":
                values = values.astype(np.float32, copy=False)
            elif dtype == "int":
                values = values.astype(np.int64, copy=False)
            record[key] = values

        yield record


def _numpy_scalar_list(value: Any) -> List[Any]:
    array = _to_numpy(value)
    if array.ndim == 0:
        return [array.item()]
    return array.reshape(-1).tolist()


def write_tfrecord_examples(
    dataset: Iterable[Dict[str, Any]], output_path: str
) -> None:
    """Write a stream of dictionaries to an uncompressed TFRecord file."""
    classes = _get_tfrecord_example_classes()
    example_cls = classes["Example"]

    with open(output_path, "wb") as handle:
        for example_dict in dataset:
            example = example_cls()
            for key, value in example_dict.items():
                if key.startswith("__"):
                    continue
                feature = example.features.feature[key]
                array = _to_numpy(value)
                if array.dtype.kind in {"f", "c"}:
                    feature.float_list.value.extend(
                        float(v) for v in _numpy_scalar_list(array)
                    )
                elif array.dtype.kind in {"i", "u", "b"}:
                    feature.int64_list.value.extend(
                        int(v) for v in _numpy_scalar_list(array)
                    )
                else:
                    encoded = array.reshape(-1).tolist()
                    if len(encoded) == 1 and isinstance(encoded[0], (bytes, bytearray)):
                        feature.bytes_list.value.extend([bytes(encoded[0])])
                    else:
                        feature.bytes_list.value.extend(
                            [str(item).encode("utf-8") for item in encoded]
                        )

            payload = example.SerializeToString()
            handle.write(struct.pack("<Q", len(payload)))
            handle.write(b"\x00\x00\x00\x00")
            handle.write(payload)
            handle.write(b"\x00\x00\x00\x00")


def build_tfrecord_description(n_groups: int) -> Dict[str, str]:
    """Build a plain dtype description for tfrecord_loader."""
    description = {
        "pos_index": "int",
        "pos": "float",
        "length": "int",
        "groups": "int",
        "time": "float",
        "time_behavior": "float",
        "indexInDat": "int",
    }
    for g in range(n_groups):
        description[f"group{g}"] = "float"
    return description


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _collate_examples(examples: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    collated: Dict[str, List[np.ndarray]] = {}
    for example in examples:
        for key, value in example.items():
            if key.startswith("__"):
                continue
            collated.setdefault(key, []).append(_to_numpy(value))

    result: Dict[str, np.ndarray] = {}
    for key, values in collated.items():
        first_value = values[0]
        if first_value.ndim == 0:
            result[key] = np.asarray(values)
        else:
            result[key] = np.stack(values, axis=0)
    return result


@dataclass
class TorchBatchDataset:
    """Minimal iterable that yields batches of numpy dictionaries."""

    batches: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]
    repeat: bool = False

    def __iter__(self) -> Iterator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        if not self.batches:
            return
        if self.repeat:
            while True:
                for batch in self.batches:
                    yield batch
        else:
            yield from self.batches

    def take(self, count: int) -> "TorchBatchDataset":
        return TorchBatchDataset(self.batches[:count], repeat=False)

    def as_numpy_iterator(
        self,
    ) -> Iterator[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
        yield from self

    def prefetch(self, *_args, **_kwargs):
        return self

    def with_options(self, *_args, **_kwargs):
        return self


def parse_serialized_sequence_torch(
    params,
    tensors: Dict[str, Any],
    count_spikes: bool = False,
    max_spikes: Optional[int] = None,
    max_spikes_per_group: Optional[int] = None,
):
    """Torch/numpy port of parse_serialized_sequence."""
    tensors = dict(tensors)
    if max_spikes is None:
        max_spikes = getattr(params, "max_nb_spikes", 512)
    if max_spikes_per_group is None:
        max_spikes_per_group = getattr(params, "max_nb_spikes_per_group", 128)

    lengths = []

    raw_groups = None
    for key in ["pos", "groups", "indexInDat"]:
        value = _to_numpy(tensors[key])
        if key == "pos":
            tensors[key] = torch.as_tensor(value, dtype=torch.float32).reshape(
                params.dimOutput
            )
            continue

        value = value.reshape(-1)
        if key == "groups":
            raw_groups = value.copy()
        if value.shape[0] > max_spikes:
            value = value[:max_spikes]
        padded = np.full((max_spikes,), -1, dtype=value.dtype)
        padded[: value.shape[0]] = value
        tensors[key] = torch.as_tensor(padded)

    for g in range(params.nGroups):
        group_key = f"group{g}"
        group_value = _to_numpy(tensors[group_key]).reshape(-1)
        spike_size = params.nChannelsPerGroup[g] * 32
        actual_spike_count = group_value.shape[0] // spike_size
        lengths.append(actual_spike_count)

        limit = min(actual_spike_count, max_spikes_per_group)
        flat_limit = limit * spike_size
        reshaped = group_value[:flat_limit].reshape(
            limit, params.nChannelsPerGroup[g], 32
        )
        padded = np.zeros(
            (max_spikes_per_group, params.nChannelsPerGroup[g], 32),
            dtype=group_value.dtype,
        )
        padded[:limit] = reshaped
        tensors[group_key] = torch.as_tensor(padded, dtype=torch.float32)

        if count_spikes:
            tensors[f"group{g}_spikes_count"] = torch.tensor(
                actual_spike_count, dtype=torch.int32
            )

    tensors["total_nb_spikes"] = torch.tensor(
        int(
            raw_groups.shape[0]
            if raw_groups is not None
            else len(_to_numpy(tensors["groups"]))
        ),
        dtype=torch.int32,
    )
    tensors["max_spikes_in_groups"] = torch.tensor(
        max(lengths) if lengths else 0, dtype=torch.int32
    )
    return tensors


def import_true_pos_torch(feature):
    feature_tensor = _to_torch_tensor(feature, dtype=torch.float32)

    def change_feature(vals):
        idx = torch.as_tensor(vals["pos_index"], dtype=torch.long).reshape(-1)
        vals["pos"] = feature_tensor[idx].reshape(feature_tensor.shape[-1])
        return vals

    return change_feature


def create_indices_torch(vals: Dict[str, Any], n_groups: int, shuffle: bool = False):
    groups = torch.as_tensor(vals["groups"], dtype=torch.int64)
    for group_id in range(n_groups):
        is_in_group = groups.eq(group_id)
        relative_indices = torch.cumsum(is_in_group.to(torch.int32), dim=0)
        indices_tensor = torch.where(
            is_in_group, relative_indices, torch.zeros_like(relative_indices)
        )

        if shuffle:
            mask = indices_tensor.gt(0)
            non_zero_indices = indices_tensor[mask]
            if non_zero_indices.numel() > 0:
                shuffled_values = non_zero_indices[
                    torch.randperm(non_zero_indices.numel())
                ]
                indices_tensor = torch.zeros_like(indices_tensor)
                indices_tensor[mask] = shuffled_values

        vals[f"indices{group_id}"] = indices_tensor
    return vals


def batch_examples(
    examples: Sequence[Dict[str, Any]],
    batch_size: int,
    drop_remainder: bool = True,
) -> List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]:
    batches: List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]] = []
    for start in range(0, len(examples), batch_size):
        batch_examples_slice = examples[start : start + batch_size]
        if drop_remainder and len(batch_examples_slice) < batch_size:
            continue

        inputs_list: List[Dict[str, Any]] = []
        targets_list: List[Dict[str, Any]] = []
        for example in batch_examples_slice:
            inputs_list.append(example[0])
            targets_list.append(example[1])

        batches.append(
            (_collate_examples(inputs_list), _collate_examples(targets_list))
        )
    return batches


def maybe_shuffle_examples(
    examples: List[Any], shuffle: bool, seed: Optional[int] = None
):
    if not shuffle:
        return examples
    shuffled = list(examples)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    return shuffled


def _to_torch_tensor(value, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """Convert numpy/torch/other to a torch.Tensor on current device.

    If value is already a torch.Tensor it is returned (cast if dtype given).
    """
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = torch.from_numpy(value)
    elif hasattr(value, "numpy"):
        try:
            arr = value.numpy()
        except Exception:
            arr = np.array(value)
        tensor = torch.from_numpy(arr)
    else:
        tensor = torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _torch_dtype(dtype_like) -> torch.dtype:
    if dtype_like is None:
        return torch.float32
    if isinstance(dtype_like, torch.dtype):
        return dtype_like
    if dtype_like in (np.float16, "float16"):
        return torch.float16
    if dtype_like in (np.float32, "float32"):
        return torch.float32
    if dtype_like in (np.float64, "float64"):
        return torch.float64
    if dtype_like in (np.int32, "int32"):
        return torch.int32
    if dtype_like in (np.int64, "int64"):
        return torch.int64
    if dtype_like in (np.bool_, bool, "bool"):
        return torch.bool
    return torch.float32


def get_device_context(device: Optional[str]):
    """Return a context manager for device placement (best-effort).

    Accepts TensorFlow-like device strings ("/CPU:0", "/GPU:0") or
    PyTorch-style ("cpu", "cuda:0"). If `device` is None a no-op
    context manager is returned.
    """
    if device is None:
        return contextlib.nullcontext()

    # Normalize common TF-style strings to PyTorch device strings
    if isinstance(device, str):
        s = device.lower()
        if s.startswith("/device:"):
            s = s.replace("/device:", "")
        if s.startswith("/cpu") or s == "cpu:0":
            return contextlib.nullcontext()
        if "gpu" in s or "cuda" in s:
            try:
                # torch.cuda.device accepts an int or device str
                # prefer using the torch.cuda.device context when available
                if torch.cuda.is_available():
                    # extract index if present
                    if ":" in s:
                        idx = int(s.split(":")[-1])
                        return torch.cuda.device(idx)
                    # default GPU 0
                    return torch.cuda.device(0)
                else:
                    logger.warning("CUDA requested but not available. Using CPU.")
                    return contextlib.nullcontext()
            except Exception:
                logger.warning("Invalid device specification '%s', using CPU.", device)
                return contextlib.nullcontext()

    # Fallback: no-op
    return contextlib.nullcontext()


def standardize_channelwise_tensor(
    tensor,
    mean,
    standard_deviation,
    axis: int = 1,
    preserve_zero_rows: bool = True,
) -> torch.Tensor:
    """Apply channel-wise standardization while preserving zero-padded rows.

    This mirrors the behaviour used in the TF implementation and is safe to
    call on CPU/GPU tensors.
    """
    tensor = _to_torch_tensor(tensor)
    rank = tensor.ndim
    axis = axis if axis >= 0 else rank + axis
    if axis < 0 or axis >= rank:
        raise ValueError(
            f"Invalid standardization axis {axis} for shape {tuple(tensor.shape)}"
        )

    mean = _to_torch_tensor(mean, dtype=tensor.dtype).to(device=tensor.device)
    standard_deviation = _to_torch_tensor(standard_deviation, dtype=tensor.dtype).to(
        device=tensor.device
    )

    broadcast_shape = [1] * rank
    broadcast_shape[axis] = mean.shape[0]
    mean = mean.reshape(broadcast_shape)
    standard_deviation = standard_deviation.reshape(broadcast_shape)

    standardized = (tensor - mean) / standard_deviation

    if not preserve_zero_rows:
        return standardized

    # Create a mask that is True where any value along non-axis dims is non-zero
    dims = [i for i in range(rank) if i != axis]
    if len(dims) == 0:
        spike_mask = tensor.ne(0.0)
    else:
        spike_mask = tensor.ne(0.0).any(dim=tuple(dims), keepdim=True)
    return torch.where(spike_mask, standardized, torch.zeros_like(standardized))


def safe_mask_creation(
    batchedInputGroups: torch.Tensor, pad_value: int = -1
) -> torch.Tensor:
    """Create boolean mask where True indicates valid (not equal to pad_value)."""
    return batchedInputGroups.ne(pad_value)


def create_attention_mask_from_padding_mask(
    padding_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Convert padding mask to attention mask for transformer layers.

    Input `padding_mask` should be boolean-like with True for valid tokens.
    Output shape: [batch, 1, seq_len] suitable for broadcasting.
    If input is None, returns None.
    """
    if padding_mask is None:
        return None
    mask = padding_mask
    if mask.dtype != torch.bool:
        mask = mask.to(torch.bool)
    # add a dim for broadcasting across queries
    return mask.unsqueeze(1)


__all__ = [
    "_to_torch_tensor",
    "_torch_dtype",
    "get_device_context",
    "standardize_channelwise_tensor",
    "safe_mask_creation",
    "create_attention_mask_from_padding_mask",
    "build_tfrecord_description",
    "TorchBatchDataset",
    "parse_serialized_sequence_torch",
    "import_true_pos_torch",
    "create_indices_torch",
    "batch_examples",
    "maybe_shuffle_examples",
]
