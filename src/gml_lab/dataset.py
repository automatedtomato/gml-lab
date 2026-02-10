from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

import lmdb
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.dataset import BaseDataset
from mmengine.registry import DATASETS, TRANSFORMS

if TYPE_CHECKING:
    from pathlib import Path


@TRANSFORMS.register_module()
class LoadImageFromLMSB(BaseTransform):
    """Decode image from bytes loaded from LMDB.

    This transform replaces the standard `LoadImageFromFile`.
    It expects 'img_bytes' in the results dict (loaded by the Dataset)
    and decodes it into a numpy array.

    Args:
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        channel_order (str): The channel order of the output image.
            Defaults to 'bgr'.
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.

    """

    def __init__(
        self,
        color_type: str = "color",
        channel_order: str = "bgr",
        to_float32: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        self.color_type = color_type
        self.channel_order = channel_order
        self.to_float32 = to_float32

    def transform(self, results: dict) -> dict[str, Any] | None:
        """Transform function to load and decode image from bytes.

        Args:
            results (dict): Result dict from :obj:`mmengine.dataset.BaseDataset`.
                Must contain key 'img_bytes'.

        Returns:
            dict: The dict contains loaded image and meta information.
                - `img`: The decoded image array (np.ndarray).
                - `img_shape`: The shape of the image (h, w).
                - `ori_shape`: The original shape of the image (h, w).

        """
        if "img_bytes" not in results:
            msg = (
                "results must contain `img_bytes`. Ensure ImageNetLMDB dataset is used."
            )
            raise KeyError(msg)

        img_bytes = results["img_bytes"]
        img = mmcv.imfrombytes(
            img_bytes, flag=self.color_type, channel_order=self.channel_order
        )
        if self.to_float32:
            img = img.astype(np.float32)
        results["img"] = img
        results["img_shape"] = img.shape[:2]
        results["ori_shape"] = img.shape[:2]
        return results


@DATASETS.register_module()
class ImageNetLMDB(BaseDataset):
    """Custom Dataset for loading ImageNet data from an LMDB file.

    This class decouples the 'indexing' phase (main process) from the
    "loading" phase (worker processes) to optimize for multi-process data loading.

    Args:
        lmdb_path (str): Path to the LMDB directory containing data.mdb.
        **kwargs: Other arguments for :class:`mmengine.dataset.BaseDataset`
            (e.g., pipeline, lazy_init).

    """

    def __init__(self, lmdb_path: Path, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.lmdb_path = lmdb_path
        # NOTE [Lazy Initialization of LMDB Environment]:
        # We explicitly set self.env to None here and initialize it lazily inside
        # `get_data_info()`.
        #
        # PyTorch `DataLoader` uses `multiprocessing` to spawn worker processes.
        # LMDB `Environment` objects are NOT picklable and cannot be shared across
        # processes. If we initialize `self.env` in `__init__` (which runs in the
        # main process), pickle will fail when spawning workers, causing a crash.
        # By initializing inside `get_data_info`, we ensure that each worker process
        # creates its own independent connection to the LMDB file.
        self.env: lmdb.Environment | None = None

    def load_data_list(self) -> list[dict]:
        """Load annotations (indices) from LMDB.

        Role:
            Executed ONCE in the main process during initialization.
            Its purpose is to build a lightweight "catalog" (list of dicts)
            that tells the dataset how many samples exist and how to retrieve them.
            It does NOT load the actual heavy image data.

        Returns:
            list[dict]: A list of dicts, where each dict contains 'sample_idx'.
                        Example: [{'sample_idx': 0}, {'sample_idx': 1}, ...]

        """
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            length_bytes = txn.get(b"length")
            if length_bytes is None:
                msg = f"Key 'length' not found in LMDB at {self.lmdb_path}"
                raise ValueError(msg)
            length = int(length_bytes.decode("ascii"))
        env.close()

        # return a lightweight list of indices
        return [{"sample_idx": i} for i in range(length)]

    def get_data_info(self, idx: int) -> dict:
        """Fetch the actual data sample corresponding to the index.

        Role:
            Executed repeatedly, often in parallel worker processes.
            This method performs the heavy I/O operations:
            1. Initializes the LMDB connection (if not already open).
            2. Retrieves the binary blob for the given index.
            3. Deserializes it into a usable dictionary.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the raw data.
                  Keys: 'img_bytes', 'gt_label', 'filename', etc.

        """
        # Lazy initialization for multiprocessing safety (See NOTE in __init__)
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        data_info = self.data_list[idx]
        sample_idx = data_info["sample_idx"]

        key = f"{sample_idx:08}".encode("ascii")

        with self.env.begin(write=False) as txn:
            byte_data = txn.get(key)
            if byte_data is None:
                msg = f"Index {key!r} not found in LMDB."
                raise ValueError(msg)

            # Deserialize the data (bytes -> dict)
            sample = pickle.loads(byte_data)

        # Merge the loaded sample data into the data_info dict
        # sample typically contains: {
        #       "img_bytes": ...,
        #       "gt_label": ...,
        #       "filename": ...
        #       }
        data_info.update(sample)

        return data_info
