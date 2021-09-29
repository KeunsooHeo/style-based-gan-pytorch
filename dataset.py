from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset


class MultiResolutionDataset(Dataset):
    """
    Object 'Dataset' provided by torch
    torch에서 제공하는 Dataset 객체
    """
    def __init__(self, path, transform, resolution=8):
        # LMDB database
        # To use database, opening is required. It will return object Environment.
        # 'path to LMDB is required to get dataset.
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        # get data on dataset related to "length"
        # dataset의 length와 관련된 정보를 가져옴.
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        # Define "transform" to transform the images
        # Define "resolution"  resolution
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        # return length of dataset.
        return self.length

    def __getitem__(self, index):
        # get "index"-th image. And return transformed one.
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img
