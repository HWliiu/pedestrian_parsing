import glob
import os.path as osp
import re

from PIL import Image
from torch.utils.data import Dataset


class CheckMixin:
    def check_before_run(self, required_files):
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))


class DukeMTMC(CheckMixin):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        self.train = self.process_dir(self.train_dir, relabel=True)
        self.query = self.process_dir(self.query_dir, relabel=False)
        self.gallery = self.process_dir(self.gallery_dir, relabel=False)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, cid = map(int, pattern.search(img_path).groups())
            assert 1 <= cid <= 8
            cid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, cid))

        return data


class Market1501(CheckMixin):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        self.train = self.process_dir(self.train_dir, relabel=True)
        self.query = self.process_dir(self.query_dir, relabel=False)
        self.gallery = self.process_dir(self.gallery_dir, relabel=False)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c(\d)")

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, cid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= cid <= 6
            cid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, cid))

        return data


class MSMT17(CheckMixin):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = osp.join(self.dataset_dir, "train")
        self.test_dir = osp.join(self.dataset_dir, "test")
        self.list_train_path = osp.join(self.dataset_dir, "list_train.txt")
        self.list_val_path = osp.join(self.dataset_dir, "list_val.txt")
        self.list_query_path = osp.join(self.dataset_dir, "list_query.txt")
        self.list_gallery_path = osp.join(self.dataset_dir, "list_gallery.txt")

        required_files = [
            self.train_dir,
            self.test_dir,
            self.list_train_path,
            self.list_val_path,
            self.list_query_path,
            self.list_gallery_path,
        ]
        self.check_before_run(required_files)

        self.train = self.process_dir(
            self.train_dir, self.list_train_path
        ) + self.process_dir(self.train_dir, self.list_val_path)
        self.query = self.process_dir(self.test_dir, self.list_query_path)
        self.gallery = self.process_dir(self.test_dir, self.list_gallery_path)

    def process_dir(self, dir_path, list_path):
        with open(list_path, "r") as txt:
            lines = txt.readlines()

        data = []
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(" ")
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split("_")[2]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid))

        return data


class PersonX(CheckMixin):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.query_dir = osp.join(self.dataset_dir, "query")
        self.gallery_dir = osp.join(self.dataset_dir, "bounding_box_test")

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        self.train = self.process_dir(self.train_dir, relabel=True)
        self.query = self.process_dir(self.query_dir, relabel=False)
        self.gallery = self.process_dir(self.gallery_dir, relabel=False)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"([-\d]+)_c([-\d]+)")
        cam2label = {3: 1, 4: 2, 8: 3, 10: 4, 11: 5, 12: 6}

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, cid = map(int, pattern.search(img_path).groups())
            assert cid in cam2label.keys()
            cid = cam2label[cid]
            cid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, cid))

        return data


class ReidDataset(Dataset):
    def __init__(self, data, transform):
        super(ReidDataset, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        fpath, pid, cid = self.data[index]
        img = Image.open(fpath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, pid, cid
