# encoding: utf-8
"""
@author: your_name
"""

import glob
import os.path as osp
import warnings

from .bases import BaseImageDataset


class OCC_OccludedReID(BaseImageDataset):
    dataset_dir = 'OccludedREID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(OCC_OccludedReID, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.query_dir = osp.join(self.dataset_dir, 'occluded_body_images')
        self.gallery_dir = osp.join(self.dataset_dir, 'whole_body_images')
        self.pid_begin = pid_begin

        self._check_before_run()

        train = self._process_dir(self.gallery_dir, relabel=True, is_query=False)
        query = self._process_dir(self.query_dir, relabel=False, is_query=True)
        gallery = self._process_dir(self.gallery_dir, relabel=False, is_query=False)

        if verbose:
            print("=> Occluded_REID loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False, is_query=True):
        img_paths = glob.glob(osp.join(dir_path, '*', '*.tif'))

        pid_container = set()
        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = int(img_name.split('_')[0])
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        camid = 0 if is_query else 1

        for img_path in img_paths:
            img_name = osp.basename(img_path)
            pid = int(img_name.split('_')[0])

            if relabel:
                pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 1))

        return dataset