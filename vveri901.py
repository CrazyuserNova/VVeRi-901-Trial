from __future__ import print_function, absolute_import
import os
import glob
import re
import sys
import urllib
import tarfile
import zipfile
import os.path as osp
from scipy.io import loadmat
import numpy as np

"""Dataset classes"""


class VVeRi901(object):
    """
    MARS

    Reference:
    Jianan Zhao et al. PhD Learning: Learning with Pompeiu-hausdorff Distances for Video-based Vehicle Re-Identification. CVPR 2021.
    
    Trial Version Dataset statistics:
    # identities: 95
    # tracklets: 128 (train) + 40 (query) + 89 (gallery)
    # cameras: unknown

    Note: 
    # gallery set must contain query set, otherwise 140 query imgs will not have ground truth.
    # gallery imgs with label=-1 can be remove, which do not influence on final performance.

    Args:
        min_seq_len (int): tracklet with length shorter than this value will be discarded (default: 0).
    """
  

    def __init__(self, root=None, min_seq_len=0, **kwargs):
        if root is None:
            root = 'E:\\dataset\\VVERI901_V1_Trial'
        self.root = root
        self.train_name_path = osp.join(root, 'train.txt')
        self.gallery_name_path = osp.join(root, 'gallery.txt')
        self.query_name_path = osp.join(root, 'query.txt')

        self._check_before_run()

        # 
        train_track_names = self._get_names(self.train_name_path)
        query_track_names = self._get_names(self.query_name_path)
        gallery_track_names = self._get_names(self.gallery_name_path)
        train, num_train_tracklets, num_train_pids, num_train_imgs = self._process_data(train_track_names, relabel=True, min_seq_len=min_seq_len)
        query, num_query_tracklets, num_query_pids, num_query_imgs = self._process_data(query_track_names, relabel=False, min_seq_len=min_seq_len)
        gallery,  num_gallery_tracklets, num_gallery_pids, num_gallery_imgs = self._process_data(gallery_track_names, relabel=False, min_seq_len=min_seq_len)
       
        num_imgs_per_tracklet = num_train_imgs + num_gallery_imgs + num_query_imgs
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_gallery_tracklets + num_query_tracklets

        print("=>  loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # tracklets")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.train_name_path):
            raise RuntimeError("'{}' is not available".format(self.train_name_path))
        if not osp.exists(self.gallery_name_path):
            raise RuntimeError("'{}' is not available".format(self.gallery_name_path))
        if not osp.exists(self.query_name_path):
            raise RuntimeError("'{}' is not available".format(self.query_name_path))

    def _get_names(self, fpath):
        names = []
        with open(fpath, 'r') as f:
            for line in f:
                new_line = line.rstrip()
                names.append(new_line)
        return names

    def _process_data(self, track_names, relabel=False, min_seq_len=0):
        pids = set()
        for i in range(len(track_names)):
            splited_path = track_names[i].split('/')
            pids.add(splited_path[2])
        pid_list = list(pids)
        num_pids = len(pid_list)

        if relabel: pid2label = {pid:label for label, pid in enumerate(pid_list)}

        track_path = []
        tracklets = []
        num_imgs_per_tracklet = []

        for tracklet_idx in range(len(track_names)):

            splited_path = track_names[tracklet_idx].split('/')
            pid = splited_path[2]
            camid = splited_path[3].split('_')[1]
            if relabel: pid = pid2label[pid]
            joined_path = osp.join(self.root, splited_path[2], splited_path[3])
            track_path.append(joined_path)
            tracklet = []
            for root, _, files in os.walk(joined_path):
                for img_idx in range(len(files)):
                    tracklet.append(osp.join(root, files[img_idx]))
            num_imgs_per_tracklet.append(len(tracklet))
            tracklets.append((tracklet, pid, camid))
        
        num_tracklets = len(tracklets)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet

if __name__ == '__main__':
    VVeRi901()

