
import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

cur_dir = osp.dirname(__file__)

add_path(osp.join(cur_dir, '../hypergcn'))
add_path(osp.join(cur_dir, '../hgnn'))
