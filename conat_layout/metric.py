from bisect import bisect_left
from typing import List

import numpy as np
import multiprocessing as mp
from itertools import chain


from scipy.optimize import linear_sum_assignment

import torch
from scipy.spatial import Rectangle
from torch_geometric.utils import to_dense_adj
from pytorch_fid.fid_score import calculate_frechet_distance

from model.layoutnet import LayoutNet
from util import convert_xywh_to_ltrb
from data.util import RelSize, RelLoc, detect_size_relation, detect_loc_relation


class Segtree:
    def __init__(self):
        self.cover = 0
        self.length = 0
        self.max_length = 0


class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        hbound = set()
        for rect in rectangles:
            # 下边界
            hbound.add(rect[1])
            # 上边界
            hbound.add(rect[3])

        hbound = sorted(hbound)
        m = len(hbound)
        # 线段树有 m-1 个叶子节点，对应着 m-1 个会被完整覆盖的线段，需要开辟 ~4m 大小的空间
        tree = [Segtree() for _ in range(m * 4 + 1)]

        def init(idx: int, l: int, r: int) -> None:
            tree[idx].cover = tree[idx].length = 0
            if l == r:
                tree[idx].max_length = hbound[l] - hbound[l - 1]
                return

            mid = (l + r) // 2
            init(idx * 2, l, mid)
            init(idx * 2 + 1, mid + 1, r)
            tree[idx].max_length = tree[idx * 2].max_length + tree[idx * 2 + 1].max_length

        def update(idx: int, l: int, r: int, ul: int, ur: int, diff: int) -> None:
            if l > ur or r < ul:
                return
            if ul <= l and r <= ur:
                tree[idx].cover += diff
                pushup(idx, l, r)
                return

            mid = (l + r) // 2
            update(idx * 2, l, mid, ul, ur, diff)
            update(idx * 2 + 1, mid + 1, r, ul, ur, diff)
            pushup(idx, l, r)

        def pushup(idx: int, l: int, r: int) -> None:
            if tree[idx].cover > 0:
                tree[idx].length = tree[idx].max_length
            elif l == r:
                tree[idx].length = 0
            else:
                tree[idx].length = tree[idx * 2].length + tree[idx * 2 + 1].length

        init(1, 1, m - 1)

        sweep = list()
        for i, rect in enumerate(rectangles):
            # 左边界
            sweep.append((rect[0], i, 1))
            # 右边界
            sweep.append((rect[2], i, -1))
        sweep.sort()

        ans = i = 0
        while i < len(sweep):
            j = i
            while j + 1 < len(sweep) and sweep[i][0] == sweep[j + 1][0]:
                j += 1
            if j + 1 == len(sweep):
                break

            # 一次性地处理掉一批横坐标相同的左右边界
            for k in range(i, j + 1):
                _, idx, diff = sweep[k]
                # 使用二分查找得到完整覆盖的线段的编号范围
                left = bisect_left(hbound, rectangles[idx][1]) + 1
                right = bisect_left(hbound, rectangles[idx][3])
                update(1, 1, m - 1, left, right, diff)

            ans += tree[1].length * (sweep[j + 1][0] - sweep[j][0])
            i = j + 1

        return ans % (10 ** 9 + 7)

class LayoutFID():
    def __init__(self, dataset_name, device='cpu'):
        #--------- by ljw 20221103----------------------
        num_label = 13 if dataset_name == 'rico' or dataset_name == 'ricoTest' else 5
        num_ricoLabel = 25 if dataset_name == 'ricoTest'or dataset_name == 'ricoTest' else -1
        self.model = LayoutNet(num_label,num_ricoLabel).to(device)
        #----------------------------------------------#


        # load pre-trained LayoutNet
        tmpl = './pretrained/layoutnet_{}.pth.tar'
        #--------------by ljw 20221103------------------#
        #加载部分预训练参数，其余新加的修改部分不加载
        state_dict = torch.load(tmpl.format(dataset_name), map_location=device)
        self.model.load_state_dict(state_dict,strict=False)
        self.model.requires_grad_(False)
        self.model.eval()
        #----------------------------------------------#

        self.real_features = []
        self.fake_features = []

    def collect_features(self, bbox, label, padding_mask, real=False,ricoLabel=-1):#by ljw 20221103
        if real and type(self.real_features) != list:
            return
        #------------------by ljw 20221103--------------#

        feats = self.model.extract_features(bbox.detach(), label, padding_mask,ricoLabel)

        #----------------------------------------------#

        features = self.real_features if real else self.fake_features
        features.append(feats.cpu().numpy())

    def compute_score(self):
        feats_1 = np.concatenate(self.fake_features)
        self.fake_features = []

        if type(self.real_features) == list:
            feats_2 = np.concatenate(self.real_features)
            self.real_features = feats_2
        else:
            feats_2 = self.real_features

        mu_1 = np.mean(feats_1, axis=0)
        sigma_1 = np.cov(feats_1, rowvar=False)
        mu_2 = np.mean(feats_2, axis=0)
        sigma_2 = np.cov(feats_2, rowvar=False)

        return calculate_frechet_distance(mu_1, sigma_1, mu_2, sigma_2)


def compute_iou(box_1, box_2):
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, torch.Tensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max),
                   lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    return iou


def __compute_maximum_iou_for_layout(layout_1, layout_2):
    score = 0.
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n = len(_bi)
        ii, jj = np.meshgrid(range(n), range(n))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, n)
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    return score / N


def __compute_maximum_iou(layouts_1_and_2):
    layouts_1, layouts_2 = layouts_1_and_2
    N, M = len(layouts_1), len(layouts_2)
    ii, jj = np.meshgrid(range(N), range(M))
    ii, jj = ii.flatten(), jj.flatten()
    scores = np.asarray([
        __compute_maximum_iou_for_layout(layouts_1[i], layouts_2[j])
        for i, j in zip(ii, jj)
    ]).reshape(N, M)
    ii, jj = linear_sum_assignment(scores, maximize=True)
    return scores[ii, jj]


def __get_cond2layouts(layout_list,ifRicoTest=False):#by ljw 20230103
    out = dict()
    if ifRicoTest is True:
        for bs, ls,rs in layout_list:#by ljw 20230103 rs为ricoLabel
            cond_key = str(sorted(ls.tolist()))
            if cond_key not in out.keys():
                out[cond_key] = [(bs, ls)]
            else:
                out[cond_key].append((bs, ls))
    else:
        for bs, ls in layout_list:
            cond_key = str(sorted(ls.tolist()))
            if cond_key not in out.keys():
                out[cond_key] = [(bs, ls)]
            else:
                out[cond_key].append((bs, ls))
    return out


def compute_maximum_iou(layouts_1, layouts_2, n_jobs=None,ifRicoTest=False):#by ljw 20230103
    c2bl_1 = __get_cond2layouts(layouts_1)
    keys_1 = set(c2bl_1.keys())
    c2bl_2 = __get_cond2layouts(layouts_2,ifRicoTest=ifRicoTest)#by ljw 20230103
    keys_2 = set(c2bl_2.keys())
    keys = list(keys_1.intersection(keys_2))
    args = [(c2bl_1[key], c2bl_2[key]) for key in keys]
    with mp.Pool(n_jobs) as p:
        scores = p.map(__compute_maximum_iou, args)
    scores = np.asarray(list(chain.from_iterable(scores)))
    return scores.mean().item()




def compute_overlap(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.3 Overlapping Loss

    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    bbox = bbox.permute(2, 0, 1)

    l1, t1, r1, b1 = convert_xywh_to_ltrb(bbox.unsqueeze(-1))
    l2, t2, r2, b2 = convert_xywh_to_ltrb(bbox.unsqueeze(-2))
    a1 = (r1 - l1) * (b1 - t1)

    # intersection
    l_max = torch.maximum(l1, l2)
    r_min = torch.minimum(r1, r2)
    t_max = torch.maximum(t1, t2)
    b_min = torch.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = torch.where(cond, (r_min - l_max) * (b_min - t_max),
                     torch.zeros_like(a1[0]))

    diag_mask = torch.eye(a1.size(1), dtype=torch.bool,
                          device=a1.device)
    ai = ai.masked_fill(diag_mask, 0)

    ar = torch.nan_to_num(ai / a1)

    return ar.sum(dim=(1, 2)) / mask.float().sum(-1)


def compute_alignment(bbox, mask):
    # Attribute-conditioned Layout GAN
    # 3.6.4 Alignment Loss

    bbox = bbox.permute(2, 0, 1)
    xl, yt, xr, yb = convert_xywh_to_ltrb(bbox)
    xc, yc = bbox[0], bbox[1]
    X = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)

    X = X.unsqueeze(-1) - X.unsqueeze(-2)
    idx = torch.arange(X.size(2), device=X.device)
    X[:, :, idx, idx] = 1.
    X = X.abs().permute(0, 2, 1, 3)
    X[~mask] = 1.
    X = X.min(-1).values.min(-1).values
    X.masked_fill_(X.eq(1.), 0.)
    X = -torch.log(1 - X)

    return X.sum(-1) / mask.float().sum(-1)


def compute_violation(bbox_flatten, data):
    device = data.x.device
    failures, valid = [], []

    _zip = zip(data.edge_attr, data.edge_index.t())
    for gt, (i, j) in _zip:
        failure, _valid = 0, 0
        b1, b2 = bbox_flatten[i], bbox_flatten[j]

        # size relation
        if ~gt & 1 << RelSize.UNKNOWN:
            pred = detect_size_relation(b1, b2)
            failure += (gt & 1 << pred).eq(0).long()
            _valid += 1

        # loc relation
        if ~gt & 1 << RelLoc.UNKNOWN:
            canvas = data.y[i].eq(0)
            pred = detect_loc_relation(b1, b2, canvas)
            failure += (gt & 1 << pred).eq(0).long()
            _valid += 1

        failures.append(failure)
        valid.append(_valid)

    failures = torch.as_tensor(failures).to(device)
    failures = to_dense_adj(data.edge_index, data.batch, failures)
    valid = torch.as_tensor(valid).to(device)
    valid = to_dense_adj(data.edge_index, data.batch, valid)

    return failures.sum((1, 2)) / valid.sum((1, 2))


#by ljw 20221223
#计算小项目减分
def compute_Irregular(bbox, label):
    minus_point=0
    index=0
    bbox_loc_x1=[]
    bbox_loc_y1 = []
    bbox_loc_x2 = []
    bbox_loc_y2 = []
    minus_reason=[]
    label_=[]
    minus_reason.append("以下建议均在2560x1440分辨率基础上进行")
    H, W = (2560, 1440)
    for box in bbox:
        if label[index]==9 or label[index]==10:
            index += 1
            continue
        x1, y1, x2, y2 = convert_xywh_to_ltrb(box)  # 得到左上右下坐标
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        bbox_loc_x1.append(x1)
        bbox_loc_x2.append(x2)
        bbox_loc_y1.append(y1)
        bbox_loc_y2.append(y2)
        label_.append(label[index])
        index+=1
    index=0
    for l in label_:
        if l==0:#导航栏,返回栏
            if bbox_loc_y2[index]-bbox_loc_y1[index]<176:
                minus_point+=4
                minus_reason.append("导航栏高度过小（-4）")
            if bbox_loc_y2[index] - bbox_loc_y1[index] > 264:
                minus_point += 4
                minus_reason.append("导航栏高度过大（-4）")
            if bbox_loc_y1[index]>min(bbox_loc_y1)+60\
                    or (bbox_loc_y2[index]>400 and bbox_loc_y2[index]<2300):#60作为背景图片余量
                minus_point+=5
                minus_reason.append("导航栏位置建议在界面最顶部或者最底部（-5）")
        # elif l==1:#图片
        #     if bbox_loc_y2[index]-bbox_loc_y1[index]<400:
        #         minus_point+=3
        #         minus_reason.append("图片高度过小（-3）")
        elif l==2:#文字
            if bbox_loc_y2[index]-bbox_loc_y1[index]<56:
                minus_point+=5
                minus_reason.append("文字过小，影响观看（-5）")
        elif l==3:#图标
            if bbox_loc_y2[index]-bbox_loc_y1[index]<56:
                minus_point+=5
                minus_reason.append("图标过小，影响观看（-5）")
        elif l==4:#按钮
            if bbox_loc_y2[index]-bbox_loc_y1[index]<56:
                minus_point+=5
                minus_reason.append("按钮过小，影响点击（-5）")
        elif l==8:#翻页器
            if (bbox_loc_x1[index]+bbox_loc_x2[index])/2<705 or (bbox_loc_x1[index]+bbox_loc_x2[index])/2>735:
                minus_point+=4
                minus_reason.append("翻页器建议居中（-4）")

        elif l==5 or l==6:#输入条或列表项
            if bbox_loc_y2[index]-bbox_loc_y1[index]<56:
                minus_point+=5
                minus_reason.append("输入框或者列表项高度过小，影响观看输入（-5）")

        index+=1
    return minus_point,minus_reason
#by ljw 20230106
def compute_coverage(bbox,label):#real data为占比0.366，返回与0.366距离，距离越小越好
    H, W = (2560, 1440)
    box_coords = []  # 其中box1, box2为n个盒子的坐标
    coverage_reason=""
    ifBackGround=False
    index=0
    for box in bbox:
        x1, y1, x2, y2 = convert_xywh_to_ltrb(box)  # 得到左上右下坐标
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        box_coords.append([int(x1),int(y1),int(x2),int(y2)])
        if label[index]==9 or label[index]==10:
            ifBackGround=True
        index+=1

    # 使用scipy库中的Rectangle函数，将n个盒子坐标构建为矩形
    tmp=Solution()
    union_area = Solution.rectangleArea(tmp,rectangles=box_coords)

    # 计算盒子的并集
    ratio=union_area/(H*W)
    if ratio<0.366:
        if  0.366-ratio>0.05:
            coverage_reason="画面过于空旷，可以考虑丰富部件内容"
        else:
            coverage_reason = "画面部件占比较为合适，可以考虑再适当丰富。"
        return 0.366-ratio,coverage_reason
    else:
        if  ratio-0.366>0.05 and ifBackGround is False:
            coverage_reason="画面过于拥挤，可以考虑删减部件或避免重叠"
        else:
            coverage_reason = "画面部件占比较为合适，可以考虑再适当删减。"
        return ratio,coverage_reason


class Segtree:
    def __init__(self):
        self.cover = 0
        self.length = 0
        self.max_length = 0
class Solution:
    def rectangleArea(self, rectangles: List[List[int]]) -> int:
        hbound = set()
        for rect in rectangles:
            hbound.add(rect[1])# 下边界
            hbound.add(rect[3])# 上边界
        hbound = sorted(hbound)
        m = len(hbound)
        # 线段树有 m-1 个叶子节点，对应着 m-1 个会被完整覆盖的线段，需要开辟 ~4m 大小的空间
        tree = [Segtree() for _ in range(m * 4 + 1)]
        def init(idx: int, l: int, r: int) -> None:
            tree[idx].cover = tree[idx].length = 0
            if l == r:
                tree[idx].max_length = hbound[l] - hbound[l - 1]
                return
            mid = (l + r) // 2
            init(idx * 2, l, mid)
            init(idx * 2 + 1, mid + 1, r)
            tree[idx].max_length = tree[idx * 2].max_length + tree[idx * 2 + 1].max_length
        def update(idx: int, l: int, r: int, ul: int, ur: int, diff: int) -> None:
            if l > ur or r < ul:
                return
            if ul <= l and r <= ur:
                tree[idx].cover += diff
                pushup(idx, l, r)
                return
            mid = (l + r) // 2
            update(idx * 2, l, mid, ul, ur, diff)
            update(idx * 2 + 1, mid + 1, r, ul, ur, diff)
            pushup(idx, l, r)
        def pushup(idx: int, l: int, r: int) -> None:
            if tree[idx].cover > 0:
                tree[idx].length = tree[idx].max_length
            elif l == r:
                tree[idx].length = 0
            else:
                tree[idx].length = tree[idx * 2].length + tree[idx * 2 + 1].length
        init(1, 1, m - 1)
        sweep = list()
        for i, rect in enumerate(rectangles):
            sweep.append((rect[0], i, 1))# 左边界
            sweep.append((rect[2], i, -1))# 右边界
        sweep.sort()

        ans = i = 0
        while i < len(sweep):
            j = i
            while j + 1 < len(sweep) and sweep[i][0] == sweep[j + 1][0]:
                j += 1
            if j + 1 == len(sweep):
                break
            # 一次性地处理掉一批横坐标相同的左右边界
            for k in range(i, j + 1):
                _, idx, diff = sweep[k]
                # 使用二分查找得到完整覆盖的线段的编号范围
                left = bisect_left(hbound, rectangles[idx][1]) + 1
                right = bisect_left(hbound, rectangles[idx][3])
                update(1, 1, m - 1, left, right, diff)
            ans += tree[1].length * (sweep[j + 1][0] - sweep[j][0])
            i = j + 1
        return ans % (10 ** 9 + 7)


