import torch
import random
from enum import IntEnum
from itertools import product, combinations

import functools


class RelSize(IntEnum):
    UNKNOWN = 0
    SMALLER = 1
    EQUAL = 2
    LARGER = 3


class RelLoc(IntEnum):
    UNKNOWN = 4
    LEFT = 5
    TOP = 6
    RIGHT = 7
    BOTTOM = 8
    CENTER = 9


REL_SIZE_ALPHA = 0.1


def detect_size_relation(b1, b2):
    a1, a2 = b1[2] * b1[3], b2[2] * b2[3]
    a1_sm = (1 - REL_SIZE_ALPHA) * a1
    a1_lg = (1 + REL_SIZE_ALPHA) * a1

    if a2 <= a1_sm:
        return RelSize.SMALLER

    if a1_sm < a2 and a2 < a1_lg:
        return RelSize.EQUAL

    if a1_lg <= a2:
        return RelSize.LARGER

    raise RuntimeError(b1, b2)


def detect_loc_relation(b1, b2, canvas=False):
    if canvas:
        yc = b2[1]
        y_sm, y_lg = 1. / 3, 2. / 3

        if yc <= y_sm:
            return RelLoc.TOP

        if y_sm < yc and yc < y_lg:
            return RelLoc.CENTER

        if y_lg <= yc:
            return RelLoc.BOTTOM

    else:
        l1, t1, r1, b1 = convert_xywh_to_ltrb(b1)
        l2, t2, r2, b2 = convert_xywh_to_ltrb(b2)

        if b2 <= t1:
            return RelLoc.TOP

        if b1 <= t2:
            return RelLoc.BOTTOM

        if t1 < b2 and t2 < b1:
            if r2 <= l1:
                return RelLoc.LEFT

            if r1 <= l2:
                return RelLoc.RIGHT

            if l1 < r2 and l2 < r1:
                return RelLoc.CENTER

    raise RuntimeError(b1, b2, canvas)


def get_rel_text(rel, canvas=False):
    if type(rel) == RelSize:
        index = rel - RelSize.UNKNOWN - 1
        if canvas:
            return [
                'within canvas',
                'spread over canvas',
                'out of canvas',
            ][index]

        else:
            return [
                'larger than',
                'equal to',
                'smaller than',
            ][index]

    else:
        index = rel - RelLoc.UNKNOWN - 1
        if canvas:
            return [
                '', 'at top',
                '', 'at bottom',
                'at middle',
            ][index]

        else:
            return [
                'right to', 'below',
                'left to', 'above',
                'around',
            ][index]


class LexicographicSort():
    def __call__(self, data):
        assert not data.attr['has_canvas_element']
        l, t, _, _ = convert_xywh_to_ltrb(data.x.t())
        _zip = zip(*sorted(enumerate(zip(t, l)), key=lambda c: c[1:]))
        idx = list(list(_zip)[0])
        data.x_orig, data.y_orig = data.x, data.y
        data.x, data.y = data.x[idx], data.y[idx]
        return data


class HorizontalFlip():
    def __call__(self, data):
        data.x = data.x.clone()
        data.x[:, 0] = 1 - data.x[:, 0]
        return data


class AddCanvasElement():
    def __init__(self):
        self.x = torch.tensor([[.5, .5, 1., 1.]], dtype=torch.float)
        self.y = torch.tensor([0], dtype=torch.long)
        self.ricoLabel = torch.tensor([0], dtype=torch.long)#by ljw 20221109

    def __call__(self, data):
        if not data.attr['has_canvas_element']:
            data.x = torch.cat([self.x, data.x], dim=0)
            data.y = torch.cat([self.y, data.y + 1], dim=0)
            #----------by ljw 20221109-------------------#
            if hasattr(data,"ricoLabel"):
                data.ricoLabel = torch.cat([self.ricoLabel, data.ricoLabel + 1], dim=0)#by ljw 20230105


            #--------------------------------------------#
            data.attr = data.attr.copy()
            data.attr['has_canvas_element'] = True
        return data


class AddRelation():
    def __init__(self, seed=None, ratio=0.1):
        self.ratio = ratio
        self.generator = random.Random()
        if seed is not None:
            self.generator.seed(seed)

    def __call__(self, data):
        N = data.x.size(0)
        has_canvas = data.attr['has_canvas_element']

        rel_all = list(product(range(2), combinations(range(N), 2)))
        size = int(len(rel_all) * self.ratio)
        rel_sample = set(self.generator.sample(rel_all, size))

        edge_index, edge_attr = [], []
        rel_unk = 1 << RelSize.UNKNOWN | 1 << RelLoc.UNKNOWN
        # for i, j in combinations(range(N), 2):
        #     bi, bj = data.x[i], data.x[j]
        #     canvas = data.y[i] == 0 and has_canvas
        #
        #     if (0, (i, j)) in rel_sample:
        #         rel_size = 1 << detect_size_relation(bi, bj)
        #     else:
        #         rel_size = 1 << RelSize.UNKNOWN
        #
        #     if (1, (i, j)) in rel_sample:
        #         rel_loc = 1 << detect_loc_relation(bi, bj, canvas)
        #     else:
        #         rel_loc = 1 << RelLoc.UNKNOWN
        #
        #     rel = rel_size | rel_loc
        #     if rel != rel_unk:
        #         edge_index.append((i, j))
        #         edge_attr.append(rel)

        #-----------by ljw 20221227------------#
        #按照规则精简布局条件
        edge_index, edge_attr = [], []
        bbox=data.x.numpy().tolist()
        label = data.y.numpy().tolist()
        edge_index_irr, edge_attr_irr=const_Irregular(bbox, label)
        edge_index_refine,edge_attr_refine=const_refine(bbox,edge_index_irr,edge_attr_irr)
        edge_index,edge_attr=edge_index_refine+edge_index_irr,edge_attr_refine+edge_attr_irr
        #--------------------------------------#


        data.edge_index = torch.as_tensor(edge_index).long()
        data.edge_index = data.edge_index.t().contiguous()
        data.edge_attr = torch.as_tensor(edge_attr).long()

        return data
#by ljw 2021228
def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]
#by ljw 20221228
def closest_plus(mylist, Number,real_index):
    answer_plus = []
    for i in mylist:
        if(Number-i>=0):
            answer_plus.append(abs(Number-i))
    if len(answer_plus)>0:
        return real_index[answer_plus.index(min(answer_plus))]
    else:
        return -1
#by ljw 20221228
def closest_minus(mylist, Number,real_index):
    answer_minus= []
    for i in mylist:
        if (Number - i < 0):
            answer_minus.append(abs(Number - i))
    if len(answer_minus) > 0:
        return real_index[answer_minus.index(min(answer_minus))]
    else:
        return -1


# 自定义排序规则
def my_compare(x, y):
    if x.WC > y.WC:
        return 1
    elif x.WC < y.WC:
        return -1
    return 0

#by ljw 20230105
#提炼原有布局上下关系
def const_refine(bbox,edge_index_irr, edge_attr_irr):
    H, W = (2560, 1440)
    bbox_all=[]
    bbox_layered= []
    edge_index, edge_attr = [], []

    box_index=0
    for box in bbox:
        x1, y1, x2, y2 = convert_xywh_to_ltrb(box)  # 得到左上右下坐标
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        tmp_box = {
            'id':box_index,
            'x1':x1,'x2':x2,'y1':y1,'y2':y2,
            'H':y2-y1,'W':x2-x1,
            'xC':(x2+x1)/2, 'yC': (y2+y1)/2,
        }
        bbox_all.append(tmp_box)
        box_index+=1
    bbox_all = sorted(bbox_all, key=lambda r: r['yC'])#按照盒子中心高度由低到高排序

    #把中心距离在56px以内的下标放到一起
    index = 0
    while(index<len(bbox_all)):
        tmp=[]
        end_index=index
        for x_index in range(index,len(bbox_all)):
            if bbox_all[x_index]['yC'] >= bbox_all[index]['yC'] and bbox_all[x_index]['yC'] <= bbox_all[index]['yC'] + 56:
                tmp.append(bbox_all[x_index])
                end_index = x_index
        tmp=sorted(tmp, key=lambda r: r['xC'])#按照盒子中心由左到右排序
        bbox_layered.append(tmp)

        index=end_index+1

    #对分层bbox加入edge_index条件
    layer_index=0
    for layer in bbox_layered:
       if len(layer)>1:
           rel_loc = 1 << RelLoc.RIGHT
           rel_size = 1 << RelSize.UNKNOWN
           rel = rel_size | rel_loc

           for tmpi in range(0,len(layer)):
               if tmpi==0:
                   continue
               index1=layer[tmpi-1]['id']
               index2=layer[tmpi]['id']
               if  if_Overlap(layer[tmpi],layer[tmpi-1]) is False and \
                       if_Irregular(index1,index2,rel,edge_index_irr, edge_attr_irr) is False:
                   edge_index.append((index1,index2))  # 同层次左右关系,不可重叠，不可包含于irregular
                   edge_attr.append(rel)

       if layer_index!=0 and len(bbox_layered[layer_index-1])>0:
           rel_loc = 1 << RelLoc.BOTTOM
           rel_size = 1 << RelSize.UNKNOWN
           rel = rel_size | rel_loc
           tmpBox1=bbox_layered[layer_index][0]
           tmpBox2=bbox_layered[layer_index-1][0]
           index1 = tmpBox1['id']
           index2 = tmpBox2['id']
           if if_Overlap(tmpBox1,tmpBox2) is False and \
                   if_Irregular(index1, index2, rel, edge_index_irr, edge_attr_irr) is False:
               edge_index.append((index1, index2))  # 同层次左右关系,不可重叠，不可包含于irregular
               edge_attr.append(rel)
       layer_index+=1
    return edge_index, edge_attr
#by ljw 20230222
#此关系是否已经存在
def if_Irregular(index1,index2,rel,edge_index_irr, edge_attr_irr):
    for index in range(0,len(edge_index_irr)):
        if (index1,index2)==edge_index_irr[index] and rel==edge_attr_irr[index]:
            return True
    return False
#by ljw 20230222
#两盒子是否重叠
def if_Overlap(tmpBox1,tmpBox2):
    if (max(tmpBox1['x1'],tmpBox1['x2']) < min(tmpBox2['x1'],tmpBox2['x2']) or
            max(tmpBox1['y1'],tmpBox1['y2']) < min(tmpBox2['y1'],tmpBox2['y2']) or
            min(tmpBox1['x1'],tmpBox1['x2']) > max(tmpBox2['x1'],tmpBox2['x2']) or
            min(tmpBox1['y1'],tmpBox1['y2']) > max(tmpBox2['y1'],tmpBox2['y2'])):
        return True
    return False
#by ljw 20221227
#汇总违反的小项目条件
def const_Irregular(bbox, label):
    minus_point=0
    index=0
    H, W = (2560, 1440)
    bbox_loc_x1=[]
    bbox_loc_y1 = []
    bbox_loc_x2 = []
    bbox_loc_y2 = []
    bbox_WC = []
    bbox_H = []
    label_ = []
    real_index=[]
    edge_index, edge_attr = [], []
    minus_reason = []
    rel_unk = 1 << RelSize.UNKNOWN | 1 << RelLoc.UNKNOWN
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
        bbox_H.append(y2-y1)
        bbox_WC.append((x2+x1)/2)
        label_.append(label[index])
        real_index.append(index)#去掉背景，获取真正下标by ljw 20230221
        index += 1
    index=0
    for l in label_:
        if l==0:#导航栏,返回栏
            if bbox_H[index]<176 or bbox_H[index]>264:
                # 导航栏高度
                index_box_plus = closest_plus(bbox_H, 176,real_index)
                index_box_minus = closest_minus(bbox_H, 176,real_index)
                rel_loc = 1 << RelLoc.UNKNOWN
                if index_box_plus!=-1:
                    rel_size = 1 << RelSize.SMALLER
                    if bbox_H[real_index.index(index_box_plus)]>=156 and bbox_H[real_index.index(index_box_plus)]<=196:
                        rel_size = 1 << RelSize.EQUAL
                    rel = rel_size | rel_loc
                    if rel != rel_unk:
                        edge_index.append((index_box_plus, real_index[index]))  # 后面和前面比较的关系
                        edge_attr.append(rel)
                if index_box_minus != -1:
                    rel_size = 1 << RelSize.LARGER
                    if bbox_H[real_index.index(index_box_minus)]>=156 and bbox_H[real_index.index(index_box_minus)]<=196:
                        rel_size = 1 << RelSize.EQUAL
                    rel = rel_size | rel_loc
                    if rel != rel_unk:
                        edge_index.append((index_box_minus, real_index[index]))  # 后面和前面比较的关系
                        edge_attr.append(rel)
            if bbox_loc_y1[index]>min(bbox_loc_y1)+60 or (bbox_loc_y2[index]>400 and bbox_loc_y2[index]<2300):
                #导航栏位于顶部或者底部
                # if (bbox_loc_y2[index]-bbox_loc_y1[index])/2<=H/2:
                    #靠上,去top
                index_box_top=bbox_loc_y1.index(min(bbox_loc_y1))
                rel_loc = 1 << RelLoc.TOP
                rel_size = 1 << RelSize.UNKNOWN
                rel = rel_size | rel_loc
                if rel != rel_unk:
                    edge_index.append((index_box_top, real_index[index]))  # 后面和前面比较的关系
                    edge_attr.append(rel)
                # else:
                #     #靠下,去bottom
                #     index_box_bottom = bbox_loc_y1.index(min(bbox_loc_y1))
                #     rel_loc = 1 << RelLoc.TOP
                #     rel_size = 1 << RelSize.UNKNOWN
                #     rel = rel_size | rel_loc
                #     if rel != rel_unk:
                #         edge_index.append((index_box_bottom, index))  # 后面和前面比较的关系
                #         edge_attr.append(rel)


        # elif l==1:#图片
        #     if bbox_H[index]<400:
        #         #图片高度
        #         index_box_plus = closest_plus(bbox_H, 400,real_index)
        #         index_box_minus = closest_minus(bbox_H, 400,real_index)
        #         rel_loc = 1 << RelLoc.UNKNOWN
        #         if index_box_plus != -1:
        #             rel_size = 1 << RelSize.SMALLER
        #             rel = rel_size | rel_loc
        #             if rel != rel_unk:
        #                 edge_index.append((index_box_plus, real_index[index]))  # 后面和前面比较的关系
        #                 edge_attr.append(rel)
        #         if index_box_minus != -1:
        #             rel_size = 1 << RelSize.LARGER
        #             rel = rel_size | rel_loc
        #             if rel != rel_unk:
        #                 edge_index.append((index_box_minus, real_index[index]))  # 后面和前面比较的关系
        #                 edge_attr.append(rel)

        elif l>=2 and l<=6:#文字/图标/按钮/列表项/输入框
            if bbox_H[index]<56:
                #文字大小
                index_box_plus = closest_plus(bbox_H, 56,real_index)
                index_box_minus = closest_minus(bbox_H, 56,real_index)
                rel_loc = 1 << RelLoc.UNKNOWN
                if index_box_plus != -1:
                    rel_size = 1 << RelSize.SMALLER
                    if bbox_H[real_index.index(index_box_plus)]>=50 and bbox_H[real_index.index(index_box_plus)]<=76:
                        rel_size = 1 << RelSize.EQUAL
                    rel = rel_size | rel_loc
                    if rel != rel_unk:
                        edge_index.append((index_box_plus, real_index[index]))  # 后面和前面比较的关系
                        edge_attr.append(rel)
                if index_box_minus != -1:
                    rel_size = 1 << RelSize.LARGER
                    if bbox_H[real_index.index(index_box_minus)] >= 50 and bbox_H[real_index.index(index_box_minus)] <= 76:
                        rel_size = 1 << RelSize.EQUAL
                    rel = rel_size | rel_loc
                    if rel != rel_unk:
                        edge_index.append((index_box_minus, real_index[index]))  # 后面和前面比较的关系
                        edge_attr.append(rel)

        elif l==8:#翻页器
            if bbox_WC[index]<705 or bbox_WC[index]>735:#720+-15
                #翻页器居中
                index_box_plus = closest_plus(bbox_WC, 720,real_index)
                index_box_minus = closest_minus(bbox_WC, 720,real_index)
                rel_size = 1 << RelLoc.UNKNOWN
                if index_box_plus != -1:
                    rel_loc = 1 << RelLoc.CENTER
                    rel = rel_size | rel_loc
                    if rel != rel_unk:
                        edge_index.append((index_box_plus, real_index[index]))  # 后面和前面比较的关系
                        edge_attr.append(rel)
                if index_box_minus != -1:
                    rel_loc = 1 << RelLoc.CENTER
                    rel = rel_size | rel_loc
                    if rel != rel_unk:
                        edge_index.append((index_box_minus, real_index[index]))  # 后面和前面比较的关系
                        edge_attr.append(rel)
        index+=1
    return edge_index, edge_attr



if __name__ == '__main__':
    print("a")
    # print(closest([1.5,2.0,3.3],3.1))