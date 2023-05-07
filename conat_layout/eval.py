import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.utils import to_dense_batch

from data import get_dataset
from metric import LayoutFID, compute_maximum_iou, \
    compute_overlap, compute_alignment, compute_Irregular,compute_coverage
import pandas as pd

from util import convert_layout_to_image
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def average(scores):
    return sum(scores) / len(scores)
#ricoTest output/generated_layouts_ricoTest.pkl
#tensorboard --logdir=F:\GradeThree\YuQianLab\GraduateDesign\const_layout_remote\output\ricoTest\LayoutGAN++\20221108175822771427

def print_scores(score_dict):
    for k, v in score_dict.items():
        if k in ['Alignment', 'Overlap']:
            v = [_v * 100 for _v in v]
        if len(v) > 1:
            mean, std = np.mean(v), np.std(v)
            print(f'\t{k}: {mean:.2f} ({std:.2f})')
        else:
            print(f'\t{k}: {v[0]:.2f}')

#------------by ljw 20221223----------#
#分数转正态分布百分制
import scipy
import random
import numpy as np
from scipy.stats import norm

#计算相应百分值对应的正态分布函数
def norm_fx(x1,x1_p,x2,x2_p):
    tmp1=norm.ppf(x1_p,0,1)
    tmp2 = norm.ppf(x2_p, 0, 1)
    scale=(x1-x2)/(tmp1-tmp2)
    loc=x1-tmp1*scale
    return loc,scale

def norm_point(loc_,scale_,pre_point):
    point=norm.cdf(pre_point,loc=loc_,scale=scale_)*100
    return point
#-------------------------------------#
def layout_json_thing(coverage,alignment,overlap,minus_point,count):
    alignment_tmp = norm_fx(-0.72, 0.60, -0.26, 0.9999)
    alignment_point = norm_point(alignment_tmp[0], alignment_tmp[1], alignment * -100)

    overlap_tmp = norm_fx(-60.45, 0.60, -50.58, 0.9999)
    overlap_point = norm_point(overlap_tmp[0], overlap_tmp[1], overlap * 100 * -1)

    coverage_tmp = norm_fx(0.316, 0.60, 0.366, 0.9999)
    coverage_point = norm_point(coverage_tmp[0], coverage_tmp[1], coverage)

    total=coverage_point / 3 + alignment_point / 3 + overlap_point / 3 - minus_point / count
    return coverage_point,alignment_point,overlap_point,total

#python eval.py ricoTest output/generated_layouts_ricoTest.pkl
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default='ricoTest', help='dataset name',
                        choices=['rico', 'publaynet', 'magazine','ricoTest'])
    parser.add_argument('pkl_paths', type=str, default='output/generated_layouts.pkl', nargs='+',
                        help='generated pickle path')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='input batch size')#by ljw建议永远为1，否则json文件易
    parser.add_argument('--compute_real', action='store_true')

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(args.dataset, 'test')
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            num_workers=0,
                            pin_memory=True,
                            shuffle=False)
    test_layouts = [(data.x.numpy(), data.y.numpy()) for data in dataset]

    # prepare for evaluation
    # if args.dataset=="ricoTest":
    #     fid_test = LayoutFID("ricoTest", device)#by ljw 20221109
    # else:
    fid_test = LayoutFID(args.dataset, device)
    # real layouts

    args.compute_real=True
    alignment, overlap = [], []
    # 保存布局信息以及评价信息
    layouts_json = []
    original_json = {
        "layouts_json": layouts_json
    }


    for i, data in enumerate(dataloader):
        data = data.to(device)
        label, mask = to_dense_batch(data.y, data.batch)
        bbox, _ = to_dense_batch(data.x, data.batch)
        padding_mask = ~mask

        fid_test.collect_features(bbox, label, padding_mask,
                                  real=True)


    if args.compute_real:
        dataset = get_dataset(args.dataset, 'val')
        dataloader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                num_workers=0,
                                pin_memory=True,
                                shuffle=False)
        val_layouts = [(data.x.numpy(), data.y.numpy()) for data in dataset]

        real_minus_point=0#by ljw 20230223
        real_coverage_point = 0  # by ljw 20230223

        num_point=0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            bbox, _ = to_dense_batch(data.x, data.batch)
            padding_mask = ~mask
            # --------by ljw 20230103------------------#
            fid_test.collect_features(bbox, label, padding_mask)
            alignment += compute_alignment(bbox, mask).tolist()
            overlap += compute_overlap(bbox, mask).tolist()

            tmp_alignment = average(compute_alignment(bbox, mask).tolist())
            tmp_overlap = average(compute_overlap(bbox, mask).tolist())

            #计算minus_point
            label_c=torch.relu(label[:, 1:] - 1)
            mask_c=mask[:, 1:]
            bbox_c=bbox[:, 1:]

            for j in range(bbox_c.size(0)):
                num_point+=1
                mask_j = mask_c[j]
                b_init = bbox_c[j][mask_j].cpu().detach().numpy()  # by ljw 20230104
                l_init = label_c[j][mask_j].cpu().detach().numpy()  # by ljw 20230104
                tmp_point, minus_reason = compute_Irregular(b_init, l_init)  # by ljw 20230221
                tmp_coverage, coverage_reason=0,[]
                if b_init.size!=0:
                    tmp_coverage, coverage_reason = compute_coverage(b_init, l_init)
                real_minus_point+=tmp_point
                real_coverage_point+= tmp_coverage
                layout_json = {}
                layout_json["bboxs"] = b_init
                layout_json["labels"] = l_init
                layout_json["coverage"] = round(tmp_coverage, 4)
                layout_json["coverage_reason"] = coverage_reason
                layout_json["minus_point"] = tmp_point
                layout_json["minus_reason"] = minus_reason
                layout_json["alignment"] = round(tmp_alignment*100,4)
                layout_json["overlap"] = round(tmp_overlap*100,4)

                tmp_json_thing = layout_json_thing(tmp_coverage, tmp_alignment, tmp_overlap,tmp_point,num_point)
                layout_json["coverage%"] = round(tmp_json_thing[0],4)
                layout_json["alignment%"] = round(tmp_json_thing[1],4)
                layout_json["overlap%"] = round(tmp_json_thing[2],4)
                layout_json["total"] = round(tmp_json_thing[3],4)
                layouts_json.append(layout_json)
            #------------------------------------------#

        fid_score = fid_test.compute_score()
        max_iou = compute_maximum_iou(test_layouts, val_layouts)
        alignment = average(alignment)
        overlap = average(overlap)
        real_minus_point=real_minus_point/num_point
        real_coverage_point = real_coverage_point / num_point
        # -----by ljw 20221223-------------#
        # 分数百分化，全乘-1
        alignment_tmp = norm_fx(-0.72, 0.60, -0.26, 0.9999)
        alignment_point = norm_point(alignment_tmp[0], alignment_tmp[1], alignment * -100)

        overlap_tmp = norm_fx(-60.45, 0.60, -50.58, 0.9999)
        overlap_point = norm_point(overlap_tmp[0], overlap_tmp[1], overlap * 100 * -1)

        coverage_tmp = norm_fx(0.316, 0.60, 0.366, 0.9999)
        coverage_point = norm_point(coverage_tmp[0], coverage_tmp[1], real_coverage_point)

        # ------by ljw 20230410-----#
        # todo 记录整体评估信息
        original_json["average_alignment"] = round(alignment*100,4)
        original_json["average_overlap"] = round(overlap*100,4)
        original_json["average_coverage"] = round(real_coverage_point,4)
        original_json["average_alignment%"] = round(alignment_point,4)
        original_json["average_overlap%"] = round(overlap_point,4)
        original_json["average_coverage%"] = round(coverage_point,4)
        original_json["average_minus_point"] = round(real_minus_point,4)
        original_json["average_Total"] = round((coverage_point / 3 + alignment_point / 3 + overlap_point / 3 - real_minus_point),4)
        original_json["FID"] = round(fid_score,4)
        original_json["max_iou"] = round(max_iou,4)
        original_json["layouts_json"]=layouts_json

        with open("./eval_original_ricoTest.json", 'w') as f:
            json.dump(original_json, f, cls=NpEncoder)
        # -----------------------------------------------#









        print('Real data:')
        print_scores({
            'FID': [fid_score],
            'Max. IoU': [max_iou],
            'Alignment': [alignment],
            'Overlap': [overlap],
            'minus_point':[real_minus_point]
        })
        print()

    # generated layouts
    scores = defaultdict(list)
    per_scores=defaultdict(list)#by ljw 20221223
    minus_point=0#by ljw 20221226
    coverage=0#by ljw 20230106
    count=0

    #保存布局信息以及评价信息
    layouts_json=[]
    res_json={
        "layouts_json": layouts_json
    }

    for pkl_path in args.pkl_paths:
        alignment, overlap = [], []
        generated_layouts=[]
        inform_layouts = []
        with Path(pkl_path).open('rb') as fb:
            #generated_layouts = pd.read_parquet(path=pkl_path)  # by ljw 20220920
            generated_layouts_ = pickle.load(fb)
        for tmp in generated_layouts_:
            generated_layouts.append((tmp[0],tmp[1]))
            inform_layouts.append(tmp[2])
        for i in range(0, len(generated_layouts), args.batch_size):
            i_end = min(i + args.batch_size, len(generated_layouts))

            # get batch from data list
            data_list = []
            for b, l in generated_layouts[i:i_end]:
                bbox = torch.tensor(b, dtype=torch.float)
                label = torch.tensor(l, dtype=torch.long)
                # ricoLabel = torch.tensor(ricoLabel,  dtype=torch.long)
                data = Data(x=bbox, y=label)
                # data.ricoLabel=ricoLabel
                data_list.append(data)

            data = Batch.from_data_list(data_list)
            data = data.to(device)
            label, mask = to_dense_batch(data.y, data.batch)
            bbox, _ = to_dense_batch(data.x, data.batch)
            padding_mask = ~mask

            # to do加入ricoLabel映射
            fid_test.collect_features(bbox, label, padding_mask)


            # --------------by ljw 20230410---------------------------#
            #todo 记录单张图片评估信息
            count += 1
            b,l=generated_layouts[i:i_end][0]
            tmp_coverage, coverage_reason = compute_coverage(b, l)
            coverage += tmp_coverage
            tmp_point, minus_reason = compute_Irregular(b, l)
            minus_point += tmp_point
            tmp_alignment = average(compute_alignment(bbox, mask).tolist())
            tmp_overlap = average(compute_overlap(bbox, mask).tolist())
            layout_json = {}
            layout_json["bboxs"] = b
            layout_json["labels"] = l
            layout_json["coverage"] = round(tmp_coverage,4)
            layout_json["coverage_reason"] = coverage_reason
            layout_json["minus_point"] = round(tmp_point,4)
            layout_json["minus_reason"] = minus_reason
            layout_json["alignment"] =round(tmp_alignment*100,4)
            layout_json["overlap"] =round( tmp_overlap*100,4)

            tmp_json_thing=layout_json_thing(tmp_coverage,tmp_alignment,tmp_overlap, minus_point, count)
            layout_json["coverage%"] = round(tmp_json_thing[0],4)
            layout_json["alignment%"] = round(tmp_json_thing[1],4)
            layout_json["overlap%"] = round(tmp_json_thing[2],4)
            layout_json["total"] = round(tmp_json_thing[3],4)

            layout_json["name"] = inform_layouts[i]["name"]
            layout_json["width"] = inform_layouts[i]["width"]
            layout_json["height"] = inform_layouts[i]["height"]
            layout_json["ricoLabel"] = inform_layouts[i]["ricoLabel"]
            layout_json["box_len"] = inform_layouts[i]["box_len"]

            layouts_json.append(layout_json)
            #----------------------------------------------------------#

            alignment += compute_alignment(bbox, mask).tolist()
            overlap += compute_overlap(bbox, mask).tolist()



        fid_score = fid_test.compute_score()
        # if args.dataset == 'ricoTest':
        #     max_iou = compute_maximum_iou(test_layouts, generated_layouts,ifRicoTest=True)#by ljw 20221109报错后续解决
        # else:
        max_iou = compute_maximum_iou(test_layouts, generated_layouts)  # by ljw 20221109报错后续解决
        alignment = average(alignment)
        overlap = average(overlap)
        coverage=coverage/count

        scores['FID'].append(fid_score)
        scores['Max. IoU'].append(max_iou)#by ljw 20221109报错后续解决
        scores['Alignment'].append(alignment)
        scores['Overlap'].append(overlap)
        scores['minus_point'].append(minus_point/count)


        # -----by ljw 20221223-------------#
        # 分数百分化，全乘-1
        FID_tmp = norm_fx(-14.43, 0.60, -4.47, 0.9999)
        FID_point = norm_point(FID_tmp[0], FID_tmp[1], fid_score * -1)

        alignment_tmp = norm_fx(-0.72, 0.60, -0.26, 0.9999)
        alignment_point = norm_point(alignment_tmp[0], alignment_tmp[1], alignment * -100)

        overlap_tmp = norm_fx(-60.45, 0.60, -50.58, 0.9999)
        overlap_point = norm_point(overlap_tmp[0], overlap_tmp[1],overlap*100*-1)

        coverage_tmp=norm_fx(0.316, 0.60, 0.366, 0.9999)
        coverage_point = norm_point(coverage_tmp[0], coverage_tmp[1], coverage)

        # ------by ljw 20230410-----#
        # todo 记录整体评估信息
        res_json["average_alignment"] = round(alignment*100,4)
        res_json["average_overlap"] = round(overlap*100,4)
        res_json["average_coverage"] = round(coverage,4)
        res_json["average_alignment%"] =round( alignment_point,4)
        res_json["average_overlap%"] =round( overlap_point,4)
        res_json["average_coverage%"] = round(coverage_point,4)
        res_json["average_minus_point"]=round(minus_point/count,4)
        res_json["average_Total"]=round((coverage_point/3+alignment_point/3+overlap_point/3-minus_point/count),4)
        res_json["FID"] =round( fid_score,4)
        res_json["max_iou"] = round(max_iou,4)
        res_json["layouts_json"] = layouts_json



        with open("./eval_ricoTest.json", 'w') as f:
            json.dump(res_json, f, cls=NpEncoder)
        # -----------------------------------------------#

        per_scores['FID'].append(FID_point)
        per_scores['Alignment'].append(alignment_point)
        per_scores['Overlap'].append(overlap_point)
        per_scores['Minus_point'].append(res_json["average_minus_point"])
        per_scores['Coverage_point'].append(coverage_point)
        per_scores['Total'].append(res_json["average_Total"])


        #----------------------------------------------------------------------#

    print(f'Input size: {len(args.pkl_paths)}')
    print(f'Dataset: {args.dataset}')
    #print_scores(scores)#by ljw 20221230
    print_scores(scores)

    print("FID")
    print(per_scores['FID'])#by ljw 20221223百分化输出

    print("Alignment")
    print(per_scores['Alignment'])  # by ljw 20221223百分化输出

    print("Overlap")
    print(per_scores['Overlap'])  # by ljw 20221223百分化输出

    print("Coverage")
    print(per_scores['Coverage_point'])  # by ljw 20221223百分化输出

    print("Minus_point")
    print(per_scores['Minus_point'])  # by ljw 20221223百分化输出

    print("Total")
    print(per_scores['Total'])  # by ljw 20221223百分化输出




if __name__ == "__main__":
    main()
