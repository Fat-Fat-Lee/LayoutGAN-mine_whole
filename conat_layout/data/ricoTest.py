import json
import os
import pickle
from pathlib import Path

import torch
from torch_geometric.data import Data

from data.base import BaseDataset


def append_child(element, elements):
    if 'children' in element.keys():
        for child in element['children']:
            elements.append(child)
            elements = append_child(child, elements)
    return elements
def get_dataset_(file_path=None,in_ricoLabel=None):
    tmp = RicoTest('gen',None)
    tmp.process_(file_path=file_path,in_ricoLabel=in_ricoLabel)
    return
#by ljw 20221102
#整个文件
class RicoTest(BaseDataset):
    labels = [
        'Toolbar',
        'Image',
        'Text',
        'Icon',
        'Text Button',
        'Input',
        'List Item',
        'Advertisement',
        'Pager Indicator',
        'Web View',
        'Background Image',
        'Drawer',
        'Modal',
    ]
    #-----by ljw 20221102---#
    #to do注明ricoLabel类型
    ricoLabels=[
        'Calendar Page','Calculator Page','Camera Page','Address Page','Translator Page','Schedule Page','Email Page',
        'Chat Page','Timer Page', 'Question Page','Weather Page','GPS Page','Install Page','Product Page','Search Page',
        'News Page','Guide Page','Start Page','Update Page','Introduction Page','Home Page','Login Page', 'Pop Window',
        'Blank Page','other']#长度为25对应num_ricoClasses的大小
    #-----------------------#

    def __init__(self, split='train', transform=None,if_small=False):
        super().__init__('ricoTest', split, transform)
        #------by ljw 20221102-----#
        #to do加入num_ricoClasses元素
        self.num_ricoClasses=25
        self.labels = [
            'Toolbar', 'Image', 'Text', 'Icon', 'Text Button', 'Input', 'List Item',
            'Advertisement', 'Pager Indicator', 'Web View', 'Background Image', 'Drawer', 'Modal',
        ]
        self.ricoLabels = [
            'Calendar Page', 'Calculator Page', 'Camera Page', 'Address Page', 'Translator Page', 'Schedule Page',
            'Email Page',
            'Chat Page', 'Timer Page', 'Question Page', 'Weather Page', 'GPS Page', 'Install Page', 'Product Page',
            'Search Page',
            'News Page', 'Guide Page', 'Start Page', 'Update Page', 'Introduction Page', 'Home Page', 'Login Page',
            'Pop Window',
            'Blank Page', 'other'
        ]  # 长度为25对应num_ricoClasses的大小

        self.index2SynzLabels = {
            0: 'alert', 1: 'button', 2: 'card', 3: 'checkbox_checked',
            4: 'checkbox_unchecked', 5: 'chip', 6: 'data_table', 7: 'dropdown_menu',
            8: 'floating_action_button', 9: 'grid_list', 10: 'image', 11: 'label',
            12: 'menu', 13: 'radio_button_checked', 14: 'radio_button_unchecked', 15: 'slider',
            16: 'switch_disabled', 17: 'switch_enabled', 18: 'text_area', 19: 'text_field',
            20: 'tooltip'
        }
        self.Synz2Rico = {
            "label": 'Text',
            "image": 'Image',
            "button": 'Text Button',
            "tooltip": 'Tooltip',
            "card": 'List Item',
            "text_field": 'Input',
            "dropdown_menu": 'Drawer',
            "chip": 'Input',
            "floating_action_button": 'Icon',
            "menu": 'Drawer',
            "data_table": 'Data_Table',
            "grid_list": 'Grid_list',
            "alert": 'Modal',
            "text_area": 'Text',
            "radio_button_unchecked": 'Text Button',
            "radio_button_checked": 'Text Button',
            "checkbox_unchecked": 'Text Button',
            "slider": 'Slider',
            "checkbox_checked": 'Text Button',
            "switch_disabled": 'Text Button',
            "switch_enabled": 'Text Button',

        }

        #--------------------------#

    def download(self):
        super().download()

    def process(self):

        data_list = []
        randn_data_list = []
        raw_dir = Path(self.raw_dir) / 'semantic_annotations'
        tmp = len(os.listdir(raw_dir))
        for json_path in sorted(raw_dir.glob('*.json')):
            with json_path.open(encoding='utf-8') as f:
                ann = json.load(f)

            B = ann['bounds']
            W, H = float(B[2]), float(B[3])
            #--------by ljw 20221102-----#
            #to do读入ricoLabel
            ricoLabel=ann['ricoLabel']
            #----------------------------#
            if B[0] != 0 or B[1] != 0 or H < W:
                continue

            def is_valid(element):
                if element['componentLabel'] not in set(self.labels):
                    return False

                x1, y1, x2, y2 = element['bounds']
                if x1 < 0 or y1 < 0 or W < x2 or H < y2:
                    return False

                if x2 <= x1 or y2 <= y1:
                    return False

                return True

            elements = append_child(ann, [])
            _elements = list(filter(is_valid, elements))
            filtered = len(elements) != len(_elements)
            elements = _elements

            N = len(elements)
            if N == 0 or 9 < N:
                continue

            boxes = []
            randn_boxes=[]#by ljw 20230221随机盒子
            labels = []
            # ----------by ljw 20221103----------#
            # 建立ricoLabel
            tmpLabel = self.ricoLabel2index[ricoLabel]
            ricoLabels = []

            for element in elements:
                # bbox
                x1, y1, x2, y2 = element['bounds']
                xc = (x1 + x2) / 2.
                yc = (y1 + y2) / 2.
                width = x2 - x1
                height = y2 - y1
                b = [xc / W, yc / H,
                     width / W, height / H]
                boxes.append(b)


                # label
                l = element['componentLabel']

                # -------by ljw 20230221---#
                # randn boxes
                randn_b=[xc / W, yc / H,width / W, height / H]
                if self.label2index[l]==0:#导航栏下移
                    randn_b=[xc / W, (yc+200) / H,width / W, height / H]
                elif self.label2index[l]==2:#文字过小
                    randn_b=[xc / W, yc/ H,50/ W, 50 / H]
                elif self.label2index[l]==8:#翻页器不居中
                    randn_b=[(xc-40) / W, yc/ H,(width-100)/ W, height / H]
                randn_boxes.append(randn_b)
                # -------------------------#


                labels.append(self.label2index[l])
                ricoLabels.append(tmpLabel)

            box_len=len(labels)
            boxes = torch.tensor(boxes, dtype=torch.float)
            randn_boxes = torch.tensor(randn_boxes, dtype=torch.float)#by ljw 20230221 randn
            labels = torch.tensor(labels, dtype=torch.long)
            ricoLabels=torch.tensor(ricoLabels, dtype=torch.long)


            data = Data(x=boxes, y=labels)
            data.attr = {
                'name': json_path.name,
                'width': W,
                'height': H,
                'filtered': filtered,
                'has_canvas_element': False,
                #-------by ljw 20221102------#
                #to do读入新特征
                'ricoLabel':ricoLabel,
                'box_len':box_len
                #----------------------------#

            }
            #-------by ljw 20221103--------#
            #加入ricoLable
            data.ricoLabel=ricoLabels
            # data.data_x2rico = data_x2rico
            # data.data_y2rico = data_y2rico
            #------------------------------#
            data_list.append(data)



            #------by ljw 20230221-----#
            #randn boxes
            randn_data = Data(x=randn_boxes, y=labels)
            randn_data.attr = {
                'name': json_path.name,
                'width': W,
                'height': H,
                'filtered': filtered,
                'has_canvas_element': False,
                # -------by ljw 20221102------#
                # to do读入新特征
                'ricoLabel': ricoLabel,
                'box_len': box_len
                # ----------------------------#

            }
            randn_data.ricoLabel = ricoLabels
            randn_data_list.append(randn_data)
            #--------------------------#

        # shuffle with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(data_list), generator=generator)
        data_list = [data_list[i] for i in indices]
        randn_data_list=[randn_data_list[i] for i in indices]#by ljw 20230221



        # train 85% / val 5% / test 10%
        N = len(data_list)
        s = [int(N * .85), int(N * .90)]
        torch.save(self.collate(data_list[:s[0]]), self.processed_paths[0])
        torch.save(self.collate(data_list[s[0]:s[1]]), self.processed_paths[1])
        torch.save(self.collate(data_list[s[1]:]), self.processed_paths[2])

        #randn 不规则版本测试集 by ljw 20230221

        torch.save(self.collate(randn_data_list[s[1]:]), self.processed_paths[3])

    def process_(self,file_path,in_ricoLabel):

        # 打开 pickle 文件，以二进制模式读取
        with open(file_path, 'rb') as f:
            # 读取 pickle 文件中的对象
            obj = pickle.load(f)


        data_list = []

        for tmp in obj:

            H,W = float(tmp["img_shape"][0]), float(tmp["img_shape"][1])
            labels_list=tmp["pred_instances"]["labels"].tolist()
            bboxes_list = tmp["pred_instances"]["bboxes"].tolist()
            scores_list = tmp["pred_instances"]["scores"].tolist()
            if(len(labels_list)==0):
                continue
            if(in_ricoLabel=={} or in_ricoLabel is None):
                ricoLabel="other"
            else:
                img_path=self.ricoLabel2index[os.path.basename(tmp['img_path'])]
                ricoLabel=in_ricoLabel[img_path]

            elements=[]
            filtered=False

            for tmp_index in range(0,len(labels_list)):
                element = {}
                if(scores_list[tmp_index]<0.8):
                    continue
                synz_index=labels_list[tmp_index]
                synz_label=self.index2SynzLabels[synz_index]
                rico_label=self.Synz2Rico[synz_label]

                element["componentLabel"]=rico_label
                element["bounds"]=(bboxes_list[tmp_index][0],bboxes_list[tmp_index][1],
                                   bboxes_list[tmp_index][2],bboxes_list[tmp_index][3])

                if element["componentLabel"] not in self.labels:
                    filtered = True
                    continue
                x1, y1, x2, y2 = element['bounds']
                if x1 < 0 or y1 < 0 or W < x2 or H < y2:
                    filtered=True
                    continue
                if x2 <= x1 or y2 <= y1:
                    filtered = True
                    continue
                elements.append(element)


            N = len(elements)
            if N == 0 or 9 < N:
                continue

            boxes = []
            labels = []
            # ----------by ljw 20221103----------#
            # 建立ricoLabel
            tmpLabel = self.ricoLabel2index[ricoLabel]
            ricoLabels = []

            for element in elements:
                # bbox
                x1, y1, x2, y2 = element['bounds']
                xc = (x1 + x2) / 2.
                yc = (y1 + y2) / 2.
                width = x2 - x1
                height = y2 - y1
                b = [xc / W, yc / H,
                     width / W, height / H]
                boxes.append(b)

                # label
                l = element['componentLabel']
                labels.append(self.label2index[l])
                ricoLabels.append(tmpLabel)

            box_len=len(labels)
            boxes = torch.tensor(boxes, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.long)
            ricoLabels=torch.tensor(ricoLabels, dtype=torch.long)


            data = Data(x=boxes, y=labels)
            data.attr = {
                'name': os.path.basename(tmp['img_path']),
                'width': W,
                'height': H,
                'filtered': filtered,
                'has_canvas_element': False,
                #-------by ljw 20221102------#
                #to do读入新特征
                'ricoLabel':ricoLabel,
                'box_len':box_len
                #----------------------------#

            }
            #-------by ljw 20221103--------#
            #加入ricoLable
            data.ricoLabel=ricoLabels
            # data.data_x2rico = data_x2rico
            # data.data_y2rico = data_y2rico
            #------------------------------#
            data_list.append(data)



        # shuffle with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(data_list), generator=generator)
        data_list = [data_list[i] for i in indices]

        # train 85% / val 5% / test 10%
        if len(data_list)!=0:
            torch.save(self.collate(data_list), self.processed_paths[4])
        else:
            print("This dataset is empty,please check your dataset！")
