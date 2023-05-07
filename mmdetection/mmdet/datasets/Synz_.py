import csv
import json

import pandas as pd


def rect_to_polygon(x, y, w, h):
    """
    Convert rectangle to polygon format
    :param x: x-coordinate of top-left corner
    :param y: y-coordinate of top-left corner
    :param w: width of rectangle
    :param h: height of rectangle
    :return: polygon list in [[x1, y1], [x2, y2], ... [xn, yn]] format
    """
    x1, y1 = x, y
    x2, y2 = x + w, y
    x3, y3 = x + w, y + h
    x4, y4 = x, y + h

    return [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

def Synz2Coco(json_path,out_file):
    label2labelID={
       'Text':1,
       'Image':2,
       'Text Button':3,
       'Tooltip':4,
       'List Item':5,
       'Input':6,
       'Drawer':7,
       'Icon':8,
       'Data_Table':9,
       'Grid_list':10,
       'Modal':11,
       'Slider':12,
    }
    rico_uisketch_map ={
        "label":'Text',
        "image":'Image',
        "button":'Text Button',
        "tooltip":'Tooltip',
        "card":'List Item',
        "text_field ":'Input',
        "dropdown_menu":'Drawer',
        "chip ":'Input',
        "floating_action_button":'Icon',
        "menu":'Drawer',
        "data_table":'Data_Table',
        "grid_list":'Grid_list',
        "alert":'Modal',
        "text_area":'Text',
        "radio_button_unchecked":'Text Button',
        "radio_button_checked":'Text Button',
        "checkbox_unchecked":'Text Button',
        "slider":'Slider',
        "checkbox_checked":'Text Button',
        "switch_disabled":'Text Button',
        "switch_enabled":'Text Button',

    }

    # 读取JSON文件
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # json_data["categories"]=[
    #     {'id': 1, 'supercategory': 'none', 'name':'alert'},
    #     {'id': 2, 'supercategory': 'none', 'name': 'button'},
    #     {'id': 3, 'supercategory': 'none', 'name': 'card'},
    #     {'id': 4, 'supercategory': 'none', 'name': 'checkbox_checked'},
    #     {'id': 5, 'supercategory': 'none', 'name': 'checkbox_unchecked'},
    #     {'id': 6, 'supercategory': 'none', 'name': 'chip'},
    #     {'id': 7, 'supercategory': 'none', 'name': 'data_table'},
    #     {'id': 8, 'supercategory': 'none', 'name': 'dropdown_menu'},
    #     {'id': 9, 'supercategory': 'none', 'name': 'floating_action_button'},
    #     {'id': 10, 'supercategory': 'none', 'name': 'grid_list'},
    #     {'id': 11, 'supercategory': 'none', 'name': 'image'},
    #     {'id': 12, 'supercategory': 'none', 'name': 'label'},
    #     {'id': 13, 'supercategory': 'none', 'name': 'menu'},
    #     {'id': 14, 'supercategory': 'none', 'name': 'radio_button_checked'},
    #     {'id': 15, 'supercategory': 'none', 'name': 'radio_button_unchecked'},
    #     {'id': 16, 'supercategory': 'none', 'name': 'slider'},
    #     {'id': 17, 'supercategory': 'none', 'name': 'switch_disabled'},
    #     {'id': 18, 'supercategory': 'none', 'name': 'switch_enabled'},
    #     {'id': 19, 'supercategory': 'none', 'name': 'text_area'},
    #     {'id': 20, 'supercategory': 'none', 'name': 'text_field'},
    #     {'id': 21, 'supercategory': 'none', 'name': 'tooltip'}
    # ]
    for tmp in json_data["annotations"]:
        tmp["segmentation"]=rect_to_polygon(tmp["bbox"][0], tmp["bbox"][1],tmp["bbox"][2],tmp["bbox"][3])
        tmp["area"]=float(tmp["bbox"][2]*tmp["bbox"][3])
        tmp["iscrowd"]=0
    # 将修改后的JSON数据写回原始文件
    with open(out_file, 'w') as f:
        json.dump(json_data, f)











if __name__ == '__main__':
    json_path=r"F:\GradeThree\YuQianLab\GraduateDesign\mmdetection-main\data\Synz\annotations\val.json"
    out_file = r"F:\GradeThree\YuQianLab\GraduateDesign\mmdetection-main\data\Synz\annotations\val_.json"
    Synz2Coco(json_path, out_file)

    json_path = r"F:\GradeThree\YuQianLab\GraduateDesign\mmdetection-main\data\Synz\annotations\train.json"
    out_file = r"F:\GradeThree\YuQianLab\GraduateDesign\mmdetection-main\data\Synz\annotations\train_.json"
    Synz2Coco(json_path, out_file)

    json_path = r"F:\GradeThree\YuQianLab\GraduateDesign\mmdetection-main\data\Synz\annotations\test.json"
    out_file = r"F:\GradeThree\YuQianLab\GraduateDesign\mmdetection-main\data\Synz\annotations\test_.json"
    Synz2Coco(json_path, out_file)

    # Synz2Coco(csv_path)


    # # 读取Excel文件，指定表格名和列名
    # df = pd.read_csv(csv_path, header=None)
    #
    # # 计算词频并输出
    # word_counts = df[2].value_counts()
    # print(word_counts)