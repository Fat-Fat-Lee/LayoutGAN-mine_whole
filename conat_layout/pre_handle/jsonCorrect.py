import json
import os
import pandas as pd
import shutil
# 获取json里面数据
def get_json_data(file,ricoLabel):
    with open(file, 'rb') as f:  # 使用只读模型，并定义名称为f
        params = json.load(f)  # 加载json文件中的内容给params
        params["ricoLabel"] = ricoLabel # imp字段对应的deeplink的值修改为end
    return params  # 返回修改后的内容


# 写入json文件# 使用写模式，名称定义为r
def write_json_data(file,params):
    with open(file, 'w') as r:
        # 将params写入名称为r的文件中
        json.dump(params, r)

#复制文件到指定路径
def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)  # 创建路径
        shutil.copy(srcfile,os.path.join(dstpath,fname))          # 复制文件
        #print ("copy %s -> %s"%(srcfile, dstpath + fname))


if __name__ == '__main__':
    df = pd.read_csv('./pre_handle/ricoLabel.csv')  # 读入
    dir_path=r'/data/tmp011/projects/data/dataset/ricoTest/raw/semantic_annotations'
    #dst_dir=r'F:\GradeThree\YuQianLab\GraduateDesign\const_layout-master\data\dataset\rico_correct\raw\semantic_annotations'

    screenId_list=df['screenId'].to_list()

    res = os.listdir(dir_path)
    for tmp in res:
        if tmp.endswith(".json"):
            p=dir_path+"/"+tmp
            #print(tmp)
            the_revised_dict = get_json_data(p,"other")

            write_json_data(p, the_revised_dict)
        #write_json_data(tmp, "other")
    for tmpi in range(0,len(screenId_list)):
        tmpId=str(screenId_list[tmpi])+".json"
        tmpPic=str(screenId_list[tmpi])+".png"

        file=os.path.join(dir_path,tmpId)
        pic_file=os.path.join(dir_path,tmpPic)
        #修改添加json字段并写入
        the_revised_dict = get_json_data(file,df.iloc[tmpi,1])
        write_json_data(file,the_revised_dict)

        # #复制图片和json到指定文件夹
        # mycopyfile(file,dst_dir)
        # mycopyfile(pic_file, dst_dir)
