import pickle

def merge_pkl_files(train_file, val_file, output_file):
    # 读取 train pkl
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    
    # 读取 val pkl
    with open(val_file, 'rb') as f:
        val_data = pickle.load(f)

    trainval_data = {'metainfo': train_data['metainfo'],'data_list': train_data['data_list'] + val_data['data_list']}

    # 将合并后的数据保存到新的文件
    with open(output_file, 'wb') as f:
        pickle.dump(trainval_data, f)

if __name__ == "__main__":
    data_root = '/home/hisham/zongyan/datasets/oneformer3d_scannet/'
    train_file = data_root+'scannet_oneformer3d_infos_train.pkl'
    val_file = data_root+'scannet_oneformer3d_infos_val.pkl'
    output_file = data_root+'scannet_oneformer3d_infos_trainval.pkl'

    merge_pkl_files(train_file, val_file, output_file)
    print(f"合并完成，输出文件：{output_file}")
