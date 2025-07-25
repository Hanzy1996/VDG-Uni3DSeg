import torch
import clip
from PIL import Image
import os
from tqdm import tqdm
import numpy as np
import json
# 加载 CLIP 模型和预处理器
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-L/14@336px', device=device)


scannet200_classes = ['wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office_chair', 'bed', 'pillow', 'sink', 'picture_frame', 'window', 'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair', 'coffee_table', 'box', 'refrigerator', 'lamp', 'kitchen_cabinet', 'towel', 'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion', 'plant', 'ceiling', 'bathtub', 'end_table', 'dining_table', 'keyboard', 'bag', 'backpack', 'toilet_paper', 'printer', 'tv_stand', 'whiteboard', 'blanket', 'shower_curtain', 'trash_can', 'closet', 'stairs', 'microwave', 'stove', 'shoe', 'computer_tower', 'bottle', 'bin', 'ottoman', 'bench', 'board', 'washing_machine', 'mirror', 'copier', 'basket', 'sofa_chair', 'file_cabinet', 'fan', 'laptop', 'shower', 'paper', 'person', 'paper_towel_dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard', 'piano', 'suitcase', 'rail', 'radiator', 'recycling_bin', 'container', 'wardrobe', 'soap_dispenser', 'telephone', 'bucket', 'clock', 'stand', 'light', 'laundry_basket', 'pipe', 'clothes_dryer', 'guitar', 'toilet_paper_holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder', 'bathroom_stall', 'shower_wall', 'cup', 'jacket', 'storage_bin', 'coffee_maker', 'dishwasher', 'paper_towel_roll', 'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin_board', 'ironing_board', 'fireplace', 'soap_dish', 'kitchen_counter', 'doorframe', 'toilet_paper_dispenser', 'mini_fridge', 'fire_extinguisher', 'ball', 'hat', 'shower_curtain_rod', 'water_cooler', 'paper_cutter', 'tray', 'shower_door', 'pillar', 'ledge', 'toaster_oven', 'mouse', 'toilet_seat_cover_dispenser', 'furniture', 'cart', 'storage_container', 'scale', 'tissue_box', 'light_switch', 'crate', 'power_outlet', 'decoration', 'sign', 'projector', 'closet_door', 'vacuum_cleaner', 'candle', 'plunger', 'stuffed_animal', 'headphones', 'dish_rack', 'broom', 'guitar_case', 'range_hood', 'dustpan', 'hair_dryer', 'water_bottle', 'handicap_bar', 'purse', 'vent', 'shower_floor', 'water_pitcher', 'mailbox', 'bowl', 'paper_bag', 'alarm_clock', 'music_stand', 'projector_screen', 'divider', 'laundry_detergent', 'bathroom_counter', 'object', 'bathroom_vanity', 'closet_wall', 'laundry_hamper', 'bathroom_stall_door', 'ceiling_light', 'trash_bin', 'dumbbell', 'stair_rail', 'tube', 'bathroom_cabinet', 'cd_case', 'closet_rod', 'coffee_kettle', 'structure', 'shower_head', 'keyboard_piano', 'case_of_water_bottles', 'coat_rack', 'storage_organizer', 'folded_chair', 'fire_alarm', 'power_strip', 'calendar', 'poster', 'potted_plant', 'luggage', 'mattress']

s3dis_classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

scannet_classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture_frame', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture']





def extract_clip_features(dataset, class_names, topk=5):
    image_features = []
    image_files = {}
    not_enough_class = []

    # 遍历文件夹中的所有图片
    for i, i_class in tqdm(enumerate(class_names)):
        i_class_features = []
        tokens = clip.tokenize(i_class.replace('_', ', ')).to(device)
        i_text_feat = (model.encode_text(tokens)) 
        i_class_img_dir = os.path.join('google-images-download/images', dataset, i_class)
        i_class_files = os.listdir(i_class_img_dir)
        if len(i_class_files) < 10:
            print(f"Skipping class {i_class} because it has less than 10 images")
            not_enough_class
            continue
        for img_name in i_class_files:
            img_path = os.path.join(i_class_img_dir, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(device)
            except:
                print(f"Error loading image: {img_path}")
                continue
            # 使用 CLIP 提取图片特征
            with torch.no_grad():
                feature = model.encode_image(image_input)
            # 将特征保存并归一化
            # feature /= feature.norm(dim=-1, keepdim=True)
            i_class_features.append(feature)
        
        i_class_features = torch.cat(i_class_features, dim=0)
        i_class_features
        i_class_sim = (i_class_features@(i_text_feat.T)).squeeze()

        _, sorted_indices = torch.sort(i_class_sim, descending=True)
        image_features.append(i_class_features[sorted_indices[:topk]].cpu().numpy())
        if i_class == 'picture_frame':
            i_class = 'picture'
        image_files[i_class]= [i_class_files[i] for i in sorted_indices[:topk]]
    
    print(f"Classes with less than 10 images: {not_enough_class}")

    # add for the unlabelled class
    image_features.append(np.random.randn(topk, 768))

    image_features = np.concatenate(image_features, axis=0)

    print(f"The size of the image features: {image_features.shape}")

    assert image_features.shape[0] == (len(class_names)+1) * topk

    filename = f'image_clip_feat/{dataset}_image_clip_top{topk}.json'
    file = {'features': image_features.tolist(), 'files': image_files}
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(file, fp)
    print(f"Image features saved to {filename}")
    return 


extract_clip_features('s3dis', s3dis_classes, topk=5)
extract_clip_features('scannet', scannet_classes, topk=5)
extract_clip_features('scannet200', scannet200_classes, topk=5)
