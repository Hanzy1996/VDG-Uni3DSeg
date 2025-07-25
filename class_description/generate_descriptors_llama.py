import os
import json
from tqdm import tqdm  
import time
import itertools


scannet200_classes = [
    'wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet', 'shelf',
    'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
    'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair',
    'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel',
    'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
    'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard',
    'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard',
    'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave',
    'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench',
    'board', 'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
    'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
    'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard',
    'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
    'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand',
    'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar',
    'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
    'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin',
    'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
    'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board',
    'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
    'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball',
    'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray',
    'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
    'toilet seat cover dispenser', 'furniture', 'cart', 'storage container',
    'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
    'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner',
    'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
    'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer',
    'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor',
    'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
    'music stand', 'projector screen', 'divider', 'laundry detergent',
    'bathroom counter', 'object', 'bathroom vanity', 'closet wall',
    'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin',
    'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case',
    'closet rod', 'coffee kettle', 'structure', 'shower head',
    'keyboard piano', 'case of water bottles', 'coat rack',
    'storage organizer', 'folded chair', 'fire alarm', 'power strip',
    'calendar', 'poster', 'potted plant', 'luggage', 'mattress', 'unlabeled'
]

s3dis_classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter', 'unlabeled']

scannet_classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub', 'otherfurniture', 'unlabeled']

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

from typing import List, Optional
from llama import Dialog, Llama


def llama_generation(given_class, generator, temperature = 0.6, top_p = 0.9,max_gen_len = None):
    """
    Examples to run with the models finetuned for chat. Prompts correspond of chat
    turns between the user and assistant with the final one always being the user.

    An optional system prompt at the beginning to control how the model should respond
    is also supported.

    The context window of llama3 models is 8192 tokens, so `max_seq_len` needs to be <= 8192.

    `max_gen_len` is optional because finetuned models are able to stop generations naturally.
    """

    dialogs: List[Dialog] = [
        [{"role": "user", "content": "In a 3D point cloud of an indoor scene, provide concise visual descriptions that a 3D model should focus on to identify and differentiate the input object. The descriptions should contain 10 types, including Type, Color, Shape, Texture, Size, Distinctive Features, Material, Position/Spatial, Lighting/Reflectivity and Associated Objects. Please make each description precise and strictly follow the output JSON format as guided. Input: television"},
         {"role": "assistant", "content": 
         """Output: {"name": "television", "descriptions": [
    "Type: electronic device", 
    "Color: Typically black or grey.",
    "Shape: Large, rectangular screen. Often flat and thin.",
    "Texture: Smooth glass screen with matte or slightly textured plastic frame.",
    "Size: Large, typically 32-85 inches diagonal.",
    "Distinctive Features: Input ports for connecting to other devices.",
    "Material: Plastic frame, glass, or LED/LCD screen.",
    "Position and Orientation: Freestanding or wall-mounted, typically horizontal and at eye level.",
    "Lighting and Reflectivity: Reflective screen surface, may catch ambient light or create glare.",
    "Associated Objects: Remote control, sound system, external devices."
  ]}"""},
        {"role": "user", "content": f"Input: {given_class}"},
        ],]

    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,)
    try:
        result = json.loads(results[0]['generation']['content'][8:])
    except:
        if results[0]['generation']['content'][8:][-2:] == '"]':
            result = json.loads(results[0]['generation']['content'][8:]+'}')
        else:
            import pdb; pdb.set_trace()
    print(result)

    if len(result['descriptions'])<10:
        print(f"Warning: {given_class} has less than 10 descriptions for at least one level")
        import pdb; pdb.set_trace()

    return {'descriptions': result['descriptions']}


def obtain_descriptors_from_llama(filename, class_list, 
                                  ckpt_dir='./llama_model/Llama3.1-8B-Instruct', 
                                  tokenizer_path='./llama_model/Llama3.1-8B-Instruct/tokenizer.model', 
                                  temperature: float = 0.6,
                                  top_p: float = 0.9,
                                  max_seq_len: int = 4096,
                                  max_batch_size: int = 4,
                                  max_gen_len: Optional[int] = None):
    descriptors = {}
    
    descriptors_list= []
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    for i_class in tqdm(class_list):
        i_des = llama_generation(i_class, generator, temperature, top_p, max_gen_len)

        descriptors_list.append(i_des)
    # response_texts = [r['message'].content for resp in responses for r in resp['choices']]
    # descriptors_list = [stringtolist(response_text) for response_text in response_texts]

    descriptors = {cat: descr for cat, descr in zip(class_list, descriptors_list)}

    # save descriptors to json file
    if not filename.endswith('.json'):
        filename += '.json'
    with open(filename, 'w') as fp:
        json.dump(descriptors, fp)

if __name__ == "__main__":

    obtain_descriptors_from_llama('./descriptors/scannet200_llama_v3', scannet200_classes)
    print('scannet200_llama done!')


# obtain_descriptors_and_save_gpt4('scannet_gpt4', scannet_classes)
# print('scannet_gpt4 done!')



