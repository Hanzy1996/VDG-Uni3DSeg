from load_llama import *
import torchmetrics
from tqdm import tqdm


# seed_everything(hparams['seed'])

# bs = hparams['batch_size']
# dataloader = DataLoader(dataset, bs, shuffle=True, num_workers=16, pin_memory=True)

print("Loading model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
# load model
model, preprocess = clip.load('ViT-L/14@336px', device=device, jit=False)
model.eval()
model.requires_grad_(False)

# hparams['dataset'] = 'scannet'

print("Encoding descriptions...")
description_encodings = compute_description_encodings(model)
# import pdb; pdb.set_trace()
# save the encodings from orderedic to dictionary
description_encodings = dict(description_encodings)

description_encode = [v.cpu() for k, v in description_encodings.items()]
description_encode = torch.cat(description_encode, dim=0).numpy()

np.save("./clip_embedding/"+ hparams['dataset'] + '_' + hparams['gpt_type']+"_description.npy", description_encode)
print('description_encode:', description_encode.shape)

# label_encodings = compute_label_encodings(model)
# # save the encoding from tensor to numpy array as file
# np.save("./clip_embedding/"+ hparams['dataset'] + '_' + hparams['gpt_type']+"_class.npy", label_encodings.cpu().numpy())
# print('label_encodings:', label_encodings.shape)

print("Encoding from datasets: "+hparams['descriptor_fname']+ " descriptions done!")

