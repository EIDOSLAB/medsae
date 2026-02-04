


from torch.utils.data import Dataset
import torch as t
import torchvision
import data_preprocess
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer
import clip
from tqdm.auto import tqdm
import torch.nn.functional as F
from collections import defaultdict
import numpy as np
import argparse
import config
import os
from scipy.optimize import nnls
import utils
import pandas as pd
import json
from PIL import Image

def create_composite_image_superpose(img_a, img_b, canvas_size=(224, 224)):
    # Resize both images to canvas size
    img_a_resized = img_a.resize(canvas_size).convert("RGBA")
    img_b_resized = img_b.resize(canvas_size).convert("RGBA")

    # Set alpha to 128 (50% opacity) for both
    img_a_resized.putalpha(128)
    img_b_resized.putalpha(128)

    # Composite the two images by alpha blending
    composite = Image.alpha_composite(img_a_resized, img_b_resized)

    # Convert back to RGB if needed
    return composite.convert("RGB")

def create_composite_image(img_a, img_b, canvas_size=(224,224), quadrant_size=(112,112)):
    canvas = Image.new('RGB', canvas_size, color=(0,0,0))
    img_a_resized = img_a.resize(quadrant_size)
    img_b_resized = img_b.resize(quadrant_size)
    canvas.paste(img_a_resized, (0,0))
    canvas.paste(img_b_resized, (canvas_size[0]-quadrant_size[0], canvas_size[1]-quadrant_size[1]))
    return canvas

def generate_pairwise_composites(image_list):
    pairs = []
    composites = []
    for i in range(len(image_list)):
        for j in range(i+1, len(image_list)):
            composite_img = create_composite_image(image_list[i], image_list[j], canvas_size=(256,256), quadrant_size=(128,128))
            pairs.append((i,j))
            composites.append(composite_img)
    return pairs, composites


def main(args):
    device = args.device 

    batch_size = args.batch_size
    model_name = args.model_name
    composite_batch_size = 100
    device=args.device
    #dataloader = utils.get_dataset(args.dataset, batch_size)
    dataloader, labels = utils.get_dataset(args.dataset, batch_size, labels=True, shuffled_version=True)
    labels = labels[args.split]
    #Model import
    print(f"Importing model {model_name}...")
    model, processor = utils.get_model(model_name)
    #Buffer set-up
    print(f"Setting up activation buffer...")
    (precomputed, path) = utils.activations_already_precomputed(model_name, args.dataset, args.split, args.layer)

    activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader[args.split],
        modality='img',
        model=model,
        layer=None if not(args.layer) else utils.rgetattr(model.vision_model, args.layer),
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        return_raw_data=True,
        load_activations_path=args.load_activations_path if args.load_activations_path else (path if precomputed else None),
        save_activations_path=args.save_activations_path if args.save_activations_path else (path if not(precomputed) else None),
    )

    total = len(activation_buffer)
    all_individual_embeddings = []
    all_imgs = []
    for i, (imgs, embeddings_individual) in tqdm(enumerate(activation_buffer), total=total, desc="Collecting embeddings"):
        if i == 10: break
        all_individual_embeddings.append(embeddings_individual)
        all_imgs += imgs
    all_individual_embeddings = t.cat(all_individual_embeddings, dim=0)  # [total_individual, embed_dim]

    del activation_buffer

    composites = []
    individual_embeddings_a = []
    individual_embeddings_b = []

    num_pairs = 100000

    # Generate composites
    print(f"Generating composite {num_pairs} images...")
    for i in range(num_pairs):
        sampled_indices = np.random.choice(len(all_imgs), size=2, replace=False)
        img_a = all_imgs[sampled_indices[0]]
        img_b = all_imgs[sampled_indices[1]]
        individual_embeddings_a.append(all_individual_embeddings[sampled_indices[0]])
        individual_embeddings_b.append(all_individual_embeddings[sampled_indices[1]])
        composite_img = create_composite_image(img_a, img_b, canvas_size=(256,256), quadrant_size=(128,128))
        if i ==0:
            composite_img.save("test.jpg")
            img_a.save("testa.jpg")
            img_b.save("testb.jpg")
        composites.append(composite_img)

    individual_embeddings_a = t.stack(individual_embeddings_a, dim=0)
    individual_embeddings_b = t.stack(individual_embeddings_b, dim=0)
    

    print("Computing embeddings of composite images in batches...")
    embeddings_composite = []
    for i in tqdm(range(0, len(composites), composite_batch_size)):
        batch = composites[i:i + composite_batch_size]
        if model_name == "MedCLIP-RN50":
            inputs = processor(images=batch, return_tensors="pt", padding=True).to(device)
            embs = model.encode_image(inputs['pixel_values']).detach().cpu()
        elif model_name == "CLIP-RN50":
            inputs = [processor(x) for x in batch]
            inputs = t.tensor(np.stack(inputs)).to(device)
            embs = model.encode_image(inputs).detach().cpu()
        embeddings_composite.append(embs)

    embeddings_composite = t.cat(embeddings_composite, dim=0)
    
    individual_embeddings_mean = t.cat((individual_embeddings_a,individual_embeddings_b), dim=0).mean(dim=0, keepdim=True)  # [1, embed_dim]
    #individual_embeddings_mean = all_individual_embeddings.mean(dim=0, keepdim=True)  # [1, embed_dim]
    embeddings_composite_mean = embeddings_composite.mean(dim=0, keepdim=True)   # [1, embed_dim]

    # all_individual_normed = (all_individual_embeddings - ind_mean) / ind_std
    # all_composite_normed = (all_composite_embeddings - comp_mean) / comp_std

    # embeddings : [n_sample, embed_dim]

    # embeddings_composite = (embeddings_composite - embeddings_composite_mean)
    # individual_embeddings_a = (individual_embeddings_a - individual_embeddings_mean) 
    # individual_embeddings_b = (individual_embeddings_b - individual_embeddings_mean) 

    # embeddings_composite = (embeddings_composite - embeddings_composite_mean) / embeddings_composite.norm(dim=1, keepdim=True)  # Normalize
    # individual_embeddings_a = (individual_embeddings_a - individual_embeddings_mean)  / individual_embeddings_a.norm(dim=1, keepdim=True)  # Normalize
    # individual_embeddings_b = (individual_embeddings_b - individual_embeddings_mean)  / individual_embeddings_b.norm(dim=1, keepdim=True)  # Normalize

    
    weights_a = []
    weights_b = []
    cosine_sims = []
    num_pairs = len(embeddings_composite)
    embed_dim = embeddings_composite.shape[1]

    print("Computing batch matrices...")
    # Prepare batch matrices
    Z = t.zeros((num_pairs, embed_dim, 2))
    z_ab = t.zeros((num_pairs, embed_dim, 1))

    for idx in range(num_pairs):

        Z[idx, :, 0] = individual_embeddings_a[idx]
        Z[idx, :, 1] = individual_embeddings_b[idx]
        z_ab[idx, :, 0] = embeddings_composite[idx]
    

    # # Batch least squares: solve Z @ w = z_ab for w
    # solution = t.linalg.lstsq(Z, z_ab)
    # w = solution.solution  # shape [num_pairs, 2, 1]
    # w = w.squeeze(-1)      # shape [num_pairs, 2]

    # w_a = w[:, 0]  # [num_pairs]
    # w_b = w[:, 1]

    # # Predicted embeddings: batch multiply
    # z_ab_pred = w_a.unsqueeze(1) * Z[:, :, 0] + w_b.unsqueeze(1) * Z[:, :, 1]  # [num_pairs, embed_dim]

    # print("Computing cosine similarities...")

    # # Compute cosine similarities in batch
    # cos_sims = F.cosine_similarity(z_ab_pred, z_ab.squeeze(-1), dim=1)  # [num_pairs]

    # weights_a.extend(w_a.cpu().tolist())
    # weights_b.extend(w_b.cpu().tolist())
    # cosine_sims.extend(cos_sims.cpu().tolist())

    for idx in range(num_pairs):
        Z_i = Z[idx].cpu().numpy()            # shape: [embed_dim, 2]
        z_i = z_ab[idx].squeeze(-1).cpu().numpy()  # shape: [embed_dim]

        w_i, _ = nnls(Z_i, z_i)               # solve non-negative least squares
        w_i = t.tensor(w_i, device=Z.device, dtype=Z.dtype)  # back to tensor

        z_pred_i = w_i[0] * Z[idx, :, 0] + w_i[1] * Z[idx, :, 1]
        cos_sim_i = F.cosine_similarity(z_pred_i.unsqueeze(0), z_ab[idx].squeeze(-1).unsqueeze(0), dim=1).item()


        weights_a.append(w_i[0].item())
        weights_b.append(w_i[1].item())
        cosine_sims.append(cos_sim_i)
    
    #save results to a CSV file
    results = {
        'weights_a': weights_a,
        'weights_b': weights_b,
        'cosine_sims': cosine_sims
    }
    results_df = pd.DataFrame(results)
    results_path = os.path.join(config.RESULTS, f"{model_name}_{args.dataset}_{args.split}_linearity_sanity_check.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    # Aggregate results for reporting
    def mean_std(arr):
        arr_np = np.array(arr)
        return np.mean(arr_np), np.std(arr_np)

    mean_wa, std_wa = mean_std(weights_a)
    mean_wb, std_wb = mean_std(weights_b)
    mean_cos, std_cos = mean_std(cosine_sims)

    

    print(f"CheXpert: w_a = {mean_wa:.4f} ± {std_wa:.4f}, w_b = {mean_wb:.4f} ± {std_wb:.4f}, cosine = {mean_cos:.4f} ± {std_cos:.4f}")
            

            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that computes the linearity sanity check of a model.")
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="val")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the images-report pairs', default=512)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)


    args = parser.parse_args()
    main(args)




