


from torch.utils.data import Dataset
import torch as t
import torchvision
import data_preprocess
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer
import clip
from tqdm.auto import tqdm
import torch.nn.functional as F
from collections import defaultdict
from scipy.optimize import nnls

import numpy as np
import argparse
import config
import os
import utils
import pandas as pd
import json
from PIL import Image
from itertools import combinations

def create_composite_4_images(img_a, img_b, img_c, img_d, canvas_size=(448,448), quadrant_size=(224,224)):
    canvas = Image.new('RGB', canvas_size, color=(0,0,0))
    canvas.paste(img_a.resize(quadrant_size), (0, 0))  # Top-left
    canvas.paste(img_b.resize(quadrant_size), (quadrant_size[0], 0))  # Top-right
    canvas.paste(img_c.resize(quadrant_size), (0, quadrant_size[1]))  # Bottom-left
    canvas.paste(img_d.resize(quadrant_size), (quadrant_size[0], quadrant_size[1]))  # Bottom-right
    return canvas

def generate_quadrant_composites(image_list):
    quadruples = list(combinations(range(len(image_list)), 4))
    composites = []
    for (i, j, k, l) in quadruples:
        composite_img = create_composite_4_images(image_list[i], image_list[j], image_list[k], image_list[l])
        composites.append(composite_img)
    return quadruples, composites


def main(args):
    device = args.device 

    batch_size = args.batch_size
    model_name = args.model_name
    device=args.device
    #dataloader = utils.get_dataset(args.dataset, batch_size)
    dataloader, labels = utils.get_dataset(args.dataset, batch_size, labels=True, shuffled_version=False)
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
    all_composite_embeddings = []
    all_quadruples = []
    all_img_counts = []

    for i, (imgs, embeddings_individual) in tqdm(enumerate(activation_buffer), total=total, desc="Collecting embeddings"):
        if len(imgs) < 4:
            print(f"Skipping batch {i} with {len(imgs)} images (less than 4 required for composites).")
            continue
        quadruples, composite_images = generate_quadrant_composites(imgs)
        inputs_composites = processor(images=composite_images, return_tensors="pt", padding=True, truncation=True).to(device)
        embeddings_composite  = model.encode_image(pixel_values=inputs_composites['pixel_values']).detach().cpu()
        embeddings_individual = embeddings_individual.cpu()

        all_individual_embeddings.append(embeddings_individual)
        all_composite_embeddings.append(embeddings_composite)
        all_quadruples.append(quadruples)
        all_img_counts.append(len(embeddings_individual))

    # Concatenate embeddings
    all_individual_embeddings = t.cat(all_individual_embeddings, dim=0)
    all_composite_embeddings = t.cat(all_composite_embeddings, dim=0)

    ind_mean = all_individual_embeddings.mean(dim=0, keepdim=True)
    ind_std = all_individual_embeddings.std(dim=0, keepdim=True)
    comp_mean = all_composite_embeddings.mean(dim=0, keepdim=True)
    comp_std = all_composite_embeddings.std(dim=0, keepdim=True)

    # all_individual_normed = (all_individual_embeddings - ind_mean) / ind_std
    # all_composite_normed = (all_composite_embeddings - comp_mean) / comp_std
    all_individual_normed = all_individual_embeddings
    all_composite_normed = all_composite_embeddings

    weights = []
    cosine_sims = []

    offset = 0
    composite_offset = 0
    for i, quadruples in tqdm(enumerate(all_quadruples)):
        num_quads = len(quadruples)
        embed_dim = all_individual_normed.shape[1]

        Z = t.zeros((num_quads, embed_dim, 4))
        z_ab = t.zeros((num_quads, embed_dim, 1))

        for idx, (idx_a, idx_b, idx_c, idx_d) in enumerate(quadruples):
            global_idx_a = offset + idx_a
            global_idx_b = offset + idx_b
            global_idx_c = offset + idx_c
            global_idx_d = offset + idx_d
            global_comp_idx = composite_offset + idx

            Z[idx, :, 0] = all_individual_normed[global_idx_a]
            Z[idx, :, 1] = all_individual_normed[global_idx_b]
            Z[idx, :, 2] = all_individual_normed[global_idx_c]
            Z[idx, :, 3] = all_individual_normed[global_idx_d]

            z_ab[idx, :, 0] = all_composite_normed[global_comp_idx]

        # Solve NNLS for each quadruple individually
        for idx in range(num_quads):
            Z_i = Z[idx].cpu().numpy()           # shape: [embed_dim, 4]
            z_i = z_ab[idx].squeeze(-1).cpu().numpy()  # shape: [embed_dim]

            w_i, _ = nnls(Z_i, z_i)              # solve non-negative least squares
            w_i = t.tensor(w_i, device=Z.device, dtype=Z.dtype)  # convert back to tensor

            # Predicted embedding from weighted sum
            z_pred_i = (w_i[0] * Z[idx, :, 0] +
                        w_i[1] * Z[idx, :, 1] +
                        w_i[2] * Z[idx, :, 2] +
                        w_i[3] * Z[idx, :, 3])

            # Cosine similarity between predicted and actual composite embedding
            cos_sim_i = F.cosine_similarity(z_pred_i.unsqueeze(0), z_ab[idx].squeeze(-1).unsqueeze(0), dim=1).item()

            weights.append(w_i.cpu().tolist())
            cosine_sims.append(cos_sim_i)

        offset += all_img_counts[i]
        composite_offset += num_quads
    weights_np = np.array(weights)  # shape: [num_samples, 4]

    results = {
        'weight_a': weights_np[:, 0],
        'weight_b': weights_np[:, 1],
        'weight_c': weights_np[:, 2],
        'weight_d': weights_np[:, 3],
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

    mean_wa, std_wa = mean_std(results['weight_a'])
    mean_wb, std_wb = mean_std(results['weight_b'])
    mean_wc, std_wc = mean_std(results['weight_c'])
    mean_wd, std_wd = mean_std(results['weight_d'])
    mean_cos, std_cos = mean_std(results['cosine_sims'])

    print(f"CheXpert: "
        f"w_a = {mean_wa:.4f} ± {std_wa:.4f}, "
        f"w_b = {mean_wb:.4f} ± {std_wb:.4f}, "
        f"w_c = {mean_wc:.4f} ± {std_wc:.4f}, "
        f"w_d = {mean_wd:.4f} ± {std_wd:.4f}, "
        f"cosine = {mean_cos:.4f} ± {std_cos:.4f}")

            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that computes the linearity sanity check of a model.")
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="val")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the images-report pairs', default=4)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)


    args = parser.parse_args()
    main(args)




