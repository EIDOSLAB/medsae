


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
import utils
import pandas as pd
import json
from PIL import Image

def create_composite_image(img_a, img_b, canvas_size=(448,448), quadrant_size=(224,224)):
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
            composite_img = create_composite_image(image_list[i], image_list[j])
            pairs.append((i,j))
            composites.append(composite_img)
    return pairs, composites


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
    weights_a = []
    weights_b = []
    cosine_sims = []


    for i, (imgs, embeddings_individual) in tqdm(enumerate(activation_buffer), total=total, desc="Computing model embeddings"):
        pairs, composite_images = generate_pairwise_composites(imgs)
        inputs_composites = processor(images=composite_images, return_tensors="pt", padding=True, truncation=True).to(device)
        embeddings_composite  = model.encode_image(pixel_values=inputs_composites['pixel_values'])
        embeddings_individual = embeddings_individual.to(device)

        num_pairs = len(pairs)
        embed_dim = embeddings_individual.shape[1]

        # Prepare batch matrices for lstsq
        # For each pair, stack z_a and z_b along last dim
        Z = t.zeros((num_pairs, embed_dim, 2), device=device)
        z_ab = t.zeros((num_pairs, embed_dim, 1), device=device)

        for idx, (idx_a, idx_b) in enumerate(pairs):
            Z[idx, :, 0] = embeddings_individual[idx_a]
            Z[idx, :, 1] = embeddings_individual[idx_b]
            z_ab[idx, :, 0] = embeddings_composite[idx]

        # Batch least squares: solve Z @ w = z_ab for w
        solution = t.linalg.lstsq(Z, z_ab)
        w = solution.solution  # shape [num_pairs, 2, 1]
        w = w.squeeze(-1)      # shape [num_pairs, 2]

        w_a = w[:, 0]  # [num_pairs]
        w_b = w[:, 1]

        # Predicted embeddings: batch multiply
        z_ab_pred = w_a.unsqueeze(1) * Z[:, :, 0] + w_b.unsqueeze(1) * Z[:, :, 1]  # [num_pairs, embed_dim]

        # Compute cosine similarities in batch
        cos_sims = F.cosine_similarity(z_ab_pred, z_ab.squeeze(-1), dim=1)  # [num_pairs]

        # Append results to global lists (convert to CPU numpy if needed)
        weights_a.extend(w_a.cpu().tolist())
        weights_b.extend(w_b.cpu().tolist())
        cosine_sims.extend(cos_sims.cpu().tolist())
    
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
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the images-report pairs', default=20)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)


    args = parser.parse_args()
    main(args)




