


from torch.utils.data import Dataset
import torch as t
import torchvision
import data_preprocess
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer
import clip
from tqdm.auto import tqdm
import argparse
import config
import os
import utils
import pandas as pd
from PIL import Image
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import mm, inch
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from contextlib import nullcontext



def create_pdf(input, output_path, images, additional_infos, grid_size=(2, 3)):
    c = canvas.Canvas(output_path, pagesize=letter)
    page_width, page_height = letter
    padding = 10  # Space around titles and images

    # Layout: 2 rows x 2 columns (for 4 grids total)
    grids_per_row = 2
    grids_per_col = 2
    grid_width = (page_width - (grids_per_row + 1) * padding) / grids_per_row
    grid_height = (page_height - (grids_per_col + 1) * padding) / grids_per_col

    # Image size is based on number of images in the grid
    img_width = (grid_width - (grid_size[0] + 1) * padding) / grid_size[0]
    img_height = (grid_height - (grid_size[1] + 1) * padding - 20) / grid_size[1]  # 20 for title

    def draw_grid(idx, title, image_list, grid_position):
        row = grid_position // grids_per_row
        col = grid_position % grids_per_row
        origin_x = padding + col * (grid_width + padding)
        origin_y = page_height - ((row + 1) * (grid_height + padding))

        # Draw the title
        c.setFont("Helvetica", 10)
        lines = title.split('\n')
        c.drawString(origin_x, origin_y + grid_height - 15, lines[0])
        c.drawString(origin_x, origin_y + grid_height - 30, lines[1])

        # Draw the images in the grid
        for j, img in enumerate(image_list):
            if j >= grid_size[0] * grid_size[1]:
                break
            r = j // grid_size[0]
            c_idx = j % grid_size[0]

            # Target box position
            box_x = origin_x + padding + c_idx * (img_width + padding)
            box_y = origin_y + grid_height - 30 - (r + 1) * (img_height + padding)

            try:
                image = ImageReader(img)
                orig_width, orig_height = img.size

                # Compute scale factor to fit in the box while preserving aspect ratio
                scale = min(img_width / orig_width, img_height / orig_height)
                draw_w = orig_width * scale
                draw_h = orig_height * scale

                # Center the image in its box
                offset_x = (img_width - draw_w) / 2
                offset_y = (img_height - draw_h) / 2

                c.drawImage(image, box_x + offset_x, box_y + offset_y, width=draw_w, height=draw_h)
            except Exception as e:
                print(f"Could not draw image {idx}: {e}")


    if isinstance(input, pd.DataFrame):
        for i, (_, row) in enumerate(input.iterrows()):
            if i >= 4:
                break
            idx = row['idx']
            concept_name = row['concept_name']
            sim_value = row['sim_value']
            draw_grid(idx, f"{concept_name} - sim value : {round(sim_value,4)} - index: {idx} -  log sparsity : {additional_infos[i]['log_sparsity']:.3f}\nvalues {[ round(elem, 3) for elem in additional_infos[i]['activations'].tolist() ]}", images[i], i)
    else:
        for i, idx in enumerate(input):
            if i >= 4:
                break
            draw_grid(idx, f"index: {idx} -  log sparsity : {additional_infos[i]['log_sparsity']:.3f} \nvalues {[ round(elem, 3) for elem in additional_infos[i]['activations'].tolist() ]}", images[i], i)

    c.save()

def main(args):
    sae_path = os.path.join(config.SAE_CKPTS,args.sae_path)

    (SaeArchitecture, _) = utils.get_sae_architecture(args.sae_type_name)
    device = args.device 

    sae = SaeArchitecture.from_pretrained(sae_path).to(device)
    activation_dim = sae.activation_dim
    dict_size = sae.dict_size
    expansion_factor = dict_size/activation_dim
    grid_size = (3,2)
    k = 100
    

    batch_size = args.batch_size
    model_name = args.model_name
    device=args.device
    #dataloader = utils.get_dataset(args.dataset, batch_size)
    dataloader, labels = utils.get_dataset(args.dataset, batch_size, labels=True)
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
    sae_name = args.sae_path.split('.pt')[0]
    (precomputed, path) = utils.features_already_precomputed(sae_name, args.dataset, args.split, args.layer)
    print(f"Setting up feature buffer...")
    feature_buffer = FeatureBuffer(
        activation_buffer=activation_buffer,
        sae=sae,
        out_batch_size=batch_size,
        device=device,
        load_features_path=os.path.join(config.FEATURES,args.load_features_path) if args.load_features_path else None,
        save_features_path=os.path.join(config.FEATURES,args.save_features_path) if args.save_features_path else None,
        return_raw_data=True,
    )

    total = len(feature_buffer)
    features_dataset = None

    for i, (imgs, features) in tqdm(enumerate(feature_buffer), total=total, desc="Computing SAE Features"):
        if i == 5: break
        if features_dataset is None:
            features_dataset = features.to(device)
        else:
            features_dataset = t.cat((features_dataset, features.to(device)), dim=0)

    topk_activating_imgs_values, topk_activating_imgs_idx = t.topk(features_dataset, k=k, dim=0, sorted=True)
    log_sparsity = t.log10(t.mean((features_dataset != 0).to(t.float32), dim=0) + 1e-10).cpu().numpy()
    variances = t.var(features_dataset, dim=0)
    # Identify dead features with zero variance
    dead_features_indices = t.where(variances == 0)[0]
    print(len(dead_features_indices))
    print(dead_features_indices)

    def get_topk_activating_imgs(feature_idx,k):
        return [dataloader[args.split].dataset[i] for i in topk_activating_imgs_idx[:,feature_idx][:k]]
    
    chunk_size = 4

    if args.assigned_concepts:
        concepts = pd.read_csv(os.path.join(config.CONCEPTS,args.assigned_concepts))
        concepts.sort_values(by=['sim_value'], ascending=False, inplace=True)
        for start in range(0, args.nb_of_print, chunk_size):
            end = start + chunk_size
            chunk = concepts[start:end]
            output_dir = os.path.join(config.RESULTS,args.sae_path.split('.')[0],args.dataset)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir,f"sim_value={max(chunk['sim_value']):.4}[{', '.join(chunk['concept_name'])}]__{args.dataset}.pdf")
            additional_infos = [{'activations' : topk_activating_imgs_values[:,idx][:grid_size[0]*grid_size[1]], 'log_sparsity' : log_sparsity[idx]} for idx in chunk['idx']]
            create_pdf(chunk, output_path, [get_topk_activating_imgs(idx, grid_size[0]*grid_size[1]) for idx in chunk['idx']], additional_infos,grid_size)
    elif args.indexes:
        indexes = [int(idx) for idx in args.indexes.split(",")]
        assert len(indexes)%4 == 0, "The number of indexes must be a multiple of 4"
        for start in range(0, int(len(indexes)/chunk_size), chunk_size):
            end = start + chunk_size
            chunk = indexes[start:end]
            output_dir = os.path.join(config.RESULTS,args.sae_path.split('.')[0],args.dataset)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{str(chunk)}_{args.split}.pdf")
            print(labels.dataset[topk_activating_imgs_idx[:,chunk[0]][:grid_size[0]*grid_size[1]].cpu()])
            print(labels.dataset[topk_activating_imgs_idx[:,chunk[1]][:grid_size[0]*grid_size[1]].cpu()])
            print(labels.dataset[topk_activating_imgs_idx[:,chunk[2]][:grid_size[0]*grid_size[1]].cpu()])
            print(labels.dataset[topk_activating_imgs_idx[:,chunk[3]][:grid_size[0]*grid_size[1]].cpu()])
            additional_infos = [{'activations' : topk_activating_imgs_values[:,idx][:grid_size[0]*grid_size[1]], 'log_sparsity' : log_sparsity[idx]} for idx in chunk]
            create_pdf(chunk, output_path, [get_topk_activating_imgs(idx, grid_size[0]*grid_size[1]) for idx in chunk], additional_infos ,grid_size=grid_size)
    else:
        for start in range(0, args.nb_of_print, chunk_size):
            end = start + chunk_size
            chunk = [i for i in range(start,end)]
            output_dir = os.path.join(config.RESULTS,args.sae_path.split('.')[0],args.dataset)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{str(chunk)}_{args.split}.pdf")
            additional_infos = [{'activations' : topk_activating_imgs_values[:,idx][:grid_size[0]*grid_size[1]], 'log_sparsity' : log_sparsity[idx]} for idx in chunk]
            create_pdf(chunk, output_path, [get_topk_activating_imgs(idx, grid_size[0]*grid_size[1]) for idx in chunk], additional_infos ,grid_size=grid_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that plots the top activating images corresponding to the SAE's features.")
    parser.add_argument('--sae_path', type=str, required=False, default="ae.pt", help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--assigned_concepts', type=str, required=False, help=f'[Optional] The path of the .csv file of the concept names/features association table starting from {config.CONCEPTS}')
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="val")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the vocab activations', default=1024)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--nb_of_print', type=int, required=False, help=f'The number of concepts printed.', default=50)
    parser.add_argument('--indexes', type=str, required=False, help=f"[Optional] The indexes of the features you want to plot. If not provided, the top k features will be plotted. Must be in the format : '152,243,564,... The number of indexes must be a multiple of 4.", default=None)
    parser.add_argument('--load_features_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the sae's features starting from {config.FEATURES}.", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features starting from {config.FEATURES}.", default=None)
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)


    args = parser.parse_args()
    main(args)




