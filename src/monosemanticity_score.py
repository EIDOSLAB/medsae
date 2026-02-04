import torch
import os.path
import argparse
import os
import config
import utils
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch.nn.functional as F
from dictionary_learning.buffer import CLIPActivationBuffer, FeatureBuffer

def get_args_parser():
    parser = argparse.ArgumentParser("Measure monosemanticity via weighted pairwise cosine similarity", add_help=False)
    parser.add_argument('--sae_path', type=str, required=False, default=None, help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the vocab activations', default=512)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features. If not provided it will automatically save the sae's features to {config.FEATURES}.", default=None)
    parser.add_argument('--load_features_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the sae's features. If not provided it will automatically check if features can be loaded from {config.FEATURES}.", default=None)
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="val")   
    parser.add_argument('--layer', type=str, required=False, help=f'[Optional] The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)

    return parser

def main(args):
    model_name = args.model_name
    batch_size = args.batch_size
    device=args.device

    dataset_name = args.dataset
    dataloader = utils.get_dataset(args.dataset, batch_size)[args.split]

    print(f"Importing model {model_name}...")
    model, processor = utils.get_model(model_name)

    print(f"Setting up activation buffer...")
    (precomputed, path) = utils.activations_already_precomputed(model_name, dataset_name, args.split)

    

    activation_buffer = CLIPActivationBuffer.get(
        model_name,
        dataloader=dataloader,
        modality='img',
        model=model,
        layer=None if not(args.layer) else utils.rgetattr(model.vision_model, args.layer),
        processor=processor,
        out_batch_size= batch_size,
        device=device,
        precompute=False,
        load_activations_path=args.load_activations_path if args.load_activations_path else (path if precomputed else None),
        save_activations_path=args.save_activations_path if args.save_activations_path else (path if not(precomputed) else None),
    )

    if args.sae_path is not None:
        sae_path = os.path.join(config.SAE_CKPTS,args.sae_path)
        sae_name = args.sae_path.split('.')[0]

        (SaeArchitecture, _) = utils.get_sae_architecture(args.sae_type_name)

        sae = SaeArchitecture.from_pretrained(sae_path).to(device)
        activation_dim = sae.activation_dim
        dict_size = sae.dict_size
        expansion_factor = dict_size/activation_dim

        print(f"Setting up feature buffer...")

        (precomputed, path) = utils.features_already_precomputed(sae_name, dataset_name, args.split, args.layer)
        feature_buffer = FeatureBuffer(
            activation_buffer=activation_buffer,
            sae=sae,
            out_batch_size=batch_size,
            device=device,
            load_features_path=args.load_features_path if args.load_features_path else (path if precomputed else None),
            save_features_path=args.save_features_path if args.save_features_path else (path if not(precomputed) else None),
        )


        for i, _ in tqdm(enumerate(feature_buffer), total=len(feature_buffer), desc=("Computing SAE Features" if activation_buffer.load_activations else "Computing SAE Features and dataset activations")):
                pass
        
        activations = feature_buffer.features
        embeddings = activation_buffer.activations
        output_dir = os.path.join(config.RESULTS, args.sae_path.split('.')[0])
    else:
        for i, _ in tqdm(enumerate(activation_buffer), total=len(activation_buffer), desc=("Computing activations")):
                pass
        activations = activation_buffer.activations
        embeddings = activation_buffer.activations
        output_dir = os.path.join(config.RESULTS, model_name.lower())

    # Scale to 0-1 per neuron
    min_values = activations.min(dim=0, keepdim=True)[0]
    max_values = activations.max(dim=0, keepdim=True)[0]
    activations = (activations - min_values) / (max_values - min_values)

    # embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    num_images, embed_dim = embeddings.shape
    num_neurons = activations.shape[1]

    # Initialize accumulators
    weighted_cosine_similarity_sum = torch.zeros(num_neurons, device=torch.device(args.device))
    weight_sum = torch.zeros(num_neurons, device=torch.device(args.device))
    #weights_total = torch.empty((num_images,num_images,num_neurons), device=torch.device(args.device))

    for i in tqdm(range(num_images), desc="Processing image pairs"):
        #weights_i = torch.empty(num_neurons, device=torch.device(args.device))
        for j_start in range(i + 1, num_images, batch_size):  # Process in batches
            j_end = min(j_start + batch_size, num_images)

            embeddings_i = embeddings[i].cuda()  # (embedding_dim)
            embeddings_j = embeddings[j_start:j_end].cuda()  # (batch_size, embedding_dim)
            activations_i = activations[i].cuda()  # (num_neurons)
            activations_j = activations[j_start:j_end].cuda()  # (batch_size, num_neurons)

            # Compute cosine similarity
            cosine_similarities = F.cosine_similarity(
                embeddings_i.unsqueeze(0).expand(j_end - j_start, -1),  # Expanding to (batch_size, embedding_dim)
                embeddings_j,
                dim=1
            )


            # Compute weights and weighted similarities
            # Expanding activations_i to (1, num_neurons)
            weights = activations_i.unsqueeze(0) * activations_j  # (batch_size, num_neurons)

            #weights_i = torch.cat([weights,weights_i],dim=0)
            weighted_cosine_similarities = weights * cosine_similarities.unsqueeze(1)  # (batch_size, num_neurons)

            weighted_cosine_similarities = torch.sum(weighted_cosine_similarities, dim=0).to(device)  # (num_neurons)
            weighted_cosine_similarity_sum += weighted_cosine_similarities

            weights = torch.sum(weights, dim=0).to(device)   # (num_neurons)
            weight_sum += weights
        #weights_total = torch.cat([weights_total,weights_i.unsqueeze(0)],dim=0)
        
    monosemanticity = torch.where(weight_sum != 0, weighted_cosine_similarity_sum / weight_sum, torch.nan).cpu()

    
    os.makedirs(output_dir, exist_ok=True)
    torch.save(monosemanticity, os.path.join(output_dir, f"monosemanticity_score_{args.dataset}_{args.split}.pth"))
    #torch.save(weights_total, os.path.join(output_dir, f"activations_weights_{args.dataset}_{args.split}.pth"))

    is_nan = torch.isnan(monosemanticity)
    nan_count = is_nan.sum()
    monosemanticity_mean = torch.mean(monosemanticity[~is_nan])
    monosemanticity_std = torch.std(monosemanticity[~is_nan])
    monosemanticity_min = torch.min(monosemanticity[~is_nan])
    monosemanticity_max = torch.max(monosemanticity[~is_nan])

    print(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()} (min = {monosemanticity_min}, max = {monosemanticity_max})")
    print(f"Dead neurons:", nan_count.item())
    print(f"Total neurons:", num_neurons)

    # Filter out NaNs
    valid_indices = ~torch.isnan(monosemanticity)
    valid_monosemanticity = monosemanticity[valid_indices]
    valid_indices = torch.nonzero(valid_indices).squeeze()

    # Get top 10 highest and lowest monosemantic neurons
    top_10_values, top_10_indices = torch.topk(valid_monosemanticity, 10)
    bottom_10_values, bottom_10_indices = torch.topk(valid_monosemanticity, 10, largest=False)

    # Map indices back to original positions
    top_10_indices = valid_indices[top_10_indices]
    bottom_10_indices = valid_indices[bottom_10_indices]

    # Print results
    print("Top 10 most monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(top_10_indices, top_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    print("\nBottom 10 least monosemantic neurons:")
    for i, (idx, val) in enumerate(zip(bottom_10_indices, bottom_10_values)):
        print(f"{i + 1}. Neuron {idx.item()} - {val.item()}")

    # Save to file
    output_path = os.path.join(output_dir, f"monosemanticity_stats_{dataset_name}_{args.split}.txt")
    with open(output_path, "w") as file:
        file.write(f"Monosemanticity: {monosemanticity_mean.item()} +- {monosemanticity_std.item()} (min = {monosemanticity_min}, max = {monosemanticity_max})\n")
        file.write(f"Dead neurons: {nan_count.item()}\n")
        file.write(f"Total neurons: {num_neurons}\n\n")

        file.write("Top 10 most monosemantic neurons:\n")
        for idx, val in zip(top_10_indices, top_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")

        file.write("\nBottom 10 least monosemantic neurons:\n")
        for idx, val in zip(bottom_10_indices, bottom_10_values):
            file.write(f"Neuron {idx.item()} - {val.item()}\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    main(args)
