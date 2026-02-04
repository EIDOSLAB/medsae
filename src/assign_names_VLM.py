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
import json
from PIL import Image
import random
from huggingface_hub import get_token
from transformers import pipeline
from huggingface_hub import notebook_login
from transformers import BitsAndBytesConfig

PROMPT = """
Below I provide you with a sequence of image-report pairs, which have been collected because they all share one concept. Your task is to identify and describe the shared concept by examining both the images and the reports. This concept could be medical, such as a specific condition or abnormality, or it could relate to non-medical aspects like patterns, artifacts, or other features present in the images.

For example, in the case of medical concepts, you might see reports mentioning pneumonia in the left lung, signs of infection in the left lung, and consolidation in the left lung. The shared concept would then relate to abnormalities in the left lung. For non-medical concepts, you might identify patterns such as image artifacts, recurring visual features, or equipment-related anomalies.

Pay close attention to both the visual elements in the images and the textual descriptions in the reports. Some concepts may be more abstract, such as patterns of disease progression, specific diagnostic features, or even technical aspects like image quality or positioning.

Further, some image-report pairs may have been included that do not relate to the concept at all. Please ignore those items.

First, please summarize the main concept in a short sentence or phrase, approximately 5 to 10 words. Be as specific as possible—it is fine if it doesn’t fit all image-report pairs. Please be precise and creative. Then, also provide a longer description elaborating on the concept.

Return the results in JSON format with the following JSON schema: {"concept": "main concept summary", "description": "longer description of the shared concept"}. The image-report pairs follow now.
"""

PROMPT_NO_REPORTS = PROMPT = """
Below I provide you with a sequence of chest X-ray images. Your task is to identify and describe the shared concept among these images. The shared concept could be medical, such as a specific condition or abnormality, or it could relate to non-medical aspects like patterns, artifacts, or other visual features.

For example, you might identify a recurring visual pattern such as opacities in a specific lung region, consistent positioning of medical devices, or artifacts that appear across the images. The shared concept could also relate to image quality, patient positioning, or technical aspects of the X-rays.

Some concepts may be more abstract, such as patterns of disease progression, specific diagnostic features, or technical aspects like image clarity or exposure.

Further, some images may have been included that do not relate to the concept at all. Please ignore those items.

First, please summarize the main concept in a short sentence or phrase, approximately 5 to 10 words. Be as specific as possible—it is fine if it doesn’t fit all images. Please be precise and creative. Then, also provide a longer description elaborating on the concept.

Return the results in JSON format with the following JSON schema: {"concept": "main concept summary", "description": "longer description of the shared concept"}. The images follow now.
"""

SYSTEM_INSTR = "You are an expert radiologist."


def init_medgemma():
    if get_token() is None:
        notebook_login()

    model_variant = "4b-it"  # @param ["4b-it", "27b-text-it"]
    model_id = f"google/medgemma-{model_variant}"

    use_quantization = True  # @param {type: "boolean"}

    # @markdown Set `is_thinking` to `True` to turn on thinking mode. **Note:** Thinking is supported for the 27B variant only.
    is_thinking = False  # @param {type: "boolean"}

    # If running the 27B variant in Google Colab, check if the runtime satisfies
    # memory requirements
    if "27b" in model_variant and google_colab:
        if not ("A100" in t.cuda.get_device_name(0) and use_quantization):
            raise ValueError(
                "Runtime has insufficient memory to run the 27B variant. "
                "Please select an A100 GPU and use 4-bit quantization."
            )

    model_kwargs = dict(
        torch_dtype=t.bfloat16,
        device_map="auto",
    )

    if use_quantization:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)


    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        model_kwargs=model_kwargs,
    )

    pipe.model.generation_config.do_sample = False

    return pipe


def get_concepts(pipe, images, reports, prompt_report=True):
    concepts = []
    descriptions = []


    messages = []
    

    for i in range(len(images)):
        
        if prompt_report == False: content = [{"type": "text", "text": PROMPT_NO_REPORTS}]
        else: content = [{"type": "text", "text": PROMPT}]
        # Iterate over the DataFrame and append image and text entries
        for img, report in zip(images[i],reports[i]):
            image_entry = {"type": "image", "image": img}
            content.append(image_entry)
            if prompt_report: 
                text_entry = {"type": "text", "text": report}
                content.append(text_entry)


        # Construct the messages list
        messages.append([
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_INSTR}]
            },
            {
                "role": "user",
                "content": content
            }
        ])
        
    for out in pipe(text=messages, max_new_tokens=300, batch_size=4):
        json_string = out[0]["generated_text"][-1]["content"]
        json_content = json_string.strip('```json\n').strip('\n```')
        try:
            out = json.loads(json_content)
            concepts.append(out['concept'])
            descriptions.append(out['description'])
        except:
            print("ERROR JSON : ", json_string)
            concepts.append("")
            descriptions.append("")
        
    return concepts, descriptions


def main(args):
    print(args.include_reports)
    sae_path = os.path.join(config.SAE_CKPTS,args.sae_path)

    (SaeArchitecture, _) = utils.get_sae_architecture(args.sae_type_name)
    device = args.device 

    sae = SaeArchitecture.from_pretrained(sae_path).to(device)
    activation_dim = sae.activation_dim
    dict_size = sae.dict_size
    expansion_factor = dict_size/activation_dim
    grid_size = (3,2)
    k = 100
    

    batch_size = 1024
    model_name = args.model_name
    device=args.device
    #dataloader = utils.get_dataset(args.dataset, batch_size)
    dataloader, labels = utils.get_dataset(args.dataset, batch_size, labels=True, shuffled_version=args.shuffle_dataset)
    labels = labels[args.split]
    #Model import
    print(f"Importing model {model_name}...")
    #model, processor = utils.get_model(model_name)
    model, processor = t.Tensor([]), None
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
        return_raw_data=False,
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
        load_features_path=args.load_features_path if args.load_features_path else (path if precomputed else None),
        save_features_path=args.save_features_path if args.save_features_path else (path if not(precomputed) else None),
        return_raw_data=False,
    )

    total = len(feature_buffer)
    features_dataset = None

    for i, features in tqdm(enumerate(feature_buffer), total=total, desc="Computing SAE Features"):
        if features_dataset is None:
            features_dataset = features.to(device)
        else:
            features_dataset = t.cat((features_dataset, features.to(device)), dim=0)

    features_dataset = activation_buffer.activations

    topk_activating_imgs_values, topk_activating_imgs_idx = t.topk(features_dataset, k=k, dim=0, sorted=True)
    l0 = t.sum((features_dataset.abs() > 1e-7).to(t.float32), dim=0)
    print(f"Number of alive features: {len(t.where(l0 > 0)[0])} out of {features_dataset.shape[1]}")
    
    variances = t.var(features_dataset, dim=0)
    # Identify dead features with zero variance
    alive_features_indices = t.where(variances != 0)[0]
    #alive_features_indices = [7745, 1291, 7036, 3324, 820, 7205, 1774, 63, 4186]

    def get_topk_activating_imgs(feature_idx,k):
        return [(i, dataloader[args.split].dataset[i]) for i in topk_activating_imgs_idx[:,feature_idx][:k]]
    

    dataset_info_path = '/scratch/medical-mi/medical-mi/datasets/chexpert/chexpert_'
    if args.split == "val": dataset_info_path += "valid"
    else: dataset_info_path += "train"
    if args.shuffle_dataset: dataset_info_path += "shuffled"
    dataset_info_path += ".csv"
    if args.include_reports:
        df = pd.read_csv(dataset_info_path)
        df_report = pd.read_csv('/scratch/medical-mi/medical-mi/datasets/chexpert/df_chexpert_plus_240401.csv')
        df['Path'] = "/scratch/medical-mi/medical-mi/"+df['Path']
        df['raw_path'] = df['Path'].str[49:]
        df['report'] = df['raw_path'].map(df_report.set_index('path_to_image')['report'])

    concepts = ["" for _ in range(features_dataset.shape[1])]
    descriptions = ["" for _ in range(features_dataset.shape[1])]
    concepts_out, descriptions_out = [], []
    images = []
    reports = []
    idx_list = []
    m = 0
    pipe = init_medgemma()
    with open(config.CONCEPTS + f"/MedCLIPONLY_reports{args.include_reports}.csv", 'w') as f:
        f.write("neuron,concepts,description\n")
        print(f"Starting to compute concepts for {len(alive_features_indices)} alive features...")
        for idx in tqdm(alive_features_indices[142:]):
            print(f"Computing concepts for feature {idx}, L0 : {l0[idx]}...")
            image_list = []
            report_list = []
            if l0[idx] > 2*args.nb_of_images_analysed:
                #try:
                imgs = get_topk_activating_imgs(idx, int(l0[idx].item()) - int(l0[idx].item()//3))
                imgs = imgs[:args.nb_of_images_analysed//2]+imgs[-args.nb_of_images_analysed//2:]
                top1, _ = imgs[0]
                bottom1, _ = imgs[-1]
                print(f"Top activating image for feature {idx} - value : {features_dataset[top1, idx]}, bottom activating image - value : {features_dataset[bottom1, idx]}")
                imgs = random.sample(imgs, len(imgs))

            # except Exception as e:
                #     print(f"Error getting top activating images for feature {idx}: {e}")
                #     imgs = get_topk_activating_imgs(idx, args.nb_of_images_analysed)
                for i, img in imgs:
                    if features_dataset[i, idx] < 1e-7:
                        print(f"Feature {idx} is dead for image {i}, warning...")
                    image_list.append(img)
                    if args.include_reports : report_list.append(df.iloc[i.item()]['report'])
                    else: report_list.append("")
                images.append(image_list)
                reports.append(report_list)
                idx_list.append(idx)
                m += 1
            if m > args.batch_size:
                m = 0
                result = get_concepts(pipe, images, reports, args.include_reports)
                concepts_out += result[0]
                descriptions_out += result[1]
                print(f"Writing {len(idx_list)} concepts to file...")
                for i in range(len(idx_list)):
                    f.write(f"{idx_list[i]},\"{result[0][i]}\",\"{result[1][i]}\"\n")
                    print(f"Neuron {idx_list[i]} : {result[0][i]} - {result[1][i][:15]}...")
                images = []
                reports = []
                idx_list = []
        result = get_concepts(pipe, images, reports, args.include_reports)
        concepts_out += result[0]
        descriptions_out += result[1]
        for i in range(len(idx_list)):
            f.write(f"{idx_list[i]},{result[0][i]},{result[1][i]}\n")

    i = 0
    for idx in alive_features_indices:
        if l0[idx] > 60:
            concepts[idx] = concepts_out[i]
            descriptions[idx] = descriptions_out[i]
            i += 1

    table = pd.DataFrame({"neuron": [i for i in range(len(concepts))],"concepts": concepts, "description": descriptions})
    output_path = os.path.join(config.CONCEPTS,f"{sae_name}_reports{args.include_reports}.csv")
    table.to_csv(output_path, index=False)
            
            
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"A script that leverages MedGemma from Meta on Huggingface to assign medical names/concepts to every dimension of a vectorial ChestXRay image representation based on the analysis of top and low activating images.")
    parser.add_argument('--sae_path', type=str, required=False, default="ae.pt", help=f'The path of the .pt file starting from {config.SAE_CKPTS}')
    parser.add_argument('--model_name', type=str, required=True, help=f"Name of the model you want to use. Should be in the list : [{', '.join(config.AVAILABLE_MODELS)} ]")
    parser.add_argument('--sae_type_name', type=str, required=False, default="Standard", help=f"The name of the SAE architecture you want to use. Should be in the list : [{', '.join(config.AVAILABLE_SAE)} ]"),
    parser.add_argument('--dataset', type=str, required=True, help=f"Name of the image dataset you want to use. Should be in the list : [{', '.join(config.AVAILABLE_DATASETS)} ]")
    parser.add_argument('--split', type=str, required=False, help=f'The split of the dataset to use', default="val")
    parser.add_argument('--batch_size', type=int, required=False, help=f'The batch size for computing the images-report pairs', default=10)
    parser.add_argument('--device', type=str, required=False, help=f'Device used for the computations', default='cuda')
    parser.add_argument('--nb_of_images_analysed', type=int, required=False, help=f'The number of concepts printed.', default=30)
    parser.add_argument('--load_features_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the sae's features starting from {config.FEATURES}.", default=None)
    parser.add_argument('--save_features_path', type=str, required=False, help=f"[Optional] Saving path of the sae's features starting from {config.FEATURES}.", default=None)
    parser.add_argument('--load_activations_path', type=str, required=False, help=f"[Optional] Path of the .pt file of the model's activations. If not provided it will automatically check if activations can be loaded from {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--save_activations_path', type=str, required=False, help=f"[Optional] Saving path of the model's activations. If not provided it will automatically save the activations to {config.ACTIVATIONS}.", default=None)
    parser.add_argument('--layer', type=str, required=False, help=f'The layer of the model to use for computing the activations. If not provided it will use the last layer.', default=None)
    parser.add_argument('--shuffle_dataset', type=bool, required=False, help=f'To use shuffle dataset version', default=False)
    parser.add_argument('--include_reports', type=bool, required=False, help=f'To include in the analysis of top activating images for the interpretation.', default=False)


    args = parser.parse_args()
    main(args)




