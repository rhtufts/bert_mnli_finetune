#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn.functional as F
import numpy as np
import pytorch_transformers as pt
from utils_glue import *
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
import wget
import zipfile
import logging

logging.basicConfig(filename='test_script.log',level=logging.DEBUG)

def fetch_weights():
    """
    checks for existence of network weights.
    if not there, attempts to download and unzip the weights.
    :return path to weights.:
    """
    weights_path = './final_training_output'
    if not os.path.exists(weights_path):
        output_dir = "./"
        url = "https://bert-mnli-fine-tune.s3-eu-west-1.amazonaws.com/bert_ft_training_weights.zip"
        logging.info("attempting to download from {}".format(url))
        zip_file = wget.download(url)
        # zip_file = "bert_ft_training_weights.zip"
        logging.info("unzipping directory")
        with zipfile.ZipFile(output_dir + zip_file, "r") as zip_ref:
            zip_ref.extractall(weights_path)

        if not os.path.exists(weights_path+"/fto"):
            logging.debug("file was not successfully unzipped.")
            exit(1)
        else:
            return weights_path+"/fto"

    else:
        return "./final_training_output/fto"
# if not os.path.exists('final_training_output'):
# #    output_dir = "./data/"
#     output_dir = "./"
#     url = "https://bert-mnli-fine-tune.s3-eu-west-1.amazonaws.com/bert_ft_training_weights.zip"
# #    zip_file = wget.download(url, out=output_dir)
#     zip_file = wget.download(url)
#     zip_file = "bert_ft_training_weights.zip"
#     with zipfile.ZipFile(output_dir+zip_file, "r") as zip_ref:
#         zip_ref.extractall("./final_training_output")

# btf = pt.BertForSequenceClassification.from_pretrained('final_training_output/fto/')
# btf.to(device)
#
# btf_tokenizer = pt.BertTokenizer.from_pretrained('final_training_output/fto/')

def predict_entailment(prod_dataloader, index2class={0:"contradiction",1:"neutral", 2:"entailment"}):
    """
    predicts whether pairs in a dataset are contradictory or not.
    """
    confidences = None
    preds = None
    for batch in prod_dataloader:
        with torch.no_grad():
            inputs = {'input_ids':  batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]
                 }

            outputs = btf(**inputs)
            if confidences is None:
                confidences = F.softmax(outputs[0], dim=1).detach().cpu().numpy()
                preds = np.argmax(confidences,axis=1)
            else:
                confidences = np.append(confidences, F.softmax(outputs[0], dim=1).detach().cpu().numpy(), axis=0)
                preds = np.append(preds,np.argmax(F.softmax(outputs[0], dim=1).detach().cpu().numpy(),axis=1), axis=0)
    return [index2class[p] for p in preds],confidences

# ex1 = InputExample(1, "Mary went to the cinema.", "Actually she went to the pictures", "contradiction")
# ex2 = InputExample(1, "Brian enjoys parties", "He likes to drink", "neutral")
#
# examples = [ex1,ex2]*10
#
# features = convert_examples_to_features(examples, ["contradiction", "entailment", "neutral"], 128, btf_tokenizer, "classification")
#
# # Convert to Tensors and build dataset
# all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
# all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
# all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(device)
# all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(device)
#
# # Create dataset and loaders
# prod_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# prod_sampler = SequentialSampler(prod_dataset)
# # prod_dataloader = DataLoader(prod_dataset, sampler=prod_sampler, batch_size=args.eval_batch_size)
# prod_dataloader = DataLoader(prod_dataset, sampler=prod_sampler, batch_size=8)
#
# # return preds,confidences
# print(predict_entailment(prod_dataloader))


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights_path = fetch_weights()
    btf = pt.BertForSequenceClassification.from_pretrained(weights_path)
    btf.to(device)
    btf_tokenizer = pt.BertTokenizer.from_pretrained(weights_path)

    ex1 = InputExample(1, "Mary went to the cinema.", "Actually she went to the pictures", "contradiction")
    ex2 = InputExample(1, "Brian enjoys parties", "He likes to drink", "neutral")
    examples = [ex1, ex2] * 10

    features = convert_examples_to_features(examples, ["contradiction", "entailment", "neutral"], 128, btf_tokenizer,
                                            "classification")

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long).to(device)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long).to(device)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long).to(device)

    # Create dataset and loaders
    prod_dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    prod_sampler = SequentialSampler(prod_dataset)
    # prod_dataloader = DataLoader(prod_dataset, sampler=prod_sampler, batch_size=args.eval_batch_size)
    prod_dataloader = DataLoader(prod_dataset, sampler=prod_sampler, batch_size=8)

    # return preds,confidences
    print(predict_entailment(prod_dataloader))



