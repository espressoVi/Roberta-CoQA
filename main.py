import collections
import glob
import os
import torch
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup)
from processors.coqa import Extract_Features, Processor, Result,Attentions
from processors.metrics import get_predictions
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np

train_file="coqa-train-v1.0.json"
predict_file="coqa-dev-v1.0.json"
output_directory="Roberta_orig"
pretrained_model="roberta-base"

epochs = 1.0
evaluation_batch_size = 16
train_batch_size = 4
MIN_FLOAT = -1e30
max_seq_length = 512
 
class RobertaBaseModel(RobertaModel):
    def __init__(self,config, load_pre = False):
        super(RobertaBaseModel,self).__init__(config)
        self.roberta = RobertaModel.from_pretrained(pretrained_model, config=config,) if load_pre else RobertaModel(config)

        hidden_size = config.hidden_size

        self.fc = nn.Linear(hidden_size,hidden_size, bias = False)
        self.fc2 = nn.Linear(hidden_size,hidden_size, bias = False)
        self.rationale_modelling = nn.Linear(hidden_size,1, bias = False)
        self.attention_modelling = nn.Linear(hidden_size,1, bias = False)
        self.span_modelling = nn.Linear(hidden_size,2,bias = False)
        self.unk_modelling = nn.Linear(2*hidden_size,1, bias = False)
        self.yes_no_modelling = nn.Linear(2*hidden_size,2, bias = False)
        self.relu = nn.ReLU()

        self.beta = 5.0

    def forward(self,input_ids,segment_ids=None,input_masks=None,start_positions=None,end_positions=None,rationale_mask=None,cls_idx=None,attn = False):
        #RoBERTa outputs
        if attn:
            outputs = self.roberta(input_ids,attention_mask=input_masks,output_attentions=True)
            output_vector, roberta_pooled_output, attentions = outputs
            attentions = list(attentions)
        else:
            outputs = self.roberta(input_ids,attention_mask=input_masks)
            output_vector, roberta_pooled_output = outputs

        start_end_logits = self.span_modelling(output_vector)
        start_logits, end_logits = start_end_logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        #Rationale modelling 
        rationale_logits = self.relu(self.fc(output_vector))
        rationale_logits = self.rationale_modelling(rationale_logits)
        rationale_logits = torch.sigmoid(rationale_logits)

        output_vector = output_vector * rationale_logits

        attention  = self.relu(self.fc2(output_vector))
        attention  = (self.attention_modelling(attention)).squeeze(-1)
        input_masks = input_masks.type(attention.dtype)
        attention = attention*input_masks + (1-input_masks)*MIN_FLOAT
        attention = F.softmax(attention, dim=-1)
        if attn:
            attentions.append(attention)
        attention_pooled_output = (attention.unsqueeze(-1) * output_vector).sum(dim=-2)
        cls_output = torch.cat((attention_pooled_output,roberta_pooled_output),dim = -1)

        rationale_logits = rationale_logits.squeeze(-1)

        unk_logits = self.unk_modelling(cls_output)
        yes_no_logits = self.yes_no_modelling(cls_output)
        yes_logits, no_logits = yes_no_logits.split(1, dim=-1)


        if self.training:
            start_positions, end_positions = start_positions + cls_idx, end_positions + cls_idx
            start = torch.cat((yes_logits, no_logits, unk_logits, start_logits), dim=-1)
            end = torch.cat((yes_logits, no_logits, unk_logits, end_logits), dim=-1)

            Entropy_loss = CrossEntropyLoss()
            start_loss = Entropy_loss(start, start_positions)
            end_loss = Entropy_loss(end, end_positions)

            rationale_positions = rationale_mask.type(attention.dtype)
            rationale_loss = -rationale_positions*torch.log(rationale_logits + 1e-8) - (1-rationale_positions)*torch.log(1-rationale_logits + 1e-8)

            rationale_loss = torch.mean(rationale_loss)
            total_loss = (start_loss + end_loss) / 2.0 + rationale_loss * self.beta

            return total_loss
        if attn:
            return start_logits, end_logits, yes_logits, no_logits, unk_logits,attentions
        else:
            return start_logits, end_logits, yes_logits, no_logits, unk_logits

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(train_dataset, model, tokenizer, device):

    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    t_total = len(train_dataloader) // 1 * epochs

    # Preparing optimizer and scheduler
    
    optimizer_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],"weight_decay": 0.01,},
                            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_parameters,lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(pretrained_model, "optimizer.pt")) and os.path.isfile(os.path.join(pretrained_model, "scheduler.pt")):
        optimizer.load_state_dict(torch.load(
            os.path.join(pretrained_model, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(
            os.path.join(pretrained_model, "scheduler.pt")))

    counter = 1
    epochs_trained = 0
    train_loss, loss = 0.0, 0.0
    model.zero_grad()
    iterator = trange(epochs_trained, int(epochs), desc="Epoch", disable=False)
    for _ in iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False)
        for i,batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = { "input_ids": batch[0],"segment_ids": None,
                  "input_masks": batch[2],"start_positions": batch[3],
                  "end_positions": batch[4],"rationale_mask": batch[5],"cls_idx": batch[6]}
            loss = model(**inputs)
            loss.backward()
            train_loss += loss.item()
             #   optimizing training parameters
            optimizer.step()
            scheduler.step()  
            model.zero_grad()
            counter += 1
            epoch_iterator.set_description("Loss :%f" % (train_loss/(4*counter)))
            epoch_iterator.refresh()

            #   Saving model weights every 1000 iterations
            if counter % 1000 == 0:
                output_dir = os.path.join(output_directory, "model_weights")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(model, "module") else model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    return train_loss/counter


def Write_predictions(model, tokenizer, device, dataset_type = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True,dataset_type = dataset_type)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    #   wrtiting predictions once training is complete
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    mod_results = []
    for batch in tqdm(evaluation_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],"segment_ids": batch[1],"input_masks": batch[2]}
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            output = [convert_to_list(output[i]) for output in outputs]
            start_logits, end_logits, yes_logits, no_logits, unk_logits = output
            result = Result(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits, yes_logits=yes_logits, no_logits=no_logits, unk_logits=unk_logits)
            mod_results.append(result)

    # Get predictions for development dataset and store it in predictions.json
    output_prediction_file = os.path.join(output_directory, "predictions.json")
    get_predictions(examples, features, mod_results, 20, 30, True, output_prediction_file, False, tokenizer)

def Write_attentions(model, tokenizer, device, dataset_type = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True,dataset_type = dataset_type)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    #   wrtiting predictions once training is complete
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    attn_results = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for batch in tqdm(evaluation_dataloader, desc="Attn"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],"segment_ids": batch[1],"input_masks": batch[2],"attn":True}
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            attentions = outputs[-1]
            attentions = [output[i].detach().cpu().numpy() for output in attentions]
            rational_mask = np.array(eval_feature.rational_mask)
            length =len(np.where(np.array(eval_feature.input_mask) == 1)[0])
            _ones = np.where(rational_mask == 1)[0]
            try:
                r_start,r_end = _ones[0],_ones[-1]+1
            except:
                continue

            #for j in range(12):
            #    attn_results[j].append(attention_res(attentions, j, 0,r_start,r_end, length))
            print(np.sum(attentions[12][r_start:r_end]))
    #attn_results = np.array(attn_results)
    #print('Mean: \n')
    #print(np.mean(attn_results,axis = 1))
    #print('STD: \n')
    #print(np.std(attn_results, axis = 1))

def attention_res(attentions,block,head,r_start,r_end,length):
    assert block < len(attentions)
    assert head < len(attentions[block])
    attention = attentions[block][head]
    #np.save( "attention.npy",attention)
    assert attention.shape == (max_seq_length,max_seq_length)
    su,su_r,su_nr = [],[],[]
    for i in range(length):
        eta = np.sum(attention[i][r_start:r_end])
        eta = (eta*length) / (r_end - r_start)
        if r_start <= i <= r_end:
            su_r.append(eta)
        else:
            su_nr.append(eta)
        su.append(eta)
    return np.mean(su), np.mean(su_r),np.mean(su_nr)
    



def load_dataset(tokenizer, evaluate=False, dataset_type = None):
    #   converting raw coqa dataset into features to be processed by ROBERTA   
    input_dir = "data" if "data" else "."
    if evaluate:
        cache_file = os.path.join(input_dir,"roberta-base_dev")
    else:
        cache_file = os.path.join(input_dir,"roberta-base_train")

    if os.path.exists(cache_file):# and False: #for new dataset
        print("Loading cache",cache_file)
        features_and_dataset = torch.load(cache_file)
        features, dataset, examples = (
            features_and_dataset["features"],features_and_dataset["dataset"],features_and_dataset["examples"])
    else:
        print("Creating features from dataset file at", input_dir)

        if not "data" and ((evaluate and not predict_file) or (not evaluate and not train_file)):
            raise ValueError("predict_file or train_file not found")
        else:
            processor = Processor()
            if evaluate:
                examples = processor.get_examples("data", 2,filename=predict_file, threads=12, dataset_type = dataset_type)
            else:
                #examples = processor.get_examples("data", 2,filename=train_file, threads=12,dataset_type = dataset_type)
                examples = processor.get_examples("data", 2,filename=train_file, threads=12,dataset_type = "TS")
                examples.extend(processor.get_examples("data", 2,filename=train_file, threads=12,dataset_type = None))
                examples.extend(processor.get_examples("data", 2,filename=train_file, threads=12,dataset_type = 'RG'))

        features, dataset = Extract_Features(examples=examples,
                tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=not evaluate, threads=12)
    #   caching it in a cache file to reduce time
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cache_file)
    if evaluate:
        return dataset, examples, features
    return dataset


def main(isTraining, attn = False):
    assert torch.cuda.is_available()
    assert not (isTraining and attn)
    device = torch.device('cuda')
    config = RobertaConfig.from_pretrained(pretrained_model)

    if isTraining:
        #   initialize configurations and tokenizer of Roberta model 
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        model = RobertaBaseModel(config, load_pre = True)
        model.to(device)

        if os.path.exists(output_directory):
            raise ValueError(f"Output directory {output_directory}  already exists, Change output_directory name")
        else:
            os.makedirs(output_directory)
        
        train_dataset = load_dataset(tokenizer, evaluate=False)
        train_loss = train(train_dataset, model, tokenizer, device)
        #   save Model
        tokenizer.save_pretrained(output_directory)
        torch.save(model.state_dict(), os.path.join(output_directory,'tweights.pt'))

    else:
        if attn:
            model = RobertaBaseModel(config)
            model.load_state_dict(torch.load(os.path.join(output_directory,'tweights.pt')))
            model.to(device)
            model.eval()
            tokenizer = RobertaTokenizer.from_pretrained(output_directory, do_lower_case=True)
            Write_attentions(model, tokenizer, device, dataset_type = None)
        else:
            model = RobertaBaseModel(config)
            model.load_state_dict(torch.load(os.path.join(output_directory,'tweights.pt')))
            model.to(device)
            model.eval()
            tokenizer = RobertaTokenizer.from_pretrained(output_directory, do_lower_case=True)
            Write_predictions(model, tokenizer, device, dataset_type = 'RG')

if __name__ == "__main__":
    #main(isTraining = True)
    main(isTraining = False, attn = True)
