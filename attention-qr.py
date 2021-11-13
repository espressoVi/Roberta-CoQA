import os
import torch
import pickle
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup)
from processors.coqa import Extract_Features, Processor
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
import getopt,sys

predict_file="coqa-dev-v1.0.json"
pretrained_model="roberta-base"
evaluation_batch_size = 16
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

    def forward(self,input_ids,segment_ids=None,input_masks=None,start_positions=None,end_positions=None,rationale_mask=None,cls_idx=None):
        outputs = self.roberta(input_ids,attention_mask=input_masks,output_attentions=True)
        output_vector, roberta_pooled_output, attentions = outputs
        attentions = list(attentions)

        start_end_logits = self.span_modelling(output_vector)
        start_logits, end_logits = start_end_logits.split(1, dim=-1)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)
        rationale_logits = self.relu(self.fc(output_vector))
        rationale_logits = self.rationale_modelling(rationale_logits)
        rationale_logits = torch.sigmoid(rationale_logits)

        output_vector = output_vector * rationale_logits

        attention  = self.relu(self.fc2(output_vector))
        attention  = (self.attention_modelling(attention)).squeeze(-1)
        input_masks = input_masks.type(attention.dtype)
        attention = attention*input_masks + (1-input_masks)*MIN_FLOAT
        attention = F.softmax(attention, dim=-1)
        attentions.append(attention)

        attention_pooled_output = (attention.unsqueeze(-1) * output_vector).sum(dim=-2)
        cls_output = torch.cat((attention_pooled_output,roberta_pooled_output),dim = -1)

        rationale_logits = rationale_logits.squeeze(-1)

        unk_logits = self.unk_modelling(cls_output)
        yes_no_logits = self.yes_no_modelling(cls_output)
        yes_logits, no_logits = yes_no_logits.split(1, dim=-1)

        return attentions

def Write_attentions(model, tokenizer, device, dataset_type = None, output_directory = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True,dataset_type = dataset_type)
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    qr_results = [[],[],[],[],[],[],[],[],[],[],[],[]]
    sep_results = [[],[],[],[],[],[],[],[],[],[],[],[]]
    for batch in tqdm(evaluation_dataloader, desc="Attn"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],"segment_ids": batch[1],"input_masks": batch[2]}
            example_indices = batch[3]
            outputs = model(**inputs)
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            doc_tok = eval_feature.tokens
            q_start = 1
            q_end = [i for i,j in enumerate(doc_tok) if j == '</s>'][0]
            seps = [i for i,j in enumerate(doc_tok) if j == '</s>']
            attentions = outputs
            attentions = [output[i].detach().cpu().numpy() for output in attentions]
            rational_mask = np.array(eval_feature.rational_mask)
            length =len(np.where(np.array(eval_feature.input_mask) == 1)[0])
            _ones = np.where(rational_mask == 1)[0]
            try:
                r_start,r_end = _ones[0],_ones[-1]+1
            except:
                continue
            for j in range(12):
                qr_results[j].append(attention_qr(attentions[j], -1, r_start,r_end,q_start,q_end, length))
                sep_results[j].append(attention_sep(attentions[j], -1, seps))
    qr_results = np.array(qr_results)
    sep_results = np.array(sep_results)
    
    Mean_qr = np.mean(qr_results,axis = 1)
    STD_qr = np.std(qr_results, axis = 1)
    Mean_sep = np.mean(sep_results,axis = 1)
    STD_sep = np.std(sep_results, axis = 1)
    qrFile = os.path.join(output_directory,f'etaQR_{dataset_type}.txt')
    sepFile = os.path.join(output_directory,f'pSEP_{dataset_type}.txt')
    with open(qrFile,'w') as f:
        f.write('Block \tMean \tSTD\n')
        for i in range(12):
            f.write(f'{i} \t{Mean_qr[i]} \t{STD_qr[i]}\n')
    with open(sepFile,'w') as f:
        f.write('Block \tMean \tSTD\n')
        for i in range(12):
            f.write(f'{i} \t{Mean_sep[i]} \t{STD_sep[i]}\n')

def attention_qr(attention,head,r_start,r_end,q_start,q_end,length):
    assert head < len(attention)
    if head == -1:
        attention = np.mean(attention, axis = 0)
    else:
        attention = attention[head]
    assert attention.shape == (max_seq_length,max_seq_length)
    su = []
    for i in range(r_start,r_end):
        eta = np.sum(attention[i][q_start:q_end])
        eta = (eta*length) / (q_end - q_start)
        su.append(eta)
    return np.mean(su)

def attention_sep(attention,head,seps):
    assert head < len(attention)
    if head == -1:
        attention = np.mean(attention, axis = 0)
    else:
        attention = attention[head]
    assert attention.shape == (max_seq_length,max_seq_length)
    p = 0
    for i in seps:
        p += np.mean(attention[:,i])
    return p

def load_dataset(tokenizer, evaluate=True, dataset_type = None):
    cache_file = os.path.join('data',"roberta-base_dev")
    processor = Processor()
    examples = processor.get_examples("data", 2,filename=predict_file, threads=12, dataset_type = dataset_type, attention = True)
    features, dataset = Extract_Features(examples=examples,
            tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=not evaluate, threads=12)
    return dataset, examples, features

def manager(model_dir, dataset_type):
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    config = RobertaConfig.from_pretrained(pretrained_model)
    model = RobertaBaseModel(config)
    model.load_state_dict(torch.load(os.path.join(model_dir,'tweights.pt')))
    model.to(device)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir, do_lower_case=True)
    for j in dataset_type:
        print(model_dir,j)
        Write_attentions(model, tokenizer, device, dataset_type = j,output_directory = model_dir)

def main():
    output_directory = "Roberta"
    argumentList = sys.argv[1:]
    options = "ho:"
    long_options = ["help", "output="]
    try:
        arguments, values = getopt.getopt(argumentList, options, long_options)
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--Help"):
                print ("""python attetion-qr.py --output [directory name]
                        --output [dir_name] is the output directory to load weights from and write
                        values to.
                        e.g. python main.py --output Roberta_comb
                        for eta and p_sep values for model stored at Roberta_comb""")
                return
     
            elif currentArgument in ("-o", "--output"):
                output_directory = currentValue
            else:
                print('See "python main.py --help" for usage')
                return

    except getopt.error as err:
        print (str(err))
    manager(model_dir = output_directory, dataset_type = ['RG','TS',None])

if __name__ == "__main__":
    main()
