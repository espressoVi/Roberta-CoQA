import os
import torch
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
evaluation_batch_size = 64
MIN_FLOAT = -1e30
 
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

    def forward(self,input_ids,segment_ids=None,input_masks=None,start_positions=None,end_positions=None,rationale_mask=None,cls_idx=None, block = -1):
        outputs = self.roberta(input_ids,attention_mask=input_masks,output_hidden_states=True)
        _, roberta_pooled_output, hidden_states = outputs
        output_vector = hidden_states[block] 
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
        return attention

def Write_attentions(model, tokenizer, device, dataset_type = None, output_directory = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True,dataset_type = dataset_type)
    evalutation_sampler = SequentialSampler(dataset)
    evaluation_dataloader = DataLoader(dataset, sampler=evalutation_sampler, batch_size=evaluation_batch_size)
    res = []
    for block in tqdm(range(1,13),desc = "Evaluating with block"):
        su = []
        for batch in evaluation_dataloader:
            model.eval()
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0],"segment_ids": batch[1],"input_masks": batch[2],"block":block}
                example_indices = batch[3]
                outputs = model(**inputs)
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                doc_tok = eval_feature.tokens
                attention = outputs[i].detach().cpu().numpy()
                rational_mask = np.array(eval_feature.rational_mask)
                length =len(np.where(np.array(eval_feature.input_mask) == 1)[0])
                _ones = np.where(rational_mask == 1)[0]
                try:
                    r_start,r_end = _ones[0],_ones[-1]+1
                except:
                    continue
                su.append(np.sum(attention[r_start:r_end]))
        res.append((np.mean(su),np.std(su)))

    blockFile = os.path.join(output_directory,f'pRationale_{dataset_type}.txt')
    with open(blockFile,'w') as f:
        f.write('Block \tMean \tSTD\n')
        for i in range(12):
            f.write(f"{i}\t {res[i][0]:.8f}\t {res[i][1]:.8f}\n")

def load_dataset(tokenizer, evaluate=False, dataset_type = None):
    processor = Processor()
    examples = processor.get_examples("data", 2,filename=predict_file, threads=12, dataset_type = dataset_type,attention=True)
    features, dataset = Extract_Features(examples=examples,
            tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=not evaluate, threads=12)
    return dataset, examples, features

def manager(model_dir,dataset_type):
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    config = RobertaConfig.from_pretrained(pretrained_model)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir, do_lower_case=True)
    model = RobertaBaseModel(config)
    model.load_state_dict(torch.load(os.path.join(model_dir,'tweights.pt')))
    model.to(device)
    for i in dataset_type:
        print(model_dir,i)
        Write_attentions(model, tokenizer, device, dataset_type = i,output_directory = model_dir)

def main():
    output_directory = "Roberta"
    argumentList = sys.argv[1:]
    options = "ho:"
    long_options = ["help", "output="]
    try:
        arguments, values = getopt.getopt(argumentList, options, long_options)
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--Help"):
                print ("""python attetion-block.py --output [directory name]
                        --output [dir_name] is the output directory to load weights from and write
                        values to.
                        e.g. python main.py --output Roberta_comb
                        for p_rationale values for model stored at./Roberta_comb""")
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
