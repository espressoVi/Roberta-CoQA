import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from transformers import (AdamW, AutoConfig, AutoTokenizer, get_linear_schedule_with_warmup)
from processors.coqa import Extract_Features, Processor, Result
from processors.metrics import get_predictions
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import getopt,sys

train_file="coqa-train-v1.0.json"
predict_file="coqa-dev-v1.0.json"
pretrained_model="roberta-base"
epochs = 1.0
evaluation_batch_size = 16
train_batch_size = 4
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

    def forward(self,input_ids,segment_ids=None,input_masks=None,start_positions=None,end_positions=None,rationale_mask=None,cls_idx=None):
        #RoBERTa outputs
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
        return start_logits, end_logits, yes_logits, no_logits, unk_logits

def convert_to_list(tensor):
    return tensor.detach().cpu().tolist()

def train(train_dataset, model, tokenizer, device):

    train_sampler = RandomSampler(train_dataset) 
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)
    t_total = len(train_dataloader) // 1 * epochs
    optimizer_parameters = [{"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "LayerNorm.weight"])],"weight_decay": 0.01,},
                            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in ["bias", "LayerNorm.weight"])], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_parameters,lr=1e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=2000, num_training_steps=t_total)
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
            optimizer.step()
            scheduler.step()  
            model.zero_grad()
            counter += 1
            epoch_iterator.set_description("Loss :%f" % (train_loss/(4*counter)))
            epoch_iterator.refresh()

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

def Write_predictions(model, tokenizer, device, dataset_type = None, output_directory = None):
    dataset, examples, features = load_dataset(tokenizer, evaluate=True,dataset_type = dataset_type)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
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

    output_prediction_file = os.path.join(output_directory, "predictions.json")
    get_predictions(examples, features, mod_results, 20, 30, True, output_prediction_file, False, tokenizer)


def load_dataset(tokenizer, evaluate=False, dataset_type = None):
    input_dir = "data"
    print("Creating features from dataset file at", input_dir)

    if ((evaluate and not predict_file) or (not evaluate and not train_file)):
        raise ValueError("predict_file or train_file name not found")
    else:
        processor = Processor()
        if evaluate:
            examples = processor.get_examples("data", 2,filename=predict_file, threads=12, dataset_type = dataset_type)
        else:
            examples = []
            for datas in dataset_type:
                examples.extend(processor.get_examples("data", 2,filename=train_file, threads=12,dataset_type = datas))

    features, dataset = Extract_Features(examples=examples,
            tokenizer=tokenizer,max_seq_length=512, doc_stride=128, max_query_length=64, is_training=not evaluate, threads=12)
    if evaluate:
        return dataset, examples, features
    return dataset


def manager(isTraining, dataset_type, output_directory):
    assert torch.cuda.is_available()
    device = torch.device('cuda')
    config = RobertaConfig.from_pretrained(pretrained_model)

    if isTraining:
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model)
        model = RobertaBaseModel(config, load_pre = True)
        model.to(device)
        if os.path.exists(output_directory):
            raise ValueError(f"Output directory {output_directory}  already exists, Change output_directory name")
        else:
            os.makedirs(output_directory)
        
        train_dataset = load_dataset(tokenizer, evaluate=False, dataset_type = dataset_type)
        train_loss = train(train_dataset, model, tokenizer, device)
        tokenizer.save_pretrained(output_directory)
        torch.save(model.state_dict(), os.path.join(output_directory,'tweights.pt'))

    else:
        model = RobertaBaseModel(config)
        model.load_state_dict(torch.load(os.path.join(output_directory,'tweights.pt')))
        model.to(device)
        tokenizer = RobertaTokenizer.from_pretrained(output_directory, do_lower_case=True)
        Write_predictions(model, tokenizer, device, dataset_type = dataset_type[0], output_directory = output_directory)

def main():
    isTraining,isEval = False, False
    train_dataset_type, eval_dataset_type = [],[]
    output_directory = "Roberta"
    argumentList = sys.argv[1:]
    options = "ht:e:o:"
    long_options = ["help", "train=","eval=", "output="]
    try:
        arguments, values = getopt.getopt(argumentList, options, long_options)
        for currentArgument, currentValue in arguments:
            if currentArgument in ("-h", "--Help"):
                print ("""python main.py --train [O|C] --eval [O|TS|RG] --output [directory name]\n
                        --train O for original training C for combined training \n
                        --eval for eval on (O) original (TS) truncated and (RG) for TS-R dataset as defined in paper\n
                        --output [dir_name] is the output directory to write weights and predictions in, 
                        and in case of eval to load weights from.
                        e.g. python main.py --train C --eval RG --output Roberta_comb
                        for combined training followed by eval on RG and writing to ./Roberta_comb""")
                return
     
            elif currentArgument in ("-t", "--train"):
                isTraining = True
                opts = {'O':[None],'C':[None, 'TS','RG']}
                if currentValue in opts:
                    train_dataset_type = opts[currentValue]
                else:
                    print('See "python main.py --help" for usage')
                    return
            elif currentArgument in ("-e", "--eval"):
                opts = {'O':[None],'TS':['TS'], 'RG':['RG']}
                if currentValue in opts:
                    eval_dataset_type = opts[currentValue]
                    isEval = True
                else:
                    print('See "python main.py --help" for usage')
                    return
            elif currentArgument in ("-o", "--output"):
                output_directory = currentValue

    except getopt.error as err:
        print (str(err))

    if isTraining:
        manager(isTraining = True, dataset_type = train_dataset_type, output_directory = output_directory)
    if isEval:
        manager(isTraining = False, dataset_type = eval_dataset_type, output_directory = output_directory)
if __name__ == "__main__":
    main()
