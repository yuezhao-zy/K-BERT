# -*- encoding:utf -*-
"""
  This script provides an K-BERT example for NER.
"""
import random
import argparse
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from uer.model_builder import build_model
from uer.utils.config import load_hyperparam
from uer.utils.optimizers import  BertAdam
from uer.utils.constants import *
from uer.utils.vocab import Vocab
from uer.utils.seed import set_seed
from uer.model_saver import save_model
import numpy as np
from save_log import  Save_Log
from brain import KnowledgeGraph
import datetime
import os
from tensorboardX import SummaryWriter
from my_logging import init_logger
import torch

import torch.functional as F
from torchcrf import CRF



class BertTagger(nn.Module):
    def __init__(self, args, model): #传参传入了model
        super(BertTagger, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder



        self.target = model.target
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        example:
            src size: torch.Size([8, 128])
            output size: torch.Size([8, 128, 768])
            output size: torch.Size([8, 128, 15])
            output size: torch.Size([1024, 15])
            output size: torch.Size([1024, 15])
            label size: torch.Size([1024, 1])
            onehot size: torch.Size([1024, 15])
        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        output = self.encoder(emb, mask, vm)

        # Target.
        output = self.output_layer(output)


        output = output.contiguous().view(-1, self.labels_num)

        output = self.softmax(output)

        label = label.contiguous().view(-1,1)

        label_mask = (label > 0).float().to(torch.device(label.device))
        one_hot = torch.zeros(label_mask.size(0),  self.labels_num). \
                  to(torch.device(label.device)). \
                  scatter_(1, label, 1.0)

        numerator = -torch.sum(output * one_hot, 1)
        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        numerator = torch.sum(label_mask * numerator)
        denominator = torch.sum(label_mask) + 1e-6
        loss = numerator / denominator

        predict = output.argmax(dim=-1)

        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        )
        return loss, correct, predict, label

class BertTagger_with_LSTMCRF(nn.Module):
    def __init__(self, args, model):  # 传参传入了model
        super(BertTagger_with_LSTMCRF, self).__init__()
        self.embedding = model.embedding
        self.encoder = model.encoder
        self.target = model.target
        self.args = args
        self.need_birnn = args.need_birnn
        self.labels_num = args.labels_num
        out_dim = args.hidden_size

        # 如果为False，则不要BiLSTM层
        if self.need_birnn:
            self.birnn = nn.LSTM(args.hidden_size, args.rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = args.rnn_dim * 2

        self.output_layer = nn.Linear(out_dim, self.labels_num)
        self.dropout = nn.Dropout(args.dropout)

        self.crf = CRF(args.labels_num, batch_first=True)



    def forward(self, src, label, mask, pos=None, vm=None):
        """
        Args:
            src: [batch_size x seq_length]
            label: [batch_size x seq_length]
            mask: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            correct: Number of labels that are predicted correctly.
            predict: Predicted label.
            label: Gold label.
        example:
            src size: torch.Size([8, 128])
            output size: torch.Size([8, 128, 768])
            output size: torch.Size([8, 128, 256])
            output size: torch.Size([8, 128, 256])
            output size: torch.Size([8, 128, 15])
            output size: torch.Size([8, 128])
            output size: torch.Size([1024, 1])
            label size: torch.Size([1024, 1])
            label size: torch.Size([1024])

        """
        # Embedding.
        emb = self.embedding(src, mask, pos)
        # Encoder.
        output = self.encoder(emb, mask, vm)
        if(self.need_birnn):
            output, _ = self.birnn(output)

        # Target.
        output = self.dropout(output)

        output = self.output_layer(output)

        loss = -1*self.crf(output,label, mask=mask.byte())
        output = torch.LongTensor(np.array(self.crf.decode(output))).to(self.args.device)

        output = output.contiguous().view(-1, 1)

        label = label.contiguous().view(-1, 1)

        label_mask = (label > 0).float().to(torch.device(label.device))



        label_mask = label_mask.contiguous().view(-1)
        label = label.contiguous().view(-1)
        predict = output.contiguous().view(-1)
        correct = torch.sum(
            label_mask * (predict.eq(label)).float()
        ) #torch nb

        return loss, correct, predict, label
#
def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--pretrained_model_path", default=None, type=str,
                        help="Path of the pretrained model.")
    parser.add_argument("--output_path", default="./models/tagger_model.bin", type=str,
                        help="Path of the output model.")
    parser.add_argument("--vocab_path", default="./models/google_vocab.txt", type=str,
                        help="Path of the vocabulary file.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--dev_path", type=str, required=True,
                        help="Path of the devset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")
    parser.add_argument("--config_path", default="./models/google_config.json", type=str,
                        help="Path of the config file.")

    # Model options.
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch_size.")
    parser.add_argument("--seq_length", default=128, type=int,
                        help="Sequence length.")
    parser.add_argument("--encoder", choices=["bert", "lstm", "gru", \
                                                   "cnn", "gatedcnn", "attn", \
                                                   "rcnn", "crnn", "gpt", "bilstm"], \
                                                   default="bert", help="Encoder type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    
    # Subword options.
    parser.add_argument("--subword_type", choices=["none", "char"], default="none",
                        help="Subword feature type.")
    parser.add_argument("--sub_vocab_path", type=str, default="models/sub_vocab.txt",
                        help="Path of the subword vocabulary file.")
    parser.add_argument("--subencoder", choices=["avg", "lstm", "gru", "cnn"], default="avg",
                        help="Subencoder type.")
    parser.add_argument("--sub_layers_num", type=int, default=2, help="The number of subencoder layers.")

    # Optimizer options.
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate.")
    parser.add_argument("--warmup", type=float, default=0.1,
                        help="Warm up value.")

    # Training options.
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout.")
    parser.add_argument("--epochs_num", type=int, default=5,
                        help="Number of epochs.")
    parser.add_argument("--report_steps", type=int, default=100,
                        help="Specific steps to print prompt.")
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed.")

    # kg
    parser.add_argument("--kg_name", required=True, help="KG name or path")
    parser.add_argument("--log_file",help='记录log信息')
    parser.add_argument('--task_name',default=None,type=str)
    parser.add_argument("--mode",default='regular',type=str)
    parser.add_argument('--run_time',default=None,type=str)
    parser.add_argument("--commit_id",default=None,type=str)
    parser.add_argument("--fold_nb",default=0,type=str)
    parser.add_argument("--tensorboard_dir",default=None)

    parser.add_argument("--need_birnn",default=False,type=bool)
    parser.add_argument("--rnn_dim",default=128,type=int)
    parser.add_argument("--model_name",default='bert',type=str)
    parser.add_argument("--pku_model_name",default='default',type=str)
    parser.add_argument("--has_token",default=False)

    parser.add_argument("--do_train",default=False,type=bool)
    parser.add_argument("--do_test",default=True,type=bool)

    args = parser.parse_args()
    args.run_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    set_seed(args.seed)

    s = Save_Log(args)
    logger = init_logger(args.log_file)

    print(args)
    logger.info(args)

    os.makedirs(args.output_path,exist_ok=True)
    writer = SummaryWriter(logdir=os.path.join(args.tensorboard_dir, "eval",'{}_{}_{}_{}'.format(args.task_name,args.fold_nb,args.run_time,args.commit_id)), comment="Linear")

    labels_map = {"[PAD]": 0, "[ENT]": 1}
    begin_ids = []

    # Find tagging labels
    with open(args.train_path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                continue
            labels = line.strip().split("\t")[1].split()
            for l in labels:
                if l not in labels_map:
                    if l.startswith("B") or l.startswith("S"):
                        begin_ids.append(len(labels_map))
                    labels_map[l] = len(labels_map)
    
    print("Labels: ", labels_map)
    logger.info(labels_map)
    args.labels_num = len(labels_map)
    id2label = {labels_map[key]:key for key in labels_map}
    print("id2label:",id2label)
    logger.info(id2label)
    # Load vocabulary.
    vocab = Vocab()
    vocab.load(args.vocab_path)
    args.vocab = vocab

    # Build knowledge graph.
    if args.kg_name == 'none':
        spo_files = []
    else:
        spo_files = [args.kg_name]
    kg = KnowledgeGraph(spo_files=spo_files,pku_model_name= args.pku_model_name,predicate=False)

    # Build bert model.
    # A pseudo target is added.
    args.target = "bert"
    model = build_model(args)

    # Load or initialize parameters.
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)  
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if 'gamma' not in n and 'beta' not in n:
                p.data.normal_(0, 0.02)
    
    # Build sequence labeling model.
    if(args.model_name=='bert'):
    # model = BertTagger_with_LSTMCRF(args, model)
        model = BertTagger(args, model)
    elif(args.model_name == 'bertcrf'):
        model = BertTagger_with_LSTMCRF(args, model)
    logger.info(model)
    # print("model:",model)

    # print("model bert Tagger:",model)
    # For simplicity, we use DataParallel wrapper to use multiple GPUs.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model)

    model = model.to(device)
    args.device = device

    # Datset loader.
    def batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids):
        instances_num = input_ids.size()[0]
        for i in range(instances_num // batch_size):
            input_ids_batch = input_ids[i*batch_size: (i+1)*batch_size, :]
            label_ids_batch = label_ids[i*batch_size: (i+1)*batch_size, :]
            mask_ids_batch = mask_ids[i*batch_size: (i+1)*batch_size, :]
            pos_ids_batch = pos_ids[i*batch_size: (i+1)*batch_size, :]
            vm_ids_batch = vm_ids[i*batch_size: (i+1)*batch_size, :, :]
            tag_ids_batch = tag_ids[i*batch_size: (i+1)*batch_size, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch
        if instances_num > instances_num // batch_size * batch_size:
            input_ids_batch = input_ids[instances_num//batch_size*batch_size:, :]
            label_ids_batch = label_ids[instances_num//batch_size*batch_size:, :]
            mask_ids_batch = mask_ids[instances_num//batch_size*batch_size:, :]
            pos_ids_batch = pos_ids[instances_num//batch_size*batch_size:, :]
            vm_ids_batch = vm_ids[instances_num//batch_size*batch_size:, :, :]
            tag_ids_batch = tag_ids[instances_num//batch_size*batch_size:, :]
            yield input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch

    # Read dataset.
    def read_dataset(path):
        dataset = []
        with open(path, mode="r", encoding="utf-8") as f:
            f.readline()
            tokens, labels = [], []
            for line_id, line in enumerate(f):
                tokens, labels = line.strip().split("\t")
                # print("token:",tokens)
                # print("label:",labels)
                # print("len tokens:",len(tokens.split(' ')),"len labels:",len(labels.split(' ')))
                text = ''.join(tokens.split(" "))
                # print("len text:",len(text))
                tokens, pos, vm, tag = kg.add_knowledge_with_vm([text], add_pad=True, max_length=args.seq_length)
                tokens = tokens[0]
                # print("len2 text:",len(tokens),"len label:",len(labels))

                pos = pos[0]
                vm = vm[0].astype("bool")
                tag = tag[0]

                tokens = [vocab.get(t) for t in tokens]
                labels = [labels_map[l] for l in labels.split(" ")]
                # print("len3 text:",len(tokens),"len label:",len(labels))

                mask = [1] * len(tokens)
                # print('tokens:',tokens)
                # print("label:",labels)
                # assert len(tokens) == len(labels),(len(tokens),len(labels))

                new_labels = []
                j = 0
                for i in range(len(tokens)):
                    if tag[i] == 0 and tokens[i] != PAD_ID:
                        new_labels.append(labels[j])
                        j += 1
                    elif tag[i] == 1 and tokens[i] != PAD_ID:  # 是添加的实体
                        new_labels.append(labels_map['[ENT]'])
                    else:
                        new_labels.append(labels_map[PAD_TOKEN])

                dataset.append([tokens, new_labels, mask, pos, vm, tag])
        
        return dataset

    # Evaluation function.
    def evaluate(args,epoch, is_test):
        f1 = 0
        if is_test:
            dataset = read_dataset(args.test_path)
        else:
            dataset = read_dataset(args.dev_path)

        input_ids = torch.LongTensor([sample[0] for sample in dataset])
        label_ids = torch.LongTensor([sample[1] for sample in dataset])
        mask_ids = torch.LongTensor([sample[2] for sample in dataset])
        pos_ids = torch.LongTensor([sample[3] for sample in dataset])
        vm_ids = torch.BoolTensor([sample[4] for sample in dataset])
        tag_ids = torch.LongTensor([sample[5] for sample in dataset])

        instances_num = input_ids.size(0)
        batch_size = args.batch_size

        if is_test:
            print("Batch size: ", batch_size)
            print("The number of test instances:", instances_num)
 
        correct = 0
        gold_entities_num = 0
        pred_entities_num = 0

        by_type_correct = {}
        by_type_gold_nb = {}
        by_type_pred_nb = {}

        confusion = torch.zeros(len(labels_map), len(labels_map), dtype=torch.long)

        pred_labels = []
        gold_labels = []
        origin_tokens = []

        model.eval()

        for i, (input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):

            input_ids_batch = input_ids_batch.to(device)
            label_ids_batch = label_ids_batch.to(device)
            mask_ids_batch = mask_ids_batch.to(device)
            pos_ids_batch = pos_ids_batch.to(device)
            tag_ids_batch = tag_ids_batch.to(device)
            vm_ids_batch = vm_ids_batch.long().to(device)
            # print("batch size:",batch_size)
            loss, _, pred, gold = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)
            # print(pred.size(),gold.size())
            # print("pred:",pred)
            # print("gold:",gold)

            """
            pred: tensor([2, 2, 2,  ..., 2, 2, 2], device='cuda:0')
            gold: tensor([2, 2, 2,  ..., 0, 0, 0], device='cuda:0')

            """
            # print("input id batch:",input_ids_batch.size())
            for input_ids in input_ids_batch:
                for id in input_ids:
                    origin_tokens.append(vocab.i2w[id])
            for p,g in zip(pred,gold):

                pred_labels.append(id2label[int(p)] )
                gold_labels.append(id2label[int(g)])

            # pred_labels.append(pred)

            # gold_labels.append(gold)
            # print("pred label",pred_labels)
            # print("gold label:",gold_labels)

            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    gold_entities_num += 1
                    if(gold[j].item() not in by_type_gold_nb):
                        by_type_gold_nb[gold[j].item()] = 1
                    else:
                        by_type_gold_nb[gold[j].item()] += 1

 
            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"]:
                    pred_entities_num += 1
                    if (pred[j].item() not in by_type_pred_nb):
                        by_type_pred_nb[pred[j].item()] = 1
                    else:
                        by_type_pred_nb[pred[j].item()] += 1

            pred_entities_pos = []
            gold_entities_pos = []
            start, end = 0, 0

            for j in range(gold.size()[0]):
                if gold[j].item() in begin_ids:
                    start = j
                    type = gold[j].item()
                    # print("gold j item:",gold[j].item())

                    for k in range(j+1, gold.size()[0]):
                        
                        if gold[k].item() == labels_map['[ENT]']:
                            continue

                        if gold[k].item() == labels_map["[PAD]"] or gold[k].item() == labels_map["O"] or gold[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = gold.size()[0] - 1
                    gold_entities_pos.append((start, end,type))
            
            for j in range(pred.size()[0]):
                if pred[j].item() in begin_ids and gold[j].item() != labels_map["[PAD]"] and gold[j].item() != labels_map["[ENT]"]:
                    start = j
                    type = pred[j].item()
                    for k in range(j+1, pred.size()[0]):

                        if gold[k].item() == labels_map['[ENT]']:
                            continue

                        if pred[k].item() == labels_map["[PAD]"] or pred[k].item() == labels_map["O"] or pred[k].item() in begin_ids:
                            end = k - 1
                            break
                    else:
                        end = pred.size()[0] - 1
                    pred_entities_pos.append((start, end,type))

            for entity in pred_entities_pos:
                if entity not in gold_entities_pos:
                    continue
                else: 
                    correct += 1
                    if(entity[2] not in by_type_correct):
                        by_type_correct[entity[2]] = 1
                    else:
                        by_type_correct[entity[2]] += 1



        if(not is_test):

            print("Report precision, recall, and f1:")
            logger.info("Report precision, recall, and f1:")
            p = correct / pred_entities_num
            r = correct / gold_entities_num
            f1 = 2 * p * r / (p + r)
            logger.info("{:.3f}, {:.3f}, {:.3f}".format(p, r, f1))
            print("{:.3f}, {:.3f}, {:.3f}".format(p, r, f1))
            writer.add_scalar("Eval/precision", p, epoch)
            writer.add_scalar("Eval/recall", r, epoch)
            writer.add_scalar("Eval/f1_score", f1, epoch)

            for type in by_type_correct:
                p = by_type_correct[type] / by_type_pred_nb[type]
                r = by_type_correct[type] / by_type_gold_nb[type]
                f1 = 2 * p * r / (p + r)
                print("{}:{:.3f}, {:.3f}, {:.3f}".format(id2label[type][2:], p, r, f1))
                logger.info("{}:{:.3f}, {:.3f}, {:.3f}".format(id2label[type][2:], p, r, f1))
                writer.add_scalar("Eval/precision_{}".format(id2label[type][2:]), p, epoch)
                writer.add_scalar("Eval/recall_{}".format(id2label[type][2:]), r, epoch)
                writer.add_scalar("Eval/f1_score_{}".format(id2label[type][2:]), f1, epoch)

        with open(os.path.join(args.output_path,'pred_label_test1_{}.txt').format(is_test),'w',encoding='utf-8') as file:
            print("!!!!!!!! saving in ",os.path.join(args.output_path,'pred_label_test1_{}.txt'))
            i = 0
            while i < len(pred_labels):
                len_ = args.seq_length
                if('[PAD]' in origin_tokens[i:i+args.seq_length]):
                    len_ = origin_tokens[i:i+args.seq_length].index('[PAD]')
                file.write(' '.join(origin_tokens[i:i+len_]))
                # print("pred:",pred_labels[i:i+len_])
                file.write('\t'+' '.join(pred_labels[i:i+len_]))
                file.write('\t'+' '.join(gold_labels[i:i+len_])+'\n')

                i += args.seq_length

        return f1

    # Training phase.
    print("args train test:",args.do_train,args.do_test)
    if(args.do_train):
        print("Start training.")
        logger.info("Start training.")
        instances = read_dataset(args.train_path)

        input_ids = torch.LongTensor([ins[0] for ins in instances])
        label_ids = torch.LongTensor([ins[1] for ins in instances])
        mask_ids = torch.LongTensor([ins[2] for ins in instances])
        pos_ids = torch.LongTensor([ins[3] for ins in instances])
        vm_ids = torch.BoolTensor([ins[4] for ins in instances])
        tag_ids = torch.LongTensor([ins[5] for ins in instances])

        instances_num = input_ids.size(0)
        batch_size = args.batch_size
        train_steps = int(instances_num * args.epochs_num / batch_size) + 1

        logger.info("Batch size: {}".format(batch_size))
        print("Batch size: ", batch_size)
        print("The number of training instances:", instances_num)
        logger.info("The number of training instances:{}".format(instances_num))

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
        optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup,
                             t_total=train_steps)

        total_loss = 0.
        f1 = 0.0
        best_f1 = 0.0
        total_step = 0
        for epoch in range(1, args.epochs_num + 1):
            print("Epoch ", epoch)
            model.train()
            for i, (
            input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch, tag_ids_batch) in enumerate(
                    batch_loader(batch_size, input_ids, label_ids, mask_ids, pos_ids, vm_ids, tag_ids)):
                model.zero_grad()
                total_step += 1
                input_ids_batch = input_ids_batch.to(device)
                label_ids_batch = label_ids_batch.to(device)
                mask_ids_batch = mask_ids_batch.to(device)
                pos_ids_batch = pos_ids_batch.to(device)
                tag_ids_batch = tag_ids_batch.to(device)
                vm_ids_batch = vm_ids_batch.long().to(device)

                loss, _, _, _ = model(input_ids_batch, label_ids_batch, mask_ids_batch, pos_ids_batch, vm_ids_batch)
                if torch.cuda.device_count() > 1:
                    loss = torch.mean(loss)
                total_loss += loss.item()
                if (i + 1) % args.report_steps == 0:
                    logger.info("Epoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1,
                                                                                            total_loss / args.report_steps))

                    writer.add_scalar("Train/loss", total_loss / args.report_steps, total_step)

                    total_loss = 0.

                loss.backward()
                optimizer.step()

            # Evaluation phase.
            print("Start evaluate on dev dataset.")
            logger.info("Start evaluate on dev dataset.")
            f1 = evaluate(args, epoch, False)
            # print("Start evaluation on test dataset.")
            # evaluate(args, True)

            if f1 > best_f1:
                best_f1 = f1
                save_model(model, os.path.join(args.output_path, '{}.bin').format(args.task_name))
            else:
                continue

    if(args.do_test):
        # Evaluation phase.
        print("Final evaluation on test dataset.")
        logger.info("Final evaluation on test dataset.")
        if torch.cuda.device_count() > 1:
            model.module.load_state_dict(torch.load(os.path.join(args.output_path, "{}.bin".format(args.task_name))))
        else:
            model.load_state_dict(torch.load(os.path.join(args.output_path, "{}.bin".format(args.task_name))))

        evaluate(args, args.epochs_num, True)

        print("============over=================={}".format(args.fold_nb))

if __name__ == "__main__":
    main()
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # for i in range(0,10):
    #     os.makedirs(os.path.join(dir_path,'outputs/{}/bert_bs16'.format(i)))
    #
    pass