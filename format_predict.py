import json
import os
import csv

def get_entity_bio( seq):
    """Gets entities from sequence.
    note: BIO
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        get_entity_bio(seq)
        #output
        [['PER', 0,1], ['LOC', 3, 3]]
    """
    chunks = []
    chunk = [-1, -1, -1]
    for indx, tag in enumerate(seq):
        # if not isinstance(tag, str):
        #     tag = id2label[tag]
        if tag.startswith("B-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
            chunk[2] = indx
            if indx == len(seq) - 1:
                chunks.append(chunk)
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx

            if indx == len(seq) - 1:
                chunks.append(chunk)
        else:
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1]
    return chunks


# bio to csv
def bio2json(self, file, trans_path):
    with open(file) as f:
        lines = [line.strip().split("\t") for line in f.readlines()]

    label_seq = []
    token_seq = []

    texts = []
    labels = []

    for id_, line in enumerate(lines):

        if line[0] == '#' and token_seq != []:
            # print(token_seq)
            texts.append(token_seq)
            labels.append(label_seq)

            token_seq = []
            label_seq = []
            continue

        if len(line) == 2 and line[0] != '#':  # 有些奇怪的空格怎么处理
            token_seq.append(line[0])
            label_seq.append(line[-1])

    if token_seq != []:
        texts.append(token_seq)
        labels.append(label_seq)
    # "entities": [{"start_pos": 12, "end_pos": 13, "label_type": "解剖部位"},

    test_submit = []
    for id_, (token_seq, label_seq) in enumerate(zip(texts, labels)):
        # print("token seq:", token_seq)
        # print("label_seq:", label_seq)
        # cnt = 0
        # for t,l in zip(token_seq,label_seq):
        #     print(cnt,t,l)
        #     cnt += 1
        json_d = {}
        # json_d['id'] = str(id_)
        json_d['entities'] = []

        chunks = self.get_entity_bio(label_seq)
        if len(chunks) != 0:
            for subject in chunks:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                json_d['entities'].append({"start_pos": start, "end_pos": end + 1, "label_type": tag})
                # word = "".join(token_seq[start:end + 1])
                # if tag in json_d['entities']:
                #     if word in json_d['entities'][tag]:
                #         json_d['entities'][tag][word].append([start, end])
                #     else:
                #         json_d['entities'][tag][word] = [[start, end]]
                # else:
                #     json_d['entities'][tag] = {}
                #     json_d['entities'][tag][word] = [[start, end]]
        test_submit.append(json_d)

    with open(trans_path, "w") as f:
        for line in test_submit:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def tsv2json(origin_path,save_path):
    texts = []
    labels = []
    cnt = -1

    with open(origin_path, encoding='utf-8') as origin_file:
        for line_id, line in enumerate(origin_file):
            text, pred_label, gold_label = line.strip('\n').split('\t')
            text = text.split(' ')

            pred_label = pred_label.split(' ')
            gold_label = gold_label.split(' ')

            for t, p, g in zip(text, pred_label, gold_label):
                if (p == '[ENT]'):
                    continue
                if(t=='#'):
                    cnt += 1
                    texts.append([])
                    labels.append([])
                    # texts[cnt] = []
                    # labels[cnt] = []
                texts[cnt].append(t)
                labels[cnt].append(p)

    test_submit = []
    for id_, (token_seq, label_seq) in enumerate(zip(texts, labels)):
        # print("token seq:", token_seq)
        # print("label_seq:", label_seq)
        # cnt = 0
        # for t,l in zip(token_seq,label_seq):
        #     print(cnt,t,l)
        #     cnt += 1
        json_d = {}
        # json_d['id'] = str(id_)
        json_d['origin_text'] = ''.join(token_seq)
        json_d['entities'] = []

        chunks = get_entity_bio(label_seq)
        if len(chunks) != 0:
            for subject in chunks:
                tag = subject[0]
                start = subject[1]
                end = subject[2]
                json_d['entities'].append({"start_pos": start, "end_pos": end + 1, "label_type": tag})
                # word = "".join(token_seq[start:end + 1])
                # if tag in json_d['entities']:
                #     if word in json_d['entities'][tag]:
                #         json_d['entities'][tag][word].append([start, end])
                #     else:
                #         json_d['entities'][tag][word] = [[start, end]]
                # else:
                #     json_d['entities'][tag] = {}
                #     json_d['entities'][tag][word] = [[start, end]]
        test_submit.append(json_d)

    with open(save_path, "w") as f:
        for line in test_submit:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def tsv2bio(origin_path,save_path):
    with open(save_path,'w',encoding='utf-8') as file:
        with open(origin_path,encoding='utf-8') as origin_file:
            for line_id,line in enumerate(origin_file):
                text,pred_label,gold_label = line.strip('\n').split('\t')
                text = text.split(' ')

                pred_label = pred_label.split(' ')
                gold_label = gold_label.split(' ')

                # if('及' not in text):
                #     continue
                # po = text.index('及')
                # if(po != 0 and po!=len(text)):
                #     # if('解剖部位' not in gold_label[po-1] or '解剖部位' not in gold_label[po+1]):
                #     #     continue
                #     if (gold_label[po - 1] == 'O' or gold_label[po + 1]=='O'):
                #         continue
                # for t,p,g in zip(text,pred_label,gold_label):
                #     # if(p == '[ENT]'):
                #     #     continu\
                #     file.write(t+" "+g+" "+p+'\n')
                #
                # file.write('\n')


                # assert (len(text) == len(pred_label))
                # assert len(pred_label) == len(gold_label)
                for t,p,g in zip(text,pred_label,gold_label):
                    if(p == '[ENT]'):
                        continue
                    file.write(t+" "+g+" "+p+'\n')
                    # print("t:({})".format(t))
                    # print("g:({})".format(g))

                file.write('\n')
                # if(line_id > 2):
                #     break

import re
def date_match(s):
    pattern = '\d+年\d*月*\d*日*'
    date = re.search(pattern,s)
    if(date is not None):
        return date[0]
    else:
        return None

def tsv2REjson(origin_file,pred_file, save_file):
    prefix_list = {}


    schema_dict = {}
    with open('/home/yzhao/sophia/Entity-Relation-Extraction/raw_data/all_50_schemas') as schema_file:
        for line in schema_file:
            # {"object_type": "人物", "predicate": "妻子", "subject_type": "人物"}
            line = json.loads(line.strip('\n'))
            # print(line)
            schema_dict[line['predicate']] = [line["object_type"], line['subject_type']]
            prefix_list[line["object_type"]+line['predicate']+line['subject_type']] = line['predicate']

    cnt_error=0
    with open(origin_file) as origin_file:
        with open(pred_file) as pred_file:
            last_text = ''
            json_list = []
            cnt = -1
            origin_file.readline()
            for line, lo in zip(pred_file, origin_file):
                # print(line)
                # print(lo)
                text, pred_label, gold_label = line.strip('\n').split('\t')

                texto, gold_labelo = lo.strip('\n').split('\t')

                text = text.split(' ')
                pred_label = pred_label.split(' ')
                gold_label = gold_label.split(' ')
                gold_labelo = gold_labelo.split(' ')

                texto = texto.split(' ')


                texto = ''.join(texto)

                prefix = ''
                # print(texto)
                # print(gold_labelo)
                # print(gold_label)
                for id, (t, g) in enumerate(zip(texto, gold_label)):

                    if (g == '[PAD]'):
                        prefix += t
                    else:
                        break
                # print("prefix:",prefix,"new text:",texto,"==",texto[len(prefix):],'last text:',last_text)
                if (texto[len(prefix):] != last_text):
                    last_text = texto[len(prefix):]
                    # print("add line",last_text)
                    cnt += 1
                    json_list.append({})
                    json_list[cnt]['text'] = texto[len(prefix):]
                    json_list[cnt]['spo_list'] = []
                    # print(json_list[cnt])


                chunks = get_entity_bio(pred_label)
                if ('Date' in prefix and date_match(texto) is not None):
                    date = date_match(texto)
                    if(['OBJ',texto.find(date),texto.find(date)+len(date)] not in chunks):
                        chunks.append(['OBJ',texto.find(date),texto.find(date)+len(date)-1])
                if len(chunks) == 2:
                    dict_ = {}
                    for subject in chunks:
                        tag = subject[0]
                        start = subject[1]
                        end = subject[2]
                        dict_[tag] = texto[start:end + 1]
                    if ('SUB' in dict_ and 'OBJ' in dict_):
                        predicate = prefix_list[prefix]
                        sub_type = schema_dict[predicate][1]
                        obj_type = schema_dict[predicate][0]
                        # {"predicate": "作曲", "object_type": "人物", "subject_type": "歌曲", "object": "张宇", "subject": "离开"}]}
                        if({'predicate': predicate, 'object_type': obj_type, 'subject_type': sub_type,
                                              'object': dict_['OBJ'], 'subject': dict_['SUB']} not in json_list[cnt]['spo_list']):
                            json_list[cnt]['spo_list'].append({'predicate': predicate, 'object_type': obj_type, 'subject_type': sub_type,
                                              'object': dict_['OBJ'], 'subject': dict_['SUB']})
                    else:
                        print(prefix,last_text)
                        print("sub not in or obj not in",chunks)

                        for chunk in chunks:
                            print(chunk[0], texto[chunk[1]:chunk[2] + 1], end='\t')
                        print()

                        cnt_error += 1
                        pass
                else:
                    if len(chunks) < 2:
                        print('nb!=2', prefix, "  ", last_text, end='\t')
                        for chunk in chunks:
                            print(chunk[0], texto[chunk[1]:chunk[2] + 1], end='\t')
                        print()


                    sub_nb = 0
                    obj_nb = 0
                    sub_list = []
                    obj_list = []

                    for chunk in chunks:
                        sub_nb += chunk[0] == 'SUB'
                        obj_nb += chunk[0] == 'OBJ'
                        if (chunk[0] == 'SUB'):
                            sub_list.append(texto[chunk[1]:chunk[2] + 1])
                        elif (chunk[0] == 'OBJ'):
                            obj_list.append(texto[chunk[1]:chunk[2] + 1])

                    if (sub_nb == 1):

                        predicate = prefix_list[prefix]
                        sub_type = schema_dict[predicate][1]
                        obj_type = schema_dict[predicate][0]
                        sub_mention = sub_list[0]

                        for obj_mention in obj_list:
                            if({'predicate': predicate, 'object_type': obj_type, 'subject_type': sub_type,
                                 'object': obj_mention, 'subject': sub_mention} not in json_list[cnt]['spo_list']):
                                json_list[cnt]['spo_list'].append(
                                {'predicate': predicate, 'object_type': obj_type, 'subject_type': sub_type,
                                 'object': obj_mention, 'subject': sub_mention})

                            pass
                    elif (obj_nb == 1):
                        predicate = prefix_list[prefix]
                        sub_type = schema_dict[predicate][1]
                        obj_type = schema_dict[predicate][0]
                        obj_mention = obj_list[0]

                        for sub_mention in sub_list:
                            if({'predicate': predicate, 'object_type': obj_type, 'subject_type': sub_type,
                                 'object': obj_mention, 'subject': sub_mention} not in json_list[cnt]['spo_list']):
                                json_list[cnt]['spo_list'].append(
                                {'predicate': predicate, 'object_type': obj_type, 'subject_type': sub_type,
                                 'object': obj_mention, 'subject': sub_mention})
                    elif(len(sub_list)==len(obj_list)):
                        predicate = prefix_list[prefix]
                        sub_type = schema_dict[predicate][1]
                        obj_type = schema_dict[predicate][0]

                        for sub_mention,obj_mention in zip(sub_list,obj_list):
                            if ({'predicate': predicate, 'object_type': obj_type, 'subject_type': sub_type,
                                 'object': obj_mention, 'subject': sub_mention} not in json_list[cnt]['spo_list']):
                                json_list[cnt]['spo_list'].append(
                                    {'predicate': predicate, 'object_type': obj_type, 'subject_type': sub_type,
                                     'object': obj_mention, 'subject': sub_mention})



                    else:
                        print(prefix,last_text)
                        print("sublist:",sub_list)
                        print("objlist:",obj_list)

                    pass


                    cnt_error += 1

    with open(save_file,'w') as save_file:
        for line in json_list:
            save_file.write(json.dumps(line,ensure_ascii=False)+'\n')

    print("cnt error:",cnt_error)



def validate(origin_file,pred_file):
    origin_lines = []
    with open(origin_file) as origin_file:
        for lineid,line in enumerate(origin_file):
            origin_lines.append(line.strip('\n'))

    pred_lines = []
    with open(pred_file) as pred_file:
        for line in pred_file:
            pred_lines.append(line.strip('\n'))


    assert len(origin_lines) == len(pred_lines),(len(origin_lines),len(pred_lines))
    tp = 0
    all_p = 0
    all_t = 0
    tp_predicate = 0
    tp_seq = 0

    for example_id,(o,p) in enumerate(zip(origin_lines,pred_lines)):
        ospo = json.loads(o)['spo_list']
        pspo = json.loads(p)['spo_list']
        all_p+=len(pspo)
        all_t+=len(ospo)
        for psp in pspo:
            if(psp in ospo):
                tp+=1

    r = tp/all_t
    p = tp/all_p
    f = 2*r*p/(r+p)
    print("recall:",r,"precision:",p,"f:",f)

    # r = tp_predicate/all_t
    # p = tp_predicate/all_p
    # f = 2*r*p/(r+p)
    # print("recall:",r,"precision:",p,"f:",f)


def get_total_triple():

    root_path = '/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data'
    triple_path = '/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/all_triple.txt'
    node_path = '/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/node.csv'
    relation_path = '/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/relatin.csv'
    corpuses = ['train','dev']
    # w.writerow(("id:ID","name",":LABEL"))
    # w.writerow((":START_ID", ":END_ID", ":TYPE", "name"))
    csvnode = open(node_path,'w',newline='',encoding = 'utf-8')
    wnode = csv.writer(csvnode)
    wnode.writerow(("id:ID","name",":LABEL"))

    csvrelation = open(relation_path,'w',newline='',encoding = 'utf-8')
    wrelation = csv.writer(csvrelation)
    wrelation.writerow((":START_ID", ":END_ID", ":TYPE", "name"))
    idx_entity = 0
    entity_dic = dict()

    with open(triple_path,'w') as triple:
        for corpus in corpuses:
            with open(os.path.join(root_path, "{}_data.json".format(corpus))) as file:
                for line in file:
                    line_dic = json.loads(line.strip('\n'))
                    spolist = line_dic['spo_list']
                    for spo in spolist:
                        triple.write(spo['subject'] + '\t' + spo['predicate'] + '\t' + spo['object'] + '\n')

                    if(spo['subject_type'] not in entity_dic):
                        entity_dic[spo['subject_type']] = idx_entity
                        wnode.writerow((str(idx_entity), spo['subject_type'], 'Entity'))

                        idx_entity += 1

                    if (spo['object_type'] not in entity_dic):
                        entity_dic[spo['object_type']] = idx_entity
                        wnode.writerow((str(idx_entity), spo['object_type'], 'Entity'))
                        idx_entity += 1


                    if (spo['subject'] not in entity_dic):
                        entity_dic[spo['subject']] = idx_entity
                        wnode.writerow((str(idx_entity), spo['subject'], 'Entity'))
                        idx_entity += 1
                        wrelation.writerow((str(entity_dic[spo['subject']]), str(entity_dic[spo['subject_type']]), 'Relation',
                                            '类型'))

                    if (spo['object'] not in entity_dic):
                        entity_dic[spo['object']] = idx_entity
                        wnode.writerow((str(idx_entity), spo['object'], 'Entity'))
                        idx_entity += 1
                        wrelation.writerow((str(entity_dic[spo['object']]), str(entity_dic[spo['object_type']]), 'Relation',
                                            '类型'))

                    wrelation.writerow((str(entity_dic[spo['subject']]), str(entity_dic[spo['object']]), 'Relation', spo['predicate']))

    pass

def pred_question():
    prefix_list = {}

    origin_file = '/home/yzhao/sophia/K-BERT/datasets/CKBQA_data/valid_for_ner.txt'
    pred_file = '/home/yzhao/sophia/K-BERT/outputs/CKBQA/pred_label_test_True.txt'
    save_file = '/home/yzhao/sophia/K-BERT/outputs/CKBQA/pred_label_test_True.json'
    with open(origin_file) as origin_file:
        with open(pred_file) as pred_file:
            last_text = ''
            json_list = []
            cnt = -1
            origin_file.readline()
            for line, lo in zip(pred_file, origin_file):
                # print(line)
                # print(lo)
                text, pred_label, gold_label = line.strip('\n').split('\t')

                texto, gold_labelo = lo.strip('\n').split('\t')

                text = text.split(' ')
                pred_label = pred_label.split(' ')
                gold_label = gold_label.split(' ')
                gold_labelo = gold_labelo.split(' ')

                texto = texto.split(' ')

                # assert len(text) == len(texto),(len(text),len(texto),text,texto)

                texto = ''.join(texto)

                # print(texto)
                # print(gold_labelo)
                # print(gold_label)

                cnt += 1
                json_list.append({})
                json_list[cnt]['text'] = texto
                json_list[cnt]['spo_list'] = []

                    # print(json_list[cnt])

                chunks = get_entity_bio(pred_label)
                for subject in chunks:
                    tag = subject[0]
                    start = subject[1]
                    end = subject[2]
                    json_list[cnt]['spo_list'].append(
                        {'tag': tag, 'mention': texto[start:end + 1]})





    with open(save_file, 'w') as save_file:
        for line in json_list:
            save_file.write(json.dumps(line, ensure_ascii=False) + '\n')





import pandas as pd
if __name__ == '__main__':
    #
    # # tsv2REjson('/home/yzhao/sophia/K-BERT/datasets/RE/pred_valid.txt',
    # #            '/home/yzhao/sophia/K-BERT/outputs/RE/pred_label_test_True.txt',
    # #            '/home/yzhao/sophia/K-BERT/outputs/RE/pred_label_test_True.json')
    #
    # # validate('/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/dev_data.json',
    # #          '/home/yzhao/sophia/K-BERT/outputs/RE/pred_label_test_True.json')
    #
    # # get_total_triple()f
    #
    # relation_path = '/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/relatin.csv'
    # relation = pd.read_csv(relation_path)
    # # print(len(relation))
    # #
    # # relation = relation.drop_duplicates()
    # # print(len(relation))
    # # relation.to_csv('/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/relation.csv')
    # node_path = '/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/node.csv'
    # node = pd.read_csv(node_path)
    # # print(len(node))
    # #
    # # node = node.drop_duplicates()
    # # print(len(node))
    # #:START_ID :END_ID :TYPE name
    # entity_dict = {}
    # #"id:ID","name",":LABEL"
    # for id,name in zip(node['id:ID'],node['name']):
    #     entity_dict[id] = name
    # for s,e,t,n in zip(relation[':START_ID'],relation[':END_ID'],relation[':TYPE'],relation['name']):
    #     if(s == 2 or e == 2):
    #         print(s,e,t,n,entity_dict[s],entity_dict[e])

    # pred_question()

    # get_total_triple()d
    pred_file = '/home/yzhao/sophia/K-BERT/outputs/RE/task1_RE/pred_label_test_True.txt'
    origin_file = '/home/yzhao/sophia/K-BERT/datasets/RE/pred_valid.txt'
    save_file = '/home/yzhao/sophia/K-BERT/outputs/RE/task1_RE/pred_label_test_True.json'
    tsv2REjson(origin_file,pred_file,save_file)

    # origin_json_file = '/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/dev_data.json'

    origin_file = '/home/yzhao/sophia/Entity-Relation-Extraction/total_raw_data/dev_data.json'
    validate(origin_file,save_file)

