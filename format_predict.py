import json

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


#


if __name__ == '__main__':
    for i in range(2,10):
        tsv2bio('/home/yzhao/sophia/K-BERT/outputs/{}/pred_label_test_True.txt'.format(i),
            '/home/yzhao/sophia/K-BERT/outputs/{}/bio_pred_label_test_True.txt'.format(i))
    pass