
def tsv2bio(origin_path,save_path):
    with open(save_path,'w',encoding='utf-8') as file:
        with open(origin_path,encoding='utf-8') as origin_file:
            for line in origin_file:
                text,pred_label,gold_label = line.split('\t')
                text = text.split(' ')
                pred_label = pred_label.split(' ')
                gold_label = gold_label.split(' ')
                for t,p,g in zip(text,pred_label,gold_label):
                    if(p == '[ENT]'):
                        continue
                    file.write(t+" "+p+" "+g+'\n')

                file.write('\n')


#


if __name__ == '__main__':
    tsv2bio('/home/yzhao/sophia/K-BERT/outputs/0/pred_label_test_True.txt',
            '/home/yzhao/sophia/K-BERT/outputs/0/bio_pred_label_test_True.txt')
    pass