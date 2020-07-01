
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

                if(pred_label == gold_label):
                    print("-==")
                    continue
                assert (len(text) == len(pred_label))
                assert len(pred_label) == len(gold_label)
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
    tsv2bio('/home/yzhao/sophia/K-BERT/outputs/0/pred_label_test_True.txt',
            '/home/yzhao/sophia/K-BERT/outputs/0/bio_pred_label_test_True_debug.txt')
    pass