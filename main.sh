BERT_BASE_DIR=/home/yzhao/chinese_roberta_wwm_ext_pytorch/
for (( FOLD_NB = 0; FOLD_NB <= 9; FOLD_NB++ ))
do

BERT_BASE_DIR=/home/yzhao/.cache/torch/transformers/bert-base-chinese/

#    --pretrained_model_path ${BERT_BASE_DIR}/pytorch_model.bin \
#    --config_path ${BERT_BASE_DIR}/config.json \
#    --vocab_path ${BERT_BASE_DIR}/v ocab.txt \
FOLD_NB=9
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_kbert_ner.py \
    --commit_id 0421c92\
    --task_name subtask1\
    --mode debug\
    --fold_nb $FOLD_NB\
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --seq_length 128 \
    --train_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_msl_126/train.txt \
    --dev_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_msl_126/dev.txt \
    --test_path ./datasets/CCKS/subtask1/test_kbert_126.txt \
    --log_file ./outputs/$FOLD_NB/logs/\
    --epochs_num 16 --batch_size 32 --kg_name CCKS \
    --output_path ./outputs/$FOLD_NB/roberta/ \
    --tensorboard_dir ./outputs/$FOLD_NB/ \
#    > ./outputs/$FOLD_NB/roberta/kbert_CCKS.log &

done