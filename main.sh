BERT_BASE_DIR=/home/yzhao/chinese_roberta_wwm_ext_pytorch/
for FOLD_NB in {1..9}
do

CUDA_VISIBLE_DEVICES='0' python3 -u run_kbert_ner.py \
    --commit_id d5f9bb5\
    --task_name subtask1\
    --mode debug\
    --fold_nb $FOLD_NB\
    --pretrained_model_path ./models/roberta/google_model.bin \
    --config_path ./models/bert/google_config.json \
    --vocab_path ./models/roberta/vocab.txt \
    --seq_length 128 \
    --train_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_msl_126/train.txt \
    --dev_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_msl_126/dev.txt \
    --test_path ./datasets/CCKS/subtask1/test_kbert_126.txt \
    --log_file ./outputs/$FOLD_NB/logs/\
    --epochs_num 16 --batch_size 32 --kg_name CCKS \
    --output_path ./outputs/$FOLD_NB/roberta/ \
    --tensorboard_dir ./outputs/$FOLD_NB/ \

done