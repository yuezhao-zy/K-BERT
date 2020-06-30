BERT_BASE_DIR=/home/yzhao/.cache/torch/transformers/bert-base-chinese/
for FOLD_NB in {1..9}
do

#    --pretrained_model_path ${BERT_BASE_DIR}/pytorch_model.bin \
#    --config_path ${BERT_BASE_DIR}/config.json \
#    --vocab_path ${BERT_BASE_DIR}/vocab.txt \

CUDA_VISIBLE_DEVICES='3' nohup python3 -u run_kbert_ner.py \
    --commit_id 3dc5937\
    --task_name subtask1\
    --mode debug\
    --fold_nb 0\
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --seq_length 128 \
    --train_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_msl_126/train.txt \
    --dev_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_msl_126/dev.txt \
    --test_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_msl_126/dev.txt \
    --log_file ./outputs/$FOLD_NB/\
    --epochs_num 16 --batch_size 16 --kg_name CCKS \
    --output_path ./outputs/$FOLD_NB/ \
    --tensorboard_dir ./outputs/$FOLD_NB/ \
    > ./outputs/$FOLD_NB/kbert_CCKS.log &

done