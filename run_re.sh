TASK_NAME=RE
FOLD_NB=0

CUDA_VISIBLE_DEVICES='0'   nohup python3 -u run_kbert_ner.py \
    --commit_id 5311a28\
    --task_name $TASK_NAME\
    --mode debug\
    --fold_nb $FOLD_NB\
    --pretrained_model_path ./models/bert/google_model.bin \
    --config_path ./models/bert/google_config.json \
    --vocab_path ./models/bert/google_vocab.txt \
    --seq_length 128 \
    --train_path ./datasets/RE/train.txt \
    --dev_path ./datasets/RE/valid.txt \
    --test_path ./datasets/RE/valid2.txt \
    --log_file ./outputs/RE/logs/\
    --epochs_num -1 \
    --batch_size 16  \
    --kg_name none\
    --output_path ./outputs/$TASK_NAME \
    --tensorboard_dir ./outputs/ \
    --model_name bert\
    --need_birnn False\
    > ./outputs/RE/pred.log
#done