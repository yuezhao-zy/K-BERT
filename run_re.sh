TASK_NAME=bertlstmcrf
FOLD_NB=0
CUDA_VISIBLE_DEVICES='3'   nohup python3 -u run_kbert_ner.py \
    --commit_id 5311a28\
    --task_name $TASK_NAME\
    --mode debug\
    --fold_nb $FOLD_NB\
    --pretrained_model_path ./models/bert/google_model.bin \
    --config_path ./models/bert/google_config.json \
    --vocab_path ./models/bert/google_vocab.txt \
    --seq_length 128 \
    --train_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_dict_msl_126/train.txt \
    --dev_path ./datasets/CCKS/subtask1/k_fold/$FOLD_NB/kbert_dict_msl_126/dev.txt \
    --test_path ./datasets/CCKS/subtask1/test_kbert_dict_126.txt \
    --log_file ./outputs/$FOLD_NB/logs/\
    --epochs_num 16 --batch_size 16 --kg_name CCKS \
    --output_path ./outputs/$FOLD_NB/$TASK_NAME \
    --tensorboard_dir ./outputs/$FOLD_NB/ \
    --model_name bertcrf\
    --need_birnn True\
    > ./outputs/$FOLD_NB/$TASK_NAME/$TASK_NAME.log
#done