BERT_BASE_DIR=/home/yzhao/.cache/torch/transformers/bert-base-chinese/

#    --pretrained_model_path ${BERT_BASE_DIR}/pytorch_model.bin \
#    --config_path ${BERT_BASE_DIR}/config.json \
#    --vocab_path ${BERT_BASE_DIR}/vocab.txt \

CUDA_VISIBLE_DEVICES='2' nohup python3 -u run_kbert_ner.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/CCKS/train.txt \
    --dev_path ./datasets/CCKS/dev.txt \
    --test_path ./datasets/CCKS/dev.txt \
    --epochs_num 16 --batch_size 16 --kg_name CCKS \
    --output_model_path ./outputs/CCKS.bin \
    > ./outputs/kbert_CCKS.log &