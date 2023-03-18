cd transformers/examples/pytorch/question-answering/

python run_qa.py \
  --model_name_or_path /bigscience/bloom-560m \
  --dataset_name dataset \
  --do_train \
  --do_eval \
  --per_device_train_batch_size 6 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /output/model_train01/ \
  --eval_accumulation_steps 1 \
  --version_2_with_negative \
  --overwrite_output_dir \
  --fp16