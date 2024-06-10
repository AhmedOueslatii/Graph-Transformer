export CUDA_VISIBLE_DEVICES=0
python main.py \
--n_class 3 \
--data_path "graphs/simclr_files" \
--train_set "scripts/train_set.txt" \
--val_set "scripts/test_set.txt" \
--model_path "graph_transformer/saved_models/" \
--log_path "graph_transformer/runs/" \
--task_name "GraphCAM" \
--batch_size 3 \
--train \
--log_interval_local 6 \

