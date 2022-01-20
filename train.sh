pip install -r code/requirements.txt
pip install ./code/fused_ops-0.1-cp36-cp36m-linux_x86_64.whl --force-reinstall
CUDA_VISIBLE_DEVICES=0,1 python ./code/train.py \
--ofrecord_path sample_seq_len_512_example \
--lr 1e-4 \
--epochs 10 \
--train_batch_size 2 \
--seq_length=512 \
--max_predictions_per_seq=80 \
--num_hidden_layers=24 \
--num_attention_heads=16 \
--hidden_size=2080 \
--vocab_size=30522 \
--dataset_size=1024 \
--profile=0 \
--pipeline_on=1 \
--check_correctness=0 \
--fuse_ops=1 \
--load_params=0 \
--accum_gradient=0 \
--sparse_embedding=1 \
--grad_update=1


