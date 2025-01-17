torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d ../train_decoding_parallel_traditional.py --model_name BrainTranslator \
    --task_name task2 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 5 \
    --num_epoch_step2 5 \
    -lr1 0.00005 \
    -lr2 0.0000005 \
    -b 8 \
    -cuda cuda \
    --dataset_path_task1 /vol/bulkdata/wrb15144/encoder_decoder/preprocessed_zuco_1/task1/task1-SR-dataset.json \
    --dataset_path_task2 /vol/bulkdata/wrb15144/encoder_decoder/preprocessed_zuco_1/task2/task2-NR-dataset.json \
    --dataset_path_task3 /vol/bulkdata/wrb15144/encoder_decoder/preprocessed_zuco_1/task3/task3-TSR-dataset.json \
    -s "./"


