python3 ../train_decoding.py --model_name BrainTranslator \
    --task_name task1_task2_task3 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 20 \
    --num_epoch_step2 30 \
    -lr1 0.00005 \
    -lr2 0.0000005 \
    -b 32 \
    -cuda cuda:0 \
    --dataset_path_task1 /users/wrb15144/temp_data/preprocessed_zuco_1/task1 \
    --dataset_path_task2 /users/wrb15144/temp_data/preprocessed_zuco_1/task2 \
    --dataset_path_task3 /users/wrb15144/temp_data/preprocessed_zuco_1/task3 \


