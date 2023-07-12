echo "This scirpt construct .pickle files from .mat files from ZuCo dataset."
echo "This script also generates tenary sentiment_labels.json file for ZuCo task1-SR v1.0 and ternary_dataset.json from filtered StanfordSentimentTreebank"
echo "Note: the sentences in ZuCo task1-SR do not overlap with sentences in filtered StanfordSentimentTreebank "
echo "Note: This process can take time, please be patient..."

python ../util/construct_dataset_mat_to_pickle_v1.py -t task1-SR -v "v2" -d "/users/wrb15144/temp_data/osfstorage-archive/" -o "/users/wrb15144/temp_data/preprocessed_zuco_1/task2" -m "tehr09"
