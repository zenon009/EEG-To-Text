echo "This scirpt construct .pickle files from .mat files from ZuCo dataset."
echo "This script also generates tenary sentiment_labels.json file for ZuCo task1-SR v1.0 and ternary_dataset.json from filtered StanfordSentimentTreebank"
echo "Note: the sentences in ZuCo task1-SR do not overlap with sentences in filtered StanfordSentimentTreebank "
echo "Note: This process can take time, please be patient..."

python3 ../util/construct_dataset_mat_to_pickle_v1.py -t task2-NR -v 2 -d "/users/wrb15144/temp_data/osfstorage-archive/" -o "/users/wrb15144/temp_data/preprocessed_zuco_1/task2"
