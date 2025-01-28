for task in cross
do
    python -m src.data.utils make_dummy_tasks --data_dir ./data/tokenized/$task --out_dir ./data/tokenized/$task-variants

    # python -m src.data.utils make_shuffle_tasks --data_dir ./data/tokenized/$task --out_dir ./data/tokenized/$task-variants
done
