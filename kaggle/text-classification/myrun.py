import sys
import run_tf_text_classification

if __name__ == "__main__":
    base_args = sys.argv
    print(base_args)

    base_args.extend([
    "--train_file", "data/train_formatted.csv",
    "--dev_file", "data/valid_formatted.csv",
    "--test_file", "data/test_formatted.csv",
    "--label_column_id", "0",
    #"--model_name_or_path", "bert-base-multilingual-uncased", 
    "--model_name_or_path", "tf_model", 
    "--output_dir", "model", 
    "--num_train_epochs", "4",
    "--per_device_train_batch_size", "16",
    "--per_device_eval_batch_size", "32",
    #"--do_train", 
    "--do_predict", 
    "--logging_steps", "119",
    "--evaluation_strategy", "steps", 
    "--save_steps", "119",
    "--overwrite_output_dir", 
    "--max_seq_length", "128"    
    ])
    run_tf_text_classification.main()