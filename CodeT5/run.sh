# Check if DATE is provided as a command-line argument
if [ $# -ne 1 ]; then
    echo "Usage: $0 <date> (e.g., ./run.sh 12-25-13-10PM)"
    exit 1
fi

DATE="$1"

export PYTHONPATH="/home/myoungkyu@unomaha.edu/Documents/0-research-codet5/"

nohup python run_gen.py > output-codet5-2024-$DATE.log 2>&1 &


# ======================================================
# Test Result with 1 epoch and 24 batch
# 52 minutes with 1 GPU in selab2 (RTX 4090 24GB)
# ======================================================
# (codet5) myoungkyu@oisit-selab2 √ ~/Documents/0-research-codet5/CodeT5 $ head -n 1 output-codet5-2024-06-27\=10-14pm.log 
# 06/27/2024 10:56:05 - INFO - __main__ -   Namespace(do_train=True, do_eval=True, do_eval_bleu=True, do_test=True, task='summarize', sub_task='python', model_type='codet5', data_num=-1, num_train_epochs=1, warmup_steps=1000, learning_rate=5e-05, patience=2, tokenizer_name='Salesforce/codet5-base', model_name_or_path='Salesforce/codet5-base', data_dir='/home/myoungkyu@unomaha.edu/Documents/0-research-codet5/CodeT5/data', cache_path='saved_models/summarize/python/codet5_base_all_lr5_bs24_src256_trg128_pat2_e1/cache_data', output_dir='saved_models/summarize/python/codet5_base_all_lr5_bs24_src256_trg128_pat2_e1', summary_dir='tensorboard', save_last_checkpoints=True, always_save_model=True, res_dir='saved_models/summarize/python/codet5_base_all_lr5_bs24_src256_trg128_pat2_e1/prediction', res_fn='results/summarize_codet5_base.txt', train_batch_size=24, eval_batch_size=24, max_source_length=256, max_target_length=128, lang='python', eval_task='', add_lang_ids=False, start_epoch=0, add_task_prefix=False, load_model_path=None, train_filename=None, dev_filename=None, test_filename=None, config_name='', do_lower_case=False, no_cuda=False, gradient_accumulation_steps=1, beam_size=10, weight_decay=0.0, adam_epsilon=1e-08, max_grad_norm=1.0, save_steps=-1, log_steps=-1, max_steps=-1, eval_steps=-1, train_steps=-1, local_rank=-1, seed=1234)
# (codet5) myoungkyu@oisit-selab2 √ ~/Documents/0-research-codet5/CodeT5 $ 
# (codet5) myoungkyu@oisit-selab2 √ ~/Documents/0-research-codet5/CodeT5 $ tail -n 1 output-codet5-2024-06-27\=10-14pm.log 
# 06/27/2024 11:48:50 - INFO - __main__ -   Finish and take 52m
# (codet5) myoungkyu@oisit-selab2 √ ~/Documents/0-research-codet5/CodeT5 $ 
# (codet5) myoungkyu@oisit-selab2 √ ~/Documents/0-research-codet5/CodeT5 $ 
# (codet5) myoungkyu@oisit-selab2 √ ~/Documents/0-research-codet5/CodeT5 $ tail -n 10 output-codet5-2024-06-27\=10-14pm.log 
# 06/27/2024 11:42:10 - INFO - __main__ -     Num examples = 14918
# 06/27/2024 11:42:10 - INFO - __main__ -     Batch size = 24
# Eval bleu for test set: 100%|██████████| 622/622 [06:25<00:00,  1.61it/s]
# Total: 14918
# 06/27/2024 11:48:50 - INFO - __main__ -   ***** Eval results *****
# 06/27/2024 11:48:50 - INFO - __main__ -     bleu = 20.46
# 06/27/2024 11:48:50 - INFO - __main__ -     em = 1.7831
# 06/27/2024 11:48:50 - INFO - __main__ -   [best-bleu] bleu-4: 20.46, em: 1.7831, codebleu: 0.0000

# 06/27/2024 11:48:50 - INFO - __main__ -   Finish and take 52m
