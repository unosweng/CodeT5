#exp_with_args.sh
import os
import argparse
import subprocess

WORKDIR = "/home/myoungkyu@unomaha.edu/Documents/0-research-codet5/CodeT5"
PY_DIR = "/home/myoungkyu@unomaha.edu/Documents/0-research-codet5/"
os.environ["PYTHONPATH"] = PY_DIR

def main():

# --do_train 
# --do_eval 
# --do_eval_bleu 
# --do_test 
# --task summarize 

# --sub_task python 
# --model_type codet5 
# --data_num -1 
# --num_train_epochs 15 
# --warmup_steps 1000 

# --learning_rate 5e-5 
# --patience 2 
# --tokenizer_name=Salesforce/codet5-base 
# --model_name_or_path=Salesforce/codet5-base 
# --data_dir /home/myoungkyu@unomaha.edu/Documents/0-research-codet5/CodeT5/data 

# --cache_path saved_models/summarize/python/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15/cache_data 
# --output_dir saved_models/summarize/python/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15 
# --summary_dir tensorboard 
# --save_last_checkpoints 
# --always_save_model 

# --res_dir saved_models/summarize/python/codet5_base_all_lr5_bs48_src256_trg128_pat2_e15/prediction 
# --res_fn results/summarize_codet5_base.txt 
# --train_batch_size 48 
# --eval_batch_size 48 
# --max_source_length 256 

# --max_target_length 128"


    parser = argparse.ArgumentParser(description="Script to set up and run a machine learning task.")
    parser.add_argument('--do_train', action='store_true', default=True, help="Flag to indicate training")
    parser.add_argument('--do_eval', action='store_true', default=True, help="Flag to indicate evaluation")
    parser.add_argument('--do_eval_bleu', action='store_true', default=True, help="Flag to indicate BLEU evaluation")
    parser.add_argument('--do_test', action='store_true', default=True, help="Flag to indicate testing")
    parser.add_argument('--task', type=str, default="summarize", help="Task name")

    parser.add_argument('--sub_task', type=str, default="python", help="Sub task name")
    # if args.model_tag == 'codet5_base':
    parser.add_argument('--model_type', type=str, default="codet5", help="Model type")
    # parser.add_argument('--model_tag', type=str, help="Model tag", required=True)
    parser.add_argument('--data_num', type=int, default=-1, help="Data number")
    parser.add_argument('--num_train_epochs', type=int, default=15, help="Number of training epochs")
    parser.add_argument('--warmup_steps', type=int, default=1000, help="Number of warmup steps")

    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--patience', type=int, default=2, help="Patience for early stopping")
    # if args.model_tag == 'codet5_base':
    parser.add_argument('--tokenizer_name', type=str, default="Salesforce/codet5-base", help="Tokenizer name")
    parser.add_argument('--model_name_or_path', type=str, default="Salesforce/codet5-base", help="Model name or path")
    parser.add_argument('--data_dir', type=str, default=f"{WORKDIR}/data", help="Data directory")

    # should be changed for different experiments.
    data_tag = 'all'
    model_tag = 'codet5_base'
    model_dir = 'saved_models'
    task = 'summarize'
    sub_task = 'python'

    # if args.data_num == -1:
    #     data_tag = 'all'
    # else:
    #     data_tag = str(args.data_num)
    #     args.epoch = 1

    lr = 5 #str(args.learning_rate)[0]
    bs = 48 #args.train_batch_size
    src_len = 256 #args.max_source_length
    trg_len = 128 #args.max_target_length
    patience = 2 #args.patience
    epoch = 15 #args.num_train_epochs

    # if args.task == 'multi_task':
    #     full_model_tag = f"{model_tag}_{data_tag}_lr{args.lr}_s{args.max_steps}"
    # else:
    full_model_tag = f"{model_tag}_{data_tag}_lr{lr}_bs{bs}_src{src_len}_trg{trg_len}_pat{patience}_e{epoch}"

    # if args.sub_task == 'none':
    #     output_dir = os.path.join(model_dir, task, full_model_tag)
    # else:
    output_dir = os.path.join(model_dir, task, sub_task, full_model_tag)
    cache_dir = os.path.join(output_dir, "cache_data")
    res_dir = os.path.join(output_dir, "prediction")
    log = os.path.join(output_dir, "train.log")

    parser.add_argument('--cache_path', type=str, default=f"{cache_dir}", help="Cache directory")
    parser.add_argument('--output_dir', type=str, default=f"{output_dir}", help="Output directory")
    parser.add_argument('--summary_dir', type=str, default="tensorboard", help="Summary directory")
    parser.add_argument('--save_last_checkpoints', action='store_true', default=True, help="Flag to save last checkpoints")
    parser.add_argument('--always_save_model', action='store_true', default=True, help="Flag to always save model")

    parser.add_argument('--res_dir', type=str, default=f"{res_dir}", help="Result directory")
    res_fn='{}/{}_{}.txt'.format('results', task, model_tag)
    parser.add_argument('--res_fn', type=str, default=f"{res_fn}", help="Result filename")
    parser.add_argument('--train_batch_size', type=int, default=48, help="Training batch size")
    parser.add_argument('--eval_batch_size', type=int, default=48, help="Evaluation batch size")
    parser.add_argument('--max_source_length', type=int, default=256, help="Maximum source length")
    parser.add_argument('--max_target_length', type=int, default=128, help="Maximum target length")

    # parser.add_argument("--gpu", type=int, default=0, help='index of the gpu to use in a cluster')
    # parser.add_argument("--model_dir", type=str, default='saved_models', help='directory to save fine-tuned models')

    args = parser.parse_args()

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # next, run followings
    # CUDA_VISIBLE_DEVICES={args.gpu}
    # main function of run_gen.py

###############################
    # # Optional arguments for multi_task
    # parser.add_argument('--max_steps', type=int, help="Max steps for multi-task", default=None)
    # parser.add_argument('--save_steps', type=int, help="Save steps for multi-task", default=None)
    # parser.add_argument('--log_steps', type=int, help="Log steps for multi-task", default=None)
#################################


    # if args.model_tag == 'roberta':
    #     model_type = 'roberta'
    #     tokenizer = 'roberta-base'
    #     model_path = 'roberta-base'
    # elif args.model_tag == 'codebert':
    #     model_type = 'roberta'
    #     tokenizer = 'roberta-base'
    #     model_path = 'microsoft/codebert-base'
    # elif args.model_tag == 'bart_base':
    #     model_type = 'bart'
    #     tokenizer = 'facebook/bart-base'
    #     model_path = 'facebook/bart-base'
    # elif args.model_tag == 'codet5_small':
    #     model_type = 'codet5'
    #     tokenizer = 'Salesforce/codet5-small'
    #     model_path = 'Salesforce/codet5-small'
    # elif args.model_tag == 'codet5_base':
    #     model_type = 'codet5'
    #     tokenizer = 'Salesforce/codet5-base'
    #     model_path = 'Salesforce/codet5-base'
    # elif args.model_tag == 'codet5_large':
    #     model_type = 'codet5'
    #     tokenizer = 'Salesforce/codet5-large'
    #     model_path = 'Salesforce/codet5-large'

    # if args.task == 'multi_task':
    #     run_fn = os.path.join(WORKDIR, "run_multi_gen.py")
    #     multi_task_aug = f"--max_steps {args.max_steps} --save_steps {args.save_steps} --log_steps {args.log_steps}"
    # elif args.task == 'clone':
    #     run_fn = os.path.join(WORKDIR, "run_clone.py")
    #     multi_task_aug = ''
    # elif args.task == 'defect' and model_type in ['roberta', 'bart']:
    #     run_fn = os.path.join(WORKDIR, "run_defect.py")
    #     multi_task_aug = ''
    # else:
    #     run_fn = os.path.join(WORKDIR, "run_gen.py")
    #     multi_task_aug = ''

    # command = f"""
    # CUDA_VISIBLE_DEVICES={args.gpu} \\
    # python {run_fn} {multi_task_aug} \\
    # --do_train --do_eval --do_eval_bleu --do_test \\
    # --task {args.task} --sub_task {args.sub_task} --model_type {model_type} --data_num {args.data_num} \\
    # --num_train_epochs {args.epoch} --warmup_steps {args.warmup} --learning_rate {args.lr}e-5 --patience {args.patience} \\
    # --tokenizer_name={tokenizer} --model_name_or_path={model_path} --data_dir {WORKDIR}/data \\
    # --cache_path {cache_dir} --output_dir {output_dir} --summary_dir {args.summary_dir} \\
    # --save_last_checkpoints --always_save_model --res_dir {res_dir} --res_fn {args.res_fn} \\
    # --train_batch_size {args.bs} --eval_batch_size {args.bs} --max_source_length {args.src_len} --max_target_length {args.trg_len} \\
    # 2>&1 | tee {log}
    # """

    # print(command)

    # subprocess.run(command, shell=True, check=True)

if __name__ == "__main__":
    main()
 