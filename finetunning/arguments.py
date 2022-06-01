import argparse


def str2bool(v):
    # arguments에 유사한 값들끼리 다 boolean 처리 되도록

    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_path", type=str, default='./configs/first_para.yaml', help="output dir")

    # basic environment args & output , logging saving
    parser.add_argument("--model_name", type=str, default='klue/roberta-large', help="output dir")
    parser.add_argument("--model_save", type=str, default='./results', help="output dir")
    parser.add_argument("--logging_steps", type=int, default=100, help="logging steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="saving limit")
    parser.add_argument("--save_steps", type=int, default=100, help="saving steps")
    parser.add_argument("--eval_steps", type=int, default=100, help="evaluation steps")
    parser.add_argument("--evaluation_strategy", type=str, default='steps', help="evaluation strategy")



    parser.add_argument("--num_train_epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--train_bs", type=int, default=32, help="batch size per device during training")
    parser.add_argument("--eval_bs", type=int, default=32, help="batch size per device during training")

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate (default:5e-5)")
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()
    breakpoint()
    print(args)