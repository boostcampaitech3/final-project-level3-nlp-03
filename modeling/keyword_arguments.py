import argparse

def get_args():
    parser = argparse.ArgumentParser()
    # basic environment args & output , logging saving
    parser.add_argument("--sentence_data_path", type=str, default='./data', help="sentence data path")
    parser.add_argument("--sentence_file_name", type=str, default='', help="sentence file name")
    parser.add_argument("--keyword_data_path", type=str, default='', help="keyword data path")
    parser.add_argument("--keyword_file_name", type=str, default='', help="keyword file name")
    parser.add_argument("--model_name", type=str, default='fast_text_ko', help="model name")
    return args

if __name__=='__main__':
    args = get_args()
    breakpoint()
    print(args)