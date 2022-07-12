import argparse
import pandas as pd
import json

def get_config():
    parser = argparse.ArgumentParser()
    ## Dataset 
    parser.add_argument('-d', '--data_root', type=str, required=False)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('-a', '--a_data', type=str, default='trainA')
    parser.add_argument('-b', '--b_data', type=str, default='trainB')
    parser.add_argument('-c', '--capacity', type=int, default=1000)
    parser.add_argument('--unaligned', action='store_true')    
    parser.add_argument('-m', '--in_memory', action='store_true')        
    parser.add_argument('--data_swap', action='store_true')

    ## Dataset & Train
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('-e', '--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)

    ## Logging
    parser.add_argument('-n', '--save_name', type=str, required=False)
    parser.add_argument('--start_save', type=int, default=40)
    parser.add_argument('--n_sample', type=int, default=5)
    parser.add_argument('--interval_test', type=int, default=5)
    parser.add_argument('--interval_save', type=int, default=5)
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('-s', '--save', action='store_true')

    ## Model
    parser.add_argument('--input_nc', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--attention', action='store_false')    
    parser.add_argument('--lr_G', type=float, default=0.0002)
    parser.add_argument('--lr_D', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--buffer_size', type=int, default=50)
    parser.add_argument('--gan_loss', type=str, default='lsgan')
    parser.add_argument('-l', '--load_name', type=str)
    parser.add_argument('--warmup_steps', type=int, default=6)
    parser.add_argument('--g_downsampling', type=int, default=3)
    parser.add_argument('--g_bottleneck', type=int, default=6)
    parser.add_argument('--g_tanh', action='store_true')
    parser.add_argument('--g_type', type=str, default='pdgan', choices=['pdgan', 'cyclegan'])
    parser.add_argument('--d_type', type=str, default='pdgan', choices=['pdgan', 'cyclegan'])
    parser.add_argument('--reduction_ratio', type=int, default=16)

    parser.add_argument('--lambda_cycle', type=int, default=10)
    parser.add_argument('--lambda_idt', type=int, default=5)
    parser.add_argument('--lambda_background', type=int, default=10)

    parser.add_argument('-t', '--test', action='store_true')

    return parser.parse_args()

def read_csv(csvfile):
    print('read_csv(): type(csvfile)) = {}'.format(csvfile))
    print('')

    foo_df = pd.read_csv(csvfile)

    return foo_df

if __name__ == '__main__':
    from pprint import pprint
    args = get_config()
    pprint(args.__dict__, indent=2)

    #args.__setattr__('img_size', params['img_size'])

    
    #with open('commandline_args.txt', 'w') as f:
    #    json.dump(args.__dict__, f, indent=2)


    #with open('commandline_args.txt', 'r') as f:
    #    args.__dict__ = json.load(f)

    #print(type(args_dict['lr_G']))
    #print(args_dict)

    #df_args = pd.DataFrame.from_dict(args_dict, orient='index', columns=['value'])
    #print(df_args.head())
    #print(df_args.info())
    #df_args.to_csv('test.csv', index=True)

