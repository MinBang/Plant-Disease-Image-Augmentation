import argparse
import pandas as pd

def get_config():
    parser = argparse.ArgumentParser()
    ## Dataset 
    parser.add_argument('--data_root', type=str, required=False)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--a_data', type=str, default='trainA')
    parser.add_argument('--b_data', type=str, default='trainB')
    parser.add_argument('--capacity', type=int, default=float("inf"))
    parser.add_argument('--unaligned', action='store_true')    
    parser.add_argument('--in_memory', action='store_true')        
    parser.add_argument('--data_swap', action='store_true')

    ## Dataset & Train
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)

    ## Logging
    parser.add_argument('--save_name', type=str, required=False)
    parser.add_argument('--start_save', type=int, default=30)
    parser.add_argument('--n_sample', type=int, default=5)
    parser.add_argument('--interval_test', type=int, default=5)
    parser.add_argument('--interval_save', type=int, default=5)
    parser.add_argument('--tensorboard', action='store_true')

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
    parser.add_argument('--load_name', type=str)
    parser.add_argument('--warmup_steps', type=int, default=3)
    parser.add_argument('--g_downsampling', type=int, default=3)
    parser.add_argument('--g_bottleneck', type=int, default=6)
    parser.add_argument('--g_tanh', action='store_false')
    parser.add_argument('--d_type', type=str, default='patch')

    parser.add_argument('--lambda_cycle', type=int, default=10)
    parser.add_argument('--lambda_idt', type=int, default=5)
    parser.add_argument('--lambda_background', type=int, default=10)

    parser.add_argument('--test', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_config()
    args_dict = vars(args)
    print(args_dict)

    df_args = pd.DataFrame.from_dict(args_dict, orient='index', columns=['option', 'value'])
    df_args.to_csv('test.csv', index=True)

