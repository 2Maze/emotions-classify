from src.train.train_func import train_func
from src.train.tune.tune_controller import start_tuning


def main():
    start_tuning()  # запустить тюнинг
    train_func(
        {
            'lr': 1e-3,
            'batch_size': 64,
            'num_workers': 1,
            'alpha': 0 * 0.25,
            "gamma": 0 * 2.0,
            "reduction": "mean",
            "from_logits": False,
        }

        # {
        #     'layer_1_size': 1024,
        #  'layer_2_size': 64,
        #  'lr': 0.008769148370041635,
        #  'batch_size': 64, 'conv_h_count': 16,
        #  'conv_w_count': 2, 'num_workers': 1,
        #  'alpha': 1, "gamma": 0.25,
        #  "reduction": "none",
        #  "from_logits": False,
        #  }

        # {
        #     'batch_size': 16,
        #     'conv_h_count': 8,
        #     'conv_w_count': 0,
        #     'layer_1_size': 512,
        #     'layer_2_size': 256,
        #     'lr': 0.0035923573027211784,
        #     'num_workers': 1
        # }
        # {'batch_size': 8, 'conv_h_count': 2, 'conv_w_count': 0,
        #  'layer_1_size': 1024, 'layer_2_size': 1024,
        #  'lr': 0.006854079146121017, 'num_workers': 1}
    )


if __name__ == '__main__':
    # check_tuning_res('tune_analyzing_results_20240416-141925')
    main()

'''
(0.8125, {'train_loop_config': {'batch_size': 16, 'conv_h_count': 8, 'conv_w_count': 0, 'layer_1_size': 512, 'layer_2_size': 256, 'lr': 0.0035923573027211784, 'num_workers': 1}})
(0.7954545617103577, {'train_loop_config': {'batch_size': 8, 'conv_h_count': 2, 'conv_w_count': 0, 'layer_1_size': 1024, 'layer_2_size': 1024, 'lr': 0.006854079146121017, 'num_workers': 1}})
(0.7954545617103577, {'train_loop_config': {'batch_size': 8, 'conv_h_count': 0, 'conv_w_count': 0, 'layer_1_size': 2048, 'layer_2_size': 256, 'lr': 0.00012928256749833867, 'num_workers': 1}})
(0.7727272510528564, {'train_loop_config': {'batch_size': 8, 'conv_h_count': 4, 'conv_w_count': 0, 'layer_1_size': 1024, 'layer_2_size': 1024, 'lr': 0.003983555045340102, 'num_workers': 1}})
(0.7613636255264282, {'train_loop_config': {'batch_size': 4, 'conv_h_count': 8, 'conv_w_count': 2, 'layer_1_size': 1024, 'layer_2_size': 64, 'lr': 0.0007337747082560383, 'num_workers': 1}})
(0.7272727489471436, {'train_loop_config': {'batch_size': 4, 'conv_h_count': 16, 'conv_w_count': 0, 'layer_1_size': 256, 'layer_2_size': 128, 'lr': 0.0013735981461098664, 'num_workers': 1}})

{'acc': 0.796875, 'epoch': 9, 'id': '55119_00068', 'layer_1_size': 1024, 'layer_2_size': 64, 'lr': 0.008769148370041635, 'batch_size': 64, 'conv_h_count': 16, 'conv_w_count': 2, 'num_workers': 1}
 {'acc': 0.796875, 'epoch': 9, 'id': '55119_00473', 'layer_1_size': 64, 'layer_2_size': 64, 'lr': 0.0036585834709462668, 'batch_size': 32, 'conv_h_count': 8, 'conv_w_count': 16, 'num_workers': 1}
{'acc': 0.796875, 'epoch': 9, 'id': '55119_00484', 'layer_1_size': 128, 'layer_2_size': 64, 'lr': 0.002930486390263172, 'batch_size': 32, 'conv_h_count': 8, 'conv_w_count': 0, 'num_workers': 1}
 {'acc': 0.7875000238418579, 'epoch': 9, 'id': '55119_00315', 'layer_1_size': 64, 'layer_2_size': 64, 'lr': 0.005135463347355345, 'batch_size': 16, 'conv_h_count': 8, 'conv_w_count': 4, 'num_workers': 1}
 {'acc': 0.78125, 'epoch': 5, 'id': '55119_00045', 'layer_1_size': 1024, 'layer_2_size': 32, 'lr': 0.004106611253125662, 'batch_size': 32, 'conv_h_count': 2, 'conv_w_count': 1, 'num_workers': 1}
 {'acc': 0.78125, 'epoch': 8, 'id': '55119_00200', 'layer_1_size': 128, 'layer_2_size': 256, 'lr': 0.003912079786909736, 'batch_size': 64, 'conv_h_count': 2, 'conv_w_count': 1, 'num_workers': 1}
 {'acc': 0.78125, 'epoch': 9, 'id': '55119_00211', 'layer_1_size': 512, 'layer_2_size': 128, 'lr': 0.0038767867780932245, 'batch_size': 32, 'conv_h_count': 2, 'conv_w_count': 0, 'num_workers': 1}
{'acc': 0.75, 'epoch': 6, 'id': '55119_00089', 'layer_1_size': 1024, 'layer_2_size': 128, 'lr': 0.01002987738490128, 'batch_size': 32, 'conv_h_count': 2, 'conv_w_count': 4, 'num_workers': 1}
 {'acc': 0.75, 'epoch': 9, 'id': '55119_00348', 'layer_1_size': 1024, 'layer_2_size': 32, 'lr': 0.0024426879784382126, 'batch_size': 16, 'conv_h_count': 8, 'conv_w_count': 8, 'num_workers': 1}
 {'acc': 0.75, 'epoch': 5, 'id': '55119_00482', 'layer_1_size': 2048, 'layer_2_size': 32, 'lr': 0.003605858112253073, 'batch_size': 16, 'conv_h_count': 2, 'conv_w_count': 0, 'num_workers': 1}

'''
