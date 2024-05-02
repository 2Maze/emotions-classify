import json
import os
from datetime import datetime

from train.train_func import train_func
from train.tune.tune_controller import start_tuning
from data_controller.emotion_dataset import EmotionSpectrogramDataset
from config.constants import ROOT_DIR, PADDING_SEC
from utils.tuning_res_viewer import check_tuning_res
from utils.cli_argparce import parse_args
import sys

def main():
    # start_tuning()  # запустить тюнинг
    # saved_checkpoint = join(
    #     ROOT_DIR, "tmp", "tune",
    #     "tune_analyzing_results_20240425-101338",
    #     "TorchTrainer_55119_00068_68_batch_size=64,conv_h_count=16,conv_w_count=2,layer_1_size=1024,layer_2_size=64,lr=0.0088_2024-04-16_14-19-27",
    #     "checkpoint_000009",
    #     "checkpoint.ckpt")
    train_func(
        {'lr': 0.0007505403591860749, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0,
         'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False,
         'conv_lr': 0.001, 'layer_1_size': 32, 'layer_2_size': 256, 'patch_transformer_size': 64,
         'transformer_depth': 2, 'transformer_attantion_head_count': 8}

        | {'is_tune': False,
           'enable_tune_features': False, },
        saved_checkpoint=None
    )
    #     {
    #         'lr': 1e-3,
    #         'batch_size': 16,
    #         'num_workers': 1,
    #         'alpha': 0 * 0.25,
    #         "gamma": 0 * 2.0,
    #         "reduction": "mean",
    #         "from_logits": False,
    #         "padding_sec": PADDING_SEC,
    #         "is_tune": False,
    #         "conv_lr": 1e-3,
    #         'layer_1_size': 2048,
    #         'layer_2_size': 1024,
    #         'patch_transformer_size': 16,
    #         'transformer_depth': 6,
    #         'transformer_attantion_head_count': 16
    #         "enable_tune_features": False,
    #     }

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
    # )
def config_builder(config: dict | list | str,  parent_config, __deep=0, __rules=None):
    rules = __rules or  {
        "$$ROOT_DIR$$": ROOT_DIR,
        "$$DATETIME$$": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "$$MODEL_NAME$$": config.get("model"),
    }


    if isinstance(config, list):
        for index, i in enumerate(config[:]):
            config[index] = config_builder(config[index], parent_config, __deep=__deep+1, __rules=rules)
    if isinstance(config, dict):
        for k, v in config.items():
            config[k] = config_builder(v, parent_config, __deep=__deep+1, __rules=rules)
        if __deep == 0:
            config['load_dataset_workers_num'] = parent_config['load_dataset_workers_num']
            # config |= config.get('model_architecture', dict())
            # config |= config.get('learn_params', dict())
            # config |= {k: os.path.join(v) for k, v in config.get('saving_data_params', dict()).items()}
            # config |= config.get('tune', dict())
            # config.pop('model_architecture')
            # config.pop('learn_params')
            # config.pop('saving_data_params')
            # config.pop('tune')
    if isinstance(config, str):
        config = rules.get(config, config)
    return config



if __name__ == '__main__':
    # print(sys.argv)
    results = parse_args()
    print(results)
    if results.parent_param == 'train':
        with open( results.config_file, 'r') as f:
            config = json.load(f)
            for conf in config['pipeline']:
                conf = config_builder(conf, config)
                if conf['type'] == 'tune':
                    start_tuning(conf)
                elif conf['type'] == 'train':
                    train_func(conf)
                elif conf['type'] == 'print_tune_res':
                    check_tuning_res(os.path.join(*conf['res_path']))
                print(conf)


    # check_tuning_res('tune_analyzing_results_20240425-101338')
    # main()

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


Трансформеры
{'acc': 0.5625, 'epoch': 15, 'id': 'ce66c_00214', 'lr': 0.0002906524622802242, 'batch_size': 16, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 512, 'layer_2_size': 128, 'patch_transformer_size': 16, 'transformer_depth': 2, 'transformer_attantion_head_count': 4, 'patch_size': 32}
2024-04-25T07:09:20.518306251Z {'acc': 0.546875, 'epoch': 18, 'id': 'ce66c_00450', 'lr': 0.00010281695165161455, 'batch_size': 32, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 2048, 'layer_2_size': 256, 'patch_transformer_size': 16, 'transformer_depth': 2, 'transformer_attantion_head_count': 20, 'patch_size': 4}
2024-04-25T07:09:20.518309141Z {'acc': 0.546875, 'epoch': 14, 'id': 'ce66c_00456', 'lr': 0.0005385481427808325, 'batch_size': 32, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 512, 'layer_2_size': 64, 'patch_transformer_size': 16, 'transformer_depth': 3, 'transformer_attantion_head_count': 20, 'patch_size': 32}
2024-04-25T07:09:20.518319501Z {'acc': 0.5441176295280457, 'epoch': 11, 'id': 'ce66c_00242', 'lr': 0.0007585375476117381, 'batch_size': 4, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 128, 'layer_2_size': 128, 'patch_transformer_size': 16, 'transformer_depth': 2, 'transformer_attantion_head_count': 20, 'patch_size': 4}
2024-04-25T07:09:20.518322402Z {'acc': 0.53125, 'epoch': 5, 'id': 'ce66c_00492', 'lr': 0.00019671742140950458, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 256, 'layer_2_size': 64, 'patch_transformer_size': 16, 'transformer_depth': 2, 'transformer_attantion_head_count': 14, 'patch_size': 16}
2024-04-25T07:09:20.518324542Z {'acc': 0.529411792755127, 'epoch': 11, 'id': 'ce66c_00183', 'lr': 0.00015483770021752362, 'batch_size': 4, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 512, 'layer_2_size': 256, 'patch_transformer_size': 16, 'transformer_depth': 3, 'transformer_attantion_head_count': 10, 'patch_size': 32}
2024-04-25T07:09:20.518326552Z {'acc': 0.515625, 'epoch': 17, 'id': 'ce66c_00156', 'lr': 0.0003034159031252662, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 1024, 'layer_2_size': 512, 'patch_transformer_size': 16, 'transformer_depth': 5, 'transformer_attantion_head_count': 22, 'patch_size': 32}
2024-04-25T07:09:20.518328552Z {'acc': 0.515625, 'epoch': 13, 'id': 'ce66c_00290', 'lr': 0.0003029421792922331, 'batch_size': 16, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 512, 'layer_2_size': 64, 'patch_transformer_size': 16, 'transformer_depth': 5, 'transformer_attantion_head_count': 2, 'patch_size': 4}
2024-04-25T07:09:20.518330652Z {'acc': 0.5, 'epoch': 12, 'id': 'ce66c_00083', 'lr': 0.0002237768345770748, 'batch_size': 16, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 256, 'layer_2_size': 32, 'patch_transformer_size': 16, 'transformer_depth': 5, 'transformer_attantion_head_count': 18, 'patch_size': 8}
2024-04-25T07:09:20.518332762Z {'acc': 0.5, 'epoch': 7, 'id': 'ce66c_00249', 'lr': 0.00012301738161653853, 'batch_size': 16, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 2048, 'layer_2_size': 1024, 'patch_transformer_size': 16, 'transformer_depth': 6, 'transformer_attantion_head_count': 6, 'patch_size': 32}


ray.exceptions.ActorDiedError: The actor died unexpectedly before finishing this task.
2024-04-25T14:44:04.498000301Z 	class_name: RayTrainWorker
2024-04-25T14:44:04.498001631Z 	actor_id: 647f783bf08ca69c00e2c66b01000000
2024-04-25T14:44:04.498002962Z 	pid: 55131
2024-04-25T14:44:04.498004252Z 	namespace: 240551e9-cf1e-41f7-a58b-1c6d10852a8b
2024-04-25T14:44:04.498005642Z 	ip: 172.17.0.2
2024-04-25T14:44:04.498006992Z The actor is dead because its worker process has died. Worker exit type: SYSTEM_ERROR Worker exit detail: Worker unexpectedly exits with a connection error code 2. End of file. There are some potential root causes. (1) The process is killed by SIGKILL by OOM killer due to high memory usage. (2) ray stop --force is called. (3) The worker is crashed unexpectedly due to SIGSEGV or other unexpected errors.
2024-04-25T14:44:04.498965037Z 

{'acc': 0.546875, 'epoch': 10, 'id': '7dbf4_00101', 'lr': 0.00019350869810725026, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 256, 'layer_2_size': 32, 'patch_transformer_size': 16, 'transformer_depth': 8, 'transformer_attantion_head_count': 2}
{'acc': 0.5, 'epoch': 18, 'id': '7dbf4_00103', 'lr': 0.0025941926723439, 'batch_size': 32, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 32, 'layer_2_size': 128, 'patch_transformer_size': 16, 'transformer_depth': 5, 'transformer_attantion_head_count': 6}
{'acc': 0.5, 'epoch': 13, 'id': '7dbf4_00191', 'lr': 0.00038202092742845945, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 512, 'layer_2_size': 128, 'patch_transformer_size': 32, 'transformer_depth': 7, 'transformer_attantion_head_count': 22}
{'acc': 0.484375, 'epoch': 12, 'id': '7dbf4_00187', 'lr': 0.0007505403591860749, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 32, 'layer_2_size': 256, 'patch_transformer_size': 64, 'transformer_depth': 2, 'transformer_attantion_head_count': 8}
{'acc': 0.46875, 'epoch': 16, 'id': '7dbf4_00001', 'lr': 0.00015467077636953925, 'batch_size': 16, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 512, 'layer_2_size': 256, 'patch_transformer_size': 16, 'transformer_depth': 3, 'transformer_attantion_head_count': 8}
{'acc': 0.46875, 'epoch': 5, 'id': '7dbf4_00090', 'lr': 0.00798665954799276, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 128, 'layer_2_size': 256, 'patch_transformer_size': 64, 'transformer_depth': 2, 'transformer_attantion_head_count': 16}
{'acc': 0.453125, 'epoch': 3, 'id': '7dbf4_00033', 'lr': 0.0003820783524266436, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 128, 'layer_2_size': 256, 'patch_transformer_size': 64, 'transformer_depth': 9, 'transformer_attantion_head_count': 12}
{'acc': 0.4375, 'epoch': 8, 'id': '7dbf4_00107', 'lr': 0.0036613498360001055, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 128, 'layer_2_size': 256, 'patch_transformer_size': 64, 'transformer_depth': 9, 'transformer_attantion_head_count': 18}
{'acc': 0.4375, 'epoch': 11, 'id': '7dbf4_00119', 'lr': 0.0037837015934034236, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 256, 'layer_2_size': 64, 'patch_transformer_size': 32, 'transformer_depth': 3, 'transformer_attantion_head_count': 22}
{'acc': 0.421875, 'epoch': 17, 'id': '7dbf4_00099', 'lr': 0.00024365241096318017, 'batch_size': 16, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 64, 'layer_2_size': 64, 'patch_transformer_size': 16, 'transformer_depth': 2, 'transformer_attantion_head_count': 8}
{'acc': 0.421875, 'epoch': 16, 'id': '7dbf4_00139', 'lr': 0.027394833531799882, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 32, 'layer_2_size': 32, 'patch_transformer_size': 64, 'transformer_depth': 5, 'transformer_attantion_head_count': 20}
{'acc': 0.421875, 'epoch': 11, 'id': '7dbf4_00142', 'lr': 0.000926208879005267, 'batch_size': 8, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 256, 'layer_2_size': 512, 'patch_transformer_size': 8, 'transformer_depth': 6, 'transformer_attantion_head_count': 2}
{'acc': 0.421875, 'epoch': 11, 'id': '7dbf4_00153', 'lr': 0.032105891220915145, 'batch_size': 16, 'num_workers': 15, 'alpha': 0.0, 'gamma': 0.0, 'reduction': 'mean', 'from_logits': False, 'padding_sec': 10, 'is_tune': True, 'enable_tune_features': False, 'conv_lr': 0.001, 'layer_1_size': 512, 'layer_2_size': 32, 'patch_transformer_size': 32, 'transformer_depth': 5, 'transformer_attantion_head_count': 20}
'''
