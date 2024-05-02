import argparse
import os

from config.constants import ROOT_DIR


def parse_args():
    parent_parser = argparse.ArgumentParser()


    subparsers  = parent_parser.add_subparsers(
        dest="parent_param"
    #     parents=[parent_parser],
    # prog='create-dataset',
    )
    # dataset_parser.add_argument(
    #     'create-dataset',
    #     type=str,
    #     help='',
    # )
    dataset_parser = subparsers.add_parser('create-dataset', help='')
    dataset_parser.add_argument(
        '--annotations-file',
        type=str,
        help='Path to annotations',
    )
    dataset_parser.add_argument(
        '--dataset-dir',
        type=str,
        help='Path to output dataset',
    )

    learn_parser = subparsers.add_parser(
    name='train',
    )
    # learn_parser.add_argument(
    #     'learn',
    #     type=str,
    #     help='',
    # )
    learn_parser.add_argument(
        '--config-file',
        type=str,
        help='Path to output config file',
        default=os.path.join(ROOT_DIR, 'config', 'example.config.json')
    )

    # r1 =  parent_parser.parse_args(['create-dataset', '--annotations-file', "./config.json"])
    # r2 =  parent_parser.parse_args(['create-dataset', '--annotations-file', "./config.json", '--dataset-dir', "./some/dir"])
    # r3 =  parent_parser.parse_args(['learn'])

    # if r3.parent_param == 'learn':
    #     print("learn")
    # else:
    #     print("not learn")
    #
    # print(r1, r2, r3, sep='\n')
    return parent_parser.parse_args()


if __name__ == '__main__':
    parse_args()

