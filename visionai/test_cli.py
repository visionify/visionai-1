import argparse
from rich import print


usage = '%(prog)s [list_devices | [list_scenarios ? camera]]'
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument('--list-devices')
parser.add_argument('--list-scenarios')
parser.add_argument('--camera')
args=parser.parse_args()

print(args)

