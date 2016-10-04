import json, argparse

"""
Utility script to print the training options for a JSON checkpoint
"""

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default='data/checkpoint.json')


def main(args):
  with open(args.checkpoint, 'r') as f:
    opts = json.load(f)['opt']
  for k, v in sorted(opts.iteritems()):
    print k, v


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

