import sys, os
import argparse

parser = argparse.ArgumentParser(description='Video Anomaly Detection')
parser.add_argument('--n', default='', type=str, help='file name')
args = parser.parse_args()

file_name =args.n
if not os.path.isdir(args.n[:-4]):
    os.mkdir(args.n[:-4])

save_name = './' + args.n[:-4] + '/%05d.jpg'

os.system('ffmpeg -i %s -r 25 -q:v 2 -vf scale=320:240 %s'%(file_name, save_name))
