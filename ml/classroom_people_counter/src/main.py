import argparse
from src.utils.config import load_config
from src.runners import RealtimeRunner, VideoRunner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['realtime','video'], default='realtime')
    parser.add_argument('--source', default='0', help='camera index or video path')
    parser.add_argument('--config', default='config/app_config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.mode == 'realtime':
        runner = RealtimeRunner(cfg)
        runner.run(args.source)
    else:
        runner = VideoRunner(cfg)
        runner.run(args.source)

if __name__ == '__main__':
    main()
