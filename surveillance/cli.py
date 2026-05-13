import argparse
import sys

from surveillance import setup_logging


def main():
    parser = argparse.ArgumentParser(
        prog='surveillance',
        description='Person activity tracking from video analysis',
    )
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable debug-level logging')
    sub = parser.add_subparsers(dest='action', required=True)

    p = sub.add_parser('surveillance', help='Monitor video folder and track activity continuously')
    p.add_argument('--init', action='store_true', help='Use default YOLO model if no trained model exists')
    p = sub.add_parser('onboard', help='Generate training data from a video')
    p.add_argument('video', help='Path to the input video file')
    p.add_argument('classname', help='Generic class to detect (e.g., person)')
    p.add_argument('username', help='Username to assign (e.g., dad)')
    p.add_argument('--init', action='store_true', help='Use default YOLO model if no trained model exists')

    p = sub.add_parser('train', help='Train a YOLO model from onboarded data')
    p.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    p.add_argument('--imgsz', type=int, default=640, help='Image size (default: 640)')
    p.add_argument('--batch', type=int, default=16, help='Batch size (default: 16)')
    p.add_argument('--export-only', action='store_true',
                   help='Export dataset to training/dataset/ and exit without training')

    p = sub.add_parser('report', help='Query activity records')
    p.add_argument('from_dt', help='Start datetime (YYYY-MM-DD HH:MM:SS)')
    p.add_argument('to_dt', help='End datetime (YYYY-MM-DD HH:MM:SS)')
    p.add_argument('--class', dest='class_filter', help='Filter by class name')

    p = sub.add_parser('find', help='Find last-seen info for a class')
    p.add_argument('classname', help='Class name to look up')

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if args.action == 'surveillance':
        from surveillance.surveil import run_surveillance
        sys.exit(run_surveillance(init_mode=args.init))
    elif args.action == 'onboard':
        from surveillance.onboard import run_onboard
        sys.exit(run_onboard(args.video, args.classname, args.username, init_mode=args.init))
    elif args.action == 'train':
        from surveillance.train_action import run_train
        sys.exit(run_train(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                           export_only=args.export_only))
    elif args.action == 'report':
        from surveillance.report import run_report
        sys.exit(run_report(args.from_dt, args.to_dt, args.class_filter))
    elif args.action == 'find':
        from surveillance.find_action import run_find
        sys.exit(run_find(args.classname))
