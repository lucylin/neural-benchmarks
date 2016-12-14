import argparse
import logging

# Module-specific
import defaults
from lm import LanguageModel


def get_argparser():
    """Return an ArgumentParser with default arguments."""

    p = argparse.ArgumentParser()

    p.add_argument('--DEBUG', action='store_true', help="Log on DEBUG mode")

    p.add_argument("--data_path", metavar="PATH",
                   help="path where the data is stored as a Pandas DataFrame")
    p.add_argument("--test", metavar="PATH",
                   help="tests a model loaded from a specified path (uses \
                         gold standard inputs)")
    p.add_argument("--train", metavar="PATH",
                   help="trains a model and saves it to a specified path")

    # Hyper parameters ish
    p.add_argument("--learning_rate", metavar="LR", type=float,
                   default=defaults.LEARNING_RATE,
                   help="Learning rate for AdamOptimizer.")
    p.add_argument("--hidden_size", dest="hidden_size", type=int,
                   default=defaults.HIDDEN_SIZE,
                   help="Size of tensors in hidden layer.")
    p.add_argument("-b", "--batch_size", dest="batch_size", type=int,
                   default=defaults.BATCH_SIZE,
                   help="Batch size to pass through the graph.")
    p.add_argument("--max_epoch", dest="max_epoch", type=int,
                   default=defaults.MAX_EPOCH,
                   help="Determines how many passes throught the training \
                         data to do")

    return p


def main():
    p = get_argparser()
    args = p.parse_args()

    lm = LanguageModel()
    lm.configure_logger(level=logging.DEBUG if args.DEBUG else logging.INFO,
                        write_file=True)

    if args.train and args.data_path:
        lm.train(args.data_path, output_path=args.train,
                 learning_rate=args.learning_rate,
                 hidden_size=args.hidden_size,
                 batch_size=args.batch_size, max_epoch=args.max_epoch)

    elif args.test and args.data_path:
        lm.predict(args.test, args.data_path)

    else:
        # Well, this is silly.
        p.print_help()
        exit(2)

if __name__ == '__main__':
    main()
