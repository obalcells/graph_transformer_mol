import argparse

def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for creating model."
    )

    # fmt: off
    parser.add_argument("--N", type=int, metavar="N", help="num layers")
    parser.add_argument("--d_model", type=int, metavar="N", help="hidden emb dimension")
    parser.add_argument("--max-nodes", type=int, help="max number of nodes in a graph")
    parser.add_argument("--h", type=int, help="num heads")
    parser.add_argument("--d_ff", type=int, help="hidden layer size in feedforward network")
    parser.add_argument("--dropout", type=float, help="dropout prob")
    # fmt: on
    return parser