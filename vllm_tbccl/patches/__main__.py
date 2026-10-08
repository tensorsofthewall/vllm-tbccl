import argparse
import sys

from . import directory, names, path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m vllm_tbccl.patches", description="Locate the patch files shipped with vllm-tbccl.")
    ap.add_argument("--list", action="store_true", help="print the patch names")
    ap.add_argument("--path", nargs="?", const="", metavar="NAME", help="print the patch directory, or the path of the named patch")
    a = ap.parse_args(argv)
    if a.list:
        print("\n".join(names()))
    elif a.path is not None:
        try:
            print(path(a.path) if a.path else directory())
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            return 1
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
