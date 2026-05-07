import argparse

from buddy_matcher import run_assign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the employee buddy matching POC.")
    parser.add_argument("--new-employee", default="new_employee.json")
    parser.add_argument("--demo", action="store_true", help="Show a high-level demo-friendly process overview.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_assign(args.new_employee, demo_mode=args.demo)


if __name__ == "__main__":
    main()
