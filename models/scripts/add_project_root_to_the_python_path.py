import sys
import os


def get_project_root() -> str:
    """Returns the absolute path of the project root directory."""
    current_dir = os.path.abspath(os.curdir)
    while not os.path.isfile(os.path.join(current_dir, "README.md")):
        current_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    return current_dir


if __name__ == "__main__":
    sys.path.insert(0, get_project_root())
