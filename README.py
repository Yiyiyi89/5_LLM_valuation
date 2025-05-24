import os
from pathlib import Path

# Symbols for visual tree representation
space = '    '
branch = '│   '
tee = '├── '
last = '└── '

def tree(dir_path, prefix: str = ''):
    """A recursive generator that, given a directory Path object,
    will yield a visual tree structure line by line
    with each line prefixed by the same characters.
    """
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # Extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix + extension)

# Set dir_path to the current working directory and convert it to Path object
dir_path = Path(os.getcwd())

# Print the directory tree
for line in tree(dir_path):
    print(line)
