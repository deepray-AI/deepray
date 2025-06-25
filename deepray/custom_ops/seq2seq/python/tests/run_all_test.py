from pathlib import Path
import sys

import pytest

if __name__ == "__main__":
  dirname = Path(__file__).absolute().parent
  # sys.exit(pytest.main([str(dirname)]))
  sys.exit(pytest.main(["-n 20", "-s", "-v", str(dirname)]))
