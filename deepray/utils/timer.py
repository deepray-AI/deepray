import time
from functools import wraps


class Timer:
  """Useage
  if __name__ == "__main__":
  with Timer():
      # ...
  """

  def __enter__(self):
    self._enter_time = time.time()

  def __exit__(self, *exc_args):
    self._exit_time = time.time()
    print(f"{self._exit_time - self._enter_time:.2f} seconds elapsed")


def timer(func):
  """Useage
  @timer
  def your_function():
      # ...
  """

  @wraps(func)
  def inner(*args, **kwargs):
    start_time = time.time()
    retval = func(*args, **kwargs)
    print(f"{time.time() - start_time:.2f} seconds elapsed")
    return retval

  return inner
