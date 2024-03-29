#!/usr/bin/env python
from subprocess import check_call, CalledProcessError


def check_bash_call(string):
  check_call(["bash", "-c", string])


def _run_format_and_yapf():
  files_changed = False

  try:
    check_bash_call("find . -name '*.py' -print0 | xargs -0 yapf --style=./.yapf -dr")
  except CalledProcessError:
    check_bash_call("find . -name '*.py' -print0 | xargs -0 yapf --style=./.yapf -ir")
    files_changed = True

  if files_changed:
    print("Some files have changed.")
    print("Please use 'find . -name '*.py' -print0 | xargs -0 yapf --style=./.yapf -ir' before commit.")
  else:
    print("No formatting needed.")

  if files_changed:
    exit(1)


def run_format_and_yapf():
  try:
    _run_format_and_yapf()
  except CalledProcessError as error:
    print("Yapf check returned exit code", error.returncode)
    exit(error.returncode)


if __name__ == "__main__":
  run_format_and_yapf()
