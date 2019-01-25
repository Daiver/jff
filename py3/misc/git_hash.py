import os

import subprocess

def get_git_revision_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    except subprocess.CalledProcessError as e:
        return None

print(get_git_revision_hash())

