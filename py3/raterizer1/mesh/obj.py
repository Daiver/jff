from .mesh import Mesh


def from_obj_file(file_name):
    with open(file_name) as f:
        string = f.read()
        return from_obj_string(string)


def from_obj_string(string):
    pass
