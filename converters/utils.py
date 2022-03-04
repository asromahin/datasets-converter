from uuid import uuid4


def get_default_filename(extension='.jpg'):
    return uuid4().hex+extension
