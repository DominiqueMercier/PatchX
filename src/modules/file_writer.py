def write_to_file(filename, content, append=False):
    """Generic file writer to write content to a txt file.

    Args:
        filename (str): path to the file
        content (str): content of the file as string
        append (bool, optional): flag to append instead of overwrite the file. Defaults to False.
    """
    with open(filename, 'w' if not append else 'a') as f:
        f.write(content)