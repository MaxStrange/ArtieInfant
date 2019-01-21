def save_file(contents):
    with open("path_to_save_the_file.wav", 'wb') as f:
        f.write(contents)
    return "path_to_save_the_file.wav"
