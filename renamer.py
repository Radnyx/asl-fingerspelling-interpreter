import os

for i in range(ord('a'), ord('y') + 1):
    if chr(i) != 'j':
        if chr(i) == 'k':
            path = './dataset5/E/' + chr(i)
            files = os.listdir(path)
            for index, file in enumerate(files):
                os.rename(os.path.join(path, file), os.path.join(path, 'E'+file))