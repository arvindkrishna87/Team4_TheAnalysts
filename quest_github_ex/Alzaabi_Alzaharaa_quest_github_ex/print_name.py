# change this to your name
name = 'Alzaharaa Alzaabi'
with open('name.txt', 'w') as f:
    for i in range(len(name) + 1):
        f.write(name[:i] + '\n')











