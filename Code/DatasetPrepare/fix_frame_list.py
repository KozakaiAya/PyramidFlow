import os

dataset_path = '../../../Dataset'
valset_path = '../../../ValSet'


def fix(path):
    with open(os.path.join(path, 'frame_list.txt'), 'r') as f:
        lines = f.readlines()
        name_list = []
        for line in lines:
            flag = True
            for idx in range(5):
                if not os.path.exists(os.path.join(path, line.strip() + '_' + str(idx) + '.png')):
                    flag = False
                    break
            if flag:
                name_list.append(line)

    with open(os.path.join(path, 'frame_list.txt'), 'w') as f:
        f.writelines(name_list)

fix(dataset_path)
fix(valset_path)
