import csv
from sys import argv


def csv2dict(csv_path: str, dict_path: str) -> None:
    f_csv = open(csv_path, 'r')
    f_dict = open(dict_path, 'w')
    csv_iter = csv.reader(f_csv)
    first_row = True
    label_col = -1
    for row in csv_iter:
        if first_row:
            for i in range(len(row)):
                if row[i] == 'label':
                    label_col = i
                    break
            first_row = False
            continue
        else:
            f_dict.write('%s 1\n' % row[label_col].strip('\"'))
    f_csv.close()
    f_dict.close()

