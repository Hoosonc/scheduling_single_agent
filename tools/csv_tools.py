# -*- coding: utf-8 -*-
# @Time    : 2023/8/21 19:01
# @Author  : hxc
# @File    : csv_tools.py
# @Software: PyCharm
import csv


class csv_writer:
    def __init__(self):
        self.encoding = 'utf-8-sig'

    def write_headers(self, file_name, headers, path):
        with open(f'{path}/{file_name}.csv', mode='w+', encoding=self.encoding, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def write_result(self, data_list, file_name, path):
        with open(f'{path}/{file_name}.csv', mode='a+', encoding=self.encoding, newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_list)
