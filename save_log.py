import argparse
import os

class Save_Log(object):
    def __init__(self,args):
        self.args = args
        self.argsDict = args.__dict__

        print(self.argsDict)

        omit_attrs = []
        self.set_omit_attr(omit_attrs)

        add_attrs = ['memo','recall','precision','F1','最好epoch']
        self.add_extra_attr(add_attrs)
        self.save()

    def set_omit_attr(self,attrs):
        for attr in attrs:
            self.argsDict.pop(attr)

    def add_extra_attr(self,attrs):
        for attr in attrs:
            self.argsDict[attr] = ''


    def save(self):
        os.makedirs(os.path.join(self.args.log_file, 'save_log'),exist_ok=True)
        print("saving log in ",os.path.join(self.args.log_file, 'save_log'))
        if('record.csv' not in os.listdir(os.path.join(self.args.log_file,'save_log'))):
            with open(os.path.join(self.args.log_file, 'save_log', 'record.csv'), 'a+', encoding='utf-8') as file:
                for key in self.argsDict:
                    file.write(key+',')
                file.write('\n')
        with open(os.path.join(self.args.log_file,'save_log','record.csv'),'a+',encoding='utf-8') as file:
            for key in self.argsDict:
                if(self.argsDict[key] is None):
                    file.write(',')
                else:
                    file.write('{}'.format(self.argsDict[key])+',')
            file.write("\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name',default='debug',type=str)
    parser.add_argument('--run_time',default='12345678',type=str)
    parser.add_argument("--commit_id",default=False,type=bool)
    parser.add_argument("--log_file",default='output/2020/logs',type=str)

    args = parser.parse_args()
    save_log = Save_Log(args)

    pass