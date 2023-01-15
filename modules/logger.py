import os
from datetime import datetime

class Logger:

    def __init__(self, args, w_id, g_id, is_detailed=True):
        self.args = args
        self.w_id = w_id
        self.g_id = g_id
        self.is_server = True
        self.is_detailed = is_detailed
        
    def switch(self, c_id):
        self.is_server = False
        self.c_id = c_id

    def print(self, message):
        now = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        msg = f'[{now}]'
        # msg += f'[p:{os.getppid()}]' if self.is_detailed else ''
        msg += f'[{self.args.trial}]'
        msg += f'[{self.args.model}]'
        msg += f'[{self.args.task}]'
        msg += f'[permuted]' if self.args.permuted else '[conven]'
        # msg += f'[w:{self.w_id}]' if self.is_detailed and not self.is_server else ''
        msg += f'[g:{self.g_id}]' if self.is_detailed and not self.is_server else ''
        msg += f'[server]'if self.is_server else f'[c:{self.c_id}]'
        msg += f' {message}'
        print(msg)
