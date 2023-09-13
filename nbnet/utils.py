import time

class TimeManager:
    def __init__(self):
        self.init_time = None
        self.curr_time = None
        self.avg_time = None
        
        self.total_step = None
        self.curr_step = None
        
    def start(self):
        print("Timer starts...")
        self.init_time = time.time()
        self.curr_time = time.time()
        
    def renew(self, curr_step, total_step):
        time_consumed = time.time() - self.curr_time
        print("Time Consumed: ", time_consumed)
        self.curr_time = time.time()
        

        
class Logger:
    def __init__(self, logging_path):
        print("Logging...")
        print("Logging path is: ", logging_path)
        self.logging_path = logging_path
        
    def logging(self, text):
        print(text)
        f = open(self.logging_path, 'a')
        f.write(text + "\n")
        f.close()