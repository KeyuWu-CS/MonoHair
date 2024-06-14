import termcolor
from easydict import EasyDict as edict
import time

def red(message,**kwargs): return termcolor.colored(str(message),color="red",attrs=[k for k,v in kwargs.items() if v is True])
def green(message,**kwargs): return termcolor.colored(str(message),color="green",attrs=[k for k,v in kwargs.items() if v is True])
def blue(message,**kwargs): return termcolor.colored(str(message),color="blue",attrs=[k for k,v in kwargs.items() if v is True])
def cyan(message,**kwargs): return termcolor.colored(str(message),color="cyan",attrs=[k for k,v in kwargs.items() if v is True])
def yellow(message,**kwargs): return termcolor.colored(str(message),color="yellow",attrs=[k for k,v in kwargs.items() if v is True])
def magenta(message,**kwargs): return termcolor.colored(str(message),color="magenta",attrs=[k for k,v in kwargs.items() if v is True])
def grey(message,**kwargs): return termcolor.colored(str(message),color="grey",attrs=[k for k,v in kwargs.items() if v is True])



def get_time(sec):
    d = int(sec//(24*60*60))
    h = int(sec//(60*60)%24)
    m = int((sec//60)%60)
    s = int(sec%60)
    return d,h,m,s


class Log:
    def __init__(self): pass
    def process(self,pid):
        print(grey("Process ID: {}".format(pid),bold=True))
    def title(self,message,opt):
        print(yellow(message,bold=True,underline=True))
        if opt.isTrain:
            with open("{0}/log.txt".format(opt.output_path), "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)
    def info(self,message):
        print(magenta(message,bold=True))
    def options(self,opt,level=0):
        for key,value in sorted(opt.items()):
            if isinstance(value,(dict,edict)):
                print("   "*level+cyan("* ")+green(key)+":")
                self.options(value,level+1)
            else:
                print("   "*level+cyan("* ")+green(key)+":",yellow(value))
    # def loss_train(self,opt,ep,lr,loss,timer):
    #     message = grey("[train] ",bold=True)
    #     message += "epoch {}/{}".format(cyan(ep,bold=True),opt.max_epoch)
    #     message += ", lr:{}".format(yellow("{:.2e}".format(lr),bold=True))
    #     message += ", loss:{}".format(red("{:.3e}".format(loss),bold=True))
    #     message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)),bold=True))
    #     message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
    #     print(message)
    def loss_val(self,opt,loss):
        message = grey("[val] ",bold=True)
        message += "loss:{}".format(red("{:.3e}".format(loss),bold=True))
        print(message)

    def loss_train(self,opt,it,loss):
        with open("{0}/log.txt".format(opt.output_path),'a+')as f:
            message="iter:{} ".format(it)
            for k, v in loss.items():
                # print(v)
                # if v != 0:
                v = v.mean().float()
                message += '%s: %.3f ' % (k, v)
            print(message)
            f.write('%s\n' % message)
log = Log()