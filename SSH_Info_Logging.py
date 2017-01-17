#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 2015年4月20日

@author: lianjiaxin
'''
import os
import threading
import time
from time import sleep
from SSHLibrary import SSHLibrary

class SSH_Info_Logging(SSHLibrary):
    '''
    将SSH打印信息记录到文本
    '''


    def __init__(self):
        '''
        Constructor
        '''
        for base in SSH_Info_Logging.__bases__:
            base.__init__(self)
        #self.exitflag=False
        self.stop_save_log=False#作为线程结束的标识

    def save_SSH_logInfo(self,filename='1.1.1.1',log_dir='F:\\sshlog',delay=5):
        '''
        保存SSH打印信息的目录
        '''
        getDay=time.strftime('%Y-%m-%d',time.localtime(time.time()))#获取当前日期
        filepath=log_dir+'\\'+filename+'_'+getDay+'.log'
        if os.path.exists(log_dir)==False:
            os.mkdir(log_dir)
        while self.stop_save_log==False:
            #print 'aaa++++++++++++++++++++++++++++++'
            getInfo=self.read(loglevel=None, delay=delay)
            #print 'bbbbbbbbbbbbb+++++++++++++++'
            fp=open(filepath,'a')
            fp.write(getInfo)
        print 'stop_save_log++++++++++++++++++++++++++++++++++++++'
        #self.close_connection()#保存打印信息完毕后，关闭SSH连接
    def start_saving_SSH_Info(self,filetag='1.1.1.1',log_dir='F:\\sshlog',delay=5):
        '''
        开启保存设备SSH打印信息线程
        filetag：文件标签,如命名test，则产生日志为：test_2015-04-05.log
        log_dir:日志保存目录
        delay:每次写入日志文本的时间频率，默认值为5秒
        '''
        print 'start logging sshInfo'
        t=threading.Thread(target=self.save_SSH_logInfo,args=(filetag,log_dir,delay))
        t.setDaemon(False)#False，不用等待线程执行结束
        t.start()
        #t.join(timeout=20)
    def stop_saving_SSH_Info(self):
        '''
            停止保存SSH打印信息
        '''
        self.stop_save_log=True
if __name__ == '__main__':
    pf=SSH_Info_Logging()
    pf.open_connection('10.8.4.201')
    #pf.ssh_open_connection('10.8.4.201')
    pf.login('admin', '12345', 5)
    #pf.ssh_login('admin', '12345', 5)
    pf.write('outputClose')
    pf.write('outputOpen')
    pf.write('ifconfig')
    pf.write('setDebug -m all -l 7 -d lll')
    #pf.ssh_set_command('outputClose')
    #pf.ssh_set_command('outputOpen')
    #pf.ssh_set_command('ifconfig')
    pf.start_saving_SSH_Info('10.8.4.201', 'F:\\sshlog')
    sleep(10)
    pf.stop_saving_SSH_Info()
