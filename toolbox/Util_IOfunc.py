__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import time, os
import yaml

from pprint import pprint
import csv

#...!...!..................
def read_yaml(ymlFn,verb=1):
        if verb: print('  read  yaml:',ymlFn)
        start = time.time()
        ymlFd = open(ymlFn, 'r')
        bulk=yaml.load( ymlFd, Loader=yaml.CLoader)

        ymlFd.close()
        if verb>1: print(' done  elaT=%.1f sec'%(time.time() - start))
        
        return bulk

#...!...!..................
def write_yaml(rec,ymlFn,verb=1):
        start = time.time()
        ymlFd = open(ymlFn, 'w')
        yaml.dump(rec, ymlFd, Dumper=yaml.CDumper)
        ymlFd.close()
        xx=os.path.getsize(ymlFn)/1024
        if verb:
                print('  closed  yaml:',ymlFn,' size=%.1f kB'%xx,'  elaT=%.1f sec'%(time.time() - start))

   
#...!...!..................
def read_one_csv(fname,delim=','):
    print('read_one_csv:',fname)
    tabL=[]
    with open(fname) as csvfile:
        drd = csv.DictReader(csvfile, delimiter=delim)
        print('see %d columns'%len(drd.fieldnames),drd.fieldnames)
        for row in drd:
            tabL.append(row)
            
        print('got %d rows \n'%(len(tabL)))
    #print('LAST:',row)
    return tabL,drd.fieldnames

#...!...!..................
def write_one_csv(fname,rowL,colNameL):
    print('write_one_csv:',fname)
    print('export %d columns'%len(colNameL), colNameL)
    with open(fname,'w') as fou:
        dw = csv.DictWriter(fou, fieldnames=colNameL)#, delimiter='\t'
        dw.writeheader()
        for row in rowL:
            dw.writerow(row)    


#...!...!..................
def expand_dash_list(inpL):
    # expand list if '-' are present
    outL=[]
    for x in inpL:
        if '-' not in x:
            outL.append(x) ; continue
        # must contain string: xxxx[n1-n2]
        ab,c=x.split(']')
        assert len(ab)>3
        a,b=ab.split('[')
        print('abc',a,b,c)
        nL=b.split('-')
        for i in range(int(nL[0]),int(nL[1])+1):
            outL.append('%s%d%s'%(a,i,c))
    print('EDL:',inpL,'  to ',outL)
    return outL
''' - - - - - - - - - 
Offset-aware time, usage:

*) get current date:
t1=time.localtime()   <type 'time.struct_time'>

*) convert to string: 
timeStr=dateT2Str(t1)

*) revert to struct_time
t2=dateStr2T(timeStr)

*) compute difference in sec:
t3=time.localtime()
delT=time.mktime(t3) - time.mktime(t1)
delSec=delT.total_seconds()
or delT is already in seconds
'''

#...!...!..................
def dateT2Str(xT=None):  # --> string
    if xT==None: xT=time.localtime(time.time()) # use time now
    nowStr=time.strftime("%Y%m%d_%H%M%S_%Z",xT)
    return nowStr

#...!...!..................
def dateStr2T(xS):  #  --> datetime
    yT = time.strptime(xS,"%Y%m%d_%H%M%S_%Z")
    return yT


#...!...!..................
def get_cpu_model():
    import subprocess
    try:
        # Run lscpu to get detailed CPU information
        result = subprocess.run(['lscpu'], capture_output=True, text=True)
        for line in result.stdout.splitlines():
            if "Model name:" in line:
                return line.split(":")[1].strip()
    except Exception as e:
        return "Unknown"

#...!...!..................
def get_cpu_info(verb=1):
    import platform
    import psutil  # for CPU processor

    # Get CPU information
    cpu_type = platform.processor()
    cpu_model = get_cpu_model()
    logical_cpus = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    # Number of sockets (assuming one CPU per socket)
    num_sockets = logical_cpus // physical_cores

    # Get RAM information
    memory_info = psutil.virtual_memory()
    total_memory = memory_info.total / (1024 ** 2)  # Convert bytes to MB
    used_memory = memory_info.used / (1024 ** 2)    # Convert bytes to MB
    free_memory = memory_info.available / (1024 ** 2)  # Convert bytes to MB

    memGB=float( '%.1f'%(total_memory/1000.))
    md={'phys_cores':physical_cores,'logic_cpus':logical_cpus,'tot_mem_gb': memGB,'cpu_type':cpu_type,'cpu_model':cpu_model,'num_sockets':num_sockets}
    
    if verb>0:
        print(f"CPU Type: {cpu_type}")
        print(f"CPU Model: {cpu_model}")
        print(f"Number of Logical CPUs: {logical_cpus}")
        print(f"Number of Physical Cores: {physical_cores}")
        print(f"Number of Sockets: {num_sockets}")
        print(f"Total RAM: {total_memory:.2f} MB")
        print(f"Used RAM: {used_memory:.2f} MB")
        print(f"Free RAM: {free_memory:.2f} MB")
    return md


#...!...!..................
def get_gpu_info(verb=1):
    import pynvml  # for inpection of GPUs
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        # Get PCI bus ID (similar to nvidia-smi format)
        pci_bus_id = pynvml.nvmlDeviceGetPciInfo(handle).busId.decode('utf-8')
        
        # Get GPU model name
        model_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
        
        # Get GPU memory information
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = memory_info.total / (1024 ** 2)  # Convert bytes to MB
        used_memory = memory_info.used / (1024 ** 2)    # Convert bytes to MB
        free_memory = memory_info.free / (1024 ** 2)    # Convert bytes to MB

        memGB=float( '%.1f'%(total_memory/1000.))
        
        if i==0:
                md={'gpu_model':model_name, 'tot_mem_gb': memGB, 'pci_bus':[pci_bus_id] }
        else:  md['pci_bus'].append(pci_bus_id)
        if verb>1:
            print(f"{i}  {model_name}   bus_id: {pci_bus_id} ")
            print(f"Total Memory: {total_memory:.2f} MB")
            print(f"Used Memory: {used_memory:.2f} MB")
            print(f"Free Memory: {free_memory:.2f} MB")
            print()
    md['device_count']=device_count

    pynvml.nvmlShutdown()
    return md

