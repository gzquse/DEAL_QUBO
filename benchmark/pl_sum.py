#!/usr/bin/env python3

__author__ = "Ziqing Guo"
__email__ = "ziqinguse@gmail.com"

import numpy as np
import  time
import sys,os
from pprint import pprint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'toolbox')))
from PlotterBackbone import roys_fontset
from PlotterBackbone import PlotterBackbone
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib import font_manager
import matplotlib.patches as patches

roys_fontset(plt)
from Util_IOfunc import  read_yaml

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],help="increase output verbosity", default=1, dest='verb')
    parser.add_argument( "-Y","--noXterm", dest='noXterm',  action='store_false', default=True, help="enables X-term for interactive mode")

    parser.add_argument("-p", "--showPlots",  default='b', nargs='+',help="abc-string listing shown plots")
    parser.add_argument("-s", "--shift", type=bool, default=True, help="whether shift the dots")

    parser.add_argument("--outPath",default='out',help="all outputs from experiment")
       
    args = parser.parse_args()
    # make arguments  more flexible
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    args.showPlots=''.join(args.showPlots)

    return args


#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self,args)


    def compute(self,bigD,tag1,figId=1,shift=False):
        nrow,ncol=1,1       
        figId=self.smart_append(figId)
        fig=self.plt.figure(figId,facecolor='white',figsize=(5.5,7))        
        ax = self.plt.subplot(nrow,ncol,1)

        #TODO 
        # ax.plot(historgram)..

#...!...!....................
def find_yaml_files(directory_path, vetoL=None):
    if vetoL is None:
        vetoL = []
   
    yaml_files = []
    if not os.path.exists(directory_path):
        print(f'Warning: Directory not found: {directory_path}')
        return yaml_files
        
    print(f'Scanning path: {directory_path}')
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.yaml') and not any(veto in file for veto in vetoL):
                yaml_files.append(os.path.join(root, file))
    print(f'Found {len(yaml_files)} YAML files')
    return yaml_files

#...!...!....................
def readOne(inpF,dataD,verb=1):
    assert os.path.exists(inpF)
    # date = extract_date_from_path(inpF)
    xMD=read_yaml(inpF,verb)
    #print(inpF,xMD['num_qubit'],xMD['elapsed_time'],float(xMD['num_circ']))
    # prime the list
    bit_strings = [i for i in range(len(xMD))]
    for x in xMD:
        bit_strings=x['result'].get('solutions_bitstrings',None)
        if bit_strings is None:
            print(f'Warning: No bit_strings in {inpF}')
            return
    # count all bit strings
    # save for future
    # head=dataD[tag1][tag2][tag3]

#...!...!....................            
def sort_end_lists(d, parent_key='', sort_key='nq', val_key='runt'):
    """
    Recursively prints all keys in a nested dictionary.
    Once the sort_key is in dict it triggers sorting both keys.

    Args:
    d (dict): The dictionary to traverse.
    parent_key (str): The base key to use for nested keys (used for recursion).
    sort_key (str): The key indicating the list to sort by.
    val_key (str): The key indicating the list to sort alongside.
    """
    if sort_key in d:
        xV = d[sort_key]
        yV = d[val_key]
        xU, yU = map(list, zip(*sorted(zip(xV, yV), key=lambda x: x[0])))
        print(' %s.%s:%d' % (parent_key, sort_key, len(xU)))
        d[sort_key]=np.array(xU)
        d[val_key]=np.array(yU)
        return
    
    for k, v in d.items():
        full_key = '%s.%s' % (parent_key, k) if parent_key else k
        print(full_key)
        if isinstance(v, dict):
            sort_end_lists(v, full_key, sort_key, val_key)

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()
    # ----  just reading
    corePath='/lustre/work/ziqguo/QML_2025/benchmark'  # bare path for ziqing
    # ebuka
    # TODO
    # save for future
    # pathL=['Nov15']
    fileL=[]
    # vetoL=['r1.4','r2.4','r3.4']
    path2=f'{corePath}/out'
    fileL+=find_yaml_files(path2)
        
    num_input_files=len(fileL)
    if num_input_files == 0:
        print('Error: No input files found')
        sys.exit(1)
        
    print(f'Found {num_input_files} input files')
    if num_input_files > 0:
        print(f'Example file: {fileL[0]}')

    dataAll={}
    for i,fileN in enumerate(fileL):
        readOne(fileN,dataAll,i==0)

    
    #pprint(dataAll)
    print('\nM: all tags:')
    sort_end_lists(dataAll)
    # ----  just plotting
    args.prjName='qubo25'
    plot=Plotter(args)
    if 'a' in args.showPlots:
        plot.compute(dataAll,'hybrid', figId=1, shift=args.shift)
    if 'b' in args.showPlots:
        plot.compute(dataAll,'X', figId=2, shift=args.shift)
    if 'c' in args.showPlots:
        plot.compute(dataAll,'X2', figId=3, shift=args.shift)
    plot.display_all(png=0)
