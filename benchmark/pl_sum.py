#!/usr/bin/env python3

__author__ = "Ziqing Guo"
__email__ = "ziqinguse@gmail.com"

import numpy as np
import time
import sys, os
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
from Util_IOfunc import read_yaml

import argparse

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2], help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("-Y", "--noXterm", dest='noXterm', action='store_false', default=True, help="enables X-term for interactive mode")
    parser.add_argument("-p", "--showPlots", default='b', nargs='+', help="abc-string listing shown plots")
    
    parser.add_argument("-s", "--shift", type=bool, default=False, help="whether shift the dots")
    parser.add_argument("--outPath", default='out', help="all outputs from experiment")
       
    args = parser.parse_args()
    for arg in vars(args):
        print('myArg:', arg, getattr(args, arg))
    assert os.path.exists(args.outPath)
    args.showPlots = ''.join(args.showPlots)
    return args


#............................
#............................
#............................
class Plotter(PlotterBackbone):
    def __init__(self, args):
        PlotterBackbone.__init__(self, args)

    def compute(self, bigD, tag1, figId=1, shift=False):
        nrow, ncol = 1, 1       
        figId = self.smart_append(figId)
        fig = self.plt.figure(figId, facecolor='white', figsize=(5.5, 7))        
        ax = self.plt.subplot(nrow, ncol, 1)
        bitstring_counts = {}
        for key, data_list in bigD.items():
            for data in data_list:  
                bitstrings = data.get('result', {}).get('solutions_bitstrings', [])
                for bitstring in bitstrings:
                    bitstring_counts[bitstring] = bitstring_counts.get(bitstring, 0) + 1
        
        #Prints bitstringas and freuency
        #print("Extracted Bitstrings and Counts:", bitstring_counts)

        if not bitstring_counts:
            print("Warning: No bitstring data to plot.")
            ax.text(0.5, 0.5, 'No bitstring data available', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
            return
        
        
        bit_labels = list(bitstring_counts.keys())
        counts = list(bitstring_counts.values())

        # Plot bars
        x_positions = range(len(bit_labels))
        width = 0.8 if not shift else 0.4
        ax.bar(x_positions, counts, width=width, color='blue', alpha=0.7, label='Counts')
        
        if shift:
            ax.bar([x + 0.4 for x in x_positions], counts, width=width, 
                   color='red', alpha=0.7, label='Shifted')

        ax.set_xticks(x_positions)
        ax.set_xticklabels(bit_labels, rotation=90, fontsize=8)
        ax.set_xlabel("Bit Strings")
        ax.set_ylabel("Counts")
        ax.set_title(f"Histogram of Bit Strings - {tag1}")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if shift:
            ax.legend()
        
        self.plt.tight_layout()

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
def readOne(inpF, dataD, verb=1):
    assert os.path.exists(inpF)
    xMD = read_yaml(inpF, verb)
    
    # Use filename as key and store all iterations as a list
    file_key = os.path.basename(inpF)
    if file_key not in dataD:
        dataD[file_key] = []
    
    valid_data = False
    for x in xMD:
        bit_strings = x.get('result', {}).get('solutions_bitstrings', None)
        if bit_strings is not None:
            dataD[file_key].append(x)
            valid_data = True
        elif verb > 0:
            print(f'Warning: No bit_strings in iteration {x.get("iteration")} of {inpF}')
    
    if not valid_data and verb > 0:
        print(f'Warning: No valid bitstring data in {inpF}')

def sort_end_lists(d, parent_key='', sort_key='nq', val_key='runt'):
    if sort_key in d:
        xV = d[sort_key]
        yV = d[val_key]
        xU, yU = map(list, zip(*sorted(zip(xV, yV), key=lambda x: x[0])))
        print(' %s.%s:%d' % (parent_key, sort_key, len(xU)))
        d[sort_key] = np.array(xU)
        d[val_key] = np.array(yU)
        return
    
    for k, v in d.items():
        full_key = '%s.%s' % (parent_key, k) if parent_key else k
        print(full_key)
        if isinstance(v, dict):
            sort_end_lists(v, full_key, sort_key, val_key)

if __name__ == '__main__':
    args = get_parser()
    corePath = '.' #Path for Ebuka's output files
    fileL = []
    path2 = f'{corePath}/out'
    fileL += find_yaml_files(path2)
        
    num_input_files = len(fileL)
    if num_input_files == 0:
        print('Error: No input files found')
        sys.exit(1)
        
    print(f'Found {num_input_files} input files')
    if num_input_files > 0:
        print(f'Example file: {fileL[0]}')

    dataAll = {}
    for i, fileN in enumerate(fileL):
        readOne(fileN, dataAll, args.verb)
    
    print('\nM: all tags:')
    sort_end_lists(dataAll)
    if args.verb > 1:
        pprint(dataAll)  # Debug output if verbose
    
    args.prjName = 'qubo25'
    plot = Plotter(args)
    if 'a' in args.showPlots:
        plot.compute(dataAll, 'hybrid', figId=1, shift=args.shift)
    if 'b' in args.showPlots:
        plot.compute(dataAll, 'X', figId=2, shift=args.shift)
    if 'c' in args.showPlots:
        plot.compute(dataAll, 'X2', figId=3, shift=args.shift)
    plot.display_all(png=0)