__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
from pprint import pprint
import os,hashlib
from toolbox.Util_H5io4 import  write4_data_hdf5, read4_data_hdf5
from qiskit import qpy  # only to IO circuits
import cudaq 
from time import time
from qiskit import QuantumRegister
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


#...!...!.................... 
def harvest_circ_transpMeta(qc,md,transBackN):
    nqTot=qc.num_qubits
        
    try:  # works for qiskit  0.45.1
        physQubitLayout = qc._layout.final_index_layout(filter_ancillas=True)
    except:
        physQubitLayout =[ i for i in range(nqTot)]

    #print('physQubitLayout'); print(physQubitLayout)
    
    #.... cycles & depth ...
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )

    #.....  gate count .....
    opsOD=qc.count_ops()  # ordered dict
    opsD={ k:opsOD[k] for k in opsOD}
    
    n1q_g=0
    for xx in ['ry','h','r','u2','u3']:
        if xx not in opsD: continue
        n1q_g+=opsD[xx]
    
    n2q_g=0
    for xx in ['cx','cz','ecr']:
        if xx not in opsD: continue
        n2q_g+=opsD[xx]
    
    #... store results
    tmd={'num_qubit': nqTot,'phys_qubits':physQubitLayout}
    tmd['transpile_backend']=transBackN

    tmd['2q_gate_depth']=len2
    tmd['1q_gate_count']=n1q_g
    tmd[ '2q_gate_count']= n2q_g
    md['transpile']=tmd

    md['payload'].update({'num_qubit':nqTot , 'num_clbit':qc.num_clbits})
    print('circ_transpMeta:'); pprint(tmd)
    
#...!...!..................
def qasm_save_one_circ(qc,md):
    outPath='out'
    pmd=md['payload']
    nqubit=pmd['num_qubit']
    myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    qasmF='qcrankEh_%dqub_%s.qasm'%(nqubit,myHN)
    qasmF=os.path.join(outPath,qasmF)
    qc.qasm(filename=qasmF)    # save QASM
    #print(circEL[0])
    print('save one non-parametric transpiled circ:',qasmF)

#...!...!..................
def qasm_save_all_circ(qcL,md,bigD,args):
    #  all circ added to bigD and NO execution
    md['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    
    if args.expName==None:
        md['short_name']='ibmexp_'+md['hash']
    else:
        md['short_name']=args.expName

    nCirc=len(qcL)
    outA=np.empty((nCirc), dtype='object')
    for ic in range(nCirc):
        outA[ic]=qcL[ic].qasm()
    bigD['circ_qasm']=outA
    outF=os.path.join(args.outPath,md['short_name']+'.qasm.h5')
    write4_data_hdf5(bigD,outF,md)

    print('bigD keys:', sorted(bigD))
    pprint(md)
    print('\nNO EXECUTION, save all circ to %s\n'%outF)
    exit(0)

#...!...!..................
def export_QPY_circs(qcL,md,args):
    md['hash']=hashlib.md5(os.urandom(32)).hexdigest()[:6]
    if args.expName==None:
        md['short_name']='exp_'+md['hash']
    else:
        md['short_name']=args.expName

    outF=md['short_name']+'_circ.qpy'
    outFF=os.path.join(args.outPath,outF)

    with open(outFF, 'wb') as fd:
        qpy.dump(qcL, fd)

    md['qpy_circ_fname']=outF
    print('Saved QPY circuits:',outFF)
   
#...!...!..................
def import_QPY_circs(md,args):
    
    outF=md['short_name']+'_circ.qpy'
    outFF=os.path.join(args.outPath,outF)
    print('Reading QPY circuits:',outFF)
    T0=time()
    with open(outFF, 'rb') as fd:
        qcL=qpy.load(fd)
    elaT=time()-T0
    print('   qpy.load elaT=%.1f sec'%(elaT))
    assert len(qcL)==md['payload']['num_sample']
    return qcL
    
        
#...!...!.................... 
def pack_npquasis_to_counts(md,bigD): # historic??
    pmd=md['payload']
    nclbit=pmd['num_clbit']
    fstr='0'+str(nclbit)+'b'  #.... converts int(mbits) back to bitstring
    nImg=pmd['num_sample']
    shots=md['submit']['num_shots']

    rawQuasis=bigD['raw_quasis']
    NB=1<<nclbit
    assert nImg,NB==rawQuasis.shape
    
    countsL=[ None for _ in range(nImg) ]
    for ic in range(nImg):
         countsL[ic] = { format(ikey,fstr):int(rawQuasis[ic,ikey]*shots)  for ikey in range(NB) if rawQuasis[ic,ikey]>0}
         #if prob<=0: continue # will loose negative probs, but SPAM wass applied alred?
    return countsL    

#...!...!.................... 
def pack_counts_to_numpy(md,bigD,countsL):
    pmd=md['payload']
    nclbit=pmd['num_clbit']
    nCirc=pmd['num_sample']
    akey=sorted(countsL[0])[0]
    #print('ddd5',akey,nclbit,len(akey))
    assert nclbit==len(akey)
    
    # Find the maximum number of keys in any dictionary
    max_keys = max(len(counts) for counts in countsL)
    #print('max keys:',max_keys)

    # Initialize output arrays  for sparse data storage
    assert nclbit <31  # change raw_iket storage to int64
    raw_ikey = np.full((nCirc, max_keys), -1, dtype=np.int32)  # Padded with -1
    raw_mshot = np.zeros((nCirc, max_keys), dtype=np.int32)    # Padded with 0
    raw_nkey = np.zeros(nCirc, dtype=np.int32)                 # Number of valid elements

    # Populate the arrays without the inner loop
    # can handle 2 types of keys:  [(0, 1, 1), (0, 0, 1),..] OR ['010', '110', ...] 
    for ic in range(nCirc):
        counts=countsL[ic]
        # Sort counts dictionary by the number of occurrences
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

        # Extract keys and counts into separate lists
        keys, counts = zip(*sorted_counts)

        # Check the type of the first element in the list and convert tuple to bitstring , if needed
        if keys and isinstance(keys[0], tuple):
            keys2= [''.join(str(bit) for bit in tup) for tup in keys]
            keys=keys2
        # Convert bitstring keys to integers and store in raw_key array
        raw_ikey[ic, :len(keys)] = np.array([int(k, 2) for k in keys], dtype=np.int32)

        # Store counts in raw_mshot array
        raw_mshot[ic, :len(counts)] = np.array(counts, dtype=np.int32)

        # Update raw_nkey with the actual number of entries
        raw_nkey[ic] = len(sorted_counts)


    # 'raw_ikey' and 'raw_mshot' are now 2D arrays padded with -1 and 0, respectively.
    # 'raw_nkey' is a 1D array indicating the number of valid entries for each circuit.


    #print('eee',raw_ikey.shape)
    bigD['raw_nkey']=raw_nkey
    bigD['raw_ikey']=raw_ikey
    bigD['raw_mshot']=raw_mshot
   
#...!...!.................... 
def unpack_numpy_to_counts(md,expD):
    pmd=md['payload']
    nclbit=pmd['num_clbit']
    fstr='0'+str(nclbit)+'b'  #.... converts int(mbits) back to bitstring
    nImg=pmd['num_sample']

    # recover raw data
    raw_nkey= expD['raw_nkey']  # number of uniqe keys per circ, rest is padded
    raw_ikey= expD['raw_ikey']  # sorted keys
    raw_mshot=expD['raw_mshot'] # sorted yields
    countsL=[ None for _ in range(nImg) ]
    for ic in range(nImg):
        nkey=raw_nkey[ic]
        ikeyV=raw_ikey[ic][:nkey]
        mshotV=raw_mshot[ic][:nkey]
        countsL[ic] = { format(ikeyV[i],fstr):mshotV[i]  for  i, val in enumerate(ikeyV)}
    return countsL
                
#...!...!....................
def circ_depth_aziz(qc,text='myCirc'):   # from Aziz @ IBMQ    summer 2023
    len1=qc.depth(filter_function=lambda x: x.operation.num_qubits == 1)
    len2x=qc.depth(filter_function=lambda x: x.operation.name == 'cx')
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )
    len3=qc.depth(filter_function=lambda x: x.operation.num_qubits ==3 )
    len4=qc.depth(filter_function=lambda x: x.operation.num_qubits > 3 )
    opsOD=qc.count_ops()  # ordered dict
    opsD={ k:opsOD[k] for k in opsOD}
    opsD['qubits']=qc.num_qubits

    depthD={'cx':len2x,'2q':len2,'3q':len3,'4q+':len4,'1q':len1}
    
    print('%s depth_aziz:'%text,depthD,', ops:',opsD)
    return depthD,opsD

      
#...!...!....................
def measL_int2bits(probsIL,nclbit): # converts int(mbits) back to bitstring
    nCirc=len(probsIL)
    probsBL=[ None for i in range(nCirc) ]
    for ic in range(nCirc):
        probsI=probsIL[ic]
        #1probsI=probsI.nearest_probability_distribution() # make probs in [0,1]
        #print('MI2B: ic=%d repack %d keys'%(ic,len(probsI)))
        probs = {}
        for key, val in probsI.items():
            mbit=format(key,'0'+str(nclbit)+'b')
            probs[mbit] = val
            #print(key,mbit,val)
        probsBL[ic]=probs
    return probsBL

