import os
import argparse
import numpy as np
import cupy as cp
import gc
import time

parser = argparse.ArgumentParser(description="arg parser")

# mpath is the master path for a given run
parser.add_argument('-mp', '--masterpath', type=str, required=True)
args = parser.parse_args()

curr_run = os.path.basename(args.masterpath)

def memfp():
    mempool = cp.get_default_memory_pool()
    print("used bytes: " + str(mempool.used_bytes()))
    print("total bytes: " + str(mempool.total_bytes()))

memfp()

# preflight
print(cp.cuda.runtime.memGetInfo(), flush=True)

bwn = {}
nnb = []

print("Loading masks into memory...")

# load onto gpu
nuc_gp = cp.load(args.masterpath+"/"+curr_run+"_nuc.npy")
bound_gp = cp.load(args.masterpath+"/"+curr_run+"_bound.npy")

nuc_unique = cp.unique(nuc_gp)
bound_unique = cp.unique(bound_gp)

# subset for testing, first index removed is 0(background)
bound_unique = bound_unique[1:]

# precomp 
precomp = time.time()
# dynamic chunk size calc accounting for memory
bound_arrsize = bound_gp.nbytes
nuc_arrsize = nuc_gp.nbytes
# copied boolean arrays only take 1 byte per element vs 4
bool_arrsize = bound_arrsize / 4    
# overhead accounted for in avail_mem
avail_mem = cp.cuda.runtime.memGetInfo()[0]
# subtract overhead
avail_mem = avail_mem - (1 * 1024**3) - nuc_arrsize

chunk_size =  int(avail_mem // bool_arrsize)
print(chunk_size, flush=True)

chunks = [bound_unique[i:i + chunk_size] for i in range(0, len(bound_unique), chunk_size)]
for chunk in chunks: 
    # Create boundary masks for all boundary labels at once
    b_masks = cp.equal(bound_gp[..., cp.newaxis], chunk)  # Shape: (bound_gp.size, len(bound_unique))

    # Extract the corresponding areas from the nucleus mask for all boundaries
    n_areas = [nuc_gp[b_masks[:, :, i]] for i in range(b_masks.shape[2])]
    n_areas_u = [cp.unique(n_area) for n_area in n_areas]
    n_areas_nz = [n_area[n_area != 0] for n_area in n_areas_u]

    # Initialize best composition and labels
    best_comps = cp.full(len(chunk), 0.4, dtype=cp.float32)
    best_labels = cp.zeros(len(chunk), dtype=cp.int32)

    # Vectorize composition computation
    for i, unique_area in enumerate(n_areas_nz):
        if len(unique_area) == 0:
            continue

        # vectorized: get the composition by dividing number of pixels in the boundary mask by total number of pixels in that nucleus mask 
        n_mask_ps = cp.array([cp.sum(nuc_gp == l) for l in unique_area])
        incounts = cp.array([cp.sum(n_areas[i] == l) for l in unique_area])
        label_comps = incounts / n_mask_ps

        max_comp_idx = cp.argmax(label_comps)
        if label_comps[max_comp_idx] > best_comps[i]:
            best_comps[i] = label_comps[max_comp_idx]
            best_labels[i] = unique_area[max_comp_idx]

        del n_mask_ps
        del incounts 
        del label_comps
        del max_comp_idx
        cp._default_memory_pool.free_all_blocks()
    
    # Process results to update bwn and nnb
    chunk = chunk.get()
    for i, b in enumerate(chunk):
        if best_labels[i] == 0:
            # No satisfactory overlap
            nnb.append(b)
        else:
            # Found a satisfactory nucleus
            bwn[b] = int(best_labels[i])
        
    # memory cleanup 
    del b_masks
    del n_areas
    del n_areas_u
    del n_areas_nz
    del best_comps
    del best_labels
    cp._default_memory_pool.free_all_blocks()

# extract and cleanup
nuc_unique_cpu = nuc_unique.get()
bound_unique_cpu = bound_unique.get()

del nuc_unique
del bound_unique
del chunks
cp._default_memory_pool.free_all_blocks()

# 0 is background
no = nuc_unique_cpu[~np.isin(nuc_unique_cpu, list(bwn.values()))]
no = np.delete(no, 0)

# BWN IS NOT CLEANED!!! MN POSSIBLE!!!
np.savetxt(args.masterpath+"/"+curr_run+"_bwn.txt", list(bwn.items()), fmt='%s', delimiter=': ')
np.save(args.masterpath+"/"+curr_run+"_nnb.npy", np.array(nnb))
np.save(args.masterpath+"/"+curr_run+"_no.npy", no)

# this is outdated, needs update 
'''
# START SANITY CHECK 
nuc_unique_built = list(bwn.values()) + list(no) 
bound_unique_built = list(bwn.keys()) + list(nnb)

comp_nuc = set(nuc_unique_built) == set(np.delete(nuc_unique_cpu, 0))
comp_bound = set(bound_unique_built) == set(np.delete(bound_unique_cpu, 0))

print(comp_nuc, flush=True)
print(comp_bound, flush=True)

bo = bound_unique_cpu[~np.isin(bound_unique_cpu, list(bwn.keys()))]
print(set(nnb) == set(np.delete(bo, 0)), flush=True)
# END SANITY CHECK
'''

# memcheck step 
gc.collect()
memfp()
print(cp.cuda.runtime.memGetInfo(), flush=True)

print("finished precomp: " + str(time.time() - precomp), flush=True)

# relabeled arrays
nuc_relabeled = cp.zeros_like(nuc_gp)
bound_relabeled = cp.zeros_like(bound_gp)

# handle multi-nucleated cells(remove for now(simplicity), later deal with edge cases(real mn, highest % overlap, etc.))
# count nuc label occurances 
value_counts = {}
for value in bwn.values():
    value_counts[value] = value_counts.get(value, 0) + 1
# only keep matched nucs that appear once(no mn)
bwn_filt = {k: v for k, v in bwn.items() if value_counts[v] == 1}

np.savetxt(args.masterpath+"/"+curr_run+"_bwnfilt.txt", list(bwn.items()), fmt='%s', delimiter=': ')

matchedbound = time.time()
# MATCHED BOUND RELABEL 
# dynamic chunk size calc accounting for memory
bound_arrsize = bound_gp.nbytes
nuc_arrsize = nuc_gp.nbytes
# copied boolean arrays only take 1 byte per element vs 4; here we have one for nuc and one for bound 
bool_arrsize = 2*(bound_arrsize / 4)
# overhead accounted for in avail_mem
avail_mem = cp.cuda.runtime.memGetInfo()[0]
# subtract overhead
avail_mem = avail_mem - (1 * 1024**3) - nuc_arrsize - bound_arrsize

chunk_size =  int(avail_mem // bool_arrsize)
print("relab: " + str(chunk_size), flush=True)

chunks = [list(bwn_filt.items())[i:i + chunk_size] for i in range(0, len(bwn_filt), chunk_size)]
start_ind = 1

for chunk in chunks: 
    # indices for current chunk
    cinds = cp.array(range(start_ind, start_ind + len(chunk)), dtype=cp.uint32)
    start_ind += len(chunk)
    
    # get n and b segmentation masks at once 
    nuc_chunks = cp.array([e[1] for e in chunk])
    bound_chunks = cp.array([e[0] for e in chunk])
    nuc_masks = cp.equal(nuc_gp[..., cp.newaxis], nuc_chunks)
    bound_masks = cp.equal(bound_gp[..., cp.newaxis], bound_chunks)
    
    # accelerate this later, works quick enough for now 
    for i in range(nuc_masks.shape[2]):  
        nuc_relabeled[nuc_masks[:, :, i]] = cinds[i]
        bound_relabeled[bound_masks[:, :, i]] = cinds[i]

    # cleanup memory 
    del cinds
    del nuc_chunks
    del bound_chunks
    del nuc_masks
    del bound_masks
    cp._default_memory_pool.free_all_blocks()

# memcheck step 
gc.collect()
memfp()
print(cp.cuda.runtime.memGetInfo(), flush=True)

print(start_ind, flush=True)

print("finished matchedbound: " + str(time.time() - matchedbound), flush=True)

notime = time.time()
# NUCELAR ONLY FAKE BOUND RELABEL 
# dynamic chunk size calc accounting for memory
bound_arrsize = bound_gp.nbytes
nuc_arrsize = nuc_gp.nbytes
# copied boolean arrays only take 1 byte per element vs 4
bool_arrsize = (bound_arrsize / 4)
# overhead accounted for in avail_mem
avail_mem = cp.cuda.runtime.memGetInfo()[0]
# subtract overhead
avail_mem = avail_mem - (1 * 1024**3) - nuc_arrsize

chunk_size =  int(avail_mem // bool_arrsize)
print("relab: " + str(chunk_size), flush=True)

chunks = [cp.array(no[i:i + chunk_size]) for i in range(0, len(no), chunk_size)]

for chunk in chunks: 
    # indices for current chunk
    cinds = cp.array(range(start_ind, start_ind + len(chunk)), dtype=cp.uint32)
    start_ind += len(chunk)
    
    # get n and b segmentation masks at once 
    nuc_masks = cp.equal(nuc_gp[..., cp.newaxis], chunk)
    
    # accelerate this later, works quick enough for now 
    for i in range(nuc_masks.shape[2]):  
        nuc_relabeled[nuc_masks[:, :, i]] = cinds[i]
        bound_relabeled[nuc_masks[:, :, i]] = cinds[i]

    # cleanup memory 
    del cinds
    del nuc_masks
    cp._default_memory_pool.free_all_blocks()

# memcheck step 
gc.collect()
memfp()
print(cp.cuda.runtime.memGetInfo(), flush=True)

print(start_ind, flush=True)

print("finished no(fake): " + str(time.time() - notime), flush=True)

nnbtime = time.time()
# NON-NUCLEATED BOUNDS RELABEL 
# dynamic chunk size calc accounting for memory
bound_arrsize = bound_gp.nbytes
nuc_arrsize = nuc_gp.nbytes
# copied boolean arrays only take 1 byte per element vs 4
bool_arrsize = (bound_arrsize / 4)
# overhead accounted for in avail_mem
avail_mem = cp.cuda.runtime.memGetInfo()[0]
# subtract overhead
avail_mem = avail_mem - (1 * 1024**3) - bound_arrsize

chunk_size =  int(avail_mem // bool_arrsize)
print("relab: " + str(chunk_size), flush=True)

chunks = [cp.array(nnb[i:i + chunk_size]) for i in range(0, len(nnb), chunk_size)]

for chunk in chunks: 
    # indices for current chunk
    cinds = cp.array(range(start_ind, start_ind + len(chunk)), dtype=cp.uint32)
    start_ind += len(chunk)
    
    # get b segmentation masks at once 
    bound_masks = cp.equal(bound_gp[..., cp.newaxis], chunk)
    
    # accelerate this later, works quick enough for now 
    for i in range(bound_masks.shape[2]):  
        bound_relabeled[bound_masks[:, :, i]] = cinds[i]

    # cleanup memory 
    del cinds
    del bound_masks
    cp._default_memory_pool.free_all_blocks()

# memcheck step 
gc.collect()
memfp()
print(cp.cuda.runtime.memGetInfo(), flush=True)

print(start_ind, flush=True)

print("finished nnb: " + str(time.time() - nnbtime), flush=True)

# save the final relabeled outputs as the new basis(pre-XR)
np.save(args.masterpath+"/"+curr_run+"_nuc_ordered.npy", nuc_relabeled.get())
np.save(args.masterpath+"/"+curr_run+"_bound_ordered.npy", bound_relabeled.get())

# temp temp
del nuc_gp
del bound_gp 
del nuc_relabeled
del bound_relabeled
cp._default_memory_pool.free_all_blocks()

# memcheck step 
gc.collect()
memfp()
print(cp.cuda.runtime.memGetInfo())