import os
import argparse
import numpy as np
import cupy as cp
import gc
import time

start_time = time.time()

parser = argparse.ArgumentParser(description="arg parser")

###
parser.add_argument('-nm', '--nucmask', type=str, required=True)
parser.add_argument('-bm', '--boundmask', type=str, required=True)
parser.add_argument("-ts", "--tilesize", type=str, required=True)
parser.add_argument("-ovs", "--overlapsize", type=str, required=True)
parser.add_argument('-o', '--output', type=str, required=True)
args = parser.parse_args()

oname = args.output

ts = int(args.tilesize)
ovs = int(args.overlapsize)

def memfp():
    mempool = cp.get_default_memory_pool()
    print("used bytes: " + str(mempool.used_bytes()), flush=True)
    print("total bytes: " + str(mempool.total_bytes()), flush=True)
    print(cp.cuda.runtime.memGetInfo(), flush=True)

# preflight
# bwn is a bound -> nuc dictionary 
bwn = {}
nnb = []
memfp()

# tile image based on input 
# WITH overlaps 
nuc_cpu = np.load(oname+"_nuc.npy")
bound_cpu = np.load(oname+"_bound.npy")

# nuc or bound can be used for shape, should be the same 
image_shape = nuc_cpu.shape
tile_size = (ts, ts)

ovl_tiles = []
row_step = tile_size[0] - ovs
col_step = tile_size[1] - ovs

max_row = image_shape[0]
max_col = image_shape[1]

row_starts = list(range(0, max_row, row_step))

col_starts = list(range(0, max_col, col_step))

for row in row_starts:
    for col in col_starts:
        row_end = min(row + tile_size[0], max_row)
        col_end = min(col + tile_size[1], max_col)
        ovl_tiles.append((row, row_end, col, col_end))

# tile format: rowstart, rowend, colstart, colend
print("number of tiles: " + str(len(ovl_tiles)), flush=True)

# parse through tiles to derive bwn, nnb, and no 
for ind, t in enumerate(ovl_tiles): 
    print("tiling start: " + str(ind), flush=True)

    # load onto gpu
    nuc_gpu = cp.asarray(nuc_cpu[t[0]:t[1], t[2]:t[3]])
    bound_gpu = cp.asarray(bound_cpu[t[0]:t[1], t[2]:t[3]])

    bound_unique_gpu = cp.unique(bound_gpu)

    # remove 0 (background)
    bound_unique_gpu = bound_unique_gpu[bound_unique_gpu != 0]

    # implement dynamic memory allocation later (maybe not necessary given heuristic of max number of labels per one tile...?)
    chunk = bound_unique_gpu
    
    # Create boundary masks for all boundary labels at once
    b_masks = cp.equal(bound_gpu[..., cp.newaxis], chunk)  # Shape: (bound_gp.size, len(bound_unique))

    # Extract the corresponding areas from the nucleus mask for all boundaries
    n_areas = [nuc_gpu[b_masks[:, :, i]] for i in range(b_masks.shape[2])] # literal area in the nuc mask indexed with bound mask 
    n_areas_u = [cp.unique(n_area) for n_area in n_areas] # unique values in the above nuc mask 
    n_areas_nz = [n_area[n_area != 0] for n_area in n_areas_u] # unique values with background(0) subtracted

    # Initialize best composition, labels
    best_comps = cp.full(len(chunk), 0.4, dtype=cp.float32)
    best_labels = cp.zeros(len(chunk), dtype=cp.int32)
    asize = cp.array([arr.size for arr in n_areas])

    # TODO: implement support for multi-nucleated cells 
    # Vectorize composition computation
    for i, unique_area in enumerate(n_areas_nz):
        if len(unique_area) == 0:
            continue
 
        # vectorized: get the composition by dividing the total number of pixels in the nucleus mask by the number of pixels in the boundary mask 
        n_mask_ps = cp.array([cp.sum(nuc_gpu == l) for l in unique_area]) # pixel sum of a nuclear mask for a given label
        incounts = cp.array([cp.sum(n_areas[i] == l) for l in unique_area]) # pixel sum of a given label in the nuclear segmentation indexed with respective boundary mask area 
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
        if best_labels[i] != 0:
            # Found a satisfactory nucleus
            if b not in bwn:
                bwn[b] = (int(best_labels[i]), int(asize[i]))
            # We want to capture the entirety of a cell within a tile, so we check the boundary size 
            # Largest boundary size == entire cell 
            else:
                if bwn[b][1] < asize[i]:
                    bwn[b] = (int(best_labels[i]), int(asize[i]))
        
    # memory cleanup 
    del nuc_gpu
    del bound_gpu
    del bound_unique_gpu

    del chunk
    del b_masks
    del n_areas
    del n_areas_u
    del n_areas_nz
    del best_comps
    del best_labels
    del asize
    cp._default_memory_pool.free_all_blocks()

    print("tiled: " + str(ind), flush=True)

#
preprop_time = time.time() 
print("finished preprop: " + str(preprop_time - start_time), flush=True)

# need to filter back bwn to simply be bound to nuc pairs without area 
# one to one 
bwn_oto = {k: v[0] for k, v in bwn.items()}

# without filtering
np.savetxt(oname+"_bwn.txt", list(bwn_oto.items()), fmt='%s', delimiter=': ')

# handle nuclei that are matched to multiple bounds
# count nuc label occurances 
value_counts = {}
for value in bwn_oto.values():
    value_counts[value] = value_counts.get(value, 0) + 1
# only keep matched nucs that appear once (rest are discarded to nnb and no)
bwn_filt = {k: v for k, v in bwn_oto.items() if value_counts[v] == 1}

# with filtering
np.savetxt(oname+"_bwnfilt.txt", list(bwn_filt.items()), fmt='%s', delimiter=': ')


# extract and cleanup
nuc_unique_cpu = np.unique(nuc_cpu)
bound_unique_cpu = np.unique(bound_cpu)

# remove 0 (background)
nuc_unique_cpu = nuc_unique_cpu[nuc_unique_cpu != 0]
bound_unique_cpu = bound_unique_cpu[bound_unique_cpu != 0]

# derive non-nucleated bounds set (note that these reference labels in the BOUNDARY segmentation mask)
# we describe a non-nucleated bound as a boundary that does not sufficiently match to any nucleus 
nnb = [bound for bound in bound_unique_cpu if bound not in bwn_filt.keys()]
np.save(oname+"_nnb.npy", np.array(nnb))

# derive nuclear only(no) set (note that these reference lables in the NUCLEAR segmentation mask) 
# we describe a nuclear only as a nucleus that was not sufficiently matched by any boundary 
no = [nuc for nuc in nuc_unique_cpu if nuc not in set(bwn_filt.values())]
np.save(oname+"_no.npy", np.array(no))

# sanity check for label composition and the existence of multi-nucleated cells 
# TODO: multiple bound assigned to one nuclei should not be possible with current paradigm, but sanity check is good 
# START SANITY CHECK 
nuc_unique_built = list(bwn_filt.values()) + list(no) 
bound_unique_built = list(bwn_filt.keys()) + list(nnb)

comp_nuc = set(nuc_unique_built) == set(nuc_unique_cpu)
comp_bound = set(bound_unique_built) == set(bound_unique_cpu)

print("Complete nuclear composition: " + str(comp_nuc), flush=True)
print("Complete boundary composition: " + str(comp_bound), flush=True)

"""
# check for dupes
nuc_comp1 = list(bwn_filt.values()) + list(no) 
nuc_comp2 = list(set(nuc_comp1))
print(len(nuc_comp1) == len(nuc_comp2), flush=True)
"""
# END SANITY CHECK

# memcheck
gc.collect()
memfp()

time2 = time.time()

# relabeled arrays  
nuc_relabeled = np.zeros(nuc_cpu.shape, dtype=np.uint32)
bound_relabeled = np.zeros(bound_cpu.shape, dtype=np.uint32)

# build label dicts 
bwn_labels = {}
for i, (key, value) in enumerate(bwn_filt.items(), start=1):
    bwn_labels[(key, value)] = i

no_labels = {}
for i, lab in enumerate(no, start=len(bwn_filt.values())+1):
    no_labels[lab] = i

nnb_labels = {}
for i, lab in enumerate(nnb, start=(len(bwn_filt.values())+len(no))+1):
    nnb_labels[lab] = i 

# 
print("relabeling...", flush=True)

# relabel 
for ind, t in enumerate(ovl_tiles): 
    print("tiling start: " + str(ind), flush=True)

    # load onto gpu
    nuc_gpu = cp.asarray(nuc_cpu[t[0]:t[1], t[2]:t[3]])
    bound_gpu = cp.asarray(bound_cpu[t[0]:t[1], t[2]:t[3]])

    nuc_unique_gpu = cp.unique(nuc_gpu)
    bound_unique_gpu = cp.unique(bound_gpu)

    # remove 0 (background)
    nuc_unique_gpu = nuc_unique_gpu[nuc_unique_gpu != 0]
    bound_unique_gpu = bound_unique_gpu[bound_unique_gpu != 0]

    ### BWN PROCESSING ###

    # check for those in bwn 
    bwn_keys_gpu = cp.asarray(list(bwn_filt.keys()))
    mask = cp.isin(bound_unique_gpu, bwn_keys_gpu)
    bwn_chunk_bound_gpu = bound_unique_gpu[mask]

    # do dict lookup on cpu and write back 
    bwn_chunk_cpu = cp.asnumpy(bwn_chunk_bound_gpu)  
    associated_values_cpu = [bwn_filt[key] for key in bwn_chunk_cpu]
    bwn_chunk_nuc_gpu = cp.asarray(associated_values_cpu)

    # get the respective labels for the matched pairs 
    curr_bwn_labels = []
    for b, n in zip(cp.asnumpy(bwn_chunk_bound_gpu), cp.asnumpy(bwn_chunk_nuc_gpu)):
        curr_bwn_labels.append(bwn_labels[(b, n)])

    # bwn_chunk_bound_gpu and bwn_chunk_nuc_gpu are aligned cupy arrays representing matched b&n masks
    bwn_nuc_masks = cp.equal(nuc_gpu[..., cp.newaxis], bwn_chunk_nuc_gpu)
    bwn_bound_masks = cp.equal(bound_gpu[..., cp.newaxis], bwn_chunk_bound_gpu)

    # write relabeled bwn, each loop iteration represents writing a label for the current chunk
    # technically can be bwn_chunk_bound_gpu.shape[0] as well; shapes should be equivalent here
    for i in range(bwn_chunk_nuc_gpu.shape[0]):
        nmask = bwn_nuc_masks[..., i]
        bmask = bwn_bound_masks[..., i]   
        bwn_label = curr_bwn_labels[i]

        # r = row, c = col
        # l = local, g = global

        # FOR NUC
        rl_nuc, cl_nuc = cp.nonzero(nmask)

        # tile offsets
        rg_nuc = rl_nuc + t[0]
        cg_nuc = cl_nuc + t[2]
        # relabel on cpu
        rg_nuc_cpu = cp.asnumpy(rg_nuc)
        cg_nuc_cpu = cp.asnumpy(cg_nuc)
        nuc_relabeled[rg_nuc_cpu, cg_nuc_cpu] = bwn_label

        # FOR BOUND
        rl_bound, cl_bound = cp.nonzero(bmask)
        # tile offsets
        rg_bound = rl_bound + t[0]
        cg_bound = cl_bound + t[2]
        # relabel on cpu
        rg_bound_cpu = cp.asnumpy(rg_bound)
        cg_bound_cpu = cp.asnumpy(cg_bound)
        bound_relabeled[rg_bound_cpu, cg_bound_cpu] = bwn_label

        del nmask, bmask
        del rl_nuc, cl_nuc, rg_nuc, cg_nuc
        del rl_bound, cl_bound, rg_bound, cg_bound
        cp._default_memory_pool.free_all_blocks()
    
    # memory cleanup 
    del bwn_keys_gpu
    del mask
    del bwn_chunk_bound_gpu
    del bwn_chunk_nuc_gpu
    del bwn_nuc_masks
    del bwn_bound_masks
    cp._default_memory_pool.free_all_blocks()

    ### NO(nuclear only) PROCESSING ###
    
    # check for those in no 
    no_gpu = cp.asarray(no)
    mask = cp.isin(nuc_unique_gpu, no_gpu)
    no_chunk_gpu = nuc_unique_gpu[mask]

    # get the respective no labels
    curr_no_labels = []
    for l in cp.asnumpy(no_chunk_gpu):
        curr_no_labels.append(no_labels[l])

    # get nuclear only masks 
    no_nuc_masks = cp.equal(nuc_gpu[..., cp.newaxis], no_chunk_gpu)

    # write relabeled no 
    for i in range(no_chunk_gpu.shape[0]):
        nmask = no_nuc_masks[..., i]
        no_label = curr_no_labels[i]

        # r = row, c = col
        # l = local, g = global 

        # FOR NUC (no -> nuclear only)
        rl_nuc, cl_nuc = cp.nonzero(nmask)

        # tile offsets 
        rg_nuc = rl_nuc + t[0]
        cg_nuc = cl_nuc + t[2]
        # relabel on cpu 
        rg_nuc_cpu = cp.asnumpy(rg_nuc)
        cg_nuc_cpu = cp.asnumpy(cg_nuc)
        nuc_relabeled[rg_nuc_cpu, cg_nuc_cpu] = no_label

        del nmask 
        del rl_nuc, cl_nuc, rg_nuc, cg_nuc
        cp._default_memory_pool.free_all_blocks()
    
    # memory cleanup 
    del no_gpu
    del mask
    del no_chunk_gpu
    del no_nuc_masks
    cp._default_memory_pool.free_all_blocks()

    ### NNB(non-nucleated bounds) PROCESSING ### 

    # check for those in nnb 
    nnb_gpu = cp.asarray(nnb)
    mask = cp.isin(bound_unique_gpu, nnb_gpu)
    nnb_chunk_gpu = bound_unique_gpu[mask]

    # get respective nnb labels
    curr_nnb_labels = []
    for l in cp.asnumpy(nnb_chunk_gpu):
        curr_nnb_labels.append(nnb_labels[l])

    # get non-nucleated bound masks 
    nnb_bound_masks = cp.equal(bound_gpu[..., cp.newaxis], nnb_chunk_gpu)

    # write relabeled to nnb 
    for i in range(nnb_chunk_gpu.shape[0]):
        bmask = nnb_bound_masks[..., i]
        nnb_label = curr_nnb_labels[i]

        # r = row, c = col
        # l = local, g = global 

        # FOR BOUND (nnb -> non-nucleated bounds)
        rl_bound, cl_bound = cp.nonzero(bmask)

        # tile offsets 
        rg_bound = rl_bound + t[0]
        cg_bound = cl_bound + t[2]
        # relabel on cpu
        rg_bound_cpu = cp.asnumpy(rg_bound)
        cg_bound_cpu = cp.asnumpy(cg_bound)
        bound_relabeled[rg_bound_cpu, cg_bound_cpu] = nnb_label

        del bmask
        del rl_bound, cl_bound, rg_bound, cg_bound
        cp._default_memory_pool.free_all_blocks()
    
    # memory cleanup
    del nnb_gpu
    del mask
    del nnb_chunk_gpu
    del nnb_bound_masks

    # final memory cleanup
    del nuc_gpu, bound_gpu
    del nuc_unique_gpu, bound_unique_gpu
    cp._default_memory_pool.free_all_blocks()

    print("tiled: " + str(ind), flush=True)

    
print(time.time() - time2, flush=True)

# memcheck
gc.collect()
memfp()

# save out relabeled arrays 
np.save(oname+"_nuc_ordered.npy", nuc_relabeled)
np.save(oname+"_bound_ordered.npy", bound_relabeled)
