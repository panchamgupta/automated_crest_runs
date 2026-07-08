#!/bin/bash

python run_crest_from_sdf.py \
	--sdf B3GNT2_only_acyclic_indane_THF_3D_prepared.sdf \
	--outdir crest_runs \
	--crest-exe /home/pgupta11/anaconda3/envs/crest/bin/crest \
	--xtb-exe /home/pgupta11/anaconda3/envs/crest/bin/xtb \
	--use-toml \
	--search-method gfn2_gfnff_sp \
	--alpb water \
	--ewin 10.0 \
	--rthr 0.15 \
	--threads 32 \
	--dry-run \
	--no-quick \
