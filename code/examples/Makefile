.DEFAULT_GOAL := all
exec = python
pre_primal:
	mkdir -p output/primal/
pre_mixed:
	mkdir -p output/mixed/

square_str.msh: square_str.geo
	gmsh -2 -clscale 0.36 square_str.geo

primal_cr: pre_primal square_str.msh
	$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/primal/CR_rate_1.0_newtonian_nref_6_.log
	$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/primal/CR_rate_0.5_newtonian_nref_6_.log
	$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/primal/CR_rate_0.2_newtonian_nref_6_.log
	for num in 2 4 ; do \
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --smoothing --beta 1.0 2>&1 | tee "output/primal/CR_rate_1.0_p--'$$num'_thinning_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --beta 1.0 2>&1 --smoothing | tee "output/primal/CR_rate_1.0_p--'$$num'_thickening_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --smoothing --beta 0.5 2>&1 | tee "output/primal/CR_rate_0.5_p--'$$num'_thinning_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --beta 0.5 2>&1 --smoothing | tee "output/primal/CR_rate_0.5_p--'$$num'_thickening_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --smoothing --beta 0.2 2>&1 | tee "output/primal/CR_rate_0.2_p--'$$num'_thinning_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --beta 0.2 2>&1 --smoothing | tee "output/primal/CR_rate_0.2_p--'$$num'_thickening_nref_6_.log" ;\
	done

primal_dg: pre_primal square_str.msh
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/primal/DG_rate_1.0_newtonian_nref_6_.log
	$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/primal/LDG_rate_1.0_newtonian_nref_5_.log
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/primal/DG_rate_0.5_newtonian_nref_6_.log
	$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/primal/LDG_rate_0.5_newtonian_nref_5_.log
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/primal/DG_rate_0.2_newtonian_nref_6_.log
	$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/primal/LDG_rate_0.2_newtonian_nref_5_.log
	for num in 2 4 ; do \
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --smoothing --beta 1.0 2>&1 | tee "output/primal/DG_rate_1.0_p--'$$num'_thinning_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --beta 1.0 2>&1 --smoothing | tee "output/primal/DG_rate_1.0_p--'$$num'_thickening_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s $$num --beta 1.0 2>&1 --smoothing | tee "output/primal/LDG_rate_1.0_p--'$$num'_thinning_nref_5_.log" ;\
		$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s $$num --beta 1.0 2>&1 --smoothing | tee "output/primal/LDG_rate_1.0_p--'$$num'_thickening_nref_5_.log" ;\
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --smoothing --beta 0.5 2>&1 | tee "output/primal/DG_rate_0.5_p--'$$num'_thinning_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --beta 0.5 2>&1 --smoothing | tee "output/primal/DG_rate_0.5_p--'$$num'_thickening_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s $$num --beta 0.5 2>&1 --smoothing | tee "output/primal/LDG_rate_0.5_p--'$$num'_thinning_nref_5_.log" ;\
		$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s $$num --beta 0.5 2>&1 --smoothing | tee "output/primal/LDG_rate_0.5_p--'$$num'_thickening_nref_5_.log" ;\
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --smoothing --beta 0.2 2>&1 | tee "output/primal/DG_rate_0.2_p--'$$num'_thinning_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --beta 0.2 2>&1 --smoothing | tee "output/primal/DG_rate_0.2_p--'$$num'_thickening_nref_6_.log" ;\
		$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s $$num --beta 0.2 2>&1 --smoothing | tee "output/primal/LDG_rate_0.2_p--'$$num'_thinning_nref_5_.log" ;\
		$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s $$num --beta 0.2 2>&1 --smoothing | tee "output/primal/LDG_rate_0.2_p--'$$num'_thickening_nref_5_.log" ;\
	done

mixed_dg: pre_mixed square_str.msh
	$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/mixed/DG_rate_1.0_newtonian_nref_5_.log
	$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/mixed/DG_rate_0.5_newtonian_nref_5_.log
	$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/mixed/DG_rate_0.2_newtonian_nref_5_.log
	for num in 2 4 ; do \
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s $$num --smoothing --beta 1.0 2>&1 | tee "output/mixed/DG_rate_1.0_p--'$$num'_thinning_nref_5_.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s $$num --beta 1.0 2>&1 --smoothing | tee "output/mixed/DG_rate_1.0_p--'$$num'_thickening_nref_5_.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s $$num --smoothing --beta 0.5 2>&1 | tee "output/mixed/DG_rate_0.5_p--'$$num'_thinning_nref_5_.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s $$num --beta 0.5 2>&1 --smoothing | tee "output/mixed/DG_rate_0.5_p--'$$num'_thickening_nref_5_.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s $$num --smoothing --beta 0.2 2>&1 | tee "output/mixed/DG_rate_0.2_p--'$$num'_thinning_nref_5_.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s $$num --beta 0.2 2>&1 --smoothing | tee "output/mixed/DG_rate_0.2_p--'$$num'_thickening_nref_5_.log" ;\
	done

primal: primal_cr primal_dg

mixed: mixed_dg

all: primal mixed
