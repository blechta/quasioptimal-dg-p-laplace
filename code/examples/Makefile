.DEFAULT_GOAL := all
exec = python
pre_primal:
	mkdir -p output/primal/
pre_mixed:
	mkdir -p output/mixed/

square.msh: square.geo
	gmsh -2 -clscale 0.3 square.geo -smooth 3

cube.msh: cube.geo
	gmsh -3 -clscale 1.0 cube.geo -algo del3d -optimize_netgen 5 -smooth 10

primal_cg: pre_primal square.msh
	$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr newtonian --alpha 1.01 2>&1 | tee output/primal/CG_a_1.01_newtonian_nref_6.log
	for num in 1 2 3 4 5 ; do \
		$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr thinning --p-s $$num --alpha 1.01 2>&1 | tee "output/primal/CG_a_1.01_p--'$$num'_thinning_nref_6.log" ;\
		$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 | tee "output/primal/CG_a_1.01_p--'$$num'_thickening_nref_6.log" ;\
	done

primal_cr: pre_primal square.msh
	$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --alpha 1.01 2>&1 | tee output/primal/CR_a_1.01_newtonian_nref_6.log
	for num in 1 2 3 4 5 ; do \
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --alpha 1.01 2>&1 | tee "output/primal/CR_a_1.01_p--'$$num'_thinning_nref_6.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --smoothing --alpha 1.01 2>&1 | tee "output/primal/CR_a_1.01_p--'$$num'_thinning_nref_6_smoothing.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 | tee "output/primal/CR_a_1.01_p--'$$num'_thickening_nref_6.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 --smoothing | tee "output/primal/CR_a_1.01_p--'$$num'_thickening_nref_6_smoothing.log" ;\
	done

primal_dg: pre_primal square.msh
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --alpha 1.01 2>&1 | tee output/primal/DG_a_1.01_newtonian_nref_6.log
	for num in 1 2 3 4 5 ; do \
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --alpha 1.01 2>&1 | tee "output/primal/DG_a_1.01_p--'$$num'_thinning_nref_6.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --smoothing --alpha 1.01 2>&1 | tee "output/primal/DG_a_1.01_p--'$$num'_thinning_nref_6_smoothing.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 | tee "output/primal/DG_a_1.01_p--'$$num'_thickening_nref_6.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --no-shift --alpha 1.01 2>&1 | tee "output/primal/DG_a_1.01_p--'$$num'_thickening_nref_6_no-shift.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 --smoothing | tee "output/primal/DG_a_1.01_p--'$$num'_thickening_nref_6_smoothing.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 --smoothing --no-shift | tee "output/primal/DG_a_1.01_p--'$$num'_thickening_nref_6_smoothing_no-shift.log" ;\
	done

mixed_cg: pre_mixed square.msh
	$(exec) plaw_mixed_eoc.py --disc CG --nrefs 6 --cr newtonian --alpha 1.01 2>&1 | tee output/mixed/CG_a_1.01_newtonian_nref_6.log
	for num in 1 2 3 4 5 ; do \
		$(exec) plaw_mixed_eoc.py --disc CG --nrefs 6 --cr thinning --p-s $$num --alpha 1.01 2>&1 | tee "output/mixed/CG_a_1.01_p--'$$num'_thinning_nref_6.log" ;\
		$(exec) plaw_mixed_eoc.py --disc CG --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 | tee "output/mixed/CG_a_1.01_p--'$$num'_thickening_nref_6.log" ;\
	done

mixed_cr: pre_mixed square.msh
	$(exec) plaw_mixed_eoc.py --disc CR --nrefs 6 --cr newtonian --alpha 1.01 2>&1 | tee output/mixed/CR_a_1.01_newtonian_nref_6.log
	for num in 1 2 3 4 5 ; do \
		$(exec) plaw_mixed_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --alpha 1.01 2>&1 | tee "output/mixed/CR_a_1.01_p--'$$num'_thinning_nref_6.log" ;\
		$(exec) plaw_mixed_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --smoothing --alpha 1.01 2>&1 | tee "output/mixed/CR_a_1.01_p--'$$num'_thinning_nref_6_smoothing.log" ;\
		$(exec) plaw_mixed_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 | tee "output/mixed/CR_a_1.01_p--'$$num'_thickening_nref_6.log" ;\
		$(exec) plaw_mixed_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 --smoothing | tee "output/mixed/CR_a_1.01_p--'$$num'_thickening_nref_6_smoothing.log" ;\
	done

mixed_dg: pre_mixed square.msh
	$(exec) plaw_mixed_eoc.py --disc DG --nrefs 6 --cr newtonian --alpha 1.01 2>&1 | tee output/mixed/DG_a_1.01_newtonian_nref_6.log
	for num in 1 2 3 4 5 ; do \
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --alpha 1.01 2>&1 | tee "output/mixed/DG_a_1.01_p--'$$num'_thinning_nref_6.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --smoothing --alpha 1.01 2>&1 | tee "output/mixed/DG_a_1.01_p--'$$num'_thinning_nref_6_smoothing.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 | tee "output/mixed/DG_a_1.01_p--'$$num'_thickening_nref_6.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --no-shift --alpha 1.01 2>&1 | tee "output/mixed/DG_a_1.01_p--'$$num'_thickening_nref_6_no-shift.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 --smoothing | tee "output/mixed/DG_a_1.01_p--'$$num'_thickening_nref_6_smoothing.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --alpha 1.01 2>&1 --smoothing --no-shift | tee "output/mixed/DG_a_1.01_p--'$$num'_thickening_nref_6_smoothing_no-shift.log" ;\
	done

primal: primal_cg primal_cr primal_dg

mixed: mixed_cg mixed_cr mixed_dg

all: primal mixed
