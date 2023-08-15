.DEFAULT_GOAL := primal
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
	$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr newtonian 2>&1 | tee output/primal/CG_a_1.01_newtonian_nref_6.log
	for num in 1 2 ; do \
		$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr thinning --p-s $$num 2>&1 | tee output/primal/CG_a_1.01_thinning_nref_6.log ;\
		$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr thickening --p-s $$num 2>&1 | tee output/primal/CG_a_1.01_thickening_nref_6.log ;\
	done

primal_cr: pre_primal square.msh
	$(exec) plaw_eoc.py --disc CR --nrefs 2 --cr newtonian 2>&1 | tee output/primal/CR_a_1.01_newtonian_nref_6.log
	for num in 1 2 ; do \
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num 2>&1 | tee output/primal/CR_a_1.01_thinning_nref_6.log ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --smoothing 2>&1 | tee output/primal/CR_a_1.01_thinning_nref_6_smoothing.log ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num 2>&1 | tee output/primal/CR_a_1.01_thickening_nref_6.log ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num 2>&1 --smoothing | tee output/primal/CR_a_1.01_thickening_nref_6_smoothing.log ;\
	done

primal_dg: pre_primal square.msh
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr newtonian 2>&1 | tee output/primal/DG_a_1.01_newtonian_nref_6.log
	for num in 1 2 ; do \
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num 2>&1 | tee output/primal/DG_a_1.01_thinning_nref_6.log ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --smoothing 2>&1 | tee output/primal/DG_a_1.01_thinning_nref_6_smoothing.log ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num 2>&1 | tee output/primal/DG_a_1.01_thickening_nref_6.log ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --no-shift 2>&1 | tee output/primal/DG_a_1.01_thickening_nref_6_no-shift.log ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num 2>&1 --smoothing | tee output/primal/DG_a_1.01_thickening_nref_6_smoothing.log ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num 2>&1 --smoothing --no-shift | tee output/primal/DG_a_1.01_thickening_nref_6_smoothing_no-shift.log ;\
	done

primal: primal_cg primal_cr primal_dg

