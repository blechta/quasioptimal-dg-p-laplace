.DEFAULT_GOAL := all
exec = python
pre_primal:
	mkdir -p output/primal/
pre_mixed:
	mkdir -p output/mixed/
pre_test:
	mkdir -p output/tests/

square.msh: square.geo
	gmsh -2 -clscale 0.3 square.geo -smooth 3
#	gmsh -2 -clscale 0.36 square_str.geo

cube.msh: cube.geo
	gmsh -3 -clscale 0.75 cube.geo -algo del3d -optimize_netgen 5 -smooth 10

primal_cg: pre_primal square.msh
	$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr newtonian --beta 0.75 2>&1 | tee output/primal/CG_rate_0.75_newtonian_nref_6.log
	for num in 1 2 3 4 5 ; do \
		$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr thinning --p-s $$num --beta 0.75 2>&1 | tee "output/primal/CG_rate_0.75_p--'$$num'_thinning_nref_6.log" ;\
		$(exec) plaw_eoc.py --disc CG --nrefs 6 --cr thickening --p-s $$num --beta 0.75 2>&1 | tee "output/primal/CG_rate_0.75_p--'$$num'_thickening_nref_6.log" ;\
#	done

primal_cr: pre_primal square.msh
	$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/primal/CR_rate_1.0_newtonian_nref_6_smoothing.log
	for num in 1 2 3 4 5 ; do \
#		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --beta 1.0 2>&1 | tee "output/primal/CR_rate_1.0_p--'$$num'_thinning_nref_6.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s $$num --smoothing --beta 1.0 2>&1 | tee "output/primal/CR_rate_1.0_p--'$$num'_thinning_nref_6_smoothing.log" ;\
#		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --beta 1.0 2>&1 | tee "output/primal/CR_rate_1.0_p--'$$num'_thickening_nref_6.log" ;\
		$(exec) plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s $$num --beta 1.0 2>&1 --smoothing | tee "output/primal/CR_rate_1.0_p--'$$num'_thickening_nref_6_smoothing.log" ;\
	done

primal_dg: pre_primal square.msh
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/primal/DG_rate_0.5_newtonian_nref_6_smoothing.log
	$(exec) plaw_eoc.py --disc LDG --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/primal/LDG_rate_0.5_newtonian_nref_6_smoothing.log
	for num in 1 2 3 4 5 ; do \
#		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --beta 0.5 2>&1 | tee "output/primal/DG_rate_0.5_p--'$$num'_thinning_nref_6.log" ;\
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s $$num --smoothing --beta 0.5 2>&1 | tee "output/primal/DG_rate_0.5_p--'$$num'_thinning_nref_6_smoothing.log" ;\
#		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --beta 0.5 2>&1 | tee "output/primal/DG_rate_0.5_p--'$$num'_thickening_nref_6.log" ;\
#		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --no-shift --beta 0.5 2>&1 | tee "output/primal/DG_rate_0.5_p--'$$num'_thickening_nref_6_no-shift.log" ;\
		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --beta 0.5 2>&1 --smoothing | tee "output/primal/DG_rate_0.5_p--'$$num'_thickening_nref_6_smoothing.log" ;\
#		$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s $$num --beta 0.5 2>&1 --smoothing --no-shift | tee "output/primal/DG_rate_0.5_p--'$$num'_thickening_nref_6_smoothing_no-shift.log" ;\
#	$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s $$num --smoothing --beta 0.5 2>&1 | tee "output/primal/LDG_rate_0.5_p--'$$num'_thinning_nref_5_smoothing.log" ;\
		$(exec) plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s $$num --beta 0.5 2>&1 --smoothing | tee "output/primal/LDG_rate_0.5_p--'$$num'_thickening_nref_5_smoothing.log" ;\
	done

mixed_cg: pre_mixed square.msh
	$(exec) plaw_mixed_eoc.py --disc CG --nrefs 5 --cr newtonian --beta 1.0 2>&1 | tee output/mixed/CG_rate_1.0_newtonian_nref_5.log
	for num in 1 2 3 4 5 ; do \
		$(exec) plaw_mixed_eoc.py --disc CG --nrefs 5 --cr thinning --p-s $$num --beta 1.0 2>&1 | tee "output/mixed/CG_rate_1.0_p--'$$num'_thinning_nref_5.log" ;\
		$(exec) plaw_mixed_eoc.py --disc CG --nrefs 5 --cr thickening --p-s $$num --beta 1.0 2>&1 | tee "output/mixed/CG_rate_1.0_p--'$$num'_thickening_nref_5.log" ;\
	done

mixed_cr: pre_mixed square.msh
#	$(exec) plaw_mixed_eoc.py --disc CR --nrefs 5 --cr newtonian --beta 1.0 2>&1 | tee output/mixed/CR_rate_1.0_newtonian_nref_5.log
#	for num in 1 2 3 4 5 ; do \
#		$(exec) plaw_mixed_eoc.py --disc CR --nrefs 5 --cr thinning --p-s $$num --beta 1.0 2>&1 | tee "output/mixed/CR_rate_1.0_p--'$$num'_thinning_nref_5.log" ;\
#		$(exec) plaw_mixed_eoc.py --disc CR --nrefs 5 --cr thinning --p-s $$num --smoothing --beta 1.0 2>&1 | tee "output/mixed/CR_rate_1.0_p--'$$num'_thinning_nref_5_smoothing.log" ;\
#		$(exec) plaw_mixed_eoc.py --disc CR --nrefs 5 --cr thickening --p-s $$num --beta 1.0 2>&1 | tee "output/mixed/CR_rate_1.0_p--'$$num'_thickening_nref_5.log" ;\
#		$(exec) plaw_mixed_eoc.py --disc CR --nrefs 5 --cr thickening --p-s $$num --beta 1.0 2>&1 --smoothing | tee "output/mixed/CR_rate_1.0_p--'$$num'_thickening_nref_5_smoothing.log" ;\
#	done

mixed_dg: pre_mixed square.msh
	$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/mixed/DG_rate_1.0_newtonian_nref_5_smoothing.log
	for num in 1 2 3 4 5 ; do \
#		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s $$num --beta 1.0 2>&1 | tee "output/mixed/DG_rate_1.0_p--'$$num'_thinning_nref_5.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s $$num --smoothing --beta 1.0 2>&1 | tee "output/mixed/DG_rate_1.0_p--'$$num'_thinning_nref_5_smoothing.log" ;\
#		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s $$num --beta 1.0 2>&1 | tee "output/mixed/DG_rate_1.0_p--'$$num'_thickening_nref_5.log" ;\
#		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s $$num --no-shift --beta 1.0 2>&1 | tee "output/mixed/DG_rate_1.0_p--'$$num'_thickening_nref_5_no-shift.log" ;\
		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s $$num --beta 1.0 2>&1 --smoothing | tee "output/mixed/DG_rate_1.0_p--'$$num'_thickening_nref_5_smoothing.log" ;\
#		$(exec) plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s $$num --beta 1.0 2>&1 --smoothing --no-shift | tee "output/mixed/DG_rate_1.0_p--'$$num'_thickening_nref_5_smoothing_no-shift.log" ;\
	done

test_primal: pre_test
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.25 2>&1 | tee output/tests/primalDG_nref_6_beta_0.25_newtonian.log
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.25 --smoothing 2>&1 | tee output/tests/primalDG_nref_6_beta_0.25_newtonian_smoothing.log
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning -p-s 1 --beta 0.25 2>&1 | tee output/tests/primalDG_nref_6_beta_0.25_thinning.log
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thinning -p-s 1 --beta 0.25 --smoothing 2>&1 | tee output/tests/primalDG_nref_6_beta_0.25_thinning_smoothing.log
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening -p-s 1 --beta 0.25 2>&1 | tee output/tests/primalDG_nref_6_beta_0.25_thickening.log
	$(exec) plaw_eoc.py --disc DG --nrefs 6 --cr thickening -p-s 1 --beta 0.25 --smoothing 2>&1 | tee output/tests/primalDG_nref_6_beta_0.25_thickening_smoothing.log

primal: primal_cg primal_cr primal_dg

mixed: mixed_cg mixed_cr mixed_dg

all: primal mixed
