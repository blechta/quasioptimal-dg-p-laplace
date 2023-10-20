all: 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60

square2.msh: square2.geo
	gmsh -save $<

output/:
	mkdir -p $@

01: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/CR_rate_1.0_p_2.0.log
02: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/CR_rate_0.5_p_2.0.log
03: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/CR_rate_0.2_p_2.0.log
04: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 1.0 2>&1 | tee output/CR_rate_1.0_p_1.7.log
05: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 2 --beta 1.0 2>&1 --smoothing | tee output/CR_rate_1.0_p_3.0.log
06: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.5 2>&1 | tee output/CR_rate_0.5_p_1.7.log
07: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 2 --beta 0.5 2>&1 --smoothing | tee output/CR_rate_0.5_p_3.0.log
08: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.2 2>&1 | tee output/CR_rate_0.2_p_1.7.log
09: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 2 --beta 0.2 2>&1 --smoothing | tee output/CR_rate_0.2_p_3.0.log
10: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 1.0 2>&1 | tee output/CR_rate_1.0_p_1.5.log
11: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 4 --beta 1.0 2>&1 --smoothing | tee output/CR_rate_1.0_p_4.5.log
12: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.5 2>&1 | tee output/CR_rate_0.5_p_1.5.log
13: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 4 --beta 0.5 2>&1 --smoothing | tee output/CR_rate_0.5_p_4.5.log
14: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.2 2>&1 | tee output/CR_rate_0.2_p_1.5.log
15: square2.msh | output/
	python -u plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 4 --beta 0.2 2>&1 --smoothing | tee output/CR_rate_0.2_p_4.5.log
16: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/DG_rate_1.0_p_2.0.log
17: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/LDG_rate_1.0_p_2.0.log
18: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/DG_rate_0.5_p_2.0.log
19: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/LDG_rate_0.5_p_2.0.log
20: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/DG_rate_0.2_p_2.0.log
21: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/LDG_rate_0.2_p_2.0.log
22: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 1.0 2>&1 | tee output/DG_rate_1.0_p_1.7.log
23: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 1.0 2>&1 --smoothing | tee output/DG_rate_1.0_p_3.0.log
24: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thinning --p-s 2 --beta 1.0 2>&1 --smoothing | tee output/LDG_rate_1.0_p_1.7.log
25: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thickening --p-s 2 --beta 1.0 2>&1 --smoothing | tee output/LDG_rate_1.0_p_3.0.log
26: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.5 2>&1 | tee output/DG_rate_0.5_p_1.7.log
27: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 0.5 2>&1 --smoothing | tee output/DG_rate_0.5_p_3.0.log
28: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thinning --p-s 2 --beta 0.5 2>&1 --smoothing | tee output/LDG_rate_0.5_p_1.7.log
29: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thickening --p-s 2 --beta 0.5 2>&1 --smoothing | tee output/LDG_rate_0.5_p_3.0.log
30: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.2 2>&1 | tee output/DG_rate_0.2_p_1.7.log
31: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 0.2 2>&1 --smoothing | tee output/DG_rate_0.2_p_3.0.log
32: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thinning --p-s 2 --beta 0.2 2>&1 --smoothing | tee output/LDG_rate_0.2_p_1.7.log
33: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thickening --p-s 2 --beta 0.2 2>&1 --smoothing | tee output/LDG_rate_0.2_p_3.0.log
34: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 1.0 2>&1 | tee output/DG_rate_1.0_p_1.5.log
35: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 1.0 2>&1 --smoothing | tee output/DG_rate_1.0_p_4.5.log
36: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thinning --p-s 4 --beta 1.0 2>&1 --smoothing | tee output/LDG_rate_1.0_p_1.5.log
37: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thickening --p-s 4 --beta 1.0 2>&1 --smoothing | tee output/LDG_rate_1.0_p_4.5.log
38: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.5 2>&1 | tee output/DG_rate_0.5_p_1.5.log
39: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 0.5 2>&1 --smoothing | tee output/DG_rate_0.5_p_4.5.log
40: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thinning --p-s 4 --beta 0.5 2>&1 --smoothing | tee output/LDG_rate_0.5_p_1.5.log
41: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thickening --p-s 4 --beta 0.5 2>&1 --smoothing | tee output/LDG_rate_0.5_p_4.5.log
42: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.2 2>&1 | tee output/DG_rate_0.2_p_1.5.log
43: square2.msh | output/
	python -u plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 0.2 2>&1 --smoothing | tee output/DG_rate_0.2_p_4.5.log
44: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thinning --p-s 4 --beta 0.2 2>&1 --smoothing | tee output/LDG_rate_0.2_p_1.5.log
45: square2.msh | output/
	python -u plaw_eoc.py --disc LDG --nrefs 6 --cr thickening --p-s 4 --beta 0.2 2>&1 --smoothing | tee output/LDG_rate_0.2_p_4.5.log
46: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/mDG_rate_1.0_p_2.0.log
47: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/mDG_rate_0.5_p_2.0.log
48: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/mDG_rate_0.2_p_2.0.log
49: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 1.0 2>&1 | tee output/mDG_rate_1.0_p_1.7.log
50: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 1.0 2>&1 --smoothing | tee output/mDG_rate_1.0_p_3.0.log
51: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.5 2>&1 | tee output/mDG_rate_0.5_p_1.7.log
52: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 0.5 2>&1 --smoothing | tee output/mDG_rate_0.5_p_3.0.log
53: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.2 2>&1 | tee output/mDG_rate_0.2_p_1.7.log
54: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 0.2 2>&1 --smoothing | tee output/mDG_rate_0.2_p_3.0.log
55: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 1.0 2>&1 | tee output/mDG_rate_1.0_p_1.5.log
56: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 1.0 2>&1 --smoothing | tee output/mDG_rate_1.0_p_4.5.log
57: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.5 2>&1 | tee output/mDG_rate_0.5_p_1.5.log
58: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 0.5 2>&1 --smoothing | tee output/mDG_rate_0.5_p_4.5.log
59: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.2 2>&1 | tee output/mDG_rate_0.2_p_1.5.log
60: square2.msh | output/
	python -u plaw_mixed_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 0.2 2>&1 --smoothing | tee output/mDG_rate_0.2_p_4.5.log
