all: 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60

01:
	python plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/primal/CR_rate_1.0_newtonian_nref_6_.log
02:
	python plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/primal/CR_rate_0.5_newtonian_nref_6_.log
03:
	python plaw_eoc.py --disc CR --nrefs 6 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/primal/CR_rate_0.2_newtonian_nref_6_.log
04:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 1.0 2>&1 | tee "output/primal/CR_rate_1.0_p--2_thinning_nref_6_.log"
05:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 2 --beta 1.0 2>&1 --smoothing | tee "output/primal/CR_rate_1.0_p--2_thickening_nref_6_.log"
06:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.5 2>&1 | tee "output/primal/CR_rate_0.5_p--2_thinning_nref_6_.log"
07:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 2 --beta 0.5 2>&1 --smoothing | tee "output/primal/CR_rate_0.5_p--2_thickening_nref_6_.log"
08:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.2 2>&1 | tee "output/primal/CR_rate_0.2_p--2_thinning_nref_6_.log"
09:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 2 --beta 0.2 2>&1 --smoothing | tee "output/primal/CR_rate_0.2_p--2_thickening_nref_6_.log"
10:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 1.0 2>&1 | tee "output/primal/CR_rate_1.0_p--4_thinning_nref_6_.log"
11:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 4 --beta 1.0 2>&1 --smoothing | tee "output/primal/CR_rate_1.0_p--4_thickening_nref_6_.log"
12:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.5 2>&1 | tee "output/primal/CR_rate_0.5_p--4_thinning_nref_6_.log"
13:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 4 --beta 0.5 2>&1 --smoothing | tee "output/primal/CR_rate_0.5_p--4_thickening_nref_6_.log"
14:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.2 2>&1 | tee "output/primal/CR_rate_0.2_p--4_thinning_nref_6_.log"
15:
	python plaw_eoc.py --disc CR --nrefs 6 --cr thickening --p-s 4 --beta 0.2 2>&1 --smoothing | tee "output/primal/CR_rate_0.2_p--4_thickening_nref_6_.log"
16:
	python plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/primal/DG_rate_1.0_newtonian_nref_6_.log
17:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/primal/LDG_rate_1.0_newtonian_nref_5_.log
18:
	python plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/primal/DG_rate_0.5_newtonian_nref_6_.log
19:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/primal/LDG_rate_0.5_newtonian_nref_5_.log
20:
	python plaw_eoc.py --disc DG --nrefs 6 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/primal/DG_rate_0.2_newtonian_nref_6_.log
21:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/primal/LDG_rate_0.2_newtonian_nref_5_.log
22:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 1.0 2>&1 | tee "output/primal/DG_rate_1.0_p--2_thinning_nref_6_.log"
23:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 1.0 2>&1 --smoothing | tee "output/primal/DG_rate_1.0_p--2_thickening_nref_6_.log"
24:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s 2 --beta 1.0 2>&1 --smoothing | tee "output/primal/LDG_rate_1.0_p--2_thinning_nref_5_.log"
25:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s 2 --beta 1.0 2>&1 --smoothing | tee "output/primal/LDG_rate_1.0_p--2_thickening_nref_5_.log"
26:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.5 2>&1 | tee "output/primal/DG_rate_0.5_p--2_thinning_nref_6_.log"
27:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 0.5 2>&1 --smoothing | tee "output/primal/DG_rate_0.5_p--2_thickening_nref_6_.log"
28:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s 2 --beta 0.5 2>&1 --smoothing | tee "output/primal/LDG_rate_0.5_p--2_thinning_nref_5_.log"
29:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s 2 --beta 0.5 2>&1 --smoothing | tee "output/primal/LDG_rate_0.5_p--2_thickening_nref_5_.log"
30:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 2 --smoothing --beta 0.2 2>&1 | tee "output/primal/DG_rate_0.2_p--2_thinning_nref_6_.log"
31:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 2 --beta 0.2 2>&1 --smoothing | tee "output/primal/DG_rate_0.2_p--2_thickening_nref_6_.log"
32:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s 2 --beta 0.2 2>&1 --smoothing | tee "output/primal/LDG_rate_0.2_p--2_thinning_nref_5_.log"
33:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s 2 --beta 0.2 2>&1 --smoothing | tee "output/primal/LDG_rate_0.2_p--2_thickening_nref_5_.log"
34:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 1.0 2>&1 | tee "output/primal/DG_rate_1.0_p--4_thinning_nref_6_.log"
35:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 1.0 2>&1 --smoothing | tee "output/primal/DG_rate_1.0_p--4_thickening_nref_6_.log"
36:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s 4 --beta 1.0 2>&1 --smoothing | tee "output/primal/LDG_rate_1.0_p--4_thinning_nref_5_.log"
37:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s 4 --beta 1.0 2>&1 --smoothing | tee "output/primal/LDG_rate_1.0_p--4_thickening_nref_5_.log"
38:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.5 2>&1 | tee "output/primal/DG_rate_0.5_p--4_thinning_nref_6_.log"
39:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 0.5 2>&1 --smoothing | tee "output/primal/DG_rate_0.5_p--4_thickening_nref_6_.log"
40:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s 4 --beta 0.5 2>&1 --smoothing | tee "output/primal/LDG_rate_0.5_p--4_thinning_nref_5_.log"
41:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s 4 --beta 0.5 2>&1 --smoothing | tee "output/primal/LDG_rate_0.5_p--4_thickening_nref_5_.log"
42:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thinning --p-s 4 --smoothing --beta 0.2 2>&1 | tee "output/primal/DG_rate_0.2_p--4_thinning_nref_6_.log"
43:
	python plaw_eoc.py --disc DG --nrefs 6 --cr thickening --p-s 4 --beta 0.2 2>&1 --smoothing | tee "output/primal/DG_rate_0.2_p--4_thickening_nref_6_.log"
44:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thinning --p-s 4 --beta 0.2 2>&1 --smoothing | tee "output/primal/LDG_rate_0.2_p--4_thinning_nref_5_.log"
45:
	python plaw_eoc.py --disc LDG --nrefs 5 --cr thickening --p-s 4 --beta 0.2 2>&1 --smoothing | tee "output/primal/LDG_rate_0.2_p--4_thickening_nref_5_.log"
46:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr newtonian --beta 1.0 --smoothing 2>&1 | tee output/mixed/DG_rate_1.0_newtonian_nref_5_.log
47:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr newtonian --beta 0.5 --smoothing 2>&1 | tee output/mixed/DG_rate_0.5_newtonian_nref_5_.log
48:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr newtonian --beta 0.2 --smoothing 2>&1 | tee output/mixed/DG_rate_0.2_newtonian_nref_5_.log
49:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s 2 --smoothing --beta 1.0 2>&1 | tee "output/mixed/DG_rate_1.0_p--2_thinning_nref_5_.log"
50:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s 2 --beta 1.0 2>&1 --smoothing | tee "output/mixed/DG_rate_1.0_p--2_thickening_nref_5_.log"
51:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s 2 --smoothing --beta 0.5 2>&1 | tee "output/mixed/DG_rate_0.5_p--2_thinning_nref_5_.log"
52:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s 2 --beta 0.5 2>&1 --smoothing | tee "output/mixed/DG_rate_0.5_p--2_thickening_nref_5_.log"
53:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s 2 --smoothing --beta 0.2 2>&1 | tee "output/mixed/DG_rate_0.2_p--2_thinning_nref_5_.log"
54:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s 2 --beta 0.2 2>&1 --smoothing | tee "output/mixed/DG_rate_0.2_p--2_thickening_nref_5_.log"
55:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s 4 --smoothing --beta 1.0 2>&1 | tee "output/mixed/DG_rate_1.0_p--4_thinning_nref_5_.log"
56:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s 4 --beta 1.0 2>&1 --smoothing | tee "output/mixed/DG_rate_1.0_p--4_thickening_nref_5_.log"
57:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s 4 --smoothing --beta 0.5 2>&1 | tee "output/mixed/DG_rate_0.5_p--4_thinning_nref_5_.log"
58:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s 4 --beta 0.5 2>&1 --smoothing | tee "output/mixed/DG_rate_0.5_p--4_thickening_nref_5_.log"
59:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thinning --p-s 4 --smoothing --beta 0.2 2>&1 | tee "output/mixed/DG_rate_0.2_p--4_thinning_nref_5_.log"
60:
	python plaw_mixed_eoc.py --disc DG --nrefs 5 --cr thickening --p-s 4 --beta 0.2 2>&1 --smoothing | tee "output/mixed/DG_rate_0.2_p--4_thickening_nref_5_.log"
