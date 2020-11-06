# testing

runsole_1:
	-@./../testit.sh sole.py "-stage 3 -mx 4 -my 4 -refine 2" 1 1

runsole_2:
	-@./../testit.sh sole.py "-stage 2 -mx 5 -my 5 -refine 1" 1 2

runpool_1:
	-@./../testit.sh pool.py "-stage 1 -refine 2" 1 1

runstokesi_1:
	-@./../testit.sh stokesi.py "-almost -dirichletsmb -dta 1.0 -s_snes_type ksponly -s_ksp_converged_reason -mx 20 -refine 1" 1 1

runstokesi_2:
	-@./../testit.sh stokesi.py "-almost -dta 0.1 -s_snes_converged_reason -s_ksp_converged_reason -s_snes_rtol 1.0e-2" 2 2

runstokesi_3:
	-@./../testit.sh stokesi.py "-almost -dta 1.0 -s_snes_converged_reason -s_ksp_converged_reason -mx 4 -my 4 -s_snes_rtol 1.0e-2" 1 3

runstokesi_4:
	-@./../testit.sh stokesi.py "-almost -dta 1.0 -mx 5 -saveextra -o test4.pvd" 2 4

runstokesi_5:
	-@./../testit.sh stokesi.py "-s_snes_converged_reason -s_ksp_converged_reason -s_snes_rtol 1.0e-2" 1 5

test_sole: runsole_1 runsole_2

test_pool: runpool_1

test_stokesi: runstokesi_1 runstokesi_2 runstokesi_3 runstokesi_4 runstokesi_5

test: test_sole test_pool test_stokesi

.PHONY: clean runsole_1 runsole_2 runpool_1 runstokesi_1 runstokesi_2 runstokesi_3 runstokesi_4 runstokesi_5 test_pool test_stokesi test

clean:
	@rm -f *.pdf *.geo *.msh *.pvd *.vtu *.pvtu *.m
	@rm -f maketmp tmp difftmp
	@rm -rf __pycache__ src/__pycache__

