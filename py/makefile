# testing

runpool_1:
	-@./../testit.sh pool.py "-mx 2 -my 2 -refine 1 -schurcoarsegamg -s_ksp_converged_reason" 1 1

runstokesi_1:
	-@./../testit.sh stokesi.py "-almost -dirichletsmb -dta 1.0 -s_snes_type ksponly -s_ksp_converged_reason -mx 20 -refine 1" 1 1

runstokesi_2:
	-@./../testit.sh stokesi.py "-almost -dta 0.1 -s_snes_converged_reason -s_ksp_converged_reason -s_snes_rtol 1.0e-2" 2 2

runstokesi_3:
	-@./../testit.sh stokesi.py "-almost -s_snes_converged_reason -s_ksp_converged_reason -mx 4 -my 4 -s_snes_rtol 1.0e-2" 1 3

runstokesi_4:
	-@./../testit.sh stokesi.py "-almost -mx 5 -saveextra -o test4.pvd" 2 4

test_pool: runpool_1

test_stokesi: runstokesi_1 runstokesi_2 runstokesi_3 runstokesi_4

test: test_pool test_stokesi

.PHONY: clean runpool_1 runstokesi_1 runstokesi_2 runstokesi_3 runstokesi_4 test_pool test_stokesi test

clean:
	@rm -f *.pdf *.geo *.msh *.pvd *.vtu *.pvtu *.m
	@rm -f maketmp tmp difftmp
	@rm -rf __pycache__ src/__pycache__

