# testing

runpool_1:
	-@./../testit.sh pool.py "-mx 2 -my 2 -refine 1 -schurcoarsegamg -s_ksp_converged_reason" 1 1

runstokesi_1:
	-@./../testit.sh stokesi.py "-dirichletsmb -linear -dta 1.0 -s_snes_type ksponly -s_ksp_converged_reason -mx 10 -baserefine 1 -vertrefine 1" 1 1

runstokesi_2:
	-@./../testit.sh stokesi.py "-dta 0.1 -s_snes_converged_reason -s_ksp_converged_reason -s_snes_rtol 1.0e-2" 2 2

runstokesi_3:
	-@./../testit.sh stokesi.py "-s_snes_converged_reason -s_ksp_converged_reason -mx 4 -my 4 -s_snes_rtol 1.0e-2" 1 3

test_pool: runpool_1

test_stokesi: runstokesi_1 runstokesi_2 runstokesi_3

test: test_pool test_stokesi

.PHONY: clean runpool_1 runstokesi_1 runstokesi_2 runstokesi_3 test_pool test_stokesi test

clean:
	@rm -f *.pdf *.geo *.msh *.pvd *.vtu *.pvtu *.m
	@rm -f maketmp tmp difftmp
	@rm -rf __pycache__

