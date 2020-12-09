# testing

runmfmgstokes_1:
	-@./../testit.sh mfmgstokes.py "-m0 4 -refine 1" 1 1

runceviche_1:
	-@./../testit.sh ceviche.py "-N 3 -s_ksp_type preonly -s_pc_type lu -s_pc_factor_mat_solver_type mumps" 1 1

runceviche_2:
	-@./../testit.sh ceviche.py "-dual -rt" 1 2

runsole_1:
	-@./../testit.sh sole.py "-stage 3 -mx 4 -my 4 -refine 2 -big" 1 1

runsole_2:
	-@./../testit.sh sole.py "-stage 2 -mx 4 -my 4 -refine 1 -aggressive" 1 2

runsole_3:
	-@./../testit.sh sole.py "-stage 3 -mx 4 -my 4 -refine 1 -aggressive -prism" 2 3

runpool_1:
	-@./../testit.sh pool.py "-stage 1 -refine 2" 1 1

runpool_2:
	-@./../testit.sh pool.py "-stage 2 -mx 4 -my 4 -refine 1 -aggressive" 1 2

runpool_3:
	-@./../testit.sh pool.py "-stage 3 -mx 4 -my 4 -refine 1 -aggressive" 1 3

runpool_4:
	-@./../testit.sh pool.py "-stage 4 -mx 4 -my 4 -mz 2" 2 4

runstokesi_1:
	-@./../testit.sh stokesi.py "-dirichletsmb -dta 0.1 -s_ksp_converged_reason -mx 5 -refine 1 -s_snes_rtol 1.0e-1" 2 1

runstokesi_2:
	-@./../testit.sh stokesi.py "-quad -mx 3 -my 3 -Href 100 -s_snes_converged_reason -s_ksp_converged_reason -s_snes_rtol 1.0e-2" 1 2

runstokesi_3:
	-@./../testit.sh stokesi.py "-s_snes_converged_reason -s_ksp_converged_reason -mx 4 -my 4 -s_snes_rtol 1.0e-2 -s_ksp_rtol 1.0e-2" 1 3

runstokesi_4:
	-@./../testit.sh stokesi.py "-dta 1.0 -mx 5 -s_snes_converged_reason -saveextra -o test4.pvd" 2 4

test_ceviche: runceviche_1 runceviche_2

test_mfmgstokes: runmfmgstokes_1

test_sole: runsole_1 runsole_2 runsole_3

test_pool: runpool_1 runpool_2 runpool_3 runpool_4

test_stokesi: runstokesi_1 runstokesi_2 runstokesi_3 runstokesi_4

test: test_ceviche test_mfmgstokes test_sole test_pool test_stokesi

.PHONY: clean runceviche_1 runceviche_2 runmfmgstokes_1 runsole_1 runsole_2 runsole_3 runpool_1 runpool_2 runpool_3 runpool_4 runstokesi_1 runstokesi_2 runstokesi_3 runstokesi_4 test_pool test_stokesi test

clean:
	@rm -f *.pdf *.geo *.msh *.pvd *.vtu *.pvtu *.m
	@rm -f maketmp tmp difftmp
	@rm -rf __pycache__ src/__pycache__

