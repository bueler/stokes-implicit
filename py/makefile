# testing

runstokes2D_1:
	-@./../testit.sh stokes2D.py "-s_ksp_monitor_short -s_ksp_converged_reason -s_ksp_rtol 1.0e-8" 1 1

runstokes2D_2:
	-@./../testit.sh stokes2D.py "-nintervals 20 -layers 4 -refine 1" 1 2

runstokes2D_3:
	-@./../testit.sh stokes2D.py "-s_ksp_converged_reason -s_pc_type svd -nintervals 6 -layer 2" 1 3

test: runstokes2D_1 runstokes2D_2 runstokes2D_3

.PHONY: clean test runstokes2D_1 runstokes2D_2 runstokes2D_3

clean:
	@rm -f *.pdf *.geo *.msh *.pvd *.vtu
	@rm -f maketmp tmp difftmp

