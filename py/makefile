# testing

runstokes2D_1:
	-@./../testit.sh stokes2D.py "-s_ksp_monitor_short -s_ksp_converged_reason -s_ksp_rtol 1.0e-8" 1 1

test: runstokes2D_1

.PHONY: clean test runstokes2D_1

clean:
	@rm -f *.pdf *.geo *.msh *.pvd *.vtu
	@rm -f maketmp tmp difftmp

