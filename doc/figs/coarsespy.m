% COARSESPY  Generate figure coarsespy.pdf from run of py/stokesi.py.
% Do in firedrake:
%   $ ./stokesi.py -s_snes_converged_reason -s_ksp_converged_reason -mx 3 -mz 3 -s_ksp_view_mat :csinput.m:ascii_matlab
% Then run this in Matlab/Octave.  Then do
%   $ pdfcrop coarsespy.pdf coarsespy.pdf

csinput
A = Mat_0x84000005_0;
nu = 98;
np = 16;
nc = 16;
N = nu + np + nc;
spy(A,'k')
axis off
axis equal
hold on
del = 0.5
plot([1-del,N+del],[nu+del,nu+del],'k','linewidth',0.5)
plot([1-del,N+del],[nu+np+del,nu+np+del],'k','linewidth',0.5)
plot([nu+del,nu+del],[1-del,N+del],'k','linewidth',0.5)
plot([nu+np+del,nu+np+del],[1-del,N+del],'k','linewidth',0.5)
hold off
print -dpdf coarsespy.pdf

