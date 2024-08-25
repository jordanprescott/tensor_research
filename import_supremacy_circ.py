import quimb.tensor as qtn
import cotengra as ctg

circ = qtn.Circuit.from_qsim_file("circuit_n53_m20_s0_e0_pABCDCDAB.qsim")

circ.psi.draw()
