# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 09:27:42 2025

@author: dhuerta
Complilation of different methods
Calculation examples (can be pastesd in main to display different examples)
These methods work well with ALBA and ALBAII
"""

###############################################################################
#Calculating dORM with thick elements and assessing validity
###############################################################################

#time1 = time.perf_counter()
###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"v" ,ind_bpm, ind_cor["v"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.quad.broadcasters(0, 3)
thickv = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad)
vdRij_dqk = cORM.dRij_dqk_thin(cORM.bpm, cORM.cor, cORM.quad)
##########################################################

###### Example calculating the dORM_dq with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.bpm.broadcasters(1, 3)
cORM.cor.broadcasters(2, 3)
cORM.quad.broadcasters(0, 3)
thickh = cORM.dRij_dqk_thick23(cORM.bpm, cORM.cor, cORM.quad)
hdRij_dqk = cORM.dRij_dqk_thin(cORM.bpm, cORM.cor, cORM.quad)
##########################################################
#time2 = time.perf_counter()
#print(time2-time1)

#plot_utils.plot_both(dORMV, dORMH, vdRij_dqk, hdRij_dqk)
#plot_utils.plot_both(dORMV, dORMH, thickv, thickh)

plot_utils.plot_both_Zeus(dORMV, dORMH, vdRij_dqk, hdRij_dqk)
plot_utils.plot_both_Zeus(dORMV, dORMH, thickv, thickh)


###############################################################################
#Seeing how the dispersion formulas work well! 
###############################################################################


###########Dispersion test in bpms######################
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.bpm.broadcasters(0, 2)
cORM.dip.correct_entrance()
cORM.dip.broadcasters(1, 2)
bpmdispls = cORM.disp_i(cORM.bpm, cORM.dip)
math_utils.listPlot([cORM.bpm.dispersion ,bpmdispls ], ["bpmDisp", "bpmDispcalc"],title="bpm Dispersions" ,savename = "disps")
########################################################

###########Dispersion test in cors######################
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.cor.broadcasters(0, 2)
cORM.dip.broadcasters(1, 2)
cordispls = cORM.disp_i(cORM.cor, cORM.dip)
math_utils.listPlot([cORM.cor.dispersion ,cordispls ], ["bpmDisp", "bpmDispcalc"],title="bpm Dispersions" ,savename = "disps")
########################################################

###########Dispersion test in dips######################
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.dip1 = copy.deepcopy(cORM.dip)
cORM.dip.broadcasters(0, 2)
cORM.dip1.broadcasters(1, 2)
dipdispls = cORM.disp_i(cORM.dip, cORM.dip1)
math_utils.listPlot([cORM.dip.dispersion ,dipdispls ], ["bpmDisp", "bpmDispcalc"],title="bpm Dispersions" ,savename = "disps")
########################################################




###############################################################################
#Example displaying how the dispersion term fixes much of the extra error in the h dir in ALBA its perfect
###############################################################################

for ind in ind_cor["v"]: ring[ind].KickAngle = np.array([0,0]) ####Adding kickangles if needed!
for ind in ind_cor["h"]: ring[ind].KickAngle = np.array([0,0])

ORMV = numerical.ORM(ring, "v", ind_bpm, ind_cor["v"])
ORMH = numerical.ORM(ring, "h", ind_bpm, ind_cor["h"])

###### Example calculating the verical ORM with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"v" ,ind_bpm, ind_cor["v"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.bpm.broadcasters(0, 2)
cORM.cor.broadcasters(1, 2)
nORMV = cORM.Rab_thick2_(cORM.bpm, cORM.cor)
##########################################################

###### Example calculating the horizontal ORM with thin and thick elements!
cORM = AnaORM.AnaORM(ring,"h" ,ind_bpm, ind_cor["h"], ind_quad, ind_dip, np.array([]))
cORM.assign_optics()
cORM.bpm.broadcasters(0, 2)
cORM.cor.broadcasters(1, 2)
nORMH = cORM.Rab_thick2_(cORM.bpm, cORM.cor) + cORM.Rab_thick2_disp(cORM.bpm, cORM.cor)
##########################################################

print(math_utils.normalized_RMSE(ORMV, nORMV, (0,1)))#2.7890545467851 Amb ALBAII
print(math_utils.normalized_RMSE(ORMH, nORMH, (0,1)))#6.625965837150698 amb terme dispersió, 6.727195627562558 sense dispersió
#En resum, aquests càlculs semblen indicar que la fórmula té problemes més grans més enllà de la dispersó, és de fet el feed-down dels elements gruixuts!
#Un kick a un corrector, comporta al seu torn un canvi de moment dipolar a tots els quadrupols, tots els quadrupols esdevenen dipols també!
#Això suma contribucions a la ORM més petites corresponents a 









