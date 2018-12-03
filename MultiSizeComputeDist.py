from mpi4py import MPI
from yt.units import pc, kpc, second, Kelvin, gram, erg, cm
from matplotlib.colors import LogNorm

import yt
import matplotlib.pyplot as plt
import cPickle as pickle
import numpy as np
import compute_charge_dist as fz
import fzMPI
import time

comm = MPI.COMM_WORLD

pos       = comm.Get_rank()
num_procs = comm.Get_size()


# ---------------------------------------------------------------------------------
grain_type = "silicate"
#grain_type = "carbonaceous"

#grain_size = 5
G0         = 1.7

#sizes = [5, 10, 50, 100, 500, 1000]

sizes = [3.5, 5, 10]
#sizes = [3.5, 5, 10, 50, 100]
#sizes = [500, 1000]

percent = 0.001

include_CR = True
# --------------------------------------------------------------------------------


# Define some constant parameters to be used.
mp      = 1.6726e-24  * gram # g
mH      = 1.6733e-24  * gram
mC      = 12.011*mH
#mu      = 1.2924
kb      = 1.3806e-16  *erg / Kelvin # erg K-1
GNewton = 6.6743e-8   * cm**3 / (gram * second**2 )# cm3 g-1 s-2
Msun    = 1.9884e33   * gram
#mm      = mu*mp

ppc = 3.0856776e18

# -------------------------------------------------------------
#              Create a lot of new derived fields
# -------------------------------------------------------------

# Create a derived field.
# Hydrogen number density
def numdensH(field, data):
    nH = data["dens"]*(data["ihp "]+data["iha "]+data["ih2 "])/(1.4*mH)
    return nH

# Molecular Hydrogen number density
def numdensH2(field, data):
    nH2 = data["dens"]*(data["ih2 "])/(1.4*mH)
    return nH2

# Carbon number density
def numdensC(field, data):
    nC = data["dens"]*(data["icp "]+data["ico "])/(1.4*mC)
    return nC

# electron number density
def numdense(field, data):
    ne = data["dens"]*(data["ihp "]/(1.4*mH) + data["icp "]/(1.4*mC))
    return ne

# Ionized hydrogen fraction
def xHp(field, data):
    nH  = data["dens"]*(data["ihp "]+data["iha "]+data["ih2 "])/(1.4*mH)
    xHp = data["dens"]*data["ihp "]/(1.4*mH)
    xHp = xHp / nH
    return xHp

# Molecular hydrogen fraction
def xH2(field, data):
    nH  = data["dens"]*(data["ihp "]+data["iha "]+data["ih2 "])/(1.4*mH)
    xH2 = data["dens"]*data["ih2 "]/(1.4*mH)
    xH2 = xH2 / nH
    return xH2

# Ionized carbon fraction
def xCp(field, data):
    nC  = data["dens"]*(data["icp "]+data["ico "])/(1.4*mC)
    xCp = data["dens"]*data["icp "]/(1.4*mC) / nC
    return xCp

# electron fraction
def xe(field, data):
    nH = data["dens"]*(data["ihp "]+data["iha "]+data["ih2 "])/(1.4*mH)
    nC = data["dens"]*(data["icp "]+data["ico "])/(1.4*mC)
    ne = data["dens"]*(data["ihp "]/(1.4*mH) + data["icp "]/(1.4*mC))
    xe = ne / (nH + nC)
    return xe


yt.add_field('nH', function=numdensH,  units="1/cm**3", force_override=True)
yt.add_field('nH2',function=numdensH2, units="1/cm**3", force_override=True)
yt.add_field('nC', function=numdensC,  units="1/cm**3", force_override=True)
yt.add_field('ne', function=numdense,  units="1/cm**3", force_override=True)
yt.add_field('xHp', function=xHp,      units="dimensionless", force_override=True)
yt.add_field('xH2', function=xH2,      units="dimensionless", force_override=True)
yt.add_field('xCp', function=xCp,      units="dimensionless", force_override=True)
yt.add_field('xe', function=xe,        units="dimensionless", force_override=True)
#yt.add_field('G',  function=GG,        units="dimensionless", force_override=True)

# Input variables.
# DustBox
data_dir   = "/home/jcibanezm/codes/run/Silcc/CF_Prabesh"

#Daikaiju
#data_dir = "/data/gamera/jcibanezm/DustAnalysis/CF_Data"
filename   = data_dir + "/NL99_R8_cf_hdf5_chk_0028"

pf = yt.load("%s"%(filename))


# In[4]:

c  = [0,0,0]
le = [-4.93696000e+19, -4.93696000e+19, -4.93696000e+19]
re = [ 4.93696000e+19,  4.93696000e+19,  4.93696000e+19]

box = pf.box(le, re)


# In[5]:

min_dens = np.min(box["density"])
max_dens = np.max(box["density"])

min_nh   = np.min(box["nH"])
max_nh   = np.max(box["nH"])

min_temp = np.min(box["temperature"])
max_temp = np.max(box["temperature"])

min_ne = np.min(box["ne"])
max_ne = np.max(box["ne"])

min_xe = np.min(box["xe"])
max_xe = np.max(box["xe"])

min_Av = np.min(box["cdto"])
max_Av = np.max(box["cdto"])


if pos == 0:
    print("-----------------------------------------------------------")
    print("Some properties of the simulation:")
    print("Density,     min = %.2g,\t max = %.2g"%(min_dens, max_dens))
    print("Temperature, min = %.2g,\t\t max = %.2g"%(min_temp, max_temp))
    print("ndens H,     min = %.2g,\t max = %.2g"%(min_nh, max_nh))
    print("ndens e,     min = %.2g,\t max = %.2g"%(min_ne, max_ne))
    print("e fraction,  min = %.2g,\t max = %.2g"%(min_xe, max_xe))
    print("Av           min = %.2g,\t max = %.2g"%(min_Av, max_Av))



np.random.seed(1)
#np.random.seed(2)

ncells = len(box["nH"])
n5     = np.int(ncells * percent/100.)
#n5 = int(1.0e2)
rand_index     = np.random.randint(0, ncells, n5)
cells_per_proc = n5 // num_procs

if pos == num_procs:
    if (pos+1)*cells_per_proc < n5:
        missing_cells = (n5 - (pos+1)*cells_per_proc)
        cells_per_proc += missing_cells
        print("Adding %i cells to the last processor"%missing_cells)


# In[24]:
temp = np.array(box["temp"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
dd   = np.array(box["dens"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
nH   = np.array(box["nH"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
nH2  = np.array(box["nH2"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
nC   = np.array(box["nC"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
ne   = np.array(box["ne"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
xe   = np.array(box["xe"]  [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
xHp  = np.array(box["xHp"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
xH2  = np.array(box["xH2"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
xCp  = np.array(box["xCp"] [rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
Av   = np.array(box["cdto"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])
fH2shield = np.array(box["cdh2"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]])

Ntot = Av * 1.87e21

new_ne = np.zeros_like(nH)
new_xe = np.zeros_like(nH)

cell_mass = np.array(box["cell_mass"][rand_index[pos*cells_per_proc:(pos+1)*cells_per_proc]].in_units("Msun"))

#for grain_size in sizes:
for ksize in range(len(sizes)):

    grain_size = sizes[ksize]

    if pos == 0:
        print("===================================================")
        print("Running the charge distribution calculation for:")
        print("%s Grains"%grain_type)
        print("%i Angstrom size grain"%grain_size)
        print("")
        print("On the simulaion: %s"%filename)
        print("Resolution: %.2g pc"%np.min(box["dx"]).in_units("pc"))
        print("Total cell number = %i"%ncells)
        print("Total cells in analysis = %i" %n5)
        print("")
        print("Number of processors: %i"%num_procs)
        print("Cell per processor: %i"%cells_per_proc)

    start_time = time.time()

    Qabs = fz.get_QabsTable(grain_type, grain_size)
    # In[28]:

    zmean, zmode, zstd  =  np.zeros(cells_per_proc), np.zeros(cells_per_proc), np.zeros(cells_per_proc)
    zminmax = np.array(np.zeros(2*cells_per_proc))

    tauz    = np.zeros_like(zmean)

    fdist   = []
    newZZ   = []

    G       = np.zeros_like(zmean)
    G_CR    = np.zeros_like(zmean)

    pp = 0

    for ii in range(cells_per_proc):

        prog = ii*100. / cells_per_proc - 1
        if prog >= 10*(1+pp):
            temp_time = time.time() - start_time
            print("processor %i, progress %.2i, time %.1f "%(pos, 10*(1+pp), temp_time))
            pp +=1


        index = ii
        
        zeta = fz.get_zeta(Ntot[ii])
        #zeta = fz.get_zeta(fH2shield[ii])

        ############################################################################################
        Jpe, Je, Jh, Jc, ZZall = fz.compute_currents   ([nH[index], nC[index]], [xHp[index], xCp[index]], xH2[index], temp[index], zeta, grain_size, Ntot[index], grain_type, Qabs, G0=G0)
        JCRe, JCRpe, ZZnew     = fz.compute_CR_currents(nH[index], zeta, grain_size, grain_type, Qabs)

        zeq                    = fz.get_zeq_vec      (Jpe, Je, Jh, Jc, ZZall, grain_size, grain_type)
        new_zmin, new_zmax     = fz.get_new_zmin_zmax([nH[index], nC[index]], [xHp[index], xCp[index]], temp[index], grain_size, Ntot[index], grain_type, Qabs, zeta, zeq=zeq, G0=G0)

        ffz, ZZ                = fz.vector_fz        (Jpe, Je, Jh, Jc, JCRe, JCRpe, ZZall, new_zmin, new_zmax, includeCR=include_CR)

	new_ne[ii], new_xe[ii] = fz.compute_new_xe([nH[index], nC[index]], [xHp[index], xCp[index]], xH2[index], zeta)

        tauz[ii]  = fz.get_tauz(grain_size, grain_type, [nH[index], nC[index]], [xHp[index], xCp[index]], temp[index], Ntot[index], ZZ, ffz, xH2[index], zeta, Qabs, G0=G0, includeCR=True)

        Zm        = fz.get_Zmode(ZZ, ffz)
        zmode[ii] = Zm

        avg, std  = fz.weighted_avg_and_std(ZZ, ffz)
        zmean[ii] = avg
        zstd[ii]  = std

        zminmax[ii*2]  = new_zmin
        zminmax[ii*2+1]= new_zmax

        # Calculate
        G[index]     = fz.get_G(Ntot[index], G0)
        G_CR[index]  = fz.get_G_CR(Ntot[index])

        #fdist[ii]   = offset + ii
        fdist.append(0)
	fdist[-1] = []
        newZZ.append(0)
        newZZ[-1] = []
        for jj in range(len(ffz)):
            fdist[-1].append(ffz[jj])
            newZZ[-1].append(ZZ[jj])

    cdist = {"info": "Dictionary containing the information about the dust grain charge distribution."}

    cdist["grain_size"] = grain_size
    cdist["grain_type"] = grain_type
    cdist["dens"] = dd
    cdist["temp"] = temp
    cdist["nH"]   = nH
    cdist["nH2"]  = nH2
    cdist["nC"]   = nC
    cdist["ne"]   = ne
    cdist["xe"]   = xe
    cdist["xHp"]  = xHp
    cdist["xH2"]  = xH2
    cdist["xCp"]  = xCp
    cdist["Av"] = Av
    cdist["Ntot"] = Ntot
    cdist["fH2shield"] = fH2shield

    cdist["new_ne"] = new_ne
    cdist["new_xe"] = new_xe

    cdist["zmean"] = zmean
    cdist["zmode"] = zmode
    cdist["zstd"]  = zstd
    cdist["zminmax"] = zminmax
    cdist["G"]     = G
    cdist["G_CR"]     = G_CR
    cdist["fdist"] = fdist
    cdist["tauz"]  = tauz

    cdist["ZZ"] = newZZ

    cdist["cell_mass"] = cell_mass

    # Now I need to retrieve and concatenate all data!

    cdist = fzMPI.gather_all_data(cdist, comm)

    if pos ==0:
        Qabs_mean = fz.get_avgQabs(Qabs, G0=G0)
        cdist["Qabs_mean"] = Qabs_mean


    if pos == 0:
        # Now Save the dictionary.
        outdir  = "/home/jcibanezm/codes/run/ChargeStatisticsAnalysis/CR"
        #Daikaiju
        #outdir = "/data/gamera/jcibanezm/DustAnalysis"
        outname = "fz_%.4iAA_%s_CR_%s_%i_pcent.pkl"%(grain_size, grain_type, include_CR, percent)
        #outname = "fz_%.4iAA_%s_CR_%s.pkl"%(grain_size, grain_type, include_CR)
        print("Saving charge distribution to %s/%s"%(outdir, outname))
        outfile = open('%s/%s'%(outdir, outname), 'wb')
        pickle.dump(cdist, outfile)
        outfile.close()
        end_time = time.time()
        print("Time taken per processor to calculate the charge distribution of %i grains with %i processors = %.2f"%(n5, num_procs, end_time - start_time))
