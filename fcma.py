#Reko Penttilä 03/2021

#Based on FCI.py script (https://github.com/aoskarih/Vibrations-with-AIMS) 
#by Arttu Hyvönen 07/2019

import scipy.special as special
from scipy.misc import factorial
import numpy as np
import matplotlib.pyplot as plt
from  heapq import nlargest

import  os
import argparse


def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()
    
    # Mandatory arguments
   
    parser.add_argument("-IP", "--vertical_IP", required=True, help="Vertical ionization potential between neutral and excited state", type=float)
    parser.add_argument("-XYZ",  "--xyz_file", required=True, help="Path name of the xyz file which contains normal modes. e.g './run.xyz' ", type=str)
    parser.add_argument("-V",  "--vib_file", required=True, help="Path name of the output file which contains frequencies in a list. e.g.'./vib_post_0.0025.out' ", type=str)
    parser.add_argument("-XYZ_e",  "--xyz_file_excited", required=True, help="Path name of the xyz file which contains normal modes of the excited state. e.g './run.xyz' ", type=str)
    parser.add_argument("-V_e",  "--vib_file_excited", required=True, help="Path name of the output file which contains frequencies of the excited state in a list. e.g.'./vib_post_0.0025.out' ", type=str)
    parser.add_argument("-mass",  "--mass_file", required=True,help="name of the mass file, currently data from here not in use e.g './masses.run_0.0025.dat' " , type=str, default="")

    # Optional arguments
    parser.add_argument("-T", "--temparature", help="Temparature", type=int, default=290)
    parser.add_argument("-S", "--S_lim", help="Limit for the Huang-Rhys constant for relevant modes", type=float, default=1e-4)
    parser.add_argument("-m", "--m", help="Limitation for the combinations of the ground state.", type=int, default=4)
    parser.add_argument("-n", "--n", help="Limitation for the combinations of the excited state.", type=int, default=3)
    parser.add_argument("-o",  "--output", help="output file", type=str, default="intensity.dat")  
 
    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    return args

args = parseArguments()

# Path names to input files
xyz = args.xyz_file                # xyz file which contains normal modes
vib = args.vib_file    # output file which contains frequencies in a list
xyz_e = args.xyz_file_excited                # xyz file which contains normal modes
vib_e = args.vib_file_excited    # output file which contains frequencies in a list
vertical_IP = args.vertical_IP # 
#mas = args.mass   # mass file, currently data from here not in use
output_file = args.output  # Output filename

# constants
hbar = 6.5821e-16	# eV s
c = 2.9979e8		# m/s
pi = 3.1416
u = 1.6605e-27		# kg
kb = 8.6173e-5		# ev/K

# Limiting number of transitions
m_lim = args.m
n_lim = args.n
S_lim = args.S_lim

# Temperature in kelvins
temperature = args.temparature


#position from the xyz file
def get_positions(xyz_path):
    f = open(xyz_path, "r")
    n = int(next(f).split()[0])
    next(f)
    pos = np.zeros(3*n)
    for i in range(n):
        l = next(f).split()
        for j in range(3):
            pos[3*i+j] = float(l[j+1])
    f.close()
    return pos

# returns normal modes from the .xyz file
def get_normal_modes(xyz_path):
    f = open(xyz_path, "r")
    n = int(next(f).split()[0])
    f.seek(0)
    modes = []
    for _ in range(3*n):
        mode = np.zeros(3*n)
        next(f)
        next(f)
        for i in range(n):
            l = next(f).split()
            for j in range(3):
                mode[3*i+j] = float(l[j+4])
        modes.append(mode)
    f.close()
    return modes

# returns frequencies from the .out file
# 1/s
def get_frequencies(vib_path):
    
    f = open(vib_path, "r")
    n = 0
    for l in f:
        if "Number of atoms" in l:
            n = int(l.split()[4])
            continue
        if "Mode number" in l:
            break
    freq_list = np.zeros(3*n)
    for i in range(3*n):
        freq = float(next(f).split()[1])
        freq *= 100*c*2*pi			# 1/cm -> 1/s
        freq_list[i] = freq
    return freq_list

# returns transfromation matrix from cartesian coordinates to coordinates in given basis 
# basis vectors are assumed to be in cartesian coordinates
def get_transformation_matrix(basis_vectors):
    n = len(basis_vectors)
    T = np.zeros((n,n))
    for i in range(n):
        v = basis_vectors[i]
        l = np.linalg.norm(v)
        v = v/l
        for j in range(n):
            T[j][i] = v[j]
    return np.linalg.inv(T)

# returns reduced masses for the different vibrational modes from .xyz
# kg
def get_reduced_masses(xyz_path):
    f = open(xyz_path, "r")
    mass_list = []
    for l in f:
        if "stable frequency" in l:
            if l.split()[13] == "*****" and l.split()[18] == "*****":
		        mass = 0
            elif l.split()[13] == "*****":
		        mass = float(l.split()[18])/(float(l.split()[3])*100*c*2*pi)**2
            else:
                mass = float(l.split()[13])
		mass_list.append(mass*u)	# u -> kg
    return mass_list

# returns force constants (k) for the different vibrational modes from .xyz
# kg/s^2
def get_force_constants(xyz_path):
    f = open(xyz_path, "r")
    k_list = []
    for l in f:
        if "stable frequency" in l:
            if l.split()[13] == "*****" and l.split()[18] == "*****":
                k = 0
            elif l.split()[18] == "*****":
                k = float(l.split()[13])*u*(200*c*pi*float(l.split()[3]))**2
            else:
                k = float(l.split()[18])
            k_list.append(k*100)		# mDyne/A -> kg/s^2
        
    return k_list

#returns the zeropoint energies read from the "vib_post_0.0025.out" file
def get_zero_point_energies(vib_path):
    energy_list = [] 
    f = open(vib_path, "r")
    for l in f:
        if "Cumulative ZPE" in l:
            energy_list.append(float(l.split()[4]))
    return energy_list

def get_relaxation_energy(force_constants, displacements):
    relaxation_energy = 0
    for i in range(len(force_constants)):
        relaxation_energy = relaxation_energy+ force_constants[i]/2*displacements[i]**2
    return relaxation_energy

# returns masses from the masses.* file. Input is the path name of the mass file
def get_masses(mass_path):
    f = open(mass_path, "r")
    masses = []
    for l in f:
        m = float(l.split()[0])
        masses.append(m*u)			# u -> kg
    return masses

# returns value of Huang-Rhys parameter
def get_huang_rhys(d, mu, f):
    S = d*d*mu*f/(2*hbar)
    # unit conversion
    S *= 0.062415				# A^2 * kg / (eV * s^2) = 0.0624150...
    return S

def FCI2(S, m, n):
    I = np.exp(-S)*S**(n-m)*factorial(m)/factorial(n)*laguerre(S, m, n-m)**2
    return I

def laguerre(x, n, k):
    if k > -1:
        L = special.genlaguerre(n, k)
        return L(x)
    else:
        # scipy doesn't handle negative k so recursive equation is used
        if n == 0:
            return 1
        else:
            return laguerre(x, n, k+1) - laguerre(x, n-1, k+1)

# returns relative intensity of vibrational transition
# by passing list relative, which contains modes for which S is above some limit you can make calculation much faster
# example relevant = [9, 17, 21, 26, 31]
def intensity(d, freq, mu, m, n, T, S_limit, relevant=[]):
    I = 1
    if relevant:
        # Option 1: Relevant modes are in a relevant mode file
        for i in relevant:
            # Get S
            S = get_huang_rhys(d[i], mu[i], freq[i])
            # Calculate Franck-Condon integral
            FCI = FCI2(S, m[i], n[i])
            # Intensity is integrals for all modes multiplied together
            I *= FCI
            # Temperature term
            if T != 0:
                I *= np.exp(-hbar*m[i]*freq[i]/(kb*T))
        if I > 1e-3:
            print("I: " + str(I))
        return I
    
    # Option 2: Relevant modes are decided with huang-rhys limit
    for i in range(len(d)):
        S = get_huang_rhys(d[i], mu[i], freq[i])
        if S < S_limit: 
            #print("%2.2i  %10.6f  %10s %5.1i %5.1i  ignored" % (i, S, "", m[i], n[i]))
            continue
        FCI = FCI2(S, m[i], n[i])
        I *= FCI
        if T != 0:
            I *= np.exp(-hbar*m[i]*freq[i]/(kb*T))
        #print("%2.2i  %10.6f  %10.6f %5.1i %5.1i" % (i, S, FCI, m[i], n[i]))
    print("I: " + str(I) + "\n")
    return I

# returns energy of some state with m = [...] where E0 is the zero-point vibrational energy
# eV

def get_vib_energy(freq, m):
    E = 0
    for i, f in enumerate(freq):
        E += f*hbar*(m[i]+1/2)
    return E

# returns list of different integer combinations of length l for which the sum of integers 
# in the combination is less than m. Length of the list grows rapidly.
def combinations(m, l):
    arr = []
    ar = [0 for _ in range(l)]
    i = 0
    while i < l:
        if ar not in arr:
            arr.append(ar[:])
        ar[i] += 1
        s = sum(ar)
        if s < m + 1:
            i = 0
        else:
            i += 1
            for j in range(i):
               ar[j] = 0
    return arr

# Writes intensities to file specified with "filename"
def write_intensities(d, freq, mu, IP, T, S_limit, filename, diff_zero):
    N = len(d)
    m = np.zeros(N)
    n = np.zeros(N)
    modes = []          # relevant modes
    sl = []

    # Two choices for limiting number of calculated modes
    # 1. Take all the modes for which S > S_limit
    # 2. Take 10 modes with the largest S
    # Option 1 ensures that all relevant modes are calculated, 
    # but option 2 is more consistent in calculation time. 
    # There might be large variance in number of relevant modes between different electronic states
    #Relaxation energy
    relaxation =0   
    if 1: # 1.
        for i, f in enumerate(freq):
            S = get_huang_rhys(d[i], mu[i], f)
            relaxation += hbar*S*f
            if S > S_limit:
                sl.append((S, i))
    else: # 2.
        for i, f in enumerate(freq):
            relaxation += hbar*S*f
            S = get_huang_rhys(d[i], mu[i], f)
            sl.append((S, i))
        def comp(x): return x[0]
        sl = nlargest(10, sl, key=comp)
    
    print(len(sl), "modes")
    ss = [x[0] for x in sl]
    modes = [x[1] for x in sl]
    print("\nMinimum required S: %14.10f" % min(ss))
    print("Number of modes in calculation: %i out of %i" % (len(modes), N))
    print("%10s %10s" % ("i", "S"))
    for S, i in sl:
        print("%10.i %10.8f" % (i, S))
    
    #dat_I = []
    #dat_E = []

    # itr_m is the list of combinations for ground state and itr_n for exited state
    # first argument in the combinations function is the most important number considering 
    # the script execution time. Anything above m=3 and n=4 not recommended since python isn't that fast.
    itr_m = combinations(m_lim, len(modes)) 
    itr_n = combinations(n_lim, len(modes))
   
    for mode in modes:
        if freq[mode] < 2:
            print("Negative mode as relevant")
    
    # Assuming that the calculation is from neutral ground state 
    # to ion and it's exited states (up in energy), then use the upper block.
    # If you want the transition down in energy, then uncomment the other block and comment out the first.
    # Blocks are identical in all other ways except in the way energy is calculated.
    #"""
    ##############################
    # Neutral -> Ion

    #Calculating the 0_0 transition energy
    print(IP, relaxation, diff_zero)
    E0_0 = IP - relaxation - diff_zero
    print(E0_0)
    f = open(filename, "a")

    # Calculate intensity for all combinations of m and n
    for comb_m in itr_m:
        for i, mode in enumerate(modes):
            m[mode] = comb_m[i]
        # Get the energy of initial state
        e_m = get_vib_energy(freq, m)
        for comb_n in itr_n:
            for i, mode in enumerate(modes):
                n[mode] = comb_n[i]
            I = intensity(d, freq, mu, m, n, T, S_limit, modes)
            # Filter out irrelevant transitions
            if I < 1e-3 and sum(m) + sum(n) > 0: continue
            # Get the energy of final state
            e_n = get_vib_energy(freq, n) 
            f.write("%10.6f  %10.6f  %s\n" % (I, E0_0+e_n-e_m))
    #############################

    """
    #############################
    # Ion -> Neutral
    f = open(filename, "a")
    for comb_m in itr_m:
        for i, mode in enumerate(modes):
            m[mode] = comb_m[i]
        e_m = get_vib_energy(freq[0], m, E0[1])
        for comb_n in itr_n:
            for i, mode in enumerate(modes):
                n[mode] = comb_n[i]
            I = intensity(d, freq, mu, m, n, T, S_limit, modes)
            if I < 1e-3: continue
            e_n = get_vib_energy(freq[0], n, E0[0])
            f.write("%10.6f  %10.6f  %s\n" % (I, e_n-e_m, c))
    #############################
    #"""
    f.close()

def main():

## Path names to input files
#xyz = args.xyz_file                # xyz file which contains normal modes
#vib = args.vib_file    # output file which contains frequencies in a list
#vertical_IP
    # Now we read read input of intensity calculations from relevant files. _0 is for neutral. 
    print("Reading vibrational output")
    freq = [get_frequencies(vib), get_frequencies(vib_e)]                     
    mu = [get_reduced_masses(xyz), get_reduced_masses(xyz_e)]                    
    norm_modes = [get_normal_modes(xyz), get_normal_modes(xyz)]
    #force_constants = [get_force_constants(xyz), get_force_constants(xyz_e)]           
    T = get_transformation_matrix(norm_modes[0]) 
    zeropoint_energies = [get_zero_point_energies(vib), get_zero_point_energies(vib_e)]
    pos = [get_positions(xyz), get_positions(xyz_e)]
    
    # calculate displacement in geometries between states
    disp = pos[1]-pos[0]
    # Convert displacement to normal coordinates
    diff_normal_coordinates = np.dot(T, disp)

    write_intensities(diff_normal_coordinates, freq[0], mu[0], vertical_IP, temperature, S_lim, output_file, zeropoint_energies[0]-zeropoint_energies[i+1] )


if __name__ == "__main__":
    main()