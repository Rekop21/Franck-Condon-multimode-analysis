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



    # Optional arguments
    parser.add_argument("-T", "--temparature", help="Temparature", type=int, default=290)
    parser.add_argument("-S", "--S_lim", help="Limit for the Huang-Rhys constant for relevant modes", type=float, default=1e-4)
    parser.add_argument("-m", "--m", help="Limitation for the combinations of the ground state.", type=int, default=4)
    parser.add_argument("-n", "--n", help="Limitation for the combinations of the excited state.", type=int, default=3)
    parser.add_argument("-nm",  "--normal_modes", help="name of the xyz file which contains normal modes", type=str, default="run.xyz")
    parser.add_argument("-freq",  "--frequencies", help="name of the output file which contains frequencies in a list", type=str, default="vib_post_0.0025.out")
    parser.add_argument("-mass",  "--mass", help="name of the mass file, currently data from here not in use", type=str, default="masses.run_0.0025.dat")
    parser.add_argument("-o",  "--output", help="output file", type=str, default="intensity.dat")   
    parser.add_argument("-E", "--energy_file", help="Filename for the ionization energies. If left empty script will use energies from relaxations.", type=str, default="")
    # Print version
    parser.add_argument("--version", action="version", version='%(prog)s - Version 1.0')

    # Parse arguments
    args = parser.parse_args()

    return args

args = parseArguments()