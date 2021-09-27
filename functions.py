import sys
import numpy as np
import math
import numpy as np
from scipy import special
from scipy import linalg

class Primitive_gaussian():
      def __init__(self, alpha, coeff, coordinates, l1, l2, l3):
          self.alpha = alpha
          self.coeff = coeff
          self.coordinates =np.array(coordinates)
          self.A = (2.0*alpha/math.pi)**0.75 # other terms L1, 2, 3

def overlap(molecule):
    nbasis = len(molecule)
    S = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    N = molecule[i][k].A * molecule[j][l].A
                    p = molecule[i][k].alpha+molecule[j][l].alpha
                    q = molecule[i][k].alpha*molecule[j][l].alpha/p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q2 = np.dot(Q, Q)
                    S[i][j] += N * molecule[i][k].coeff * molecule[j][l].coeff * math.exp(-q*Q2) * (math.pi/p)**(3/2)
    return S

def kinetic(molecule):
    nbasis = len(molecule)
    T = np.zeros([nbasis, nbasis])
    for i in range(nbasis):
        for j in range(nbasis):
            nprimitives_i = len(molecule[i])
            nprimitives_j = len(molecule[j])
            for k in range(nprimitives_i):
                for l in range(nprimitives_j):
                    
                    c1c2 = molecule[i][k].coeff * molecule[j][l].coeff
                    N = molecule[i][k].A * molecule[j][l].A
                    p = molecule[i][k].alpha+molecule[j][l].alpha
                    q = molecule[i][k].alpha*molecule[j][l].alpha/p
                    Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                    Q2 = np.dot(Q, Q)
                    
                    P = molecule[i][k].alpha*molecule[i][k].coordinates + molecule[j][l].alpha*molecule[j][l].coordinates
                    Pp = P/p
                    PG = Pp - molecule[j][l].coordinates
                    PGx2 = PG[0]*PG[0]
                    PGy2 = PG[1]*PG[1]
                    PGz2 = PG[2]*PG[2]

                    s = N *  c1c2* math.exp(-q*Q2) * (math.pi/p)**(3/2)
                    T[i][j] += 3* molecule[j][l].alpha*s
                    T[i][j] -= 2* molecule[j][l].alpha* molecule[j][l].alpha*s*(PGx2+0.5/p)
                    T[i][j] -= 2* molecule[j][l].alpha* molecule[j][l].alpha*s*(PGy2+0.5/p)
                    T[i][j] -= 2* molecule[j][l].alpha* molecule[j][l].alpha*s*(PGz2+0.5/p)

    return T

def boys(x, n):
    if x == 0:
        return 1.0/(2)
    else:
        return special.gammainc(n+0.5, x) * special.gamma(n+0.5)*(1.0/(2*x**(n+0.5)))
def electron_nuclear_attraction(molecule, atom_coordinates, Z):
    natoms = len(Z)
    nbasis = len(molecule)
    V_ne = np.zeros([nbasis, nbasis])
    for atom in range(natoms):
        for i in range(nbasis):
            for j in range(nbasis):
                nprimitives_i = len(molecule[i])
                nprimitives_j = len(molecule[j])
                for k in range(nprimitives_i):
                    for l in range(nprimitives_j):

                        c1c2 = molecule[i][k].coeff * molecule[j][l].coeff
                        N = molecule[i][k].A * molecule[j][l].A
                        p = molecule[i][k].alpha+molecule[j][l].alpha
                        q = molecule[i][k].alpha*molecule[j][l].alpha/p
                        Q = molecule[i][k].coordinates - molecule[j][l].coordinates
                        Q2 = np.dot(Q, Q)

                        P = molecule[i][k].alpha*molecule[i][k].coordinates + molecule[j][l].alpha*molecule[j][l].coordinates
                        Pp = P/p
                        PG = Pp - atom_coordinates[atom]
                        PG2 = np.dot(PG, PG)

                        V_ne[i][j] += -Z[atom]*N*c1c2*math.exp(-q*Q2)*(2.0*math.pi/p)*boys(p*PG2, 0)
    return V_ne


def electron_electron_repulsion(molecule):

   nbasis = len(molecule)
   V_ee = np.zeros([nbasis, nbasis, nbasis, nbasis])

   for i in range(nbasis):
       for j in range(nbasis):
           for k in range(nbasis):
               for l in range(nbasis):
                   nprimitives_i = len(molecule[i])
                   nprimitives_j = len(molecule[j])
                   nprimitives_k = len(molecule[k])
                   nprimitives_l = len(molecule[l])

                   for ii in range(nprimitives_i):
                       for jj in range(nprimitives_j):
                           for kk in range(nprimitives_k):
                                for ll in range(nprimitives_l):

                                    N = molecule[i][ii].A*molecule[j][jj].A*molecule[k][kk].A*molecule[l][ll].A
                                    cicjckcl = molecule[i][ii].coeff*molecule[j][jj].coeff*\
                                               molecule[k][kk].coeff*molecule[l][ll].coeff

                                    pij = molecule[i][ii].alpha+molecule[j][jj].alpha
                                    pkl = molecule[k][kk].alpha+molecule[l][ll].alpha

                                    Pij = molecule[i][ii].alpha*molecule[i][ii].coordinates+\
                                          molecule[j][jj].alpha*molecule[j][jj].coordinates
                                    Pkl = molecule[k][kk].alpha*molecule[k][kk].coordinates+\
                                          molecule[l][ll].alpha*molecule[l][ll].coordinates

                                    Ppij = Pij/pij
                                    Ppkl = Pkl/pkl

                                    PpijPpkl = Ppij - Ppkl
                                    PpijPpkl2 = np.dot(PpijPpkl, PpijPpkl)
                                    denom = 1.0/pij + 1.0/pkl

                                    qij = molecule[i][ii].alpha*molecule[j][jj].alpha / pij
                                    qkl = molecule[k][kk].alpha*molecule[l][ll].alpha / pkl

                                    Qij = molecule[i][ii].coordinates - molecule[j][jj].coordinates
                                    Qkl = molecule[k][kk].coordinates - molecule[l][ll].coordinates

                                    Q2ij = np.dot(Qij, Qij)
                                    Q2kl = np.dot(Qkl, Qkl)

                                    term1 = 2.0*math.pi*math.pi/(pij*pkl)
                                    term2 = math.sqrt(math.pi/(pij+pkl) )
                                    term3 = math.exp(-qij*Q2ij)
                                    term4 = math.exp(-qkl*Q2kl)

                                    V_ee[i, j, k, l] += N* cicjckcl * term1 * term2 * term3 * term4 * boys(PpijPpkl2/denom, 0) # 3 more for p orbotals
   return V_ee

#STO-3G basis set
H1_pg1a = Primitive_gaussian(0.3425250914E+01 ,0.1543289673E+00, [0, 0, 0], 0, 0,0)
H1_pg1b = Primitive_gaussian(0.6239137298E+00 ,0.5353281423E+00 , [0, 0, 0], 0, 0,0)
H1_pg1c = Primitive_gaussian(0.1688554040E+00 ,0.4446345422E+00 , [0, 0, 0], 0, 0,0)

H2_pg1a = Primitive_gaussian(0.3425250914E+01 ,0.1543289673E+00, [1.4, 0, 0], 0, 0,0)
H2_pg1b = Primitive_gaussian(0.6239137298E+00 ,0.5353281423E+00 , [1.4, 0, 0], 0, 0,0)
H2_pg1c = Primitive_gaussian(0.1688554040E+00 ,0.4446345422E+00 , [1.4, 0, 0], 0, 0,0)
H1_1s = [H1_pg1a, H1_pg1b, H1_pg1c]
H2_1s = [H2_pg1a, H2_pg1b, H2_pg1c]
molecule = [H1_1s, H2_1s]

Z = [1.0, 1.0]
atom_coordinates = [np.array([0, 0, 0]),
                   np.array([1.4, 0, 0])]
print('\nSTO-3G basis for H2:\n')
print('overlap\n', overlap(molecule))
print('kinetic energy\n', kinetic(molecule))
print('electron nuclear attraction\n', electron_nuclear_attraction(molecule, atom_coordinates, Z))
print('electron electron repuslion\n', electron_electron_repulsion(molecule))


#6-31G basis for 1s and 2s hydrogen

H1_pg1a = Primitive_gaussian(0.1873113696E+02 ,0.3349460434E-01 , [0, 0, 0], 0, 0,0)
H1_pg1b = Primitive_gaussian(0.2825394365E+01 ,0.2347269535E+00 , [0, 0, 0], 0, 0,0)
H1_pg1c = Primitive_gaussian(0.6401216923E+00 ,0.8137573261E+00 , [0, 0, 0], 0, 0,0)
H1_pg2a = Primitive_gaussian(0.1612777588E+00 ,1.0000000 , [0, 0, 0], 0, 0,0)

H2_pg1a = Primitive_gaussian(0.1873113696E+02 ,0.3349460434E-01 , [1.4, 0, 0], 0, 0,0)
H2_pg1b = Primitive_gaussian(0.2825394365E+01 ,0.2347269535E+00 , [1.4, 0, 0], 0, 0,0)
H2_pg1c = Primitive_gaussian(0.6401216923E+00 ,0.8137573261E+00 , [1.4, 0, 0], 0, 0,0)
H2_pg2a = Primitive_gaussian(0.1612777588E+00 ,1.0000000 , [1.4, 0, 0], 0, 0,0)

H1_1s = [H1_pg1a, H1_pg1b, H1_pg1c]
H1_2s = [H1_pg2a]
H2_1s = [H2_pg1a, H2_pg1b, H2_pg1c]
H2_2s = [H2_pg2a]
molecule =[H1_1s, H1_2s, H2_1s, H2_2s]
Z = [1.0, 1.0]
atom_coordinates = [np.array([0, 0, 0]),
                   np.array([1.4, 0, 0])]
print('\n 6-31G basis for H2:\n')
print('overlap\n', overlap(molecule))
print('kinetic energy\n', kinetic(molecule))
print('electron nuclear attraction\n', electron_nuclear_attraction(molecule, atom_coordinates, Z))
print('electron electron repuslion\n', electron_electron_repulsion(molecule))

