import numpy as np
import math
def Beam2DMatrices(m, EA, EI, NodeCoord):
# Inputs:
# m         - mass per unit length [kg/m]
# EA        - axial stiffness [N]
# EI        - bending stiffness [N.m2]
# NodeCoord - ([xl, yl], [xr, yr])      
#           - left (l) and right (r) node coordinates

    # 1 - calculate length of beam (L) and orientation alpha
    xl = NodeCoord[0][0]    # x-coordinate of left node
    yl = NodeCoord[0][1]    # y-coordinate of left node
    xr = NodeCoord[1][0]    # x-coordinate of right node
    yr = NodeCoord[1][1]    # y-coordinate of rigth node
    L = np.sqrt((xr - xl)**2 + (yr - yl)**2)    # length
    alpha = math.atan2(yr - yl, xr - xl)        # angle

    # 2 - calculate transformation matrix T
    C = np.cos(alpha)
    S = np.sin(alpha)
    T = np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])
    T = np.asarray(np.bmat([[T, np.zeros((3,3))], [np.zeros((3, 3)), T]]))
    

    # 3 - calculate local stiffness and matrices
    L2 = L*L
    L3 = L*L2
    K = np.array([[EA/L, 0, 0, -EA/L, 0, 0], 
                    [0, 12*EI/L3, 6*EI/L2, 0, -12*EI/L3, 6*EI/L2], 
                    [0, 6*EI/L2, 4*EI/L, 0, -6*EI/L2, 2*EI/L], 
                    [-EA/L, 0, 0, EA/L, 0, 0], 
                    [0, -12*EI/L3, -6*EI/L2, 0, 12*EI/L3, -6*EI/L2], 
                    [0, 6*EI/L2, 2*EI/L, 0, -6*EI/L2, 4*EI/L]])
    M = m*L/420*np.array([[140, 0, 0, 70, 0, 0], 
                            [0, 156, 22*L, 0, 54, -13*L], 
                            [0, 22*L, 4*L2, 0, 13*L, -3*L2], 
                            [70, 0, 0, 140, 0, 0], 
                            [0, 54, 13*L, 0, 156, -22*L], 
                            [0, -13*L, -3*L2, 0, -22*L, 4*L2]])
    
    Q = L/420 * np.array([[140, 0, 0, 70, 0, 0], 
                            [0, 156, 22*L, 0, 54, -13*L], 
                            [0, 22*L, 4*L2, 0, 13*L, -3*L2], 
                            [70, 0, 0, 140, 0, 0], 
                            [0, 54, 13*L, 0, 156, -22*L], 
                            [0, -13*L, -3*L2, 0, -22*L, 4*L2]])

    # 4 - rotate the matrices
    K = np.matmul(np.transpose(T), np.matmul(K, T))
    M = np.matmul(np.transpose(T), np.matmul(M, T))
    Q = np.matmul(np.transpose(T), np.matmul(Q, T))
    return M, K, Q


def Beam3DMatrices(m, EA, EI, GJ, Im, NodeCoord):
# Inputs:
# m         - mass per unit length [kg/m]
# EA        - axial stiffness [N]
# EI        - bending stiffness [N.m2]
# NodeCoord - ([xl, yl, zl], [xr, yr, zr])
#           - left (l) and right (r) node coordinates

    # 1 - calculate length of beam (L) and orientation alpha
    xl = NodeCoord[0][0]    # x-coordinate of left node
    yl = NodeCoord[0][1]    # y-coordinate of left node
    zl = NodeCoord[0][2]    # z-coordinate of left node
    xr = NodeCoord[1][0]    # x-coordinate of right node
    yr = NodeCoord[1][1]    # y-coordinate of rigth node
    zr = NodeCoord[1][2]    # z-coordinate of rigth node
    L = np.sqrt((xr - xl)**2 + (yr - yl)**2 + (zr - zl)**2)    # length
    
    # 2 - calculate transformation matrix T
    C = (xr-xl)/L
    S = (yr-yl)/L
    # T in this is different from T in the above Beam 2D function
    T = np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])
    
    # Only support rotation in XY plane
    if(abs(zr - zl) > 1e-6):
        print("Error, Only supports rotation in XY plane")
        T = np.array([[1, 0, 0], [0, 1, 0], [0,0,1]])        
    
    T = np.asarray(np.bmat([[T, np.zeros((3,3))], [np.zeros((3, 3)), T]]))    
    T = np.asarray(np.bmat([[T, np.zeros((6,6))], [np.zeros((6, 6)), T]]))    
    # print(T)

    # 3 - calculate local stiffness and matrices
    L2 = L*L
    L3 = L*L2
    K = np.array([[EA/L, 0, 0, 0, 0, 0, -EA/L, 0, 0, 0, 0, 0], 
                  [0, 12*EI/L3, 0, 0, 0, 6*EI/L2, 0, -12*EI/L3, 0, 0, 0, 6*EI/L2], 
                  [0, 0, 12*EI/L3, 0, -6*EI/L2, 0, 0, 0, -12*EI/L3, 0, -6*EI/L2, 0], 
                  [0, 0, 0, GJ/L, 0, 0, 0, 0, 0, -GJ/L, 0, 0],
                  [0, 0, -6*EI/L2, 0, 4*EI/L, 0, 0, 0, 6*EI/L2, 0, 2*EI/L, 0], 
                  [0, 6*EI/L2, 0, 0, 0, 4*EI/L, 0, -6*EI/L2, 0, 0, 0, 2*EI/L], 
                  [-EA/L, 0, 0, 0, 0, 0, EA/L, 0, 0, 0, 0, 0], 
                  [0, -12*EI/L3, 0, 0, 0, -6*EI/L2, 0, 12*EI/L3, 0, 0, 0, -6*EI/L2], 
                  [0, 0, -12*EI/L3, 0, 6*EI/L2, 0, 0, 0, 12*EI/L3, 0, 6*EI/L2, 0], 
                  [0, 0, 0, -GJ/L, 0, 0, 0, 0, 0, GJ/L, 0, 0],                  
                  [0, 0, -6*EI/L2, 0, 2*EI/L, 0, 0, 0, 6*EI/L2, 0, 4*EI/L, 0],
                  [0, 6*EI/L2, 0, 0, 0, 2*EI/L, 0, -6*EI/L2, 0, 0, 0, 4*EI/L]])    
    
    M = np.array([[140, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0], 
                  [0, 156, 0, 0, 0, 22*L, 0, 54, 0, 0, 0, -13*L], 
                  [0, 0, 156, 0, 22*L, 0, 0, 0, 54, 0, 13*L, 0], 
                  [0, 0, 0, 140*Im, 0, 0, 0, 0, 0, 70*Im, 0, 0],
                  [0, 0, 22*L, 0, 4*L2, 0, 0, 0, -13*L, 0, -3*L2, 0], 
                  [0, 22*L, 0, 0, 0, 4*L2, 0, 13*L, 0, 0, 0, -3*L2], 
                  [70, 0, 0, 0, 0, 0, 140, 0, 0, 0, 0, 0], 
                  [0, 54, 0, 0, 0, 13*L, 0, 156, 0, 0, 0, -22*L], 
                  [0, 0, 54, 0, -13*L, 0, 0, 0, 156, 0, 22*L, 0], 
                  [0, 0, 0, 70*Im, 0, 0, 0, 0, 0, 140*Im, 0, 0],
                  [0, 0, 13*L, 0, -3*L2, 0, 0, 0, 22*L, 0, 4*L2, 0],
                  [0, -13*L, 0, 0, 0, -3*L2, 0, -22*L, 0, 0, 0, 4*L2]])
    M = m*L/420 * M
    
    Q = np.array([[140, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0], 
                  [0, 156, 0, 0, 0, 22*L, 0, 54, 0, 0, 0, -13*L], 
                  [0, 0, 156, 0, 22*L, 0, 0, 0, 54, 0, 13*L, 0], 
                  [0, 0, 0, 140, 0, 0, 0, 0, 0, 70, 0, 0],
                  [0, 0, 22*L, 0, 4*L2, 0, 0, 0, -13*L, 0, -3*L2, 0], 
                  [0, 22*L, 0, 0, 0, 4*L2, 0, 13*L, 0, 0, 0, -3*L2], 
                  [70, 0, 0, 0, 0, 0, 140, 0, 0, 0, 0, 0], 
                  [0, 54, 0, 0, 0, 13*L, 0, 156, 0, 0, 0, -22*L], 
                  [0, 0, 54, 0, -13*L, 0, 0, 0, 156, 0, 22*L, 0], 
                  [0, 0, 0, 70, 0, 0, 0, 0, 0, 140, 0, 0],
                  [0, 0, 13*L, 0, -3*L2, 0, 0, 0, 22*L, 0, 4*L2, 0],
                  [0, -13*L, 0, 0, 0, -3*L2, 0, -22*L, 0, 0, 0, 4*L2]])
    Q = L/420 * Q

    # 4 - rotate the matrices
    K = np.matmul(np.transpose(T), np.matmul(K, T))
    M = np.matmul(np.transpose(T), np.matmul(M, T))
    Q = np.matmul(np.transpose(T), np.matmul(Q, T))
    return M, K, Q