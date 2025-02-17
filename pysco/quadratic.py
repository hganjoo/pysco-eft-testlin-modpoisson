"""
This module implements numerical solutions for a quadratic operator in the context of the EFT of gravity,
based on the work by Cusin et al. (2017). 

The EFT model implemented here has two additional params (alphaB, alphaM). 

The density term b has to be pre-processed into box units before inserting as args here. 

Himanish Ganjoo - Nov 2024
"""

import numpy as np
import numpy.typing as npt
from numba import config, njit, prange
import mesh

@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def operator(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32, 
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32
    ) -> npt.NDArray[np.float32]:
    """Quadratic operator

    a pi^2 + b pi + c = 0 \\
    EFT from Cusin et al. (2017)\\
    
    Parameters
    ----------
    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Dimensionless Hubble param E(a) E(a)
    a : np.float32
        scale factor
    
        
    Returns
    -------
    npt.NDArray[np.float32]
        Quadratic operator(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    """
    ncells_1d = pi.shape[0]
    result = np.empty_like(pi)
    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):

                h2 = h**2
                h4 = h2**2
                aH2 = (a*H)**2
                
                onebyfour = np.float32(0.25)
                onebyeight = np.float32(0.125)
                six = np.float32(6)
                eight = np.float32(8)
                two = np.float32(2)

                
                pins = pi[-1 + i,j,k] + pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k] + pi[1 + i,j,k]
                
                av = 0.

                lin = (alphaB*(six*alphaB - two*six*alphaM) + six*C2)/h2
                #nlin = -eight*pins/(h4)
                
                bv = lin

                '''lin = (
                    2*onebyfour*(a**2*(-alphaB + alphaM )*b[i,j,k])/M**2 
                    + ((alphaB*(-alphaB + 2.*alphaM) - C2)*(pins))/h2
                )'''

                lin = (
                    (alphaM - alphaB) * b[i,j,k]
                    + ((alphaB*(-alphaB + 2.*alphaM) - C2)*(pins))/h2
                )

                # Coeff of pi^0 in Q2[pi,pi]
                '''q2offd = -onebyeight*((pi[i,-1 + j,-1 + k] - pi[i,-1 + j,1 + k] - pi[i,1 + j,-1 + k] + pi[i,1 + j,1 + k])**2 
                - 16.*((pi[i,j,-1 + k] + pi[i,j,1 + k])*(pi[i,-1 + j,k] + pi[i,1 + j,k]) + pi[-1 + i,j,k]*(pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k]) + (pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k])*pi[1 + i,j,k]) 
                + (pi[-1 + i,j,-1 + k] - pi[-1 + i,j,1 + k] - pi[1 + i,j,-1 + k] + pi[1 + i,j,1 + k])**2 
                + (pi[-1 + i,-1 + j,k] - pi[-1 + i,1 + j,k] - pi[1 + i,-1 + j,k] + pi[1 + i,1 + j,k])**2)/(h4)'''

                cv = lin 


                result[i,j,k] = bv*pi[i,j,k] + cv
    
    return result



@njit(
        ["f4(f4[:,:,::1],f4,i4,i4,i4,f4,f4,f4,f4,f4,f4,f4)"],
        fastmath=True
)
def solution_quadratic_equation(
    pi: npt.NDArray[np.float32],
    b: np.float32,
    x: np.int32,
    y: np.int32,
    z: np.int32,
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32
    ) -> np.float32:
    
    """Solution of the quadratic equation governing the pi (chi) field \\
    for the EFT parameters. 

    This computes the solution to pi[i,j,k] in terms of the density field and \\
    the neighbours of the cell [i,j,k].

    Parameters
    ----------
    pi : npt.NDArray[np.float32]
         Pi Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term at [x,y,z]
    x,y,z : np.int16
        3D indices [i,j,k]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    

    Returns
    -------
    np.float32
        Solution of the quadratic equation for Pi at location [x,y,z]
    """

    h2 = h**2
    h4 = h2**2
    aH2 = (a*H)**2
    onebyfour = np.float32(0.25)
    onebyeight = np.float32(0.125)
    six = np.float32(6)
    eight = np.float32(8)
    two = np.float32(2)

    
    pins = pi[-1 + x,y,z] + pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z] + pi[1 + x,y,z]
    
    #av = (-six*C4)/(h4 * aH2) # goes to zero for linear
    av = 0

    lin = (alphaB*(six*alphaB - two*six*alphaM) + six*C2)/h2
    #nlin = -eight*pins/(h4) # goes to zero for linear
    
    bv = lin

    '''lin = (
        2*onebyfour*(a**2*(-alphaB + alphaM )*b)/M**2 
        + ((alphaB*(-alphaB + two*alphaM) - C2)*(pins))/h2
    )'''

    lin = (
                    (alphaM - alphaB) * b
                    + ((alphaB*(-alphaB + 2.*alphaM) - C2)*(pins))/h2
                )

    # Coeff of pi^0 in Q2[pi,pi] goes to zero for linear
    '''q2offd = -onebyeight*((pi[x,-1 + y,-1 + z] - pi[x,-1 + y,1 + z] - pi[x,1 + y,-1 + z] + pi[x,1 + y,1 + z])**2 
    - 16.*((pi[x,y,-1 + z] + pi[x,y,1 + z])*(pi[x,-1 + y,z] + pi[x,1 + y,z]) + pi[-1 + x,y,z]*(pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z]) + (pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z])*pi[1 + x,y,z]) 
    + (pi[-1 + x,y,-1 + z] - pi[-1 + x,y,1 + z] - pi[1 + x,y,-1 + z] + pi[1 + x,y,1 + z])**2 
    + (pi[-1 + x,-1 + y,z] - pi[-1 + x,1 + y,z] - pi[1 + x,-1 + y,z] + pi[1 + x,1 + y,z])**2)/(h4)'''

    cv = lin

    '''if bv*bv - 4*av*cv < 0:
        print('Cvals:',av,bv,cv,bv**2 - 4*av*cv)'''
    
    '''if (x==1)&(y==1)&(z==1):
        print('Dsc111:',bv**2 - 4*av*cv)'''
    
    '''dterm = bv**2 - 4*av*cv
    if dterm>0:
        qsol =  (-bv - np.sqrt(dterm)) / (2*av)
    else:
        qsol = -0.5*bv/av'''
    
   
    return -1.*cv/bv


@njit(
        ["f4(f4[:,:,::1],f4,i4,i4,i4,f4,f4,f4,f4,f4,f4,f4,f4)"],
        fastmath=True
)
def solution_quadratic_equation_with_rhs(
    pi: npt.NDArray[np.float32],
    b: np.float32,
    x: np.int32,
    y: np.int32,
    z: np.int32,
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    rhs: np.float32
    ) -> np.float32:
    
    """Solution of the quadratic equation governing the pi (chi) field \\
    for the EFT parameters. 

    This computes the solution to pi[i,j,k] in terms of the density field and \\
    the neighbours of the cell [i,j,k].

    Parameters
    ----------
    pi : npt.NDArray[np.float32]
         Pi Field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term at [x,y,z]
    x,y,z : np.int16
        3D indices [i,j,k]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    rhs: np.float32
        right hand side of the field equation at [x,y,z]
    

    Returns
    -------
    np.float32
        Solution of the quadratic equation for Pi at location [x,y,z]
    """

    h2 = h**2
    h4 = h2**2
    aH2 = (a*H)**2
    onebyfour = np.float32(0.25)
    onebyeight = np.float32(0.125)
    six = np.float32(6)
    eight = np.float32(8)
    two = np.float32(2)

    
    pins = pi[-1 + x,y,z] + pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z] + pi[1 + x,y,z]
    
    #av = (-six*C4)/(h4 * aH2) # goes to zero for linear
    av = 0

    lin = (alphaB*(six*alphaB - two*six*alphaM) + six*C2)/h2
    #nlin = -eight*pins/(h4) # goes to zero for linear
    
    bv = lin

    '''lin = (
        2*onebyfour*(a**2*(-alphaB + alphaM )*b)/M**2 
        + ((alphaB*(-alphaB + two*alphaM) - C2)*(pins))/h2
    )'''

    lin = (
                    (alphaM - alphaB) * b
                    + ((alphaB*(-alphaB + 2.*alphaM) - C2)*(pins))/h2
                )

    # Coeff of pi^0 in Q2[pi,pi] goes to zero for linear
    '''q2offd = -onebyeight*((pi[x,-1 + y,-1 + z] - pi[x,-1 + y,1 + z] - pi[x,1 + y,-1 + z] + pi[x,1 + y,1 + z])**2 
    - 16.*((pi[x,y,-1 + z] + pi[x,y,1 + z])*(pi[x,-1 + y,z] + pi[x,1 + y,z]) + pi[-1 + x,y,z]*(pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z]) + (pi[x,-1 + y,z] + pi[x,y,-1 + z] + pi[x,y,1 + z] + pi[x,1 + y,z])*pi[1 + x,y,z]) 
    + (pi[-1 + x,y,-1 + z] - pi[-1 + x,y,1 + z] - pi[1 + x,y,-1 + z] + pi[1 + x,y,1 + z])**2 
    + (pi[-1 + x,-1 + y,z] - pi[-1 + x,1 + y,z] - pi[1 + x,-1 + y,z] + pi[1 + x,1 + y,z])**2)/(h4)'''

    cv = lin - rhs

   
    return -1.*cv/bv




@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True
)
def jacobi(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    ) -> None:
    """Gauss-Seidel quadratic equation solver \\
    Solve the roots of u in the equation: \\
    a u^2 + bu + c = 0 \\
    for the EFT in Cusin et al (2017)\\
    
    Parameters
    ----------
    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    
    """

    ncells_1d = pi.shape[0]
    pi_old = np.copy(pi) # debug?

    for ix in range(-1,ncells_1d - 1):
            for iy in range(-1,ncells_1d - 1):
                for iz in range(-1,ncells_1d - 1):
                    pi[ix,iy,iz] = solution_quadratic_equation(pi_old,b[ix,iy,iz],ix,iy,iz,h,C2,C4,alphaB,alphaM,H,a)


@njit(
    ["void(f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4, f4[:,:,::1])"],
    fastmath=True,
    cache=True
)
def jacobi_with_rhs(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    rhs: npt.NDArray[np.float32]
    ) -> None:
    """Gauss-Seidel quadratic equation solver \\
    Solve the roots of u in the equation: \\
    a u^2 + bu + c = 0 \\
    for the EFT in Cusin et al (2017)\\
    
    Parameters
    ----------
    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    rhs: npt.NDArray[np.float32]
        right hand side of the field equation [N_cells_1d, N_cells_1d, N_cells_1d]
    """

    ncells_1d = pi.shape[0]
    pi_old = np.copy(pi) # debug?

    for ix in range(-1,ncells_1d - 1):
            for iy in range(-1,ncells_1d - 1):
                for iz in range(-1,ncells_1d - 1):
                    pi[ix,iy,iz] = solution_quadratic_equation_with_rhs(pi_old,b[ix,iy,iz],ix,iy,iz,h,C2,C4,alphaB,alphaM,H,a,rhs[ix,iy,iz])
              

def smoothing(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    n_smoothing: int) -> None:
    
    """Smooth Chi field with several Jacobi iterations

    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    n_smoothing : int
        number of smoothing iterations

    """
    
    for _ in range(n_smoothing):
        jacobi(pi, b, h, C2, C4, alphaB, alphaM, H, a)

def smoothing_with_rhs(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32,
    n_smoothing: int,
    rhs: npt.NDArray[np.float32]) -> None:
    
    """Smooth Chi field with several Jacobi iterations

    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    n_smoothing : int
        number of smoothing iterations

    """
    
    for _ in range(n_smoothing):
        jacobi_with_rhs(pi, b, h, C2, C4, alphaB, alphaM, H, a, rhs)



@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32
    ) -> npt.NDArray[np.float32]:

    """Residual of Quadratic operator

    R = -(a pi^2 + b pi + c) \\
    EFT from Cusin et al. (2017)\\
    
    Parameters
    ----------
    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    M : np.float32
        time-dependent Planck mass
        
    Returns
    -------
    npt.NDArray[np.float32]
        Residual(x) [N_cells_1d, N_cells_1d, N_cells_1d]

    """

    return -1*operator(pi,b,h,C2,C4,alphaB,alphaM,H,a)
    """ ncells_1d = pi.shape[0]
    result = np.empty_like(pi)
    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):

                h2 = h**2
                h4 = h2**2
                aH2 = (a*H)**2
                
                pins = pi[-1 + i,j,k] + pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k] + pi[1 + i,j,k]
                
                av = (-6.*C4)/(h4 * aH2)

                lin = (alphaB*(6.*alphaB - 12.*alphaM) + 6.*C2)/h2
                nlin = -8*pins/(h4)
                
                bv = lin - 0.25*C4*nlin/(aH2)

                lin = (
                    (a**2*(-0.5*alphaB + 0.5*alphaM )*b[i,j,k])/M**2 
                    + ((alphaB*(-alphaB + 2.*alphaM) - C2)*(pins))/h2
                )

                # Coeff of pi^0 in Q2[pi,pi]
                q2offd = -0.125*((pi[i,-1 + j,-1 + k] - pi[i,-1 + j,1 + k] - pi[i,1 + j,-1 + k] + pi[i,1 + j,1 + k])**2 
                - 16.*((pi[i,j,-1 + k] + pi[i,j,1 + k])*(pi[i,-1 + j,k] + pi[i,1 + j,k]) + pi[-1 + i,j,k]*(pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k]) + (pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k])*pi[1 + i,j,k]) 
                + (pi[-1 + i,j,-1 + k] - pi[-1 + i,j,1 + k] - pi[1 + i,j,-1 + k] + pi[1 + i,j,1 + k])**2 
                + (pi[-1 + i,-1 + j,k] - pi[-1 + i,1 + j,k] - pi[1 + i,-1 + j,k] + pi[1 + i,1 + j,k])**2)/(h4)

                cv = lin - 0.25*C4*q2offd/(aH2)

                result[i,j,k] = -(av*pi[i,j,k]**2 + bv*pi[i,j,k] + cv)
    
    return result
 """


@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def residual_error(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32
    ) -> np.float32:

    """Error on half of the residual of the quadratic operator  \\
    residual = -(a pi^2 + b pi + c)  \\
    error = sqrt[sum(residual**2)] \\
    
    Parameters
    ----------
    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    M : np.float32
        time-dependent Planck mass
    Returns
    -------
    np.float32
        Residual error

    """
    ncells_1d = pi.shape[0]
    res = operator(pi,b,h,C2,C4,alphaB,alphaM,H,a)
    result = 0.0

    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):
                result += res[i,j,k]**2
    
    return np.sqrt(result)
    


    """ ncells_1d = pi.shape[0]
    result = 0.0
    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):

                h2 = h**2
                h4 = h2**2
                aH2 = (a*H)**2
                
                pins = pi[-1 + i,j,k] + pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k] + pi[1 + i,j,k]
                
                av = (-6.*C4)/(h4 * aH2)

                lin = (alphaB*(6.*alphaB - 12.*alphaM) + 6.*C2)/h2
                nlin = -8*pins/(h4)
                
                bv = lin - 0.25*C4*nlin/(aH2)

                lin = (
                    (a**2*(-0.5*alphaB + 0.5*alphaM )*b[i,j,k])/M**2 
                    + ((alphaB*(-alphaB + 2.*alphaM) - C2)*(pins))/h2
                )

                # Coeff of pi^0 in Q2[pi,pi]
                q2offd = -0.125*((pi[i,-1 + j,-1 + k] - pi[i,-1 + j,1 + k] - pi[i,1 + j,-1 + k] + pi[i,1 + j,1 + k])**2 
                - 16.*((pi[i,j,-1 + k] + pi[i,j,1 + k])*(pi[i,-1 + j,k] + pi[i,1 + j,k]) + pi[-1 + i,j,k]*(pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k]) + (pi[i,-1 + j,k] + pi[i,j,-1 + k] + pi[i,j,1 + k] + pi[i,1 + j,k])*pi[1 + i,j,k]) 
                + (pi[-1 + i,j,-1 + k] - pi[-1 + i,j,1 + k] - pi[1 + i,j,-1 + k] + pi[1 + i,j,1 + k])**2 
                + (pi[-1 + i,-1 + j,k] - pi[-1 + i,1 + j,k] - pi[1 + i,-1 + j,k] + pi[1 + i,1 + j,k])**2)/(h4)

                cv = lin - 0.25*C4*q2offd/(aH2)

                result += (av*pi[i,j,k]**2 + bv*pi[i,j,k] + cv)**2
    
    return np.sqrt(result)
 """

@njit(
    ["f4(f4[:,:,::1], f4[:,:,::1], f4, f4, f4, f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def truncation_error(
    pi: npt.NDArray[np.float32],
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    C4: np.float32,
    alphaB: np.float32,
    alphaM: np.float32,
    H: np.float32,
    a: np.float32
    ) -> np.float32:

    """
    Truncation error estimator \\
    As in Numerical Recipes, we estimate the truncation error as \\
    t = Operator(Restriction(Phi)) - Operator(Laplacian(Phi)) \\
    error = sqrt[sum(residual**2)] \\
    
    Parameters
    ----------
    pi : npt.NDArray[np.float32]
        Scalar field [N_cells_1d, N_cells_1d, N_cells_1d]
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, C4, alphaB, alphaM : np.float32
        EFT params
    H : np.float32
        Dimensionless Hubble param E(a)
    a : np.float32
        scale factor
    
    
    Returns
    -------
    np.float32
        Truncation error

    """

    ncells_1d = pi.shape[0] >> 1
    RLx = mesh.restriction(operator(pi,b,h,C2,C4,alphaB,alphaM,H,a))
    LRx = operator(mesh.restriction(pi), mesh.restriction(b), 2 * h ,C2,C4,alphaB,alphaM,H,a)
    result = 0.0
    for i in prange(-1, ncells_1d - 1):
        for j in prange(-1, ncells_1d - 1):
            for k in prange(-1, ncells_1d - 1):
                result += (RLx[i, j, k] - LRx[i, j, k]) ** 2
    return np.sqrt(result)



@njit(
    ["f4[:,:,::1](f4[:,:,::1], f4, f4, f4, f4)"],
    fastmath=True,
    cache=True,
    parallel=True,
)
def initialise_potential(
    b: npt.NDArray[np.float32],
    h: np.float32,
    C2: np.float32,
    alphaB: np.float32,
    alphaM: np.float32
) -> npt.NDArray[np.float32]:
    """
    HG: 14/11/2024

    VERY ROUGH VERSION - have to decide how to initialise the scalar field \\
    this might be too slow
    
    Solution for the Chi field \\
    using the linear order DE solution \\
    Laplacian[Chi] = mu_chi * delta 

    Parameters
    ----------
    b : npt.NDArray[np.float32]
        Density term [N_cells_1d, N_cells_1d, N_cells_1d]
    h : np.float32
        Grid size
    C2, alphaB0, alphaM0 : np.float32
        EFT params, basic and derived, taken from cosmotable
    

    Returns
    -------
    npt.NDArray[np.float32]
        Chi field

    
    """
    xi = alphaB - alphaM
    nu = -C2 - alphaB*(xi - alphaM)
    mu_chi = 1 + xi*xi/nu
    
    pi = np.empty_like(b)
    one_by_six = np.float32(1./6)
    ncells_1d = b.shape[0]
    for i in prange(ncells_1d):
        for j in prange(ncells_1d):
            for k in prange(ncells_1d):
                pi[i, j, k] = - one_by_six*mu_chi*h*h*b[i,j,k]
    #print('Init F:',pi)
    
    return pi


