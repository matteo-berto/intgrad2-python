import numpy as np
import numpy.matlib
from scipy.sparse import csr_matrix


def intgrad2(fx,fy,dx=1,dy=1,f11=0):
    """
    Author: Matteo Berto
    Institution: University of Padua (Università degli Studi di Padova)
    Date: 28th August 2024
    https://github.com/matteo-berto/intgrad2-python
    
    Python implementation of MATLAB function "intgrad2" by John D'Errico.
    
    John D'Errico (2024). Inverse (integrated) gradient
    (https://www.mathworks.com/matlabcentral/fileexchange/9734-inverse-integrated-gradient)
    MATLAB Central File Exchange. Retrieved August 28, 2024.
    
    intgrad2: generates a 2-d surface, integrating gradient information.
    usage: fhat = intgrad2(fx,fy)
    usage: fhat = intgrad2(fx,fy,dx,dy)
    usage: fhat = intgrad2(fx,fy,dx,dy,f11)
    
    arguments: (input)
     fx,fy - (ny by nx) arrays, as gradient would have produced. fx and
             fy must both be the same size. Note that x is assumed to
             be the column dimension of f, in the meshgrid convention.
    
             nx and ny must both be at least 2.
    
             fx and fy will be assumed to contain consistent gradient
             information. If they are inconsistent, then the generated
             gradient will be solved for in a least squares sense.
    
             Central differences will be used where possible.
                
        dx - (OPTIONAL) scalar or vector - denotes the spacing in x
             if dx is a scalar, then spacing in x (the column index
             of fx and fy) will be assumed to be constant = dx.
             if dx is a vector, it denotes the actual coordinates
             of the points in x (i.e., the column dimension of fx
             and fy.) length(dx) == nx
    
             DEFAULT: dx = 1
                
        dy - (OPTIONAL) scalar or vector - denotes the spacing in y
             if dy is a scalar, then the spacing in x (the row index
             of fx and fy) will be assumed to be constant = dy.
             if dy is a vector, it denotes the actual coordinates
             of the points in y (i.e., the row dimension of fx
             and fy.) length(dy) == ny
    
             DEFAULT: dy = 1
                
        f11 - (OPTIONAL) scalar - defines the (1,1) element of fhat
             after integration. This is just the constant of integration.

             DEFAULT: f11 = 0
    """
    if len(fx.shape)>2 or len(fy.shape)>2:
        raise Exception('fx and fy must be 2d arrays')
    [ny,nx] = fx.shape
    if nx!=np.shape(fy)[1] or ny!=np.shape(fy)[0]:
        raise Exception('fx and fy must be the same sizes')
    if nx<2 or ny<2:
        raise Exception('fx and fy must be at least 2x2 arrays')
    # if scalar spacings, expand them to be vectors
    # dx=dx.flatten()
    if np.size(dx) == 1:
        dx = np.matlib.repmat(dx,nx-1,1)
    elif np.size(dx)==nx:
        # dx was a vector, use diff to get the spacing
        dx = np.atleast_2d(np.diff(dx)).T
    else:
        raise Exception('dx is not a scalar or of length == nx')
    # dy=dy.flatten()
    if np.size(dy) == 1:
        dy = np.matlib.repmat(dy,ny-1,1)
    elif np.size(dy)==ny:
        # dy was a vector, use diff to get the spacing
        dy = np.atleast_2d(np.diff(dy)).T
    else:
        raise Exception('dy is not a scalar or of length == ny')
    if np.size(f11) > 1 or np.isnan(f11) or not np.isfinite(f11): #|| ~isnumeric(f11) ||
        raise Exception('f11 must be a finite scalar numeric variable')
    # build gradient design matrix, sparsely. Use a central difference
    # in the body of the array, and forward/backward differences along
    # the edges.
    # A will be the final design matrix. it will be sparse.
    # The unrolling of F will be with row index running most rapidly.
    rhs = np.zeros([2*nx*ny,1])
    # but build the array elements in Af
    Af = np.zeros([2*nx*ny,6])
    L = 0;
    # do the leading edge in x, forward difference
    indx = 1
    indy = np.atleast_2d((np.linspace(1, ny, num=ny))).T
    ind = indy + (indx-1)*ny
    rind = np.matlib.repmat(L+indy,1,2)
    cind = np.concatenate((ind,ind+ny),1)
    dfdx = np.matlib.repmat([-1, 1]/dx[0],ny,1)
    Af[(L+np.linspace(1, ny, num=ny)-1).astype(int), :] = np.concatenate((rind,cind,dfdx),1)
    rhs[(L+np.linspace(1, ny, num=ny)-1).astype(int)] = (fx[:,0]).reshape(ny,1)
    L = L+ny
    # interior partials in x, central difference
    if nx>2:
        [indx,indy] = np.meshgrid(np.linspace(2, nx-1, num=nx-2),np.linspace(1, ny, num=ny))
        indx = indx.T.reshape((nx-2)*ny,1)
        indy = indy.T.reshape((nx-2)*ny,1)
        ind = indy + (indx-1)*ny
        m = ny*(nx-2)

        rind = np.matlib.repmat(L+np.atleast_2d((np.linspace(1, m, num=m))).T,1,2)
        cind = np.concatenate((ind-ny,ind+ny),1)
        
        dfdx = 1/((dx[np.array(list(map(int, indx)))-2])+(dx[np.array(list(map(int, indx)))-1]))
        dfdx = dfdx*[-1, 1]

        Af[(L+np.linspace(1, m, num=m)-1).astype(int), :] = np.concatenate((rind,cind,dfdx),1)
        [row,col] = np.unravel_index(np.array(list(map(int, ind)))-1, fx.shape, 'F')
        rhs[(L+np.linspace(1, m, num=m)-1).astype(int)] = (fx[row,col]).reshape((nx-2)*ny,1)

        L = L+m
    # do the trailing edge in x, backward difference
    indx = nx
    indy = np.atleast_2d((np.linspace(1, ny, num=ny))).T
    ind = indy + (indx-1)*ny
    rind = np.matlib.repmat(L+indy,1,2)
    cind = np.concatenate((ind-ny,ind),1)
    dfdx = np.matlib.repmat([-1, 1]/dx[-1],ny,1)
    Af[(L+np.linspace(1, ny, num=ny)-1).astype(int), :] = np.concatenate((rind,cind,dfdx),1)
    rhs[(L+np.linspace(1, ny, num=ny)-1).astype(int)] = (fx[:,-1]).reshape(ny,1)
    L = L+ny
    # do the leading edge in y, forward difference
    indx = np.atleast_2d((np.linspace(1, nx, num=nx))).T
    indy = 1
    ind = indy + (indx-1)*ny
    rind = np.matlib.repmat(L+indx,1,2)
    cind = np.concatenate((ind,ind+1),1)
    dfdy = np.matlib.repmat([-1, 1]/dy[0],nx,1)
    Af[(L+np.linspace(1, nx, num=nx)-1).astype(int), :] = np.concatenate((rind,cind,dfdy),1)
    rhs[(L+np.linspace(1, nx, num=nx)-1).astype(int)] = (fy[0,:]).reshape(nx,1)
    L = L+nx;
    # interior partials in y, use a central difference
    if ny>2:
        [indx,indy] = np.meshgrid(np.linspace(1, nx, num=nx),np.linspace(2, ny-1, num=ny-2))
        indx = indx.T.reshape(nx*(ny-2),1)
        indy = indy.T.reshape(nx*(ny-2),1)
        ind = indy + (indx-1)*ny
        m = nx*(ny-2)

        rind = np.matlib.repmat(L+np.atleast_2d((np.linspace(1, m, num=m))).T,1,2)
        cind = np.concatenate((ind-1,ind+1),1)

        dfdy = 1/((dy[np.array(list(map(int, indy)))-2])+(dy[np.array(list(map(int, indy)))-1]))
        dfdy = dfdy*[-1, 1]

        Af[(L+np.linspace(1, m, num=m)-1).astype(int), :] = np.concatenate((rind,cind,dfdy),1)
        [row,col] = np.unravel_index(np.array(list(map(int, ind)))-1, fy.shape, 'F')
        rhs[(L+np.linspace(1, m, num=m)-1).astype(int)] = (fy[row,col]).reshape(nx*(ny-2),1)

        L = L+m
    # do the trailing edge in y, backward difference
    indx = np.atleast_2d((np.linspace(1, nx, num=nx))).T
    indy = ny
    ind = indy + (indx-1)*ny
    rind = np.matlib.repmat(L+indx,1,2)
    cind = np.concatenate((ind-1,ind),1)
    dfdy = np.matlib.repmat([-1, 1]/dy[-1],nx,1)
    Af[(L+np.linspace(1, nx, num=nx)-1).astype(int), :] = np.concatenate((rind,cind,dfdy),1)
    rhs[(L+np.linspace(1, nx, num=nx)-1).astype(int)] = (fy[-1,:]).reshape(nx,1)
    # finally, we can build the rest of A itself, in its sparse form.
    Af_row = Af[:,[0, 1]].reshape((4*nx*ny,1), order='F').flatten()
    Af_col = Af[:,[2, 3]].reshape((4*nx*ny,1), order='F').flatten()
    Af_data = Af[:,[4, 5]].reshape((4*nx*ny,1), order='F').flatten()
    A = csr_matrix((Af_data, (Af_row-1,Af_col-1)), shape=(2*nx*ny,nx*ny)).toarray()
    # Finish up with f11, the constant of integration.
    # eliminate the first unknown, as f11 is given.
    rhs = rhs - A[:,[0]]*f11
    # Solve the final system of equations. They will be of
    # full rank, due to the explicit integration constant.
    # Just use sparse \
    fhat_lstsq = np.linalg.lstsq(A[:,1:], rhs)
    fhat = np.insert(fhat_lstsq[0],0,[f11]).reshape((ny,nx), order='F')

    return fhat