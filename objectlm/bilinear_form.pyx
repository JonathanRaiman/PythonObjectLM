import cython
from cython cimport view
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

from cpython cimport PyCapsule_GetPointer # PyCObject_AsVoidPtr
from scipy.linalg.blas import fblas

REAL = np.float32
DOUBLE = np.float64
ctypedef np.float32_t REAL_t
ctypedef np.float64_t DOUBLE_t

cdef int ONE = 1

cdef DOUBLE_t dONEF   = <DOUBLE_t>1.0
cdef DOUBLE_t dZEROF  = <DOUBLE_t>0.0

cdef REAL_t ONEF      = <REAL_t>1.0
cdef REAL_t ZEROF     = <REAL_t>0.0

cdef REAL_t SMALL_NUM = <REAL_t>1e-6

ctypedef void (*sger_ptr) (const int *M, const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY, float *A, const int * LDA) nogil
cdef sger_ptr sger=<sger_ptr>PyCapsule_GetPointer(fblas.sger._cpointer , NULL)  # A := alpha*x*y.T + A


ctypedef void (*sgemv_ptr) (char *trans, int *m, int *n,\
                     REAL_t *alpha, REAL_t *a, int *lda, REAL_t *x,\
                     int *incx,\
                     REAL_t *beta,  REAL_t *y, int *incy) nogil
cdef sgemv_ptr sgemv=<sgemv_ptr>PyCapsule_GetPointer(fblas.cblas.sgemv._cpointer, NULL) # y := A*x + beta * y
cdef void matrix_vector_product(char * tranposed, int *M, int *N, REAL_t *matrix, REAL_t *vector, REAL_t* destination) nogil:
    sgemv(tranposed, N, M, &ONEF, matrix, N, vector, &ONE, &ZEROF, destination, &ONE)

cdef void matrix_vector_product_additive(char * tranposed, int *M, int *N, REAL_t *matrix, REAL_t *vector, REAL_t* destination) nogil:
    sgemv(tranposed, N, M, &ONEF, matrix, N, vector, &ONE, &ONEF, destination, &ONE)

ctypedef void (*dgemv_ptr) (char *trans, int *m, int *n,\
                     DOUBLE_t *alpha, DOUBLE_t *a, int *lda, DOUBLE_t *x,\
                     int *incx,\
                     DOUBLE_t *beta,  DOUBLE_t *y, int *incy) nogil
cdef dgemv_ptr dgemv=<dgemv_ptr>PyCapsule_GetPointer(fblas.cblas.dgemv._cpointer, NULL) # y := A*x + beta * y
cdef void matrix_vector_product_double(char * tranposed, int *M, int *N, DOUBLE_t *matrix, DOUBLE_t *vector, DOUBLE_t* destination) nogil:
    dgemv(tranposed, N, M, &dONEF, matrix, N, vector, &ONE, &dZEROF, destination, &ONE)

cdef void matrix_vector_product_double_additive(char * tranposed, int *M, int *N, DOUBLE_t *matrix, DOUBLE_t *vector, DOUBLE_t* destination) nogil:
    dgemv(tranposed, N, M, &dONEF, matrix, N, vector, &ONE, &dONEF, destination, &ONE)

cdef char trans  = 'T'
cdef char transN = 'N'

cdef void _quadratic_form_cython_single_ptr(REAL_t* tensor, 
    int tensor_shape_0,
    int tensor_shape_2,
    REAL_t * obs_ptr,
    REAL_t * intermediary_result,
    REAL_t * final_result) nogil:
    cdef int i
    for i in range(tensor_shape_0):
        matrix_vector_product(&trans, &tensor_shape_2, &tensor_shape_2, &tensor[i * tensor_shape_2 * tensor_shape_2], obs_ptr, &intermediary_result[i * tensor_shape_2])
    for i in range(tensor_shape_0):
        matrix_vector_product_additive(&trans, &ONE, &tensor_shape_2, &intermediary_result[i * tensor_shape_2], obs_ptr, &final_result[i])

cdef void _quadratic_form_cython_double_ptr(REAL_t* tensor, 
    int tensor_shape_0,
    int tensor_shape_2,
    DOUBLE_t * obs_ptr,
    DOUBLE_t * intermediary_result,
    DOUBLE_t * final_result) nogil:
    cdef int i
    for i in range(tensor_shape_0):
        matrix_vector_product_double(&trans, &tensor_shape_2, &tensor_shape_2, &tensor[i * tensor_shape_2 * tensor_shape_2], obs_ptr, &intermediary_result[i * tensor_shape_2])
    for i in range(tensor_shape_0):
        matrix_vector_product_double_additive(&trans, &ONE, &tensor_shape_2, &intermediary_result[i * tensor_shape_2], obs_ptr, &final_result[i])

cdef np.ndarray[DOUBLE_t, ndim=1] _quadratic_form_cython_double(np.ndarray[DOUBLE_t, ndim=3] _tensor,
                                        np.ndarray[DOUBLE_t, ndim=1] _x):
    tensor_shape = _tensor.shape
    cdef int major_output_axis = tensor_shape[0]
    cdef int object_shape = tensor_shape[2]
    cdef np.ndarray[DOUBLE_t, ndim=2] _intermediary_result
    cdef DOUBLE_t * intermediary_result
    cdef np.ndarray[DOUBLE_t, ndim=1] _final_result
    cdef DOUBLE_t * final_result
    cdef DOUBLE_t * tensor
    cdef DOUBLE_t * x
    
    _intermediary_result = np.zeros([tensor_shape[0], tensor_shape[2]], dtype=DOUBLE)
    intermediary_result = <DOUBLE_t *> np.PyArray_DATA(_intermediary_result)
    _final_result = np.zeros(tensor_shape[0], dtype=DOUBLE)
    final_result = <DOUBLE_t *> np.PyArray_DATA(_final_result)

    tensor = <DOUBLE_t *> np.PyArray_DATA(_tensor)
    x = <DOUBLE_t *> np.PyArray_DATA(_x)

    with nogil:
        _quadratic_form_cython_double_ptr(tensor, major_output_axis, object_shape, x, intermediary_result, final_result)

    return _final_result_d

cdef np.ndarray[REAL_t, ndim=1] _quadratic_form_cython_single(np.ndarray[REAL_t, ndim=3] _tensor,
                                        np.ndarray[REAL_t, ndim=1] _x):
    
    tensor_shape = _tensor.shape
    cdef int i, j
    cdef int major_output_axis = tensor_shape[0]
    cdef int object_shape = tensor_shape[2]
    cdef np.ndarray[REAL_t, ndim=2] _intermediary_result
    cdef REAL_t * intermediary_result
    cdef np.ndarray[REAL_t, ndim=1] _final_result
    cdef REAL_t * final_result
    cdef REAL_t * tensor
    cdef REAL_t * x
    
    _intermediary_result = np.zeros([tensor_shape[0], tensor_shape[2]], dtype=REAL)
    intermediary_result = <REAL_t *> np.PyArray_DATA(_intermediary_result)
    _final_result = np.zeros(tensor_shape[0], dtype=REAL)
    final_result = <REAL_t *> np.PyArray_DATA(_final_result)

    tensor = <REAL_t *> np.PyArray_DATA(_tensor)
    x = <REAL_t *> np.PyArray_DATA(_x)

    with nogil:
        _quadratic_form_cython_single_ptr(tensor, major_output_axis, object_shape, x, intermediary_result, final_result)

    return _final_result

def quadratic_form_cython(np.ndarray[cython.floating, ndim=3] _tensor,
                          np.ndarray[cython.floating, ndim=1] _x):      
    if cython.floating == REAL_t:
        return _quadratic_form_cython_single(_tensor, _x)
    else:
        return _quadratic_form_cython_double(_tensor, _x)