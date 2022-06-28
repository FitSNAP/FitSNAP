# Declare the class with cdef
ctypedef long long MKL_INT
cdef extern from "scalapack.h":
    void blacs_get_(MKL_INT*, MKL_INT*, MKL_INT*)
    void blacs_pinfo_(MKL_INT*, MKL_INT*)
    void blacs_gridinit_(MKL_INT*, char*, MKL_INT*, MKL_INT*)
    void blacs_gridmap_(MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*)
    void blacs_gridinfo_(MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*)
    void blacs_gridexit_(MKL_INT*)
    void blacs_exit_(MKL_INT*)
    MKL_INT blacs_pnum_(MKL_INT*, MKL_INT*, MKL_INT*)
    void descinit_(MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*)
    void pdpotrf_(char*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*)
    void psgels_(char*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*)
    void pdgels_(char*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, long*, MKL_INT*)
    void pdgesvd_(char*, char*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*, MKL_INT*, double*, MKL_INT*, MKL_INT*)
    MKL_INT numroc_(MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*)
    MKL_INT ilcm_(MKL_INT*, MKL_INT*)
    MKL_INT indxg2p_(MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*, MKL_INT*)
