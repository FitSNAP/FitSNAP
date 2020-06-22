# <!----------------BEGIN-HEADER------------------------------------>
# ## FitSNAP3
# A Python Package For Training SNAP Interatomic Potentials for use in the LAMMPS molecular dynamics package
#
# _Copyright (2016) Sandia Corporation. Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software. This software is distributed under the GNU General Public License_
# ##
#
# #### Original author:
#     Aidan P. Thompson, athomps (at) sandia (dot) gov (Sandia National Labs)
#     http://www.cs.sandia.gov/~athomps
#
# #### Key contributors (alphabetical):
#     Mary Alice Cusentino (Sandia National Labs)
#     Nicholas Lubbers (Los Alamos National Lab)
#     Charles Sievers (UC Davis, Sandia National Labs)
#     Adam Stephens (Sandia National Labs)
#     Mitchell Wood (Sandia National Labs)
#
# #### Additional authors (alphabetical):
#     Elizabeth Decolvenaere (D. E. Shaw Research)
#     Stan Moore (Sandia National Labs)
#     Steve Plimpton (Sandia National Labs)
#     Gary Saavedra (Sandia National Labs)
#     Peter Schultz (Sandia National Labs)
#     Laura Swiler (Sandia National Labs)
# <!-----------------END-HEADER------------------------------------->

import numpy as np
import scipy as sp
import pandas as pd
import csv
import sklearn as skl
import sklearn.linear_model

def energy_type_offset(A_energy,AtomTypes):
    """Add atom type columns to energy A matrix"""
    n_types = A_energy.shape[1]
    onehot_atoms = np.stack([np.eye(n_types)[x-1,:].sum(axis=0) for x in AtomTypes])
    onehot_fraction = onehot_atoms/onehot_atoms.sum(axis=1)[:,np.newaxis]
    A_energy = np.concatenate([onehot_fraction[:,:,np.newaxis],A_energy],axis=2)
    return A_energy

def zero_type_offset(A_other):
    """Add atom type columns to a force or virial A matrix; these columns are zeros"""
    n_rows,n_types,n_bispec = A_other.shape
    add_column = np.zeros((n_rows,n_types,1))
    return np.concatenate([add_column,A_other],axis=2)

def makeAbw_b(b_sum,Energy,ref_Energy,NumAtoms,AtomTypes,eweight,tag,offset=False,conversion=None,**kwargs):
    """Make linear system A,b,w for energy terms."""
    a =  b_sum / NumAtoms[:,np.newaxis,np.newaxis]
    b_dft = Energy/NumAtoms
    b_ref = ref_Energy/NumAtoms
    #b=list(zip(b_dft, b_ref))
    if tag=="Validation":
        b = b_ref
    else:
        b=b_dft-b_ref
    if conversion is not None: a *= conversion
    if offset: a = energy_type_offset(a,AtomTypes)
    return a,b,eweight

def makeAbw_db(db_atom,Forces,ref_Forces,NumAtoms,fweight,tag,offset=False,conversion=None,**kwargs):
    """Make linear system A,b,w for force terms"""
    db_atom_flat = np.concatenate(db_atom)
    s1,s2,s3,s4 = db_atom_flat.shape
    a = db_atom_flat.reshape(s1*s2,s3,s4)
#    b = (np.concatenate(Forces)-np.concatenate(ref_Forces)).reshape(s1*s2)
    b_dft = (np.concatenate(Forces)).reshape(s1 * s2)
    b_ref = (np.concatenate(ref_Forces)).reshape(s1 * s2)
#    b=list(zip(b_dft, b_ref))
    if tag=="Validation":
        b = b_ref
    else:
        b=b_dft-b_ref
    #Weights look like broadcasted force components, flattened
    w = np.concatenate([np.full_like(a,b) for a,b in zip(Forces,fweight)]).reshape(s1*s2)
    if conversion is not None: a*=conversion
    if offset: a = zero_type_offset(a)
    return a,b,w

_nktv2p = 1.6021765e6
def makeAbw_vb(vb_sum,Stress,ref_Stress,Volume,vweight,tag,offset=False,conversion=_nktv2p,**kwargs):
    """Make linear system A,b,w for virial terms"""
    s1,s2,s3,s4 = vb_sum.shape

    a = vb_sum.reshape(s1*s2,s3,s4) / np.repeat(Volume,6)[:,np.newaxis,np.newaxis]

    # switch from pressure tensor to flat stress (SNAP Voigt notation)

    flat_stress = Stress[:,[0,1,2,1,0,0],[0,1,2,2,2,1]]
    #b = (flat_stress-ref_Stress).reshape(s1*s2)
    b_dft = (flat_stress).reshape(s1 * s2)
    b_ref = (ref_Stress).reshape(s1 * s2)
#    b=list(zip(b_dft, b_ref))
    if tag=="Validation":
        b = b_ref
    else:
        b=b_dft-b_ref
    w = np.repeat(vweight,6)

    if conversion is not None: a*= conversion
    if offset: a = zero_type_offset(a)

    return a,b,w


def make_Abw(configs, offset, return_subsystems=True,subsystems=(True,True,True),tag=""):
    """
    Make linear system for a set of configurations.

    offset: whether to add columns to A matrices for constants in energy terms.

    subsystems: Tuple, len 3 of booleans.
    1st component: Include energy in system
    2nd component: Include force in system
    3rd component: Include virial in system

    If return_subsystems, then
    The first element is the full system, the lattr 3 are the energy, force, and virial subsystems as specified
    If not return_subsystem, then returns tuple A,b,w.
    """
    fns = tuple(f for f,needed in zip((makeAbw_b,makeAbw_db,makeAbw_vb),subsystems) if needed)
    submatrices = tuple(map(lambda x:x(**configs,offset=offset,tag=tag), fns))
    if len(fns) == 1:
        if return_subsystems:
            return submatrices
        else:
            return submatrices[0]
#    print(submatrices[0])
    A, b, w = map(np.concatenate,zip(*submatrices))
#    np.save('a.out',A)
#    np.save('b.out',b)
#    np.save('w.out',w)
    #A_load=np.load('../FitSNAP2/A.out.npy')
    #A=np.reshape(A_fs2,(np.shape(A_fs2)[0],1,np.shape(A_fs2)[1]))
    #b=np.load('../FitSNAP2/b.out.npy')
    if return_subsystems:
        return ((A, b, w), *submatrices)
    return A,b, w



def add_offset_zero_fit(x_typed):
    """Adds offset values of zero to a linear fit which does not have type offsets"""
    n_types, n_coeff = x_typed.shape
    offsets=np.zeros((n_types,1))
    x_typed = np.concatenate([offsets,x_typed],axis=1)
    return x_typed

def solve_linear_snap(A,b,w, solver, offset=False):
    """Assembles and solves configurations using a solver.
    Solver should be a solver(A,b) -> x where Ax=b, e.g. scipy.linalg.lstsq
    offset: If False, enforce energy offsets terms are zero.
    """

    n_rows, n_types, n_coeff = A.shape
    #print("rows, types, coeffs: ",n_rows,n_types,n_coeff)
    if (np.isinf(A)).any() or (np.isnan(A)).any(): print("Inf or NaN found in Bispectrum Matrix (A)")
    elif (np.isinf(b)).any() or (np.isnan(b)).any(): print("Inf or NaN found in Training Reference Data (b)")
    elif (np.isinf(w)).any() or (np.isnan(w)).any(): print("Inf or NaN found in Weighting Vector (w)")
    else: print("A,b,w look clean, sending to liner solver" )
    Aw, bw = w[:, np.newaxis] * A.reshape(n_rows, n_types * n_coeff), w*b
    x, *solver_info = solver(Aw, bw)
    x_typed = x.reshape(n_types, n_coeff)
    if not offset:
        x_typed = add_offset_zero_fit(x_typed)

    return x_typed, solver_info


##### Functions for getting error metrics of a linear fit. ######
# Note: from this point on we assume that fits have the constant terms (offset) included.

def unflatten(flat_array,shape_like):
    """
    Unshapes numpy array flat_array to look like shape_like. If shape_like is an array,
    just uses reshape. if shape_likee is a list, "unpacks" flat_array to look like the list.
    :param flat_array:
    :param shape_like:
    :return:
    """
    # covers energy and stress
    if isinstance(shape_like,np.ndarray):
        return flat_array.reshape(shape_like.shape)

    # covers forces and other 'ragged' objects
    num_elem = [np.prod(s.shape) for s in shape_like]
    flat_result = np.array_split(flat_array,np.cumsum(num_elem)[:-1])
    return [r.reshape(s.shape) for r,s in zip(flat_result,shape_like)]

def get_residuals(x,configs,subsystems=(True,True,True)):
    names = get_subsys_names(subsystems)
    submatrices = make_Abw(configs,offset=True,return_subsystems=True,subsystems=subsystems)
    if len(submatrices) > 1:
        submatrices = submatrices[1:]
        names = names[1:]

    outdict = {}
    for nm, (A,b,w) in zip(names,submatrices):
        res = get_error_metrics(x,A,b,w,include_residual=True)["residual"]
        outdict["residual_" + nm] = unflatten(res,configs["ref_"+nm])
    return outdict

def get_subsys_names(subsystems):
    sysnames = ("Combined", "Energy", "Forces", "Stress")
    needs_combined = (sum(subsystems) > 1)
    return tuple(sn for sn, needed in zip(sysnames,(needs_combined,*subsystems)) if needed)

def get_error_metrics(x, A, b, w=None,include_residual=False):
    """Compute a variety of error metrics for (wA)x = (wb), return as a dictionary.
    Use w=None for Ax=b.
    A can be typed, shape=(n_rows,n_types,n_coeff) or flat, shape=(n_rows,n_types*n_flat)"""
    A_flat = A.reshape(A.shape[0], -1)
    x_flat = x.reshape(-1)
#    print("SHAPE OF A,x,b",np.shape(A_flat),np.shape(x_flat),np.shape(b[:,0]-b[:,1]))
    true, pred = b, A_flat @ x_flat
    if w is not None:
        true,pred = w*true,w*pred
        nconfig = np.count_nonzero(w)
    else:
        nconfig = len(pred)
    res = true - pred
    mae = np.sum(np.abs(res)/nconfig)
    mean_dev = np.sum(np.abs(true - np.median(true))/nconfig)
    ssr = np.square(res).sum()
    mse = ssr / nconfig
    rmse = np.sqrt(mse)
    rsq = 1 - ssr / np.sum(np.square(true - (true/nconfig).sum()))
    error_record = {
        "ncount": nconfig,
        "mae": mae,
        "rmae": mae / mean_dev,
        "rmse": rmse,
        "rrmse": rmse / np.std(true),
        "ssr": ssr,
        "rsq": rsq
    }
    if include_residual:
        error_record["residual"] = res
    return error_record

def get_full_error_metrics(x, A, b, w=None,include_residual=False):
    """Compute detailed (wA)x = (wb), return as a dictionary.
    Use w=None for Ax=b.
    A can be typed, shape=(n_rows,n_types,n_coeff) or flat, shape=(n_rows,n_types*n_flat)"""
    A_flat = A.reshape(A.shape[0], -1)
    x_flat = x.reshape(-1)
    true, pred = b, A_flat @ x_flat
    res = true - pred
    error_record = {
        "DFT-Ref": true,
        "SNAP": pred,
    }
    return true, pred

def get_ind(v, ind):
    """get v[ind] where ind is a list; uses np arrays if v is a np array """
    return v[np.asarray(ind)] if isinstance(v, np.ndarray) else [v[i] for i in ind]

def get_subgroup(configs, group_name):
    """Extract subconfigurations associated with group group_name"""
    ind = [i for i, gi in enumerate(configs["Group"]) if gi == group_name]
    return {k: get_ind(v, ind) for k, v in configs.items()}

def group_errors(x, configs,bispec_options,subsystems=(True,True,True),tag=""):
    """Get error metrics for a set of configs by group, subsystem, and weighting.
    Returns a pandas DataFrame describing the results."""
    group_set = set(configs["Group"])

    subconfigs = {g: get_subgroup(configs, g) for g in group_set}
    if len(group_set) > 1:
        if "*All" in subconfigs: raise ValueError("groupname '*ALL' not allowed." )
        subconfigs["*ALL"] = configs
    sysnames= get_subsys_names(subsystems)
    all_records = []
    for gname, cf in subconfigs.items():
        systems = make_Abw(offset=True, return_subsystems=True, configs=cf,subsystems=subsystems,tag=tag)
        for gtype, subsys in zip(sysnames,systems):
            for w, wtype in ((subsys[2], "Weighted"), (None, "Unweighted")):
                A,b,_ = subsys
                error_record = get_error_metrics(x, A,b,w)
                error_record["Group"] = gname
                error_record["Subsystem"] = gtype
                error_record["Weighting"] = wtype
                all_records.append(error_record)
                if bispec_options["detailed_errors"] and wtype=="Unweighted" and gtype!="Combined":
                    true, pred = get_full_error_metrics(x, A,b,w)
                    f=open("%s_%s_%s_FullErrors.txt"%(gname,gtype,tag),"w")
                    output = "# Predicted\t\tTrue\t\t\tError = Predicted - True\n"
                    output += '\n'.join('\t'.join(map(str, row+tuple([row[0]-row[1]]))) for row in zip(pred, true))
                    f.write(output)
                    f.close()

    all_records = pd.DataFrame.from_records(all_records)
    all_records = all_records.set_index(["Group", "Weighting", "Subsystem", ]).sort_index()
    return all_records


def sklearn_model_to_fn(modelcls,**model_kwargs):
    """
    :param modelcls: sklearn model class, must support .coef_ attribute.
    :param model_kwargs: kwargs to build the model object
    :return: fitfn(*args,**kwargs): makes a  new model, fits it, and returns coefficients, model, results of fit call.
    """
    # Check that these qwkargs do instantiate a model.
    modelcls(**model_kwargs)

    def fitfn(*args,**kwargs):
        model = modelcls(**model_kwargs)
        fit_result = model.fit(*args,**kwargs)
#        print("Rank: ",model.rank_)
#        print("Intercept: ",model.intercept_)
#        print("Singular Values: ",model.singular_)
#        print("Condition Number",model.singular_[0]/model.singular_[len(model.singular_)-1])
#        np.savetxt('SingularValues.txt',model.singular_)
        return model.coef_, model, fit_result
    return fitfn

def get_solver_fn(solver,normweight=None,normratio=None,**kwargs):
    if normweight is not None: normweight = float(10 ** (float(normweight)))
    if normratio is not None: normratio = float(normratio)
    solver_fndict = {
        "SVD": sklearn_model_to_fn(skl.linear_model.LinearRegression,),
        #sp.linalg.lstsq,
        "LASSO":
            sklearn_model_to_fn(skl.linear_model.Lasso,
                                alpha=normweight,max_iter=1E6,fit_intercept=False),
        "RIDGE":
            sklearn_model_to_fn(skl.linear_model.Ridge,
                                alpha=normweight,max_iter=1E6,fit_intercept=False),
        "ELASTIC":
            sklearn_model_to_fn(skl.linear_model.ElasticNet,
                                alpha=normweight,l1_ratio=normratio,max_iter=1E6,fit_intercept=False)
    }
    return solver_fndict[solver]
