using LinearAlgebra, Distributions, StatsBase, Plots, Combinatorics, Tensors,BenchmarkTools, Polynomials


# LITTLE TOOLS FUNCTIONS
function rep(x,k)
    return [copy(x) for i=1:k]
end
function rep_tpl(x,k)
    return [copy(x) for i=1:k]
end
function coord_lin(pos,n)
    d = length(pos)
    return sum([n^(d-i)pos[i] for i=1:d])
end
function bool_to_10(bool)
    if bool return 1
    else return 0
    end
end
import Base.max
function max(v::Array)
    return max(v...)
end
import Base.isless
function isless(a::Complex, b::Complex)
    return isless((real(a), imag(a)), (real(b), imag(b)))
end
function concat(v::Array)
    if length(v) == 1 return v[1]
    elseif length(v) == 2 return [v[1]; v[2]]
    else return [v[1] ; concat(v[2:end])]
    end
end
cbrt_compl = x -> cbrt(abs(x)) * exp(angle(x)*1im/3)
norm3 = x -> cbrt_compl(sum(x.^3))
approx = x -> y -> (abs(x-y) < 1e-6)
egal_a = x -> (y -> (x==y))
coord = i -> v -> v[i]
coordv = a -> v -> [v[i] for i in a]
Id = p -> Diagonal(ones(p))

# TENSOR BASIC OPERATION AND CONSTRUCTORS
function pdt_tens(A::Array, B::Array)
# ex : if A is 2x3x2 and B is 4x5, T is 2x3x2x4x5 and T_{i_1,i_2,i_3,i_4,i_5} = A_{i_1,i_2,i_3} B_{i_4,i_5}
    s1 = size(A)
    m1 = length(s1)
    s2 = size(B)
    m2 = length(s2)
    T = zeros( tuple(s1...,s2...))
    dim1 = []
    for d in 1:m1
        push!(dim1,1:s1[d])
    end
    ix1s = Iterators.product(dim1...)
    dim2 = []
    for d in 1:m2
        push!(dim2, 1:s2[d])
    end
    ix2s = Iterators.product(dim2...)
    for ix1 in ix1s
        for ix2 in ix2s
            ix = tuple(ix1...,ix2...)
            T[ix...] = A[ix1...] * B[ix2...]
        end
    end
    return T
end

function pdt_tens(tpl::Tuple)
# multiple product
    if length(tpl) > 2 return pdt_tens(tpl[1],pdt_tens(tpl[2:end]))
    else pdt_tens(tpl[1],tpl[2])
    end
end

function construct_tensor_vp_orth(n, m)
# Constructs a tensor as a sum of rank one tensors built from orthogonal vectors (thus there can not be more than n elements)
    A = rand(n,n)
    Q, R = qr(A)
    λs = rand(n)
    T = zeros(rep(n,m)...)
    for i=1:n
        T += λs[i] * pdt_tens(tuple(rep(Q[:,i],m)...))
    end
    return (T, λs, Q)
end
function construct_tensor(n, m, nb_comp)
# Constructs a tensor as a sum of nb_comp rank one tensors
    A = rand(n,n)
    Q = randn(n, nb_comp)
    λs = rand(nb_comp)
    T = zeros(rep(n,m)...)
    for i=1:nb_comp
        Q[:,i] /= cbrt(sum(Q[:,i].^3))
        # Q[:,i] /= norm3(Q[:,i])
        T +=  λs[i] * pdt_tens(tuple(rep(Q[:,i],m)...))
    end
    return (T, λs, Q)
end

function contract(T, x)
# Contract one time a tensor on a vector, if T is of degree m, c is of degree m-1
    s = size(T)
    m = length(s)
    c = zeros(Complex, rep(s[1],m-1)...)
    dims = []
    for d in 1:(m-1)
        push!(dims,1:s[d])
    end
    ixs = Iterators.product(dims...)
    for ix in ixs
        c[ix...] = sum([x[i]*T[[ix... ; i]...] for i=1:s[end]])
    end
    return c
end

function contract(T, x, k)
# Contracts k times T on x
    s = size(T)
    m = length(s)
    k>m && @error("contraction impossible over the degree of the tensor !")
    if k==1 return contract(T, x)
    else return contract(contract(T, x), x, k-1)
    end
end


# FUNCTIONS TO COMPUTE THE EIGEN DECOMPOSITION OF A TENSOR
function part_to_ind_prod_matrice(part, k, m)
# Gives the matrix indices of the element of tr(A^(k*(m-1))) associated to the partition part.
# l_part is the number of different possible indices, for simplicity they belong to {1,...l_part}
    l_part = length(part)
    ind_list = rep([0,0], k*(m-1)+1)
    for i in 1:l_part
        for j in part[i]
            ind_list[j][2] = i
            ind_list[j+1][1] = i
        end
    end
    ind_list[k*(m-1)+1][2] = 1
    ind_list[1][1] = l_part
    return ind_list[2:end]
end

function test_seq(seq, l_part, k, m)
# Informs if a sequence of indices must be conserved or not in the computation of the trace of order k
    bool = true
    for i = 1:l_part
        bool &= (sum(ones(k*(m-1))[first.(seq) .== i]) % (m-1))==0
    end
    return bool
end
function ind_Aks(n,m,k)
# Provides the sequence of valid indices of tr(A^(k*(m-1)))
    inds = []
    for l_part in 1:n
        for part in partitions(1:(k*(m-1)), l_part)
            ind = part_to_ind_prod_matrice(part, k, m)
            if test_seq(ind, l_part, k, m)
                push!(inds, ind)
            end
        end
    end
    return(inds)
end
n=4
m=3
k=6
ind_Aks(n,m,k)

function coef_mult_ind(ind, n, m)
# Gives the coefficient associated to a valid sequence of indices of tr(A^(k*(m-1)))
    l_part = max(max.(ind))
    c2 = prod(factorial.(sum.([egal_a(i).(ind) for i in unique(ind)])))
    c1 = prod(factorial.(sum.(coord(1).(ind) .== i for i=1:l_part)))
    ind_is = [coord(1).(ind) .== i for i=1:l_part]
    return (m-1)^(n-1)*c2/c1
end

function perm_ind(ind, l_part, n, m)
# ind contains indices "i" belonging to {1,...,l_part}, the elements of ind_perms
# will present the indices σ(i) where σ is a one-to-one map from {1,...l_part} to
# a subset of {1,...n} of cardinal l_part
    ind_perms = []
    for perm in multiset_permutations(1:n, l_part)
        σ = v -> [coord.(v)[i](perm) for i=1:m]
        push!(ind_perms, σ.(ind))
    end
    return(ind_perms)
end

function ind_tens(ind, n, m, k)
# Gives the tensor multi-indices associated to the valid indices appearing in tr(A^(k*(m-1)))
# ex: if n = 3, m = 3 k = 2 and ind = [[1,2],[1,1],[2,1],[2,2]], i_t will contain (among many other):
# [[1,2,1], [2,1,2]], [[3,2,3], [2,3,2]], [[2,2,3], [3,2,3]] ....
# this time we see that the indices belong to the whole set of indices {1,...,n}
    l_part = max(max.(ind))
    ind_is = [coord(1).(ind) .== i for i=1:l_part]
    dict_i_t = Dict([(i,[]) for i=1:l_part])
    for i = 1:l_part
        ind_i = coord(1).(ind) .== i
        poss = coord(2).(ind[ind_i])
        uposs = unique(poss)
        nbre_posss = sum.([egal_a(i).(poss) for i in uposs])
        for perm in multiset_permutations(uposs, nbre_posss, sum(ind_i))
            push!(dict_i_t[i], [[i ; perm[(j*(m-1)+1):((j+1)*(m-1))]] for j=0:(Int(sum(ind_i)//(m-1))-1)])
        end
    end
    i_t = []
    dim = []
    for i in 1:l_part
        push!(dim, 1:length(dict_i_t[i]))
    end
    ixs = Iterators.product(dim...)
    for ix in ixs
        # ind_tens(ind_perm, k, m)] for ind_perm in perm_ind(ind, l_part, n)]
        ind_t_model = concat([dict_i_t[i][ix[i]] for i=1:l_part])
        push!(i_t, perm_ind(ind_t_model, l_part, n, m)...)
    end
    return i_t
end

function ind_Ts(inds, n, m, k)
# Gives the sequence of valid tensor multi-indices with all the possible permutation of indices.
    ind_ts = []
    for ind in inds
        push!(ind_ts, ind_tens(ind, n, m, k))
    end
    return ind_ts
end

function tce(T, ind_ts, coeffs, k)
# Compute the k-th order trace of a tensor
    s = size(T)
    m = length(s)
    n = s[1]
    tce = 0
    for i in 1:length(coeffs)
        tce += coeffs[i] * (sum([prod([T[ix...] for ix in ind_T]) for ind_T in ind_ts[i]]))
    end
    return(tce)
end

den_schr = (part,k) -> factorial(length(part))/multinomial([sum(part .== i) for i=1:k]...)
Schr = v -> sum([prod([v[d[j]] for j=1:length(d)])/den_schr(d,length(v)) for d in partitions(length(v))])

function save_coeff_traces(n_max, m_max)
# Save in  two dictionaries the coefficients and the sequence of tensor multi_indices necessary
# for the computation of the eigen value of a tensor of size nx...xn (m times)
    coeffs = Dict()
    ind_ts = Dict()
    for n=2:n_max
        for m=2:m_max
            coeffs[n,m] = []
            ind_ts[n,m] = []
            nb_vp = n*(m-1)^(n-1)
            for k = 1:nb_vp
                inds = ind_Aks(n,m,k)
                push!(ind_ts[n,m], ind_Ts(inds, n, m, k))
                push!(coeffs[n,m], [coef_mult_ind(ind, n, m) for ind in inds])
            end
        end
    end
    return (coeffs, ind_ts)
end
(coefficients, ind_ts) = save_coeff_traces(2, 3)

function x1_possibles(T, m, λ)
# Gives all the possible first coordinate of an eigen vector (x1,1) associated to the eigen value λ
    ix_t1 =[]
    ix_t2 =[]
    for k=0:(m-1)
        push!(ix_t1, [[1; ix ] for ix in multiset_permutations([1,2], [k, m-1-k], m-1)])
        push!(ix_t2, [[2; ix ] for ix in multiset_permutations([1,2], [k, m-1-k], m-1)])
    end
    ix = ix_t1[2]
    p1 = Poly(sum.([concat([T[ix...] for ix in ixs]) for ixs in ix_t1])) - Poly([zeros(m-1) ; λ])
    p2 = Poly(sum.([concat([T[ix...] for ix in ixs]) for ixs in ix_t2])) - Poly([λ])
    return (roots(p1), roots(p2))
end
function find_vecp(T, m, λ)
# finds a non-normalized eigen vector x associated to the eigen value λ
    if contract(T,[1,0],m-1) == λ * [1,0] return [1,0]
    else
        (r1, r2) = x1_possibles(T, m, λ)
        return [[x1,1] for x1 in r1[[any(approx(r).(r2)) for r in r1]]]
    end
end
function spectrum(T, m)
    s = size(T)
    all(s .== 2) || @error("the tensor must be of size 2 x...x 2")
    m <= max(collect(keys(coefficients)))[2] || @error("coefficients not computed for tensor of size $(m) use function save_coeff_traces")
    nb_vp = 2*(m-1)
    tt = [-tce(T, ind_ts[2,m][k], coefficients[2,m][k], k)/k for k=1:nb_vp]
    pol_car = Poly([[Schr(tt[1:(nb_vp-k+1)]) for k=1:(nb_vp)] ; 1])
    return sort(roots(pol_car))
end
function spectrify_tensor_2_sided(T)
# Returns the sequence of eigen values and eigen vectors associated to a tensor T
    s = size(T)
    m = length(s)
    λs = spectrum(T,m)
    xs = []
    for λ in unique(λs)
        push!(xs, find_vecp(T, m, λ)...)
    end
    return (λs, xs)
    # return (λs, xs ./ norm3.(xs))
end


n=2
m=3
# T = zeros(2,2,2)
# T[1,1,1] = T[2,2,2] = randn(1)[1]
# # T[2,2,2] = rand(1)[1]
# T[1,2,1] = T[2,1,2]  = randn(1)[1]
#  T[2,2,1] = T[1,1,2] = rand(1)[1]
# T[1,2,2] = T[2,1,1]  =rand(1)[1]
# x = randn(2) /norm(x)
# T = pdt_tens((x,x,x))
# T = ones(2,2,2)
# T[1,1,1] =T[1,2,2] = rand(1)[1]
# T[2,2,2] = T[2,1,1]  =rand(1)[1]
# T[1,2,1] = T[1,1,2]  =  rand(1)[1]
# T[1,2,2] = T[2,1,2]  = T[2,2,1] = rand(1)[1]
# (T, comps, Q) = construct_tensor(n, m, 4)
# (T, comps, Q) = construct_tensor_vp_orth(n, m)
T = randn(2,2,2)
(λs, xs) = spectrify_tensor_2_sided(T)
for i=1:length(λs)
# Visual verification of the spectral decomposition
    println(λs[i])
    println(contract(T,xs[i],2) ./ xs[i].^2)
end

# PLOTS
nb_iter = 1000
R = []
I = []
T = zeros(2,2,2)
for i= 1:nb_iter
    # T[1,1,1] = randn(1)[1]
    # T[2,2,2] =randn(1)[1]
    # T[1,2,1] = T[1,1,2]  = T[2,1,1] = randn(1)[1]
    # T[1,2,2] = T[2,1,2]  = T[2,2,1] = randn(1)[1]
    T = randn(2,2,2) + 1im * randn(2,2,2)
    λs = spectrum(T,m)
    push!(R, real.(λs)...)
    push!(I, imag.(λs)...)
end
scatter(R,I)
supreme = []
for i=1:nb_iter
    x = randn(2)
    x /= norm3(x)
    push!(supreme, abs(contract(T,x,3)[1]))
end
max(supreme)
=
