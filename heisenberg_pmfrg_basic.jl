

## auxiliary functions
using RecursiveArrayTools

setZero!(a::AbstractArray{T,N}) where {T,N} = fill!(a,zero(T))

function setZero!(PartArr::ArrayPartition)
    for arr in PartArr.x
        fill!(arr,0.)
    end
end

"""Recursively sets structure to zero"""
function setZero!(a::T) where T
    for f in fieldnames(T)
        setZero!( getfield(a,f))
    end
    return a
end




## general data structs

struct NumericalParams
    T
    N::Int
    Ngamma::Int

    accuracy
    Lam_min
    Lam_max

    lenIntw::Int
    lenIntw_acc::Int
    np_vec::Array{Int,1}
    np_vec_gamma::Array{Int,1}
end

function NumericalParams(;
    T::Real = 0.5, # Temperature
    N::Integer = 24,
    Ngamma::Integer = N, #Number of gamma frequencies

    accuracy = 1e-6,
    Lam_min = exp(-10.),
    Lam_max = exp(10.),

    lenIntw::Int = N,
    lenIntw_acc::Int = 2*maximum((N,Ngamma,lenIntw)),
    np_vec::Array{Int,1} = collect(0:N-1),
    np_vec_gamma::Array{Int,1} = collect(0:Ngamma-1)
    )

    return NumericalParams(
        T,
        N,
        Ngamma,
        accuracy,
        Lam_min,
        Lam_max,
        lenIntw,
        lenIntw_acc,
        np_vec,
        np_vec_gamma,
    )
end

struct VertexType{T}
    a::Array{T,4}
    b::Array{T,4}
    c::Array{T,4}
end

function VertexType(VDims::Tuple)
    return VertexType(
        zeros(VDims), # Gamma_a
        zeros(VDims), # Gamma_b
        zeros(VDims) # Gamma_c
    )
end
getVDims(Par) = (Par.System.Npairs,Par.NumericalParams.N,Par.NumericalParams.N,Par.NumericalParams.N)
VertexType(Par) = VertexType(getVDims(Par))

struct StateType{T}
    f_int::Array{T,1}
    γ::Array{T,2}
    Γ::VertexType{T}
end

function StateType(NUnique::Int,Ngamma::Int,VDims::Tuple,floattype = Float64::Type)
    return StateType(
        zeros(floattype,NUnique), # f int
        zeros(floattype,NUnique,Ngamma), # gamma
        VertexType(VDims)
    )
end

StateType(Par) = StateType(Par.System.NUnique,Par.NumericalParams.Ngamma,getVDims(Par),_getFloatType(Par)) 

StateType(f_int,γ,Γa,Γb,Γc) = StateType(f_int,γ,VertexType(Γa,Γb,Γc)) 

RecursiveArrayTools.ArrayPartition(x) = ArrayPartition(x.f_int,x.γ,x.Γ.a,x.Γ.b,x.Γ.c)
StateType(Arr::ArrayPartition) = StateType(Arr.x...)

struct BubbleType{T}
    a::Array{T,4} 
    b::Array{T,4}
    c::Array{T,4}

    Ta::Array{T,4} #"a-Tilde" type bubble
    Tb::Array{T,4}
    Tc::Array{T,4}
    Td::Array{T,4}
end

function BubbleType(VDims::Tuple,type = Float64)
    return BubbleType(
        zeros(type,VDims),
        zeros(type,VDims),
        zeros(type,VDims),

        zeros(type,VDims),
        zeros(type,VDims),
        zeros(type,VDims),
        zeros(type,VDims)
    )
end
BubbleType(Par) = BubbleType(getVDims(Par))

struct OptionParams
    usesymmetry::Bool
    MinimalOutput::Bool
end
OptionParams(;usesymmetry::Bool = true,MinimalOutput::Bool = false,kwargs...) = OptionParams(usesymmetry,MinimalOutput)

struct OneLoopParams
    System
    NumericalParams::NumericalParams
    Options::OptionParams
end

Params(System;kwargs...) = OneLoopParams(System,NumericalParams(;kwargs...),OptionParams(;kwargs...))

struct OneLoopWorkspace
    State::StateType  #Stores the current state
    Deriv::StateType  #Stores the derivative
    X::BubbleType #Stores the bubble function X and XTilde
    Par # Params
end

function OneLoopWorkspace(Deriv,State,X,Par)
    setZero!(Deriv)
    setZero!(X)
    return OneLoopWorkspace(
        StateType(State.x...),
        StateType(Deriv.x...),
        X,
        Par
    )
end




## 1 particle Propagators

function iG_(gamma::AbstractArray, x::Integer, Lam::Real, nw::Integer, T::Real)
    w = get_w(nw,T)
    return(w/(w^2+w*gamma_(gamma,x,nw) + Lam^2))
end

function iS_(gamma::AbstractArray, x::Integer, Lam::Real, nw::Integer, T::Real)
    w = get_w(nw,T)
    return(-iG_(gamma,x,Lam,nw,T)^2* 2*Lam/w )
end

function iSKat_(gamma::AbstractArray, Dgamma::AbstractArray, x::Integer, Lam::Real, nw::Integer, T::Real)
    w = get_w(nw,T)
    return(-iG_(gamma,x,Lam,nw,T)^2*(2*Lam/w + gamma_(Dgamma,x,nw)) )
end

function gamma_(gamma::AbstractArray, x::Integer, nw::Integer)
    Ngamma = size(gamma,2)
    s = 1
    if nw<0
        nw = -nw -1
        s = -1
    end
    iw = get_sign_iw(nw,Ngamma)
    return s*gamma[x,iw]
end

function get_sign_iw(nw::Integer,N::Integer)
    s = sign(nw)
    nw_bounds = min(abs(nw), N-1)
    return s*nw_bounds+1
end

"""given a Matsubara (!) integer, return the corresponding Matsubara frequency"""
function get_w(nw,T)
    return (2*nw+1)*pi*T
end



## Vertices

function V_(Vertex::AbstractArray, Rij::Integer, ns::Integer,nt::Integer,nu::Integer,Rji::Integer,N::Integer)
    # @assert (ns+nt+nu) %2 != 0 "$ns + $nt +  $nu = $(ns+nt+nu)"
    ns,nt,nu,swapsites = convertFreqArgs(ns,nt,nu,N)
    Rij = ifelse(swapsites,Rji,Rij)
    return Vertex[Rij,ns+1,nt+1,nu+1]
end

function convertFreqArgs(ns,nt,nu,Nw)
    # assert ns+nt+nu odd, uses pos freq <-> neg freq symmetry, cuts off freqs at Nw
    swapsites = nt*nu <0
    ns,nt,nu = abs.((ns,nt,nu))
    ns = min( ns, Nw - 1 - (ns+Nw-1)%2)
    nt = min( nt, Nw - 1 - (nt+Nw-1)%2)
    nu = min( nu, Nw - 1 - (nu+Nw-1)%2)

    return ns,nt,nu,swapsites
end




## Define flow eq
# functions are defined below

function getDeriv!(Deriv,State,setup,Lam)
    (X,Par) = setup #use pre-allocated X and XTilde to reduce garbage collector time
    Workspace = OneLoopWorkspace(Deriv,State,X,Par)

    getDFint!(Workspace,Lam)

    get_Self_Energy!(Workspace,Lam)

    getXBubble!(Workspace,Lam)

    symmetrizeBubble!(Workspace.X,Par)

    addToVertexFromBubble!(Workspace.Deriv.Γ,Workspace.X)
    symmetrizeVertex!(Workspace.Deriv.Γ,Par)
    return
end

function getDFint!(Workspace,Lam::Real)
    (;State,Deriv,Par) = Workspace 
    (;T,lenIntw_acc) = Par.NumericalParams 
    NUnique = Par.System.NUnique 
	
	γ(x,nw) = gamma_(State.γ,x,nw)
	iG(x,nw) = iG_(State.γ,x,Lam,nw,T)
	iS(x,nw) = iS_(State.γ,x,Lam,nw,T)

	Theta(Lam,w) = w^2/(w^2+Lam^2)
	
	for x in 1:NUnique
		sumres = 0.
		for nw in -lenIntw_acc:lenIntw_acc-1
			w = get_w(nw,T)
			sumres += iS(x,nw)/iG(x,nw)*Theta(Lam,w) *γ(x,nw)/w
		end
		Deriv.f_int[x] = -3/2*T*sumres
	end
end

# Get Self energy
function get_Self_Energy!(Workspace,Lam) 
	Par = Workspace.Par
	@inline iS(x,nw) = iS_(Workspace.State.γ,x,Lam,nw,Par.NumericalParams.T)/2
	compute1PartBubble!(Workspace.Deriv.γ,Workspace.State.Γ,iS,Par)
end

function compute1PartBubble!(Dgamma::AbstractArray,Γ::VertexType,Prop,Par)
    invpairs = Par.System.invpairs

	setZero!(Dgamma)

	@inline Γa_(Rij,s,t,u) = V_(Γ.a,Rij,t,u,s,invpairs[Rij],Par.NumericalParams.N) # Tilde-type can be obtained by permutation of vertices
	@inline Γb_(Rij,s,t,u) = V_(Γ.b,Rij,t,u,s,invpairs[Rij],Par.NumericalParams.N) # cTilde corresponds to b type vertex!

    addTo1PartBubble!(Dgamma,Γa_,Γb_,Prop,Par)
end

function addTo1PartBubble!(Dgamma::AbstractArray,Γa_::Function,Γb_::Function,Prop,Par)

    (;T,Ngamma,lenIntw_acc,np_vec_gamma) = Par.NumericalParams
    (;siteSum,Nsum,OnsitePairs) = Par.System

	Threads.@threads for iw1 in 1:Ngamma
		nw1 = np_vec_gamma[iw1]
    	for (x,Rx) in enumerate(OnsitePairs)
			for nw in -lenIntw_acc: lenIntw_acc-1
				jsum = 0.
				wpw1 = nw1+nw+1 #w + w1: Adding two fermionic Matsubara frequencies gives a +1 for the bosonic index
				wmw1 = nw-nw1
				for k_spl in 1:Nsum[Rx]
					(;m,ki,xk) = siteSum[k_spl,Rx]
					jsum += (Γa_(ki,wpw1,0,wmw1)+2*Γb_(ki,wpw1,0,wmw1))*Prop(xk,nw)*m
				end
				Dgamma[x,iw1] += -T *jsum #For the self-energy derivative, the factor of 1/2 must be in the propagator
			end
		end
	end
    return Dgamma
end




# Get X Bubble
function mixedFrequencies(ns,nt,nu,nwpr)
	nw1=Int((ns+nt+nu-1)/2)
    nw2=Int((ns-nt-nu-1)/2)
    nw3=Int((-ns+nt-nu-1)/2)
    nw4=Int((-ns-nt+nu-1)/2)
	wpw1 = nwpr + nw1 + 1
    wpw2 = nwpr + nw2 + 1
    wmw3 = nwpr - nw3
    wmw4 = nwpr - nw4
	# @assert (ns + wmw3 +wmw4)%2 != 0 "error in freq"
	return wpw1,wpw2,wmw3,wmw4
end

function getXBubble!(Workspace,Lam)
	Par = Workspace.Par
    (;T,N,lenIntw,np_vec) = Par.NumericalParams
    (;NUnique) = Par.System
	 
	iG(x,nw) = iG_(Workspace.State.γ,x,Lam,nw,T)
	iSKat(x,nw) = iSKat_(Workspace.State.γ,Workspace.Deriv.γ,x,Lam,nw,T)

	function getKataninProp!(BubbleProp,nw1,nw2)
		for i in 1:Par.System.NUnique, j in 1:Par.System.NUnique
			BubbleProp[i,j] = iSKat(i,nw1) *iG(j,nw2)* T
		end
		return SMatrix{NUnique,NUnique}(BubbleProp)
	end
	for is in 1:N,it in 1:N
        BubbleProp = zeros(NUnique,NUnique)
        ns = np_vec[is]
        nt = np_vec[it]
        for nw in -lenIntw:lenIntw-1 # Matsubara sum
            sprop = getKataninProp!(BubbleProp,nw,nw+ns) 
            for iu in 1:N
                nu = np_vec[iu]
                if (ns+nt+nu)%2 == 0	# skip unphysical bosonic frequency combinations
                    continue
                end
                addXTilde!(Workspace,is,it,iu,nw,sprop) # add to XTilde-type bubble functions
                if(!Par.Options.usesymmetry || nu<=nt)
                    addX!(Workspace,is,it,iu,nw,sprop)# add to X-type bubble functions
                end
            end
        end
	end
end

function addX!(Workspace, is::Integer, it::Integer, iu::Integer, nwpr::Integer, Props)
	(;State,X,Par) = Workspace 
	(;N,np_vec) = Par.NumericalParams
	(;Npairs,Nsum,siteSum,invpairs) = Par.System

    Va_(Rij,s,t,u) = V_(State.Γ.a,Rij,s,t,u,invpairs[Rij],N)
	Vb_(Rij,s,t,u) = V_(State.Γ.b,Rij,s,t,u,invpairs[Rij],N)
	Vc_(Rij,s,t,u) = V_(State.Γ.c,Rij,s,t,u,invpairs[Rij],N)
	ns = np_vec[is]
	nt = np_vec[it]
	nu = np_vec[iu]
	wpw1,wpw2,wmw3,wmw4 = mixedFrequencies(ns,nt,nu,nwpr)

	# get fields of siteSum struct as Matrices for better use of LoopVectorization
	S_ki = siteSum.ki
	S_kj = siteSum.kj
	S_xk = siteSum.xk
	S_m = siteSum.m

	for Rij in 1:Npairs
		#loop over all left hand side inequivalent pairs Rij
		Xa_sum = 0. #Perform summation on this temp variable before writing to State array as Base.setindex! proved to be a bottleneck!
		Xb_sum = 0.
		Xc_sum = 0.
		for k_spl in 1:Nsum[Rij]
			#loop over all Nsum summation elements defined in geometry. This inner loop is responsible for most of the computational effort! 
			ki,kj,m,xk = S_ki[k_spl,Rij],S_kj[k_spl,Rij],S_m[k_spl,Rij],S_xk[k_spl,Rij]
			Ptm = Props[xk,xk]*m

            Va12 = Va_(ki, ns, wpw1, wpw2)
            Vb12 = Vb_(ki, ns, wpw1, wpw2)
            Vc12 = Vc_(ki, ns, wpw1, wpw2)

            Va34 = Va_(kj, ns, wmw3, wmw4)
            Vb34 = Vb_(kj, ns, wmw3, wmw4)
            Vc34 = Vc_(kj, ns, wmw3, wmw4)

            Vc21 = Vc_(ki, ns, wpw2, wpw1)
            Vc43 = Vc_(kj, ns, wmw4, wmw3)

			Xa_sum += (
				+Va12 * Va34 
				+Vb12 * Vb34 * 2
			)* Ptm

			Xb_sum += (
				+Va12 * Vb34
				+Vb12 * Va34
				+Vb12 * Vb34
			)* Ptm
			
			Xc_sum += (
				+Vc12 * Vc34
				+Vc21 * Vc43
			)* Ptm
		end
		X.a[Rij,is,it,iu] += Xa_sum
		X.b[Rij,is,it,iu] += Xb_sum
		X.c[Rij,is,it,iu] += Xc_sum
    end
    return
end

function addXTilde!(Workspace, is::Integer, it::Integer, iu::Integer, nwpr::Integer, Props)
	(;State,X,Par) = Workspace 
	(;N,np_vec) = Par.NumericalParams
	(;Npairs,invpairs,PairTypes,OnsitePairs) = Par.System

	Va_(Rij,s,t,u) = V_(State.Γ.a,Rij,s,t,u,invpairs[Rij],N)
	Vb_(Rij,s,t,u) = V_(State.Γ.b,Rij,s,t,u,invpairs[Rij],N)
	Vc_(Rij,s,t,u) = V_(State.Γ.c,Rij,s,t,u,invpairs[Rij],N)
	ns = np_vec[is]
	nt = np_vec[it]
	nu = np_vec[iu]
	wpw1,wpw2,wmw3,wmw4 = mixedFrequencies(ns,nt,nu,nwpr)

	#Xtilde only defined for nonlocal pairs Rij != Rii
	for Rij in 1:Npairs
		Rij in OnsitePairs && continue
		#loop over all left hand side inequivalent pairs Rij
		Rji = invpairs[Rij] # store pair corresponding to Rji (easiest case: Rji = Rij)
		(;xi,xj) = PairTypes[Rij]

		#These values are used several times so they are saved locally
		Va12 = Va_(Rji, wpw1, ns, wpw2)
		Va21 = Va_(Rij, wpw2, ns, wpw1)
		Va34 = Va_(Rji, wmw3, ns, wmw4)
		Va43 = Va_(Rij, wmw4, ns, wmw3)

		Vb12 = Vb_(Rji, wpw1, ns, wpw2)
		Vb21 = Vb_(Rij, wpw2, ns, wpw1)
		Vb34 = Vb_(Rji, wmw3, ns, wmw4)
		Vb43 = Vb_(Rij, wmw4, ns, wmw3)

		Vc12 = Vc_(Rji, wpw1, ns, wpw2)
		Vc21 = Vc_(Rij, wpw2, ns, wpw1)
		Vc34 = Vc_(Rji, wmw3, ns, wmw4)
		Vc43 = Vc_(Rij, wmw4, ns, wmw3)

	    X.Ta[Rij,is,it,iu] += (
			(+Va21 * Va43
			+2*Vc21 * Vc43) * Props[xi,xj]
			+(Va12 * Va34
			+2*Vc12 * Vc34)* Props[xj,xi]
		)
		
	    X.Tb[Rij,is,it,iu] += (
			(+Va21 * Vc43
			+Vc21 * Vc43
			+Vc21 * Va43) * Props[xi,xj]

			+(Va12 * Vc34
			+Vc12 * Vc34
			+Vc12 * Va34)* Props[xj,xi]
		)
		Vb12 = Vb_(Rji, wpw1, wpw2, ns)
		Vb21 = Vb_(Rij, wpw2, wpw1, ns)
		Vb34 = Vb_(Rji, wmw3, wmw4, ns)
		Vb43 = Vb_(Rij, wmw4, wmw3, ns)

		Vc12 = Vc_(Rji, wpw1, wpw2, ns)
		Vc21 = Vc_(Rij, wpw2, wpw1, ns)
		Vc34 = Vc_(Rji, wmw3, wmw4, ns)
		Vc43 = Vc_(Rij, wmw4, wmw3, ns)


	    X.Tc[Rij,is,it,iu] += (
			(+Vb21 * Vb43
			+Vc21 * Vc43
			) * Props[xi,xj]
			+(Vb12 * Vb34
			+Vc12 * Vc34
	    	)* Props[xj,xi]
		)
    end
end

function symmetrizeBubble!(X::BubbleType,Par)
    N = Par.NumericalParams.N
    (;Npairs,OnsitePairs) = Par.System
    usesymmetry = Par.Options.usesymmetry
    # use the u <--> t symmetry
    if(usesymmetry)
        for it in 1:N
            for iu in it+1:N, is in 1:N, Rij in 1:Npairs
                X.a[Rij,is,it,iu] = -X.a[Rij,is,iu,it]
                X.b[Rij,is,it,iu] = -X.b[Rij,is,iu,it]
                X.c[Rij,is,it,iu] = (
                + X.a[Rij,is,it,iu]+
                - X.b[Rij,is,it,iu]+
                + X.c[Rij,is,iu,it])
            end
        end
    end
    #local definitions of X.Tilde vertices
    for iu in 1:N
		for it in 1:N, is in 1:N, R in OnsitePairs
			X.Ta[R,is,it,iu] = X.a[R,is,it,iu]
			X.Tb[R,is,it,iu] = X.b[R,is,it,iu]
			X.Tc[R,is,it,iu] = X.c[R,is,it,iu]
			X.Td[R,is,it,iu] = -X.c[R,is,iu,it]
		end
    end
    X.Td .= X.Ta .- X.Tb .- X.Tc
end

function addToVertexFromBubble!(Γ::VertexType,X::BubbleType)
    for iu in axes(Γ.a,4)
        for it in axes(Γ.a,3), is in axes(Γ.a,2), Rij in axes(Γ.a,1)
            Γ.a[Rij,is,it,iu] += X.a[Rij,is,it,iu] - X.Ta[Rij,it,is,iu] + X.Ta[Rij,iu,is,it]
            Γ.b[Rij,is,it,iu] += X.b[Rij,is,it,iu] - X.Tc[Rij,it,is,iu] + X.Tc[Rij,iu,is,it]
            Γ.c[Rij,is,it,iu] += X.c[Rij,is,it,iu] - X.Tb[Rij,it,is,iu] + X.Td[Rij,iu,is,it]
        end
    end
    return Γ
end

function symmetrizeVertex!(Γ::VertexType,Par)
	N = Par.NumericalParams.N
	for iu in 1:N
		for it in 1:N, is in 1:N, R in Par.System.OnsitePairs
			Γ.c[R,is,it,iu] = -Γ.b[R,it,is,iu]
		end
	end
end




## Solve flow equations

function AllocateSetup(Par::OneLoopParams)
    println("One Loop: T= ",Par.NumericalParams.T)
    ##Allocate Memory:
    X = BubbleType(Par)
    return (X,Par)
end

function InitializeState(Par)
    (;N,Ngamma) = Par.NumericalParams
    VDims = getVDims(Par)
    (;couplings,NUnique) = Par.System

    floattype = _getFloatType(Par)
    
    State = ArrayPartition( #Allocate Memory:
        zeros(floattype,NUnique), # f_int 
        zeros(floattype,NUnique,Ngamma), # gamma
        zeros(floattype,VDims), #Va
        zeros(floattype,VDims), #Vb
        zeros(floattype,VDims) #Vc
    )

    Γc = State.x[5]
    setToBareVertex!(Γc,couplings)
    return State
end
_getFloatType(Par) = typeof(Par.NumericalParams.T)

SolveFRG(Par;kwargs...) = launchPMFRG!(InitializeState(Par),AllocateSetup(Par),getDeriv!; kwargs...)

function launchPMFRG!(State,setup,Deriv!::Function;
    method = DP5()
    )

    Par = setup[end]
    (;Lam_max,Lam_min,accuracy) = Par.NumericalParams

    t0 = Lam_to_t(Lam_max)
    tend = get_t_min(Lam_min)
    Deriv_subst! = generateSubstituteDeriv(Deriv!)
    problem = ODEProblem(Deriv_subst!,State,(t0,tend),setup)
    sol = solve(problem,method,reltol = accuracy,abstol = accuracy,save_everystep = true,dt=Lam_to_t(0.2*Lam_max))
    return sol
end

t_to_Lam(t) = exp(t)
Lam_to_t(t) = log(t)

function get_t_min(Lam)
    Lam < exp(-30) && @warn "Lam_min too small! Set to exp(-30) instead."
    max(Lam_to_t(Lam),-30.)
end

function generateSubstituteDeriv(getDeriv!::Function)
    
    function DerivSubs!(Deriv,State,par,t)
        Lam = t_to_Lam(t)
        a = getDeriv!(Deriv,State,par,Lam)
        Deriv .*= Lam
        a
    end

end

function setToBareVertex!(Γc::AbstractArray{T,4},couplings::AbstractVector) where T
    for Rj in axes(Γc,1)
        Γc[Rj,:,:,:] .= -couplings[Rj]
    end
    return Γc
end




## Observables

struct Observables
    Chi
    gamma
end

getChi(State::ArrayPartition, Lam::Real,Par) = getChi(State.x[2],State.x[5], Lam,Par)

function getChi(gamma::AbstractArray,Γc::AbstractArray, Lam::Real,Par)
	(;T,N,lenIntw_acc) = Par.NumericalParams
	(;Npairs,invpairs,PairTypes,OnsitePairs) = Par.System

	iG(x,w) = iG_(gamma,x, Lam,w,T)
	Vc_(Rij,s,t,u) = V_(Γc,Rij,s,t,u,invpairs[Rij],N)

	Chi = zeros(_getFloatType(Par),Npairs)

	for Rij in 1:Npairs
		(;xi,xj) = PairTypes[Rij]
		for nK in -lenIntw_acc:lenIntw_acc-1
			if Rij in OnsitePairs
				Chi[Rij,1] += T * iG(xi,nK) ^2
			end
			for nK2 in -lenIntw_acc:lenIntw_acc-1
				npwpw2 = nK+nK2+1
				w2mw = nK2-nK
				#use that Vc_0 is calculated from Vb
				GGGG = iG(xi,nK)^2 * iG(xj,nK2)^2
				Chi[Rij] += T^2 * GGGG *Vc_(Rij,0,npwpw2,w2mw)
			end
        end
    end
	return(Chi)
end




## Execution Dimer Susc

using SpinFRGLattices,OrdinaryDiffEq,DiffEqCallbacks,RecursiveArrayTools,StructArrays
using SpinFRGLattices.StaticArrays
using CairoMakie

System = getPolymer(2) # create a structure that contains all information about the geometry of the problem. 

Par = Params( #create a group of all parameters to pass them to the FRG Solver
    System, # geometry, this is always required
    T=0.5, # Temperature for the simulation.
    Lam_max = 100.,
    Lam_min = .01,
    N = 8, # Number of positive Matsubara frequencies for the four-point vertex.
    accuracy = 1e-5,
)

@time sol = SolveFRG(Par,method = DP5());


## Evaluation Dimer Susc

tr = LinRange(3,-2,20)
chiR = [getChi(sol(t),exp(t),Par) for t in tr] # getting susceptibility
fig = Figure()
ax = Axis(fig[1,1], ylabel = L"χ",xlabel = L"Λ")

scatterlines!(ax,exp.(tr),getindex.(chiR,1))
scatterlines!(ax,exp.(tr),getindex.(chiR,2))
fig


## T sweep Dimer

using SpinFRGLattices,OrdinaryDiffEq,DiffEqCallbacks,RecursiveArrayTools,StructArrays
using SpinFRGLattices.StaticArrays
using CairoMakie

System = getPolymer(2) # create a structure that contains all information about the geometry of the problem. 

Trange = 0.1:0.1:2.5
f = []

for T in Trange
    Par = Params( #create a group of all parameters to pass them to the FRG Solver
        System, # geometry, this is always required
        T = T,
        Lam_max = 100000.,
        Lam_min = .001,
        N = 8, # Number of positive Matsubara frequencies for the four-point vertex.
        accuracy = 1e-5,
    )

    @time sol = SolveFRG(Par,method = DP5());
    append!(f,sol[end].x[1][1])
end


## discrete derivative
function discD(arrayX,arrayY)
    res = []
    for i in 2:length(arrayX)-1
        deriv = (arrayY[i+1]-arrayY[i-1])/(arrayX[i+1]-arrayX[i-1])
        append!(res,deriv)
    end

    return Float64.(res)
end

function av(array::Array)
    return [(array[i+1]+array[i-1])/2 for i in 2:length(array)-1]
end

## computation e and c
Tvals = [i for i in 0.1:0.1:2.5]
e = av(f) - av(Tvals).*discD(Tvals,f)
c = discD(Tvals[2:end-1],e)


## plot stuff
fig = Figure()
axf = Axis(fig[1,1],aspect = 7/1,ylabel = L"F/N", title = "from 'reproduce Heisenberg pmfrg'")
axe = Axis(fig[2,1],aspect = 7/1,ylabel = L"E/N")
axc = Axis(fig[3,1],aspect = 7/1,xlabel = L"T/J",ylabel = L"C/N")

scatterlines!(axf,Tvals,Float64.(f))
scatterlines!(axe,Tvals[2:end-1],e)
scatterlines!(axc,Tvals[3:end-2],c)

rowgap!(fig.layout,0.1)
fig














## Execution Square lattice

using SpinFRGLattices,OrdinaryDiffEq,DiffEqCallbacks,RecursiveArrayTools,StructArrays
using SpinFRGLattices.StaticArrays
using SpinFRGLattices.SquareLattice

NLen = 5 # Number of nearest neighbor bonds up to which correlations are treated in the lattice. For NLen = 5, all correlations C_{ij} are zero if sites i and j are separated by more than 5 nearest neighbor bonds.
J1 = 1
J2 = 0.1
couplings = [J1,J2] # Construct a vector of couplings: nearest neighbor coupling is J1 (J2) and further couplings to zero. For finite further couplings simply provide a longer array, i.e [J1,J2,J3,...]

System = getSquareLattice(NLen,couplings) # create a structure that contains all information about the geometry of the problem. 

Par = Params( #create a group of all parameters to pass them to the FRG Solver
    System, # geometry, this is always required
    T=0.5, # Temperature for the simulation.
    Lam_max = exp(10.),
    Lam_min = exp(-10.),
    N = 8, # Number of positive Matsubara frequencies for the four-point vertex.
    accuracy = 1e-3,
)

@time sol = SolveFRG(Par,method = DP5());



## Evaluation Square lattice

@time begin
    
    using PMFRGEvaluation
    using CairoMakie #for plotting. You can use whatever plotting package you like of course

    System = SquareLattice.getSquareLattice(NLen)
    Lattice = LatticeInfo(System,SquareLattice)
    let 
        chi_R = getChi(sol[end],Par.NumericalParams.Lam_min,Par)
        
        chi = getFourier(chi_R,Lattice)
        
        k = LinRange(-2pi,2pi,300)
        
        chik = [chi(x,y) for x in k, y in k]
        
        fig, ax, hm = heatmap(k,k,chik,axis = (;aspect = 1))
        Colorbar(fig[1,2],hm)
        fig
    end

end











## CHECK SYMMETRIES

gamma,Va,Vb,Vc = sol[end].x[2],sol[end].x[3],sol[end].x[4],sol[end].x[5]

function iszero(M::Array)
    for i in round.(M, digits = 15)
        if i != 0.0
            prinln("IS NOT ZERO")
        else
        end
    end

    return res
end

##

sol[5].x[2]

##

sol.t

##


## Γc_ii(s,t,u) = - Γb_ii(t,s,u)

for i in 1:Par.System.Npairs
    println("
Pair: ", Par.System.PairList[i])
    for j in 1:Par.NumericalParams.N
        println(iszero(Vc[i,:,:,j]+transpose(Vb[i,:,:,j])))
    end
end

## Γc_ij(s,u,t) = (- Γa_ij + Γb_ij + Γc_ij)(s,t,u)

for Rij in 1:Par.System.Npairs
    println("
Pair: ", Par.System.PairList[Rij])
    for s in 1:Par.NumericalParams.N
        println(iszero(transpose(Vc[Rij,s,:,:])+Va[Rij,s,:,:]-Vb[Rij,s,:,:]-Vc[Rij,s,:,:]))
    end
end

## Γa_ij(s,t,u) = Γa_ij(t,s,u) = Γa_ij(s,t,u) = 

iszero(Va[1,:,:,1] + transpose(Va[1,:,:,1]))
iszero(Va[1,:,1,:] + transpose(Va[1,:,1,:]))
iszero(Va[1,1,:,:] + transpose(Va[1,1,:,:]))

##

Vb[1,:,:,1] + transpose(Vb[1,:,:,1])

## (1,2) <-> (3,4)

println(iszero(round.(Va[1,:,1,:] - transpose(Va[1,:,1,:]), digits = 15)))
