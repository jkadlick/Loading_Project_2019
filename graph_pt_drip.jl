## function: 2 idiosyncratic 1 aggregate solver 
#
## this function will take inputs and solve the dynamic rational inattention problem (drip) for an agent 
## tracking 3 processes and taking 2 actions 
#
## NOTE: H can have at most 2 rows 
#
## Inputs: 
# 1. ω cost of nat of information in units of the household's utilty 
# 2. β discount rate 
# 3. A persistence matrix
# 4. Q0 initial Q_0 matrix
# 5. H matrix, states to actions  
# 6. deltas interval over which to loop

# packages 
include("DRIP.jl");
using Plots, LaTeXStrings, LinearAlgebra; pyplot()

function graph_pt_drip(ω,β,A,Q0,H, # primitives of drip
                    deltas      # interval over which to loop
                    )
    
    # initialize dimensions 
    (n,m) = length(size(H)) == 2 ? size(H) : (size(H,1),1);  
    # initialize eye 
    eye = Matrix{Float64}(I,n,n);
    
    sims = length(deltas); # get number of simulations
    iter = 1;              # initialize iteration 
    
    # initialize the output matricies 
    
    kalman = Array{Float64}(undef, 3, sims); # captures Kalman gain of signals on beliefs about q
    weights = Array{Float64}(undef, 3, sims); # captures the weights on each shock in x on beliefs about q 
    betas = Array{Float64}(undef, 2, sims); # captures the population beta 
    
    # run the for loop of the sims 
    
    for delta = deltas 
        
        # generate Q and solve
        Q = Q0 + delta.*[0 0 0; 0 0 0; 0 0 1]; # set Q
        ex1 = solve_drip(ω,β,A,Q,H); # solve drip
        
        # generate kalman gains 
        sqrtΣ = sqrt(ex1.Σ_1); # sqrt prior var covar
        D, U = eigen(sqrtΣ*ex1.Ω*sqrtΣ); # eigenvalue decomp of benefit matrix
        D   = getreal(D); # get real elements for comparison
        D   = diagm(abs.(D).>1e-10).*diagm(D) + (diagm(abs.(D).<= 1e-10))*1e-8; # censor small values 
        kalman[:, iter] = diag(eye - ω*eye./(max.(D,ω))); # add kalman gains to kalman 
        
        # generate weights 
        w_Matrix = ex1.K*transpose(ex1.Y); # temp weight matrix
        weights[:, iter] = w_Matrix[1, :]; # add weights on q to weights 
        
        # generate beta
        Var_p = [ 2 1 ; 1 1+(1+delta)^2 ]; # variance for beta calc 
        y = [ weights[1, iter] + weights[2, iter] ; weights[1, iter] + weights[3, iter].*(1+delta)^2 ] ; # y for beta calc
        betas[:, iter] = inv(Var_p)*y; # calc beta 
        
        # send iteration to next value 
        iter = iter + 1; 

    end 
    
# now generate the plots 
    
    p1 = plot(deltas, transpose(kalman),
            title = L"Kalman Gain to q_{t}",
            xlabel = L"\delta_{i}",
            label = [L"Kalman Gain on \lambda_{1}" L"Kalman Gain on \lambda_{2}" L"Kalman Gain on \lambda_{3}"])
    
    p2 = plot(deltas, transpose(weights),
            title = L"Weight on Belief of q_{t}",
            xlabel = L"\delta_{i}",
            label = [L"q_{t}" L"z_{1t}" L"z_{2t}"])
    
    p3 = plot(deltas, transpose(betas),
            title = L"Population \beta",
            xlabel = L"\delta_{i}",
            label = [L"\beta on \pi_{1t}" L"\beta on \pi_{2t}"]) 
    
    pfin = plot(p1, p2, p3,
            layout = (1,3),
            framestyle = :box)
    
    return(pfin)

    end 

### Auxilary function from Miguel Acosta 

function getreal(M)
    if maximum(abs.(imag.(M))) < 1e-10
        return(real.(M))
    else
        print("Your matrix has complex elements")
    end
end
    
    


