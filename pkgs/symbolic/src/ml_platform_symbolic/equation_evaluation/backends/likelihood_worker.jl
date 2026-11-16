# Data-only AST evaluation and native LikelihoodProfiler1.6.1. Never eval source strings.
using LikelihoodProfiler, OptimizationLBFGSB, ForwardDiff, CICOBase, JSON3

plain(x::Nothing) = nothing
plain(x::Symbol) = string(x)
plain(x::AbstractString) = x
plain(x::Bool) = x
plain(x::Real) = isfinite(x) ? Float64(x) : nothing
plain(x::AbstractArray) = [plain(v) for v in x]
plain(x::NamedTuple) = Dict(string(k)=>plain(v) for (k,v) in pairs(x))
plain(x) = Dict(string(k)=>plain(getfield(x,k)) for k in fieldnames(typeof(x)))

function visit(node, x, parameters, variable_ids)
    haskey(node, :const) && return node[:const]
    haskey(node, :var) && return x[findfirst(==(String(node[:var])), variable_ids)]
    haskey(node, :param) && return parameters[String(node[:param])]
    args = [visit(a,x,parameters,variable_ids) for a in node[:args]]
    all(isfinite,args) || return NaN
    op = String(node[:op])
    try
        result = if op == "add"; args[1]+args[2]
        elseif op == "sub"; args[1]-args[2]
        elseif op == "mul"; args[1]*args[2]
        elseif op == "div"; args[2]==0 ? NaN : args[1]/args[2]
        elseif op == "pow_int"; args[1]^Int(args[2])
        elseif op == "pow"; args[1]^args[2]
        elseif op == "rootn"
            n=Int(args[2]); isodd(n) ? sign(args[1])*abs(args[1])^(1/n) : args[1]^(1/n)
        elseif op == "neg"; -args[1]
        elseif op == "abs"; abs(args[1])
        elseif op == "sqrt"; sqrt(args[1])
        elseif op == "log"; log(args[1])
        elseif op == "exp"; exp(args[1])
        elseif op == "sin"; sin(args[1])
        elseif op == "cos"; cos(args[1])
        elseif op == "tan"; tan(args[1])
        elseif op == "asin"; asin(args[1])
        elseif op == "acos"; acos(args[1])
        elseif op == "atan"; atan(args[1])
        elseif op == "sinh"; sinh(args[1])
        elseif op == "cosh"; cosh(args[1])
        elseif op == "tanh"; tanh(args[1])
        elseif op == "asinh"; asinh(args[1])
        elseif op == "acosh"; acosh(args[1])
        elseif op == "atanh"; atanh(args[1])
        else; error("Unsupported AST operator: $op")
        end
        return isfinite(result) ? result : NaN
    catch err
        err isa DomainError || rethrow()
        return NaN
    end
end

function parameters_at(equation, names, theta)
    parameters=Dict{String,Any}(String(k)=>v[:value] for (k,v) in pairs(equation[:parameters]))
    for (name,value) in zip(names,theta); parameters[name]=value; end
    function resolve(name,active)
        name in active && error("Cyclic parameter alias")
        p=equation[:parameters][Symbol(name)]
        if haskey(p,:tied_to) && !isnothing(p[:tied_to])
            parameters[name]=resolve(String(p[:tied_to]),union(active,[name]))
        end
        return parameters[name]
    end
    for name in keys(parameters); resolve(name,Set{String}()); end
    return parameters
end

function curve_data(curve, scales; index=nothing)
    result=Dict("x"=>plain(curve.x),"objective"=>plain(curve.obj),"parameters"=>plain(curve.pars),
         "objective_level"=>plain(curve.obj_level),"retcodes"=>plain(curve.retcodes),
         "native_endpoints"=>plain(curve.endpoints),"stats"=>plain(curve.stats))
    result["parameters"]=[isnothing(row) ? nothing : [isnothing(v) ? nothing : v*scale for (v,scale) in zip(row,scales)] for row in result["parameters"]]
    if !isnothing(index)
        result["x"]=[isnothing(v) ? nothing : v*scales[index] for v in result["x"]]
        result["native_endpoints"]=Dict(k=>isnothing(v) ? nothing : v*scales[index] for (k,v) in result["native_endpoints"])
    end
    result
end

function profile_one(job)
    equation=job[:equation]; names=String.(job[:free_names]); variables=String.(job[:variable_ids])
    X=[Float64.(row) for row in job[:X]]; y=Float64.(job[:y]); sigma=Float64(job[:sigma])
    scales=Float64.(job[:parameter_scales])
    theta0=Float64[equation[:parameters][Symbol(n)][:value] for n in names]./scales
    lower=Float64[isnothing(v) ? -Inf : v for v in job[:lower]]./scales
    upper=Float64[isnothing(v) ? Inf : v for v in job[:upper]]./scales
    function predictions(theta)
        parameters=parameters_at(equation,names,theta.*scales)
        [visit(equation[:ast],x,parameters,variables) for x in X]
    end
    domain=job[:target_domain]
    function nll(theta,p)
        pred=predictions(theta)
        all(isfinite,pred) || return Inf
        if !isnothing(domain)
            !isnothing(domain[1]) && any(v->v<domain[1],pred) && return Inf
            !isnothing(domain[2]) && any(v->v>domain[2],pred) && return Inf
        end
        sum(((yi-prediction_i)/sigma)^2/2+log(sigma*sqrt(2*pi)) for (yi,prediction_i) in zip(y,pred))
    end
    optf=OptimizationFunction(nll,AutoForwardDiff())
    optprob=OptimizationProblem(optf,theta0;lb=lower,ub=upper)
    optimum=solve(optprob,LBFGSB();maxiters=Int(job[:maxiters]))
    theta=collect(optimum.u)
    baseline=nll(theta,nothing)
    result=Dict{String,Any}("version"=>string(pkgversion(LikelihoodProfiler)),
        "julia"=>string(VERSION),"source_predictions"=>plain(predictions(theta0)),
        "optimum"=>plain(theta.*scales),"objective_min"=>plain(baseline),"optimum_retcode"=>string(optimum.retcode),
        "optimizer_coordinates"=>Dict("scales"=>scales,"meaning"=>"internal theta=source parameter/scale; exported optimum/profile/parameters restored to source coordinates; unchanged NLL"),
        "threshold_increment"=>job[:threshold_increment],"profiles"=>Any[],"quantities"=>Any[])
    if !LikelihoodProfiler.SciMLBase.successful_retcode(optimum.retcode) || !isfinite(baseline)
        result["outcome"]="failed"; result["reason"]="Native likelihood optimization did not converge"
        return result
    end
    for (name,limits) in pairs(job[:parameter_ranges])
        index=findfirst(==(String(name)),names)
        if isnothing(index); error("Profile requested for non-free parameter"); end
        target=ParameterTarget([index],[Float64(limits[1])/scales[index]],[Float64(limits[2])/scales[index]],[Symbol(name)])
        prob=ProfileLikelihoodProblem(optprob,theta,target;threshold=Float64(job[:threshold_increment]))
        profiler=OptimizationProfiler(optimizer=LBFGSB(),stepper=FixedStep(initial_step=Float64(job[:parameter_steps][name])/scales[index]))
        solution=solve(prob,profiler;parallel_type=:none,reoptimize_init=false,maxiters=Int(job[:maxiters]))
        curve=merge(Dict("name"=>String(name),"kind"=>"native parameter profile curve; separate CICO endpoint solve"),curve_data(solution[1],scales;index=index))
        if all((theta.>lower).&(theta.<upper))
            tolerance=(Float64(limits[2])-Float64(limits[1]))*1e-10
            endpoints=solve(prob,CICOProfiler(scan_tol=tolerance/scales[index]);parallel_type=:none,reoptimize_init=false)
            curve["endpoint_refinement"]=curve_data(endpoints[1],scales;index=index)
            curve["endpoint_scan_tolerance"]=tolerance
        end
        push!(result["profiles"],curve)
    end
    for quantity in job[:quantities]
        if !all((theta.>lower).&(theta.<upper))
            push!(result["quantities"],Dict("name"=>quantity[:id],"outcome"=>"unavailable",
                "reason"=>"CICO requires an optimum strictly inside parameter bounds; no nudging"))
            continue
        end
        point=Float64.(quantity[:X])
        transform=quantity[:coordinate_conversion]
        g=(theta,p)->visit(quantity[:ast],point,parameters_at(equation,names,theta.*scales),variables)*transform[:scale]+transform[:offset]
        function_opt=OptimizationFunction(g,AutoForwardDiff())
        target=FunctionTarget([function_opt],[Float64(quantity[:range][1])],[Float64(quantity[:range][2])],[Symbol(quantity[:id])])
        prob=ProfileLikelihoodProblem(optprob,theta,target;threshold=Float64(job[:threshold_increment]))
        solution=solve(prob,CICOProfiler(scan_tol=Float64(quantity[:scan_tolerance]));parallel_type=:none,reoptimize_init=false)
        push!(result["quantities"],merge(Dict("name"=>String(quantity[:id]),"unit"=>quantity[:unit],
            "kind"=>"native functional endpoints only; not a sampled smooth curve",
            "endpoint_scan_tolerance"=>quantity[:scan_tolerance],
            "budget"=>"native CICO defaults plus external whole-worker wall deadline"),curve_data(solution[1],scales)))
    end
    result["outcome"]="ok"
    result
end

job=JSON3.read(read(ARGS[1],String))
result=profile_one(job)
open(ARGS[2],"w") do io; JSON3.write(io,result); end
