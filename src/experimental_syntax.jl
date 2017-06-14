type Trace
    fresh::Bool
    vals::Dict
    function Trace()
        new(true, Dict{String,Any}())
    end
end

macro program(args, body)
    err() = error("invalid @program definition")
    println("args: $args")
    println("args.head: $(args.head)")
    println("args.args: $(args.args)")
    println("body: $body")
    new_args = [:(T::Trace)]
    local name::Nullable{Symbol}
    if args.head == :call
        name = args.args[1]
        for arg in args.args[2:end]
            push!(new_args, arg)
        end
    elseif args.head == :(::)
        # single argument
        push!(new_args, args)
        name = Nullable{Symbol}()
    elseif args.head == :tuple
        # multiple arguments
        for arg in args.args
            push!(new_args, arg)
        end
        name = Nullable{Symbol}()
    else
        err()
    end
    arg_tuple = Expr(:tuple, new_args...)
    extended_body = quote
        if !T.fresh
            error("Trace is not fresh!")
        end
        local value = $body
        T.fresh = false
        value
    end
    if isnull(name)
        Expr(:function, arg_tuple, extended_body)
    else
        function_name = get(name)
        println("defining function $function_name")
        eval(Expr(:function,
                Expr(:call, function_name, new_args...),
                extended_body))
    end
end

foo = @program (x::Int, y::Float64) begin
    x + 2
end
    
@program foo2(x::Int, y::Float64) begin
    x + y
end

println(methods(foo2))
# for now, just introduce another macro
# @exposed 

#baz = @program (a::Int) begin
    #trace = Trace()
    #@trace(trace, foo(1, 3.5)) # trace it in a separate trace [[ using the trace macro ]]
    #@trace(foo(1, 3.5)) # traces in the ambient trace
    #foo(trace, 1, 3.5) # trace it in a separate trace { NO }
#end

# TODO add an @curtrace() macro which returns the current trace, for debugging purposes..

t = Trace()
println(t)
z = foo(t, 2, 4.)
t.fresh = true
z = foo(t, 2, 4.)
t.fresh = true
println(foo2(t, 2, 90.))

#t = Trace()
#z = bar(2)
#println(z)

# 1. create an empty trace
# 2. add some constriants, interentions, or proposals?
# 3. run the program on the trace (but it may call sub-routines)
# it is the act of running the progrma in the trace that should do all the checks
# NOTE: this is not just calling the program (which is what is done durin ghte tracing itself in sub-routines)
# 

# there needs to be a special macro for running a program in a trace, that
# takes care of resetting the log-weight and checking the constraints at the end

#@trace(trace, program(x, y)) # +> it resets the log-weight, then runs hte program, then checks keys at the end. returns the ret-val
#@trace(program(x, y)) # => returns a new trace, with a log weight (trace, retval) [[ don't include this, for simplicity ]]

#@in(trace, program(x, y))

# this suggests that 


#different than just program(trace, x, y)

















