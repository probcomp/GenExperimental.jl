##########################
# Probabilistic programs #
##########################

# Probabilistic programs are a type of Generator that use dictionary-based
# trace types and are constructed by writing Julia functions annotated with
# 'tags'.

## Trace types for probabilistic programs

"""
Trace of a `ProbabilisticProgram`.
"""
mutable struct ProgramTrace <: Trace

    # key is a single element of an address (called `addr_head` in the code)
    subtraces::Dict{Any, Trace}

    # gets reset to 0. after each call to generate!
    score::GenScalar

    # the return value addressed at () (initially is nothing)
    # TODO would adding return type information to the trace constructor be useful for the compiler?
    return_value
    
    # map from subtrace addr_head to a map from subaddres to alias
    aliases::Dict{Any, Dict{Tuple, Any}}

    tape::Tape
end

function ProgramTrace()
    subtraces = Dict{Any, Trace}()
    return_value = nothing
    aliases = Dict{Any, Dict{Any, Tuple}}()
    tape = Tape()
    score = GenScalar(0., tape)
    ProgramTrace(subtraces, score, return_value, aliases, tape)
end

# serialization
import JSON
function JSON.lower(trace::ProgramTrace)
    trace.subtraces
end


"""
Delete an address from the trace
"""
function Base.delete!(t::ProgramTrace, addr::Tuple)
    if addr == ()
        t.return_value = nothing
        t.intervened = false
    end
    addrhead = addr[1]
    if haskey(t.subtraces, addrhead)
        if length(addr) == 1
            # delete the subtrace completely
            delete!(t.subtraces, addrhead)
        else
            subtrace = t.subtraces[addrhead]
            delete!(subtrace, addr[2:end])
        end
    end
end

"""
Check if value exists for a given address.
"""
function Base.haskey(t::ProgramTrace, addr::Tuple)
    if addr == ()
        return t.return_value != nothing
    end
    addrhead = addr[1]
    if haskey(t.subtraces, addrhead)
        subtrace = t.subtraces[addrhead]
        haskey(subtrace, addr[2:end])
    else
        return false
    end
end

"""
Retrieve the value recorded at a given address.

NOTE: use `subtrace` to retrieve a subtrace at a given address.
"""
function value(t::ProgramTrace, addr::Tuple)
    if addr == ()
        return t.return_value
    end
    local subtrace::Trace
    addrhead = addr[1]
    if haskey(t.subtraces, addrhead)
        subtrace = t.subtraces[addrhead]
    else
        error("address not found: $addr")
    end
    value(subtrace, addr[2:end])
end

"Return the subtrace at the given address element"
function subtrace(t::ProgramTrace, addrhead)
    # NOTE: having a subtrace is not a part of the generic Trace interface
    if haskey(t.subtraces, addrhead)
        t.subtraces[addrhead]
    else
        error("address not found: $addr")
    end
end

"
Set the subtrace and value at the given address.

Uses the existing value, if any, or `nothing` if there is none.
"
function set_subtrace!(t::ProgramTrace, addrhead, subtrace::Trace)
    t.subtraces[addrhead] = subtrace
end

function add_alias!(t::ProgramTrace, alias, addr::Tuple)
    # cannot conflict with an existing subtrace address
    if alias == ()
        error("cannot create alias $alias")
    end
    addrhead = addr[1]
    subaddr = addr[2:end]
    if !haskey(t.aliases, addrhead)
        t.aliases[addrhead] = Dict{Any, Tuple}()
    end
    if haskey(t.aliases[addrhead], subaddr)
        existing = t.aliases[addrhead][subaddr]
        error("cannot create alias $alias of address $addr because there is already an alias: $existing")
    end
    t.aliases[addrhead][subaddr] = alias
end

function alias_to_subtrace!(t::ProgramTrace, subtrace::Trace, subaddr::Tuple, alias)
    # TODO what happens if it was already constrained? The alias overwrites the constraint?
    # It depends on the subtrace implementation. We can make the subtrace specification
    # require that an error be thrown if the same address is constrained, intervened, or proposed twice
    # (without first using release!)
    if haskey(t.subtraces, alias)
        # some directives were applied to this alias
        alias_subtrace = t.subtraces[alias]
        if alias_subtrace.mode == constrain
            constrain!(subtrace, subaddr, value(alias_subtrace, ()))
        elseif alias_subtrace.mode == intervene
            intervene!(subtrace, subaddr, value(alias_subtrace, ()))
        elseif alias_subtrace.mode == propose
            propose!(subtrace, subaddr, value_type(alias_subtrace))
        end
    end
end

function subtrace_to_alias!(t::ProgramTrace, subtrace::Trace, subaddr::Tuple, alias)
    val = value(subtrace, subaddr)
    if !haskey(t.subtraces, alias)
        t.subtraces[alias] = AtomicTrace(val)
    else
        t.subtraces[alias].value = Nullable(val)
    end
end

# TODO introduce special syntax for accesing the subtrace (like [] for value but different)

function Base.print(io::IO, trace::ProgramTrace)
    # TODO make a nice table representaiton, and sort the keys
    println(io, "Trace(")
    indent = "  "
    for (addrhead, subtrace) in trace.subtraces
        subtrace_str = isa(subtrace, AtomicTrace) ? "$subtrace" : "$(typeof(subtrace))"
        print(io, "$addrhead\t$subtrace_str\n")
    end
    println(io, ")")
end

function finalize!(t::ProgramTrace)
    if !isempty(t.aliases)
        error("not all aliases were visited")
    end
    backprop(t.score)
    previous_score = concrete(t.score)
    t.tape = Tape()
    t.score = GenScalar(0., t.tape)
    previous_score
end


"""
A generative process represented by a Julia function and constructed with `@program`.

    ProbabilisticProgram <: Generator{ProgramTrace}
"""
struct ProbabilisticProgram <: Generator{ProgramTrace}
    program::Function
end

# TOOD pass args in
empty_trace(::ProbabilisticProgram) = ProgramTrace()

# this symbol is passed as the first argument to every probabilistic program
# invocation, and each @g and @e macro expands into a function call on the trace
const trace_symbol = gensym()
const output_symbol = gensym()
const condition_symbol = gensym()
const execution_mode_symbol = gensym()

"""
Annotate an invocation of a `Generator` within a `@program` with an address.

The address should uniquely identify this point in the dynamic execution of the program.
The program can process `constrain!`, `propose!`, and `intervene!` requests for this address that are present in the trace passed to `generate!`.
"""
macro g(expr, addr)
    # NOTE: the purpose of this macro is the same as the purpose of the @generate! macro:
    # to allow use of function call syntax generator(args...) while tracing
    if expr.head == :call
        generator = expr.args[1]
        generator_args = expr.args[2:end]
        Expr(:call, 
            tagged!,
            esc(trace_symbol),
            esc(output_symbol),
            esc(condition_symbol),
            esc(execution_mode_symbol),
            esc(generator),
            esc(Expr(:tuple, generator_args...)),
            esc(addr))

    else
        error("invalid application of @g, it is only used to address generator invocations")
    end
end

"""
Annotate an arbitrary expression within a `@program` with an address.

The program can process `intervene!` requests for this address that are present in the trace passed to `generate!`.
"""
macro e(expr, addr)
    # TODO?
    Expr(:call, tagged!, esc(trace_symbol), esc(expr), esc(addr))
end

macro alias(alias, addr)
        Expr(:call, 
            add_alias!,
            esc(trace_symbol),
            esc(alias),
            esc(addr))
end


function tagged!(t::ProgramTrace, output::AddressTrie, condition::AddressTrie,
                 is_simulating::Bool, generator::Generator{T}, args::Tuple, addr_head) where {T}
    local subtrace::T
    if haskey(t.subtraces, addr_head)
        # check if the sub-trace is the right type.
        # if it's not the right type, we need to recursively copy over all the directives.
        subtrace = t.subtraces[addr_head]
    else
        subtrace = empty_trace(generator) # TODO pass args in
    end

    # forward any aliases
    if haskey(t.aliases, addr_head)
        for (subaddr, alias) in t.aliases[addr_head]
            alias_to_subtrace!(t, subtrace, subaddr, alias)
        end
    end

    # NOTE: if this was an atomic genreator and it was constrained, then value will be unchanged
    sub_output = output[addr_head]
    sub_condition = condition[addr_head]
    if is_simulating
        (score, val) = simulate!(generator, args, sub_output, sub_condition, subtrace)
    else
        (score, val) = regenerate!(generator, args, sub_output, sub_condition, subtrace)
    end
    t.score += score
    t.subtraces[addr_head] = subtrace
    @assert value(subtrace, ()) == val

    # copy back data from subtrace to aliases
    if haskey(t.aliases, addr_head)
        for (subaddr, alias) in t.aliases[addr_head]
            subtrace_to_alias!(t, subtrace, subaddr, alias)
        end
        delete!(t.aliases, addr_head)
    end

    val
end

# process a generic tagged value 
function tagged!(trace::ProgramTrace, val, addr_head)
    # NOTE: it's not necessary to create an atomic trace here
    # this can be optimized out
    local subtrace::AtomicTrace
    if haskey(trace.subtraces, addr_head)
        subtrace = trace.subtraces[addr_head]
        if subtrace.mode == record
            subtrace.value = val
        elseif subtrace.mode == constrain || subtrace.mode == propose
            error("cannot constrain or propose a non-generator invocation at $addr_head")
        end
        # if the mode is intervene, then don't change the value in the subtrace
    else
        subtrace = AtomicTrace(val)
    end
    trace.subtraces[addr_head] = subtrace
    value(subtrace, ())
end


"""
Define a probabilisic program.

The body of the program is just the body of a regular Julia function, except that the annotation macros [`@g`](@ref) and [`@e`](@ref) can be used.
"""
macro program(args, body)

    # first argument is the trace
    new_args = Any[
        :($trace_symbol::ProgramTrace),
        :($output_symbol::AddressTrie),
        :($condition_symbol::AddressTrie),
        :($execution_mode_symbol::Bool)
    ]

    # remaining arguments are the original arguments
    local name = Nullable{Symbol}()
    if isa(args, Symbol)

        # single untyped argument
        push!(new_args, args)
    elseif args.head == :call
    
        # @program name(args...)
        name = Nullable{Symbol}(args.args[1])
        for arg in args.args[2:end]
            push!(new_args, arg)
        end
    elseif args.head == :(::)

        # single typed argument
        push!(new_args, args)
    elseif args.head == :tuple

        # multiple arguments
        for arg in args.args
            push!(new_args, arg)
        end
    else
        error("invalid @program")
    end
    arg_tuple = Expr(:tuple, new_args...)

    generator_expr = Expr(:call, ProbabilisticProgram, 
                        Expr(:function, esc(arg_tuple), esc(body)))
    if isnull(name)
        generator_expr
    else
        generator_symbol = Base.get(name)
        Expr(Symbol("="), esc(generator_symbol), generator_expr)
    end
end

function regenerate!(p::ProbabilisticProgram, args::Tuple, output::AddressTrie,
                     condition::AddressTrie, trace::ProgramTrace)
    @assert isempty(trace.aliases)
    # first argument to the program is the 'is_simulating::Bool'
    val = p.program(false, trace, output, condition, args...)
    score = finalize!(trace)
    # NOTE: intervention on the return value does not modify the procedure by
    # which the score is computed. semantics: the probabilistic model is still
    # running, it is just disconnected from the output by the intervention.
    # constraints or proposals to random choices within the program will still
    # be scored
    if trace.intervened
        val = trace.return_value
    else
        trace.return_value = val
    end
    (score, val)
end

function simulate!(p::ProbabilisticProgram, args::Tuple, output::AddressTrie,
                     condition::AddressTrie, trace::ProgramTrace)
     # TODO
end


#function generate!(p::ProbabilisticProgram, args::Tuple, trace::ProgramTrace)
    #@assert isempty(trace.aliases)
    #val = p.program(trace, args...)
    #score = finalize!(trace)
    ## NOTE: intervention on the return value does not modify the procedure by
    ## which the score is computed. semantics: the probabilistic model is still
    ## running, it is just disconnected from the output by the intervention.
    ## constraints or proposals to random choices within the program will still
    ## be scored
    #if trace.intervened
        #val = trace.return_value
    #else
        #trace.return_value = val
    #end
    #(score, val)
#end

# TODO make this true for all generators:
(p::ProbabilisticProgram)(args...) = generate!(p, args, ProgramTrace())[2]

#################################################
# Encapsulate a program into an AtomicGenerator #
#################################################

# There are many ways to encapsulate a program into a generator.
# This is just one default encapsulator that produces an atomic generator whose
# value is a dictionary mapping addresses to sub-values

struct EncapsulatedProbabilisticProgram <: AtomicGenerator{Dict}
    program::ProbabilisticProgram
    address_to_type::Dict
end

encapsulate(program, address_to_type::Dict) = EncapsulatedProbabilisticProgram(program, address_to_type)

# TODO pass args in
empty_trace(::EncapsulatedProbabilisticProgram) = AtomicTrace(Dict) 

function generate!(g::EncapsulatedProbabilisticProgram, args::Tuple, trace::AtomicTrace{Dict})
    program_trace = ProgramTrace()
    if trace.mode == intervene
        # TODO implement this
        error("not yet implemented")
    end
    local val::Dict
    if !haskey(trace, ())
        @assert (trace.mode == record) || (trace.mode == propose)
        val = Dict()
        trace.value = Nullable(val)
    else
        val = get(trace.value)
    end
    if trace.mode != record
        for (addr, addr_type) in g.address_to_type
            if trace.mode == constrain
                constrain!(program_trace, addr, val[addr])
            elseif trace.mode == propose
                propose!(program_trace, addr, addr_type)
            end
        end
    end
    (score, _) = generate!(g.program, args, program_trace)

    # copy over data from inner program trace to atomic trace
    if (trace.mode == propose) || (trace.mode == record)
        for addr in keys(g.address_to_type)
            val[addr] = value(program_trace, addr)
        end
    end
    (score, value(trace))
end

#(p::EncapsulatedProbabilisticProgram)(args...) = generate!(p, args, ProgramTrace())[2]


export ProbabilisticProgram
export ProgramTrace
export @program
export @g
export @e
export @alias
export subtrace
export set_subtrace!
export tagged!
export parametrize!
export SubtraceMode
export encapsulate
