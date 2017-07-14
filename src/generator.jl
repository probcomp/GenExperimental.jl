######################
# Generic generators #
######################

# Each Generator{TraceType} should implement:
# score = generate!(generator, trace::TraceType)

abstract type Generator{TraceType} end

# subtraces can be in one of several modes:
@enum SubtraceMode record propose constrain intervene

# some generators overload generator(args) into (generator, args)
# this function allows the syntax generate!(generator(args), trace)
function generate!(generator_and_args::Tuple{Generator,Tuple}, trace)
    generate!(generator_and_args[1], generator_and_args[2], trace)
end


#####################
# Atomic generators #
#####################

# These are generators with traces that are an atomic value (i.e. there is only
# one 'address' in the trace) these correspond to 'probabilistic modules' of
# https://arxiv.org/abs/1612.04759

mutable struct AtomicTrace{T}
    value::Nullable{T}
	mode::SubtraceMode
end

AtomicTrace(value) = AtomicTrace(Nullable(value), record)
has(trace::AtomicTrace) = !isnull(trace.value)
get(trace::AtomicTrace) = Base.get(trace.value)
set!(trace::AtomicTrace{T}, value::T) where {T} = begin trace.value = Nullable{T}(value) end

function unconstrain!(trace::AtomicTrace)
	if trace.mode != constrain
		error("not constrained")
	end
	trace.mode = record
end

function constrain!(trace::AtomicTrace{T}, value::T) where {T}
	trace.mode = constrain
	trace.value = Nullable{T}(value)
end

function propose!(trace::AtomicTrace{T}) where {T}
	trace.mode = propose
end

function intervene!(trace::AtomicTrace{T}, value::T) where {T}
	trace.mode = intervene
	trace.value = Nullable{T}(value)
end

AtomicGenerator{T} = Generator{AtomicTrace{T}}


################################
# Assessable atomic generators #
################################

# These are stochastic computations whose log density can be computed:
# They should implement two methods:
#
# simulate(args...)::T
# logpdf(value::T, args...)::Any

abstract type AssessableAtomicGenerator{T} <: AtomicGenerator{T} end

function generate!(g::AssessableAtomicGenerator{T}, args::Tuple, trace::AtomicTrace{T}) where {T}
	local value::T
	if trace.mode == intervene || trace.mode == constrain
		value = get(trace)
	else
		value = simulate(g, args...)
		set!(trace, value)
	end
	if trace.mode == constrain || trace.mode == propose
		logpdf(g, value, args...)
	else
		0.
	end
end


# exports
export register_generator_shortname
export Generator
export generate!
export AtomicTrace
export AtomicGenerator
export get
export has
export simulate
export logpdf
