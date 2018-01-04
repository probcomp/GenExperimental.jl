############################################
# Generic trace type based on a dictionary #
############################################

"""
A generic `Trace` type for storing values under hierarchical addresses using subtraces.

A concrete address is resolved by removing the first element from the address, accessing the subtrace identified by that first address element, and resolving the remainder of the address relative to the subtrace.

Each top-level address element (the first element of an address) idenfities a subtrace.
Subtraces can be accessed, deleted, and mutated, using a separate set of methods from the core `Trace` interface methods, which are concerned with access and mutation of concrete values at concrete addresses, and not subtraces.

An address cannot be set with `setindex!` until the relevant subtrace has been created, with the exception of a single-element address, in which case an `AtomicTrace` subtrace of the appropriate type will be created.

The empty address `()` is only a concrete address, and does not identify a subtrace.

`S` is the type of score.
"""
mutable struct DictTrace <: Trace

    # key is a single element of an address (called `addr_first` in the code)
    subtraces::Dict{Any, Trace}

    # the value for address ()
    has_empty_address_value::Bool
    empty_address_value
end

function DictTrace()
    subtraces = Dict{Any, Trace}()
    empty_address_value = nothing
    DictTrace(subtraces, false, empty_address_value)
end


# serialization to JSON does not include the score or the empty address value
import JSON
JSON.lower(trace::DictTrace) = trace.subtraces


function Base.haskey(t::DictTrace, addr::Tuple)
    if addr == ()
        return t.has_empty_address_value
    end
    addr_first = addr[1]
    if haskey(t.subtraces, addr_first)
        subtrace = t.subtraces[addr_first]
        return haskey(subtrace, addr[2:end])
    else
        return false
    end
end

function Base.delete!(t::DictTrace, addr::Tuple)
    if addr == ()
        t.empty_address_value = nothing
        t.has_empty_address_value = false
        return
    end
    addr_first = addr[1]
    if haskey(t.subtraces, addr_first)

        # NOTE: Does not remove a subtrace from the trace, even if the address is a single-element address.
        # For a single-element address, it forwards the delete request to the subtrace, with address ()
        subtrace = t.subtraces[addr_first]
        delete!(subtrace, addr[2:end])
    end
end

function Base.getindex(t::DictTrace, addr::Tuple)
    if addr == ()
        if t.has_empty_address_value
            return t.empty_address_value
        else
            error("Address not found: $addr")
        end
    end
    addr_first = addr[1]
    if haskey(t.subtraces, addr_first)
        subtrace = t.subtraces[addr_first]
        return subtrace[addr[2:end]]
    else
        error("Address not found: $addr")
    end
end

function Base.setindex!(t::DictTrace, value, addr::Tuple)
    if addr == ()
        t.empty_address_value = value
        t.has_empty_address_value = true
        return
    end
    addr_first = addr[1]
    if haskey(t.subtraces, addr_first)
        subtrace = t.subtraces[addr_first]
        subtrace[addr[2:end]] = value
    else
        if length(addr) == 1

            # if there is a single element address, and no subtrace has been
            # created, we create an AtomicTrace of the appropriate type
            t.subtraces[addr_first] = AtomicTrace(value)
        else
            error("Address not found: $addr")
        end
    end
end

Base.haskey(t::DictTrace, addr_element) = haskey(t, (addr_element,))

Base.delete!(t::DictTrace, addr_element) = delete!(t, (addr_element,))

Base.getindex(t::DictTrace, addr_element) = t[(addr_element,)]

Base.setindex!(t::DictTrace, value, addr_element) = begin t[(addr_element,)] = value end


"""
Check if a subtrace exists at a given address prefix.
"""
has_subtrace(t::Trace, addr_first) = haskey(t.subtraces, addr_first)


"""
Retrieve a subtrace at a given address prefix.
"""
get_subtrace(t::Trace, addr_first) = t.subtraces[addr_first]


"""
Set the subtrace at a given address prefix.
"""
set_subtrace!(t::Trace, addr_first, subtrace::Trace) = begin t.subtraces[addr_first] = subtrace end


"""
Delete the subtrace at a given address prefix.
"""
delete_subtrace!(t::Trace, addr_first) = begin delete!(t.subtraces, addr_first) end


function Base.print(io::IO, trace::DictTrace)
    # TODO make a nice table representaiton, and sort the keys
    println(io, "Trace(")
    indent = "  "
    for (addr_first, subtrace) in trace.subtraces
        subtrace_str = isa(subtrace, AtomicTrace) ? "$subtrace" : "$(typeof(subtrace))"
        print(io, "$addr_first\t$subtrace_str\n")
    end
    println(io, ")")
end

export DictTrace
export has_subtrace
export get_subtrace
export set_subtrace!
export delete_subtrace!
