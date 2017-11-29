#################################################
# Generic flat trace type based on a dictionary #
#################################################

struct FlatDictTrace <: Trace
    dict::Dict{FlatAddress,Any}
end

function FlatDictTrace()
    FlatDictTrace(Dict{FlatAddress,Any}())
end

Base.haskey(t::FlatDictTrace, addr::FlatAddress) = haskey(t.dict, addr)

Base.delete!(t::FlatDictTrace, addr::FlatAddress) = delete!(t.dict, addr)

Base.getindex(t::FlatDictTrace, addr::FlatAddress) = t.dict[addr]

function Base.setindex!(t::FlatDictTrace, value, addr::FlatAddress)
    t.dict[addr] = value
end

export FlatDictTrace
