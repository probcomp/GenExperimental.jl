import DataStructures.OrderedDict

"""

An address ("foo", "bar", 2) has two meanings:

    1. It is a concrete terminal address

    2. It is the prefix of all addresses of the form ("foo, "bar", 2, ...)
       It is also the prefix of the adddress ("foo", "bar", 2), i.e. the prefix
       of itself interpreted as a concrete address
"""

mutable struct AddressTrie
    subtries::OrderedDict{Any, AddressTrie}
    has_ea::Bool
end

AddressTrie() = AddressTrie(OrderedDict{Any, AddressTrie}(), false)


"""
Test if the given address is included.
"""
function Base.in(addr::Tuple, t::AddressTrie)
    if addr == ()
        return t.has_ea
    end
    addr_first = addr[1]
    addr_rest = addr[2:end]
    if haskey(t.subtries, addr_first)
        return addr_rest in t.subtries[addr_first]
    end
    return false
end

"""
Shorthand `in` for a single element address
"""
Base.in(addr, t::AddressTrie) = (addr,) in t

"""
Add the given address.
"""
function Base.push!(t::AddressTrie, addr::Tuple)
    if addr == ()
        t.has_ea = true
        return
    end
    addr_first = addr[1]
    addr_rest = addr[2:end]
    if !haskey(t.subtries, addr_first)
        t.subtries[addr_first] = AddressTrie()
    end
    push!(t.subtries[addr_first], addr_rest)
end

"""
Shorthand `push!` for a single element address
"""
Base.push!(t::AddressTrie, addr) = push!(t, (addr,))

"""
Remove the given address.
"""
function Base.delete!(t::AddressTrie, addr::Tuple)
    if addr == ()
        t.has_ea = false
        return
    end
    addr_first = addr[1]
    addr_rest = addr[2:end]
    delete!(t.subtries[addr_first], addr_rest)
    if isempty(t.subtries[addr_first])
        delete!(t.subtries, addr_first)
    end
end

"""
Shorthand `delete!` for a single element address
"""
Base.delete!(t::AddressTrie, addr) = delete!(t, (addr,))


"""
Return the `AddressTrie` for addresses under the given prefix.
The resulting address trie may include the empty address (), which corresponds to the address `prefix`.
Note that `()` is not a valid value for `prefix`.

Warning: The `AddressTrie` that is returned is not a copy of the data.
Mutating the return value will mutate the input `AddressTrie`.
"""
function Base.getindex(t::AddressTrie, prefix::Tuple)
    prefix_head = prefix[1]
    prefix_rest = prefix[2:end]
    if !haskey(t.subtries, prefix_head)
        return AddressTrie()
    end
    if length(prefix) == 1
        return t.subtries[prefix_head]
    else
        return t.subtries[prefix_head][prefix_rest]
    end
end

"""
Shorthand `getindex` for a single element address
"""
Base.getindex(t::AddressTrie, prefix) = t[(prefix,)]

"""
Check whether there are any addresses in the trie.
"""
function Base.isempty(t::AddressTrie)
    if t.has_ea
        return false
    end
    for subtrie in values(t.subtries)
        if !isempty(subtrie)
            return false
        end
    end
    return true
end

"""
Iterator for `AddressTrie`
"""
mutable struct AddressTrieIteratorState
    dict_iterator_state::Int

    # whether the empty address has been visited
    visited_ea::Bool

    current_key::Nullable
    current_subtrie::Nullable{AddressTrie}
    current_subtrie_iterator_state::Nullable{AddressTrieIteratorState}
end

function Base.start(t::AddressTrie)
    dict_iterator_state = start(t.subtries)
    if done(t.subtries, dict_iterator_state)
        current_key = Nullable()
        current_subtrie = Nullable{AddressTrie}()
        current_subtrie_iterator_state = Nullable{AddressTrieIteratorState}()
    else
        ((current_key, current_subtrie), dict_iterator_state) = next(t.subtries, dict_iterator_state)
        current_subtrie_iterator_state = start(current_subtrie)
    end
    AddressTrieIteratorState(dict_iterator_state, false, current_key,
                             current_subtrie, current_subtrie_iterator_state)
end

function Base.done(t::AddressTrie, state::AddressTrieIteratorState)

    if t.has_ea && !state.visited_ea

        # we have the empty address, and we haven't visited it yet
        return false
    end

    if isnull(state.current_key)
        return true
    end

    @assert !isnull(state.current_key)
    @assert !isnull(state.current_subtrie)
    @assert !isnull(state.current_subtrie_iterator_state)
    current_subtrie = get(state.current_subtrie)
    current_subtrie_iterator_state = get(state.current_subtrie_iterator_state)

    if !done(current_subtrie, current_subtrie_iterator_state)

        # current element is not done
        return false
    end

    # the current element is done, we are done if there are no more elements
    return done(t.subtries, state.dict_iterator_state)
end


function Base.next(t::AddressTrie, state::AddressTrieIteratorState)

    if t.has_ea && !state.visited_ea

        # visit the empty element
        state.visited_ea = true
        return ((), state)
    end

    # get the active element
    current_key = get(state.current_key)
    current_subtrie = get(state.current_subtrie)
    current_subtrie_iterator_state = get(state.current_subtrie_iterator_state)

    if done(current_subtrie, current_subtrie_iterator_state)

        # the current subtrie is done, move to the next subtrie
        ((current_key, current_subtrie), state.dict_iterator_state) = next(t.subtries, state.dict_iterator_state)
        current_subtrie_iterator_state = start(current_subtrie)
        state.current_key = current_key
        state.current_subtrie = current_subtrie
        state.current_subtrie_iterator_state = current_subtrie_iterator_state
    end

    # retrieve the next item from the current element
    (sub_address, state.current_subtrie_iterator_state) = next(current_subtrie, current_subtrie_iterator_state)
    item = tuple(current_key, sub_address...)
    return (item, state)
end

function disjoint(t1::AddressTrie, t2::AddressTrie)
    for addr in t1
        if addr in t2
            return false
        end
    end
    return true
end

export AddressTrie
export disjoint
