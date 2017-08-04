import DataStructures.OrderedDict

"""

An address ("foo", "bar", 2) has two meanings:

    1. it is a concrete terminal address
    2. it is the prefix of all addresses of the form ("foo, "bar", 2, ...), which does NOT include
       the concrete terminal address ("foo", "bar", 2) (i.e. it is not its own prefix).

    Base.in, push!, delete! accept concrete addresses.

    Therefore, suppose the concrete address ("foo", "bar, 2) was pushed.
    Then, Base.in(t, ("foo", "bar", 2)) would return true, but
    Base.getindex(t, ("foo", "bar", 2)) would return an empty AddressTrie.

    
    
"""

mutable struct AddressTrie{T}
    dict::OrderedDict{Any, T}
end

mutable struct AddressTrieElement{T}
    root_present::Bool
    subtrie::T
end

# NOTE: using parametric types here to avoid circular type definition.  See
# https://github.com/JuliaLang/julia/issues/269#issuecomment-68421745

AddressTrie() = AddressTrie(OrderedDict{Any, AddressTrieElement}())


AddressTrieElement() = AddressTrieElement(false, AddressTrie())


"""
Test if the given address is included.
"""
function Base.in(addr::Tuple, t::AddressTrie)
    addr_first = addr[1]
    addr_rest = addr[2:end]
    if haskey(t.dict, addr_first)
        element = t.dict[addr_first]
        if addr_rest == ()
            return element.root_present
        else
            return addr_rest in element.subtrie
        end
    else
        return false
    end
end

"""
Shorthand `in` for a single element address
"""
Base.in(addr, t::AddressTrie) = in((addr,), t)

"""
Add the given address.
"""
function Base.push!(t::AddressTrie, addr::Tuple)
    addr_first = addr[1]
    addr_rest = addr[2:end]
    local element::AddressTrieElement
    if haskey(t.dict, addr_first)
        element = t.dict[addr_first]
    else
        element = AddressTrieElement()
        t.dict[addr_first] = element
    end
    if addr_rest == ()
        element.root_present = true
    else
        push!(element.subtrie, addr_rest)
    end
end

"""
Shorthand `push!` for a single element address
"""
Base.push!(t::AddressTrie, addr) = push!(t, (addr,))

"""
Remove the given address.
"""
function Base.delete!(t::AddressTrie, addr::Tuple)
    addr_first = addr[1]
    addr_rest = addr[2:end]
    local element::AddressTrieElement
    if haskey(t.dict, addr_first)
        element = t.dict[addr_first]
        if addr_rest == ()
            element.root_present = false
        else
            delete!(element.subtrie, addr_rest)
        end
        if !element.root_present && isempty(element.subtrie)
            delete!(t.dict, addr_first)
        end
    end
end

"""
Shorthand `delete!` for a single element address
"""
Base.delete!(t::AddressTrie, addr) = delete!(t, (addr,))


"""
Test if there are any addresses with the given prefix.

Note that the address prefix is not considered a prefix of itself. To test for a particular address use `in`.
"""
function Base.haskey(t::AddressTrie, prefix::Tuple)
    prefix_head = prefix[1]
    prefix_rest = prefix[2:end]
    if haskey(t.dict, prefix_head)
        element = t.dict[prefix_head]
        if prefix_rest == ()
            return !isempty(element.subtrie)
        else
            return haskey(element.subtrie, prefix_rest)
        end
    end
    return false
end

"""
Shorthand `haskey` for a single element address
"""
Base.haskey(t::AddressTrie, prefix) = haskey(t, (prefix,))

"""
Return the `AddressTrie` for addresses under the given prefix.
"""
function Base.getindex(t::AddressTrie, prefix::Tuple)
    prefix_head = prefix[1]
    prefix_rest = prefix[2:end]
    subtrie = t.dict[prefix_head].subtrie 
    if prefix_rest == ()
        return subtrie
    else
        return subtrie[prefix_rest]
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
    for (addr_first, element) in t.dict
        if element.root_present || !isempty(element.subtrie)
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

    # whether the root address for the active dict element has been visited
    visited_root::Bool

    current_key::Nullable
    current_element::Nullable{AddressTrieElement}
    current_subtrie::Nullable{AddressTrie}
    current_subtrie_iterator_state::Nullable{AddressTrieIteratorState}
end

function Base.start(t::AddressTrie)
    dict_iterator_state = start(t.dict)
    if done(t.dict, dict_iterator_state)
        current_key = Nullable()
        current_element = Nullable{AddressTrieElement}()
        current_subtrie = Nullable{AddressTrie}()
        current_subtrie_iterator_state = Nullable{AddressTrieIteratorState}()
    else
        ((current_key, current_element), dict_iterator_state) = next(t.dict, dict_iterator_state)
        current_subtrie = current_element.subtrie
        current_subtrie_iterator_state = start(current_subtrie)
    end
    AddressTrieIteratorState(dict_iterator_state, false, current_key, current_element, current_subtrie, current_subtrie_iterator_state)
end

function Base.done(t::AddressTrie, state::AddressTrieIteratorState)

    if isnull(state.current_element)

        # there were no elements at all
        return true
    end

    # there was at least one element. it never goes from not null to null
    @assert !isnull(state.current_key)
    @assert !isnull(state.current_element)
    @assert !isnull(state.current_subtrie)
    @assert !isnull(state.current_subtrie_iterator_state)

    # get the current element
    current_element = get(state.current_element)
    current_subtrie = get(state.current_subtrie)
    current_subtrie_iterator_state = get(state.current_subtrie_iterator_state)

    if current_element.root_present && !state.visited_root

        # current element has a root that we haven't visited yet
        return false
    end

    if !done(current_subtrie, current_subtrie_iterator_state)

        # current element is not done
        return false
    end

    # the current element is done, we are done if there are no more elements
    return done(t.dict, state.dict_iterator_state)
end


function Base.next(t::AddressTrie, state::AddressTrieIteratorState)

    # get the active element
    current_key = get(state.current_key)
    current_element = get(state.current_element)
    current_subtrie = get(state.current_subtrie)
    current_subtrie_iterator_state = get(state.current_subtrie_iterator_state)

    if current_element.root_present && !state.visited_root

        # visit the root of the current element
        state.visited_root = true
        item = (current_key,)
        return (item, state)
    end

    if done(current_subtrie, current_subtrie_iterator_state)

        # the current element is done, move to the next element
        ((current_key, current_element), state.dict_iterator_state) = next(t.dict, state.dict_iterator_state)
        current_subtrie = current_element.subtrie
        current_subtrie_iterator_state = start(current_subtrie)
        state.current_key = current_key
        state.current_element = current_element
        state.current_subtrie = current_subtrie
        state.current_subtrie_iterator_state = current_subtrie_iterator_state
        state.visited_root = false

        if current_element.root_present && !state.visited_root

            # visit the root of the new current element
            state.visited_root = true
            item = (current_key,)
            return (item, state)
        end
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

# test cases
using Base.Test

@testset "isempty" begin
    @test isempty(AddressTrie())
end

@testset "insert, test, and delete" begin

    t = AddressTrie()
    push!(t, ("foo",))
    @test ("foo",) in t
    @test !haskey(t, ("foo",))

    push!(t, ("foo", 1))
    @test ("foo",) in t
    @test ("foo", 1) in t
    @test haskey(t, ("foo",))
    @test !haskey(t, ("foo", 1))

    push!(t, ("foo", 2))
    @test ("foo",) in t
    @test ("foo", 1) in t
    @test ("foo", 2) in t

    push!(t, ("foo", 1, "X"))
    @test ("foo",) in t
    @test ("foo", 1) in t
    @test ("foo", 1, "X") in t
    @test ("foo", 2) in t

    @test haskey(t, ("foo",))
    subtrie = t[("foo",)]
    @test (1,) in subtrie
    @test (2,) in subtrie

    @test haskey(t, ("foo", 1))
    subtrie = t[("foo", 1)]
    @test ("X",) in subtrie

    delete!(t, ("foo",))
    @test !(("foo",) in t)
    @test ("foo", 1) in t # the subtrie is not deleted, only the one element
  
    delete!(t, ("foo", 1))
    delete!(t, ("foo", 1, "X"))
    delete!(t, ("foo", 2))
    @test isempty(t)

end


@testset "shorthand for single-element addresses" begin

    t = AddressTrie()
    push!(t, "foo")
    @test "foo" in t
    @test !haskey(t, "foo")

    push!(t, ("foo", 1))
    @test "foo" in t
    @test ("foo", 1) in t
    @test haskey(t, "foo")

    push!(t, ("foo", 2))
    @test "foo" in t
    @test ("foo", 1) in t
    @test ("foo", 2) in t

    push!(t, ("foo", 1, "X"))
    @test "foo" in t
    @test ("foo", 1) in t
    @test ("foo", 1, "X") in t
    @test ("foo", 2) in t

    @test haskey(t, "foo")
    subtrie = t["foo"]
    @test 1 in subtrie
    @test 2 in subtrie

    subtrie = t[("foo", 1)]
    @test "X" in subtrie

    delete!(t, "foo")
    @test !("foo" in t)
    @test ("foo", 1) in t
end

@testset "iterator" begin

    t = AddressTrie()
    push!(t, "foo")
    push!(t, ("foo", 1))
    push!(t, ("foo", 2))
    push!(t, ("bar", "x", "y", "z"))
    push!(t, ("foo", 2, 3))
    push!(t, ("foo", 3, 4, 5))

    state = start(t)
    @test !done(t, state)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo",)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo", 1)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo", 2)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo", 2, 3)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo", 3, 4, 5)

    item, state = next(t, state)
    @test done(t, state)
    @test item == ("bar", "x", "y", "z")

    # remove all items and start new iterator, checking it is done
    delete!(t, "foo")
    delete!(t, ("foo", 1))
    delete!(t, ("foo", 2))
    delete!(t, ("bar", "x", "y", "z"))
    delete!(t, ("foo", 2, 3))
    delete!(t, ("foo", 3, 4, 5))
    state = start(t)
    @test done(t, state)

    # add new items, check the order, which is based on insertion, except that
    # the root address gets visited first
    push!(t, ("baz",))
    push!(t, ("foo", 1))
    push!(t, "foo")

    state = start(t)
    @test !done(t, state)
    
    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("baz",)

    item, state = next(t, state)
    @test !done(t, state)
    @test item == ("foo",)

    item, state = next(t, state)
    @test done(t, state)
    @test item == ("foo", 1)

end

@testset "iterator for .. in .. " begin

    t = AddressTrie()
    push!(t, "foo")
    push!(t, ("foo", 1))
    push!(t, ("foo", 2))
    push!(t, ("bar", "x", "y", "z"))

    arr = []
    for addr in t
        push!(arr, addr)
    end
    @test length(arr) == 4
    @test arr[1] == ("foo",)
    @test arr[2] == ("foo", 1)
    @test arr[3] == ("foo", 2)
    @test arr[4] == ("bar", "x", "y", "z")
    
end

@testset "disjoint" begin

    t1 = AddressTrie()
    push!(t1, "foo")
    push!(t1, ("foo", 1))

    t2 = AddressTrie()
    push!(t2, "bar")
    @test disjoint(t1, t2)

    push!(t2, ("foo", 1))
    @test !disjoint(t1, t2)
end
