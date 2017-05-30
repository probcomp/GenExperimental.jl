local C = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
void seedtime()
{
   time_t t;
   srand((unsigned) time(&t));
}
]]

log = C.log

terra seedtime() 
    C.seedtime()
end


local Hash = require("hash")
local Address = &int8
local terra hashAddress(address : Address)
    return Hash.rawhash(address, C.strlen(address))
end

-- local HashMap = require("hashmap")
-- local Trace = HashMap(&int8, bool) -- todo handle other types

local HashMap = function(K, V, hashfn)
    local struct HM {
        log_weight : float -- todo initialize to zero
    }
    terra HM:get(name : K, val : &bool)
        return true 
    end
    terra HM:put(name : K, val : bool)
    end
    return HM
end


local Trace = HashMap(&int8, bool)


terra uniform() : float
    return ([float](C.rand()) / [float](C.RAND_MAX))
end

terra flip_trace(trace : Trace, weight : float) : {float, bool}
    return 0.0, uniform() <= weight
end

terra flip(weight : float) : bool
    return uniform() <= weight
end


terra _flip_regen(val : bool, weight : float) : float
    if val then
        return log(weight)
    else
        return log(1.0 - weight)
    end
end



-- testing out macros

local T = symbol(Trace)
local log_weight = symbol(float)
local log_weight_inc = symbol(float)
local default_regenerators = {}
default_regenerators["flip"] = _flip_regen

local tag = macro(function(name, proc, arg, regenerator)
    return quote
        var val : bool
        var found = [T]:get(name, &val)
        if found then
            [log_weight] = [log_weight] + regenerator(val, [arg])
        else
            val = [proc]([arg])
            [T]:put(name, val)
        end
    in
        val
    end
end)

local program = terra(weight : float)
    var [T]
    var [log_weight]
    var coin = tag("coin", flip, weight, _flip_regen)
    return [log_weight]
end

-- gen = 'probabilistic program'
local gen = function(body)
    return terra([T]) : float
        var [log_weight]
        [body]
        C.printf("log_weight: %f\n", [log_weight])
        return [log_weight]
    end
end

local program2 = gen(quote
    var weight : float = 0.5
    var coin1 = tag("coin1", flip, weight, _flip_regen) -- todo: clean up syntax..
    var coin2 = tag("coin2", flip, weight, _flip_regen)
    C.printf("coin1: %d, coin2: %d\n", coin1, coin2)
end)

-- print(program2:printpretty())

-- t = HashMap(&int8, bool)
-- log_weight = program2(t)
-- print(log_weight)

----
local sample = macro(function(name, proc, arg, regenerator)
    return quote
        var val : bool
        var found = [T]:get(name, &val)
        if found then
            [T].log_weight = [T].log_weight + regenerator(val, [arg])
        else
            val = [proc]([arg])
            [T]:put(name, val)
        end
    in
        val
    end
end)


-- used to pass the trace and log-weight to a trace subroutine
local trace = macro(function(proc, arg)
    return `[proc]([T], [arg])
end)



terra model1([T], weight : float)
    var coin1 = sample("coin1", flip, weight, _flip_regen)
    var coin2 = flip(weight)
    C.printf("coin1: %d, coin2: %d\n", coin1, coin2)
    return coin1, coin2
end

terra model2([T], weight : float)
    var coins = trace(model1, weight)
    var new_coin = sample("coin1", flip, weight, _flip_regen)
    return new_coin
end

print(model2:printpretty())






