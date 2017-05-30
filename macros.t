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

terra uniform() : float
    return ([float](C.rand()) / [float](C.RAND_MAX))
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

local Hash = require("hash")
local Address = &int8
local terra hashAddress(address : Address)
    return Hash.rawhash(address, C.strlen(address))
end

-- local HashMap = require("hashmap")
-- local Trace = HashMap(&int8, bool) -- todo handle other types

local HashMap = function(K, V, hashfn)
    local struct HM {
    }
    terra HM:get(name : K, val : &bool)
        return true 
    end
    terra HM:put(name : K, val : bool)
    end
    return HM
end


local trace = symbol(HashMap(&int8, bool))
local log_weight = symbol(float)
local default_regenerators = {}
default_regenerators["flip"] = _flip_regen

local tag = macro(function(name, proc, arg, regenerator)
    return quote
        var val : bool
        var found = [trace]:get(name, &val)
        if found then
            [log_weight] = [log_weight] + regenerator(val, [arg])
        else
            val = [proc]([arg])
            [trace]:put(name, val)
        end
    in
        val
    end
end)

local program = terra(weight : float)
    var [trace]
    var [log_weight]
    var coin = tag("coin", flip, weight, _flip_regen)
    return [log_weight]
end

-- gen = 'probabilistic program'
local gen = function(body)
    return terra([trace]) : float
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

print(program2:printpretty())

t = HashMap(&int8, bool)
log_weight = program2(t)
print(log_weight)
