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

terra _flip_simulate(weight : float, weight2 : float) : bool
    return uniform() <= weight
end


terra _flip_regen(val : bool, weight : float, weight2 : float) : float
    if val then
        return log(weight)
    else
        return log(1.0 - weight)
    end
end



-- testing out macros

local T = symbol(Trace)

-- t = HashMap(&int8, bool)
-- log_weight = program2(t)
-- print(log_weight)

---

-- register regenerators with function pointers separately?
-- use exotypes ---- make the procedure an object of type 'module' that 
-- has compiled with it the ability to perform two functions.

-- function application is regular application


local function makeModule(simulate, regenerate)
	local Module = terralib.types.newstruct()
	local RegenArgTypes = regenerate:gettype().parameters
	local ValType = RegenArgTypes[1]
	local ParamTypes = terralib.newlist()
	for i=2,#RegenArgTypes do ParamTypes:insert(RegenArgTypes[i]) end
	Module.metamethods.__apply = macro(function(self, name, ...)
		local args = terralib.newlist({...})
		return quote
			var val : ValType
        	var found = [T]:get([name], &val)
        	if found then
            	[T].log_weight = [T].log_weight + regenerate(val, [args])
        	else
            	val = [simulate]([args])
            	[T]:put([name], val)
        	end
		in val end
	end)
	return terralib.new(Module)
end

-- how to define custom regeneration parameters?
local flip = makeModule(_flip_simulate, _flip_regen)

-- used to pass the trace and log-weight to a trace subroutine
local traced = macro(function(proc, ...)
    local args = terralib.newlist({...})
    return `[proc]([T], [args])
end)

terra model1([T], weight : float, weight2 : float)
    var coin1 = flip("coin1", weight, weight2) -- NOTE: regeneration parameters are additional arguments to __apply
    var coin2 = _flip_simulate(weight, weight2)
    C.printf("coin1: %d, coin2: %d\n", coin1, coin2)
    return coin1, coin2
end

print(model1:printpretty())

terra model2([T], weight : float, weight2 : float)
    var coins = model1([T], weight, weight2) -- traced subroutine
    var new_coin = flip("coin2", weight, weight2) -- primitive
    return new_coin
end

print(model2:printpretty())

-------

--gen model2(weight : float)
    --var coins ~ model1(weight) -- a gen function
    --var new_coin ~ flip(weight) #coin1 -- a primitive with a regenerator
    --var other_coin = flip(weight) -- a regular terra function
    --return new_coin
--end
--
--
--trace = new Trace()
--trace.coin1 = false
--weight = model2(trace)


-------------

-- the regeneration policy should be determined at inference time, not inside the
-- model program. example:

-- fix the regenerators at compile file, but determine the parameters at runtime?

-- trace = new Trace()
-- trace.obs = 0.123
-- lw = my_model(trace, regenerators)










