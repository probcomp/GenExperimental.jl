local C = terralib.includecstring [[
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
]]

log = C.log

local rand = require("rand")
local Trace = require("trace")
local T = symbol(Trace) -- for macros

local function makeModule(simulate, regenerate)
	local Module = terralib.types.newstruct()
	local RegenArgTypes = regenerate:gettype().parameters
	local ValType = RegenArgTypes[1]
	Module.metamethods.__apply = macro(function(self, ...)
		local args = terralib.newlist({...})
        local name = args[#RegenArgTypes]
        local actual_args = terralib.newlist()
        for i=1,(#RegenArgTypes-1) do actual_args:insert(args[i]) end
        local lookup_method = Trace.methods[string.format("get_%s", tostring(ValType))]
        local put_method = Trace.methods[string.format("put_%s", tostring(ValType))]
		return quote
			var val : ValType
        	var found = [lookup_method](&[T], [name], &val)
        	if found then
            	[T].log_weight = [T].log_weight + regenerate(val, [actual_args])
        	else
            	val = [simulate]([actual_args])
            	[put_method](&[T], [name], val)
        	end
		in val end
	end)
	return terralib.new(Module)
end

-- define some primitives

terra flip_trace(trace : Trace, p: float) : {float, bool}
    return 0.0, rand.uniform() <= p 
end

terra _flip_simulate(p: float) : bool
    return rand.uniform() <= p 
end

terra _flip_regen(val : bool, p: float) : float
    if val then
        return log(p)
    else
        return log(1.0 - p)
    end
end

-- NOTE: custom regen parameters are just arguments to the function call in the program.
local flip = makeModule(_flip_simulate, _flip_regen)

terra test_model([T], weight : float)
    var coin1 = flip(weight, "coin1")
    var coin2 = _flip_simulate(weight)
    C.printf("coin1: %d, coin2: %d\n", coin1, coin2)
    return coin1
end

terra doit()
    rand.seedtime()
    var trace : Trace
    trace:init() -- TODO use better syntax here
    var result = test_model(trace, 0.5)
    C.printf("result=%d\n", result)
    C.printf("log_weight=%f\n", trace.log_weight)
end
doit()

print(test_model:printpretty())

--terra model2([T], weight : float)
    --var coins = model1([T], weight) -- traced subroutine
    --var new_coin = flip(weight, "coin2") -- primitive
    --return new_coin
--end
