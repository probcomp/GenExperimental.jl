HashMap = terralib.memoize(function(K, V)
    local struct Cell {
        k : K,
        v : V,
        next : &Cell
    }
    local struct HM {
        -- __cells : &&HashCell,
        __capacity : uint,
        __size : uint
    }
    terra HM:get(k : K, v : &V) : bool
        return false -- todo
    end
    terra HM:put(k : K, v : V)
        -- todo
    end
    return HM
end)

Trace = terralib.types.newstruct("Trace")
Trace.entries:insert{field="log_weight", type=float}
rcTypes = terralib.newlist()

function register_type(ValType)
    local getName = string.format("get_%s", tostring(ValType))
    local putName = string.format("put_%s", tostring(ValType))
    local mapIdx = #rcTypes
    local mapName = string.format("map%d", mapIdx)
    rcTypes:insert({type=ValType, getName=getName, putName=putName, mapName=mapName, mapIdx=mapIdx})
    Trace.entries:insert{field=mapName, type=HashMap(&int8, ValType)}
end

register_type(float)
register_type(bool)

-- finalize
function finalize_types()
    for i=1,#rcTypes do
        local rcType = rcTypes[i]
        local ValType = rcType.type
        local mapName = rcType.mapName
        Trace.methods[rcType.getName] = terra(self : &Trace, name : &int8, val : &ValType)
            return self.[mapName]:get(name, val)
        end
        Trace.methods[rcType.putName] = terra(self : &Trace, name: &int8, val : ValType)
            self.[mapName]:put(name, val)
        end
    end
end
finalize_types()


local T = symbol(Trace)

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

terra simulate(a : float) : bool
    return false
end
terra regenerate(val : bool, a : float) : float
    return 0.1
end
local testModule = makeModule(simulate, regenerate)

terra program([T])
    var x = testModule(0.5, "x")
    return x
end












