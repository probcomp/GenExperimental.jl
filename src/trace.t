-- used quicksand as a reference
local C = terralib.includecstring [[
#include <stdlib.h>
#include <stdio.h>
]]
local S = require("std")
local hash = require("qs.lib.hash")
local util = require("qs.lib.util")

local defaultInitialCapacity = 8
local expandFactor = 2
local loadFactor = 2.0

HashMap = terralib.memoize(function(K, V, hashfn)

    hashfn = hashfn or hash.gethashfn(K)

    local struct HashCell(S.Object) {
        key : K,
        val : V,
        next : &HashCell
    }
    
    -- todo: init other form?

    terra HashCell:__init(k: K, v: V)
        -- todo: does this matter?
        self.key = k
        -- S.copy(self.key, k)
        self.val = v
        self.next = nil
    end

    terra HashCell:__destruct() : {}
        if self.next ~= nil then
            self.next:delete()
        end
    end

    local struct HM(S.object) {
        __cells : &&HashCell,
        __capacity : uint,
        __size : uint
    }

	terra HM:__init(initialCapacity: uint) : {}
		self.__capacity = initialCapacity
		self.__cells = [&&HashCell](C.malloc(initialCapacity*sizeof([&HashCell])))
		for i=0,self.__capacity do
			self.__cells[i] = nil
		end
		self.__size = 0
	end

    terra HM:clear()
        for i=0,self.__capacity do
            if self.__cells[i] ~= nil then
                self.__cells[i]:delete()
                self.__cells[i] = nil
            end
        end
        self.__size = 0
    end

    terra HM:__destruct()
        self:clear()
        S.free(self.__cells)
    end

    terra HM:capacity() return self.__capacity end
    HM.methods.capacity:setinlined(true)

    terra HM:size() return self.__size end
    HM.methods.size:setinlined(true)

    terra HM:hash(key: K)
        return hashfn(key) % self.__capacity
    end
    HM.methods.hash:setinlined(true)

    terra HM:get(key: K, outval: &V)
        var vptr = self:getPointer(key)
        if vptr == nil then
            return false
        else
            @outval = @vptr
            return true
        end
    end

    terra HM:getPointer(key: K)
        var cell = self.__cells[self:hash(key)]
        while cell ~= nil do
            if util.equal(cell.key, key) then
                return &cell.val
            end
            cell = cell.next
        end
        return nil
    end

    -- Expand and rehash
    terra HM:__expand()
        var oldcap = self.__capacity
        var oldcells = self.__cells
        var old__size = self.__size
        self:__init(2*oldcap)
        self.__size = old__size
        for i=0,oldcap do
            var cell = oldcells[i]
            while cell ~= nil do
                var index = self:hash(cell.key)
                var nextCellToProcess = cell.next
                cell.next = self.__cells[index]
                self.__cells[index] = cell
                cell = nextCellToProcess
            end
        end
        S.free(oldcells)
    end

    terra HM:__checkExpand()
        if [float](self.__size)/self.__capacity > loadFactor then
            self:__expand()
        end
    end
    HM.methods.__checkExpand:setinlined(true)

    terra HM:put(key: K, val: V) : {}
        var index = self:hash(key)
        var cell = self.__cells[index]
        if cell == nil then
            cell = HashCell.alloc()
            cell:__init(key, val)
            self.__cells[index] = cell
        else
            -- Check if this key is already present, and if so, replace
            -- its value
            var origcell = cell
            while cell ~= nil do
                if util.equal(cell.key, key) then
                    S.rundestructor(cell.val)
                    cell.val = val
                    return
                end
                cell = cell.next
            end
            cell = origcell
            -- Otherwise, insert new cell at head of linked list
            var newcell = HashCell.alloc()
            newcell:__init(key, val)
            newcell.next = cell
            self.__cells[index] = newcell
        end
        self.__size = self.__size + 1
        self:__checkExpand()
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

-- register types of random choices that the trace should hold here (can be arbitrary Terra types, I think)
register_type(float)
register_type(bool)

function finalize_types()
    local init_macros = terralib.newlist()
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
    terra Trace:init() : {}
        self.log_weight = 0.0
        self.map0:__init(defaultInitialCapacity)
        self.map1:__init(defaultInitialCapacity)
        -- todo: programatically do this.....
    end

end
finalize_types()

return Trace
