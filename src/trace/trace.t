stdio = terralib.includec("stdio.h")
stdlib = terralib.includec("stdlib.h")
uthash = terralib.includec("uthash.h")

hashtable = terralib.includecstring [[
    #include <string.h>  /* strcpy */
    #include <stdlib.h>  /* malloc */
    #include <stdio.h>   /* printf */    
    #include "uthash.h"
    typedef struct my_struct {
        const char *name;          /* key */
        float value;
        UT_hash_handle hh;         /* makes this structure hashable */
    } table_t;
    
    table_t* new_table() {
        return NULL;
    }

    table_t* add(table_t* table, const char* name, const float value) {
        table_t* s = (table_t*) malloc(sizeof(table_t));
        s->name = name;
        s->value = value;
        HASH_ADD_KEYPTR(hh, table, s->name, strlen(s->name), s);
		return table;
    }

    table_t* get(table_t* table, const char* name) {
        table_t* s = NULL;
        HASH_FIND_STR(table, name, s);
        return s; // NULL if not found, otherwise, use s.value
    }

	int has(table_t* table, const char* name) {
		return get(table, name) != NULL;
	}

    void free_table(table_t* table) {
		table_t *s, *tmp = NULL;
    	/* free the hash table contents */
    	HASH_ITER(hh, table, s, tmp) {
      		HASH_DEL(table, s);
      		free(s);
    	}
    }

]]

--- trace is a map from names (strings) to values (currently just floats)

Trace = hashtable.table_t

terra new_trace() : &Trace
	return hashtable.new_table()
end

terra trace_add(table : &Trace, name : &int8, value : float)
	-- returns the pointer tothe hashtable which might be new if the table
	-- was empty
	return hashtable.add(table, name, value)
end

terra trace_has(table : &Trace, name : &int8)
	return [bool](hashtable.has(table, name))
end

terra trace_get(table : &Trace, name : &int8) : float
	var result = hashtable.get(table, name)
	return result.value
end

terra test_trace()
	var table = new_trace()
	table = trace_add(table, "a", 0.5)
	table = trace_add(table, "b", 0.3)
	table = trace_add(table, "xx", 0.4)
	var names : (&int8)[5]
	names[0] = "a"
	names[1] = "b"
	names[2] = "xx"
	names[3] = "asdf"
	names[4] = "asdfasfd"
	for i=0,3 do
		stdio.printf("%s %i\n", names[i], trace_has(table, names[i]))
		stdio.printf("%s %f\n", names[i], trace_get(table, names[i]))
	end
	for i=3,5 do
		stdio.printf("%s %i\n", names[i], trace_has(table, names[i]))
	end
end

--- Requests is just a HashSet of strings ----

hashset = terralib.includecstring [[
    #include <string.h>  /* strcpy */
    #include <stdlib.h>  /* malloc */
    #include <stdio.h>   /* printf */    
    #include "uthash.h"
    typedef struct my_struct {
        const char *name;          /* key */
        UT_hash_handle hh;         /* makes this structure hashable */
    } table_t;
    
    table_t* new_hashset() {
        return NULL;
    }

    table_t* hashset_add(table_t* table, const char* name) {
        table_t* s = (table_t*) malloc(sizeof(table_t));
        s->name = name;
        HASH_ADD_KEYPTR(hh, table, s->name, strlen(s->name), s);
		return table;
    }

	int hashset_has(table_t* table, const char* name) {
        table_t* s = NULL;
        HASH_FIND_STR(table, name, s);
		return s != NULL;
	}

    void hashset_free(table_t* table) {
		table_t *s, *tmp = NULL;
    	/* free the hash table contents */
    	HASH_ITER(hh, table, s, tmp) {
      		HASH_DEL(table, s);
      		free(s);
    	}
    }

]]


Request = hashset.table_t

terra new_request() : &Request
	return hashset.new_hashset()
end

terra request_add(table : &Request, name : &int8)
	-- returns the pointer tothe hashtable which might be new if the table
	-- was empty
	return hashset.hashset_add(table, name)
end

terra request_has(table : &Request, name : &int8) : bool
	return [bool](hashset.hashset_has(table, name))
end

terra test_request()
	var request = new_request()
	request = request_add(request, "a")
	request = request_add(request, "b")
	request = request_add(request, "xx")
	var names : (&int8)[5]
	names[0] = "a"
	names[1] = "b"
	names[2] = "xx"
	names[3] = "asdf"
	names[4] = "asdfasfd"
	for i=0,3 do
		stdio.printf("%s %i\n", names[i], request_has(request, names[i]))
	end
	for i=3,5 do
		stdio.printf("%s %i\n", names[i], request_has(request, names[i]))
	end
end


-- test them

test_trace()
test_request()

print("loaded!")
