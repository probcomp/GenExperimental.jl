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

terra new_table() : &Trace
	return hashtable.new_table()
end

terra add(table : &Trace, name : &int8, value : float)
	-- returns the pointer tothe hashtable which might be new if the table
	-- was empty
	return hashtable.add(table, name, value)
end

terra has(table : &Trace, name : &int8)
	return hashtable.has(table, name)
end

terra get(table : &Trace, name : &int8) : float
	var result = hashtable.get(table, name)
	return result.value
end

terra doit()
	var table = new_table()
	table = add(table, "a", 0.5)
	table = add(table, "b", 0.3)
	table = add(table, "xx", 0.4)
	var names : (&int8)[5]
	names[0] = "a"
	names[1] = "b"
	names[2] = "xx"
	names[3] = "asdf"
	names[4] = "asdfasfd"
	for i=0,3 do
		stdio.printf("%s %i\n", names[i], has(table, names[i]))
		stdio.printf("%s %f\n", names[i], get(table, names[i]))
	end
	for i=3,5 do
		stdio.printf("%s %i\n", names[i], has(table, names[i]))
	end
end
doit()
