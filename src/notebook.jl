import IJulia

global active_viewport = "undefined"

function get_active_viewport()
    global active_viewport
    return active_viewport
end

function random_viewport_name()
    "id_$(replace(string(Base.Random.uuid4()), "-", ""))"
end

# TODO: add a global variable that is the 'current viewport' (like current axis in matplotlib)
# then, there should be a version of here that doesn't require a name, and without explicitly calling attach
# on the renderer, the renderer should automatically rneder to the current viewport
# TODO: can this genrealize to other cases?
# NOTE: the default behavior for a renderer for which attach() has not been called is to render to the active viewport
# NOTE: the default behavior of here() for which an id has not been given is to generate a rnadom id and set it to the current active viewport.
# every call to here sets the current active viewport.
# sub-viewports within viewports are identified by <viewport-id>-i where i ranges from 1 to num_cols * num_rows
# NOTE: it is also possible to write custom HTML and Javascript that defines
# custom layouts (the trace rendering just renders a trace to an identified DOM
# element)
function here(id::String=random_viewport_name(); num_cols::Int=1, num_rows::Int=1,
                  width::Int=200, height::Int=200, 
                  trace_xmin::Real=0.0, trace_ymin::Real=0.0,
                  trace_width::Real=1.0, trace_height::Real=1.0)
    global active_viewport = id
    println("Setting active viewport to $id")
    HTML("""
<svg id="$(id)" width=$(width) height=$(height)></svg>
<script>
	var root = document.getElementById("$(id)");
	var tile_width = $(float(width) / num_cols);
	var tile_height = $(float(height) / num_rows);
    var i = 1;
	for (var row=0; row < $(num_rows); row++) {
		for (var col=0; col < $(num_cols); col++) {
			var id = "$(id)_" + i;
			var x = tile_width * col;
			var y = tile_height * row;
			var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
			svg.setAttribute("id", id);
			svg.setAttribute("x", x);
			svg.setAttribute("y", y);
			svg.setAttribute("width", tile_width);
			svg.setAttribute("height", tile_height);
			svg.setAttribute("viewBox", "$(trace_xmin) $(trace_ymin) $(trace_width) $(trace_height)");
			root.appendChild(svg);
            i += 1;
		}
	}

</script>
    """)
end

type JupyterInlineRenderer
    name::String # The target name for Javascript
    dom_element_id::Nullable{String} # The DOM element where the JS code should render to
    comm::IJulia.Comm # communication object to Javascript
    configuration::Dict # gets sent to the JS renderer alongside the trace (the JS renderer is currently stateless)
    
    function JupyterInlineRenderer(name::String, configuration::Dict)
        comm = IJulia.Comm(name, data=Dict())
        new(name, Nullable{String}(), comm, configuration)
    end
end

function attach(renderer::JupyterInlineRenderer, dom_element_id::String)
    renderer.dom_element_id = dom_element_id
end

function render(renderer::JupyterInlineRenderer, trace::Trace) # TODO handle DifferentiableTrace
    local id::String
    global active_viewport
    if (isnull(renderer.dom_element_id))
        if (active_viewport == "undefined")
            error("No viewport has been defined")
        end
        # use global active viewport
        id = active_viewport
    else
        # use explicitly attached viewport
        id = get(renderer.dom_element_id)
    end
    IJulia.send_comm(renderer.comm, Dict("trace" => trace,
										 "dom_element_id" => id,
                                         "conf" => renderer.configuration))
end

macro javascript_str(s) display("text/javascript", s); end

export @javascript_str
export enable_inline
export JupyterInlineRenderer
export TiledJupyterInlineRenderer
export TiledJupyterInlineRenderer
#export inline
export viewport
export render
export attach

export here
export get_active_viewport
