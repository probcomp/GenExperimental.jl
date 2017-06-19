import IJulia

type Tile
    id::String
    num_cols::Int
    num_rows::Int
    width::Int
    height::Int
    trace_xmin::Float64
    trace_ymin::Float64
    trace_width::Float64
    trace_height::Float64
end
# TODO setters

function here(tile::Tile)
    HTML("""
<div id="$(tile.id)"></id>
<script>
	var root = document.getElementById("$(tile.id)");
	var tile_width = $(float(tile.width) / tile.num_cols);
	var tile_height = $(float(tile.height) / tile.num_rows);
    var i = 1;
	for (var row=0; row < $(tile.num_rows); row++) {
		for (var col=0; col < $(tile.num_cols); col++) {
			var id = "$(tile.id)_" + i;
			var x = tile_width * col;
			var y = tile_height * row;
			var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
			svg.setAttribute("id", id);
			svg.setAttribute("x", x);
			svg.setAttribute("y", y);
			svg.setAttribute("width", tile_width);
			svg.setAttribute("height", tile_height);
			svg.setAttribute("viewBox", "$(tile.trace_xmin) $(tile.trace_ymin) $(tile.trace_width) $(tile.trace_height)");
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

function viewport(renderer::JupyterInlineRenderer, height::Int=200)
    renderer.dom_element_id = "id_$(randstring(20))"
    HTML("""
<div style="border-style: solid; border-radius: 25px; text-align: center;">
Trace Rendering ($(renderer.name))
<div id="$(get(renderer.dom_element_id))" style="height: $(height)px; resize: vertical; position: relative; overflow: hidden; text-align: center;">
</div>
</div>
    """)
end

function render(renderer::JupyterInlineRenderer, trace::Trace) # TODO handle DifferentiableTrace
    if (isnull(renderer.dom_element_id))
        error("No dom element has been defined")
    end
    IJulia.send_comm(renderer.comm, Dict("trace" => trace,
										 "dom_element_id" => renderer.dom_element_id,
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

export Tile
export here
