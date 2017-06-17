import IJulia

type JupyterInlineRenderer
    name::String # The target name for Javascript
    dom_element_id::Nullable{String} # The DOM element where the JS code should render to
    comm::IJulia.Comm # communication object to Javascript
    
    function JupyterInlineRenderer(name::String)
        comm = IJulia.Comm(name, data=Dict())
        new(name, Nullable{String}(), comm)
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
										 "dom_element_id" => renderer.dom_element_id))
end


type TiledJupyterInlineRenderer
    name::String # The target name for Javascript
    dom_element_id::Nullable{String} # The DOM element where the JS code should render to
    comm::IJulia.Comm # communication object to Javascript
	num_rows::Int
	num_cols::Int
	width::Int
	height::Int
    function TiledJupyterInlineRenderer(name::String,
										num_cols::Int, num_rows::Int,
										width::Int, height::Int)
        comm = IJulia.Comm(name, data=Dict())
        new(name, Nullable{String}(), comm, num_rows, num_cols, width, height)
    end
end

function inline(renderer::TiledJupyterInlineRenderer) # TODO add pixels as args
    # create an array of boxes, and when render gets called, iterate throgugh them..
    id = "id_$(randstring(20))"
	renderer.dom_element_id = id
HTML("""
<svg id=$(id) width=$(renderer.width) height=$(renderer.height) xmlns="http://www.w3.org/2000/svg">
</svg>
<script>
	var parent_svg = document.getElementById("$(id)");
	var tile_width = $(float(renderer.width) / renderer.num_cols);
	var tile_height = $(float(renderer.height) / renderer.num_rows);
	for (var row=0; row < $(renderer.num_rows); row++) {
		for (var col=0; col < $(renderer.num_cols); col++) {
			var id = "$(id)_" + row + "_" + col;
			var x = tile_width * col;
			var y = tile_height * row;
			var svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
			svg.setAttribute("id", id);
			svg.setAttribute("x", x);
			svg.setAttribute("y", y);
			svg.setAttribute("width", tile_width);
			svg.setAttribute("height", tile_height);
			svg.setAttribute("viewBox", "0 0 100 100");
			parent_svg.appendChild(svg);
		}
	}
</script>
""")
end

function render(renderer::TiledJupyterInlineRenderer, traces::Vector{Trace})
    if (isnull(renderer.dom_element_id))
        error("No target has been defined")
    end
    for (i, trace) in enumerate(traces)
		row = div((i - 1), renderer.num_cols)
		col = (i - 1) % renderer.num_cols
		dom_element_id = "$(get(renderer.dom_element_id))_$(row)_$(col)"
        IJulia.send_comm(renderer.comm, Dict("trace" => trace, "dom_element_id" => dom_element_id))
    end
end

macro javascript_str(s) display("text/javascript", s); end

export enable_inline
export JupyterInlineRenderer
export TiledJupyterInlineRenderer
export TiledJupyterInlineRenderer
#export inline
export viewport
export render
export attach
export @javascript_str
