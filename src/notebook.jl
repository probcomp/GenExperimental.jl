import IJulia

global active_viewport = "undefined"

function get_active_viewport()
    global active_viewport
    return "$(active_viewport)_frame"
end

function random_viewport_name()
    "id_$(replace(string(Base.Random.uuid4()), "-", ""))"
end

mutable struct Figure
    id::String
    num_cols::Int
    num_rows::Int
    width::Int
    height::Int
    trace_xmin::Real
    trace_ymin::Real
    trace_width::Real
    trace_height::Real
    margin_top::Real
    margin_bottom::Real
    margin_left::Real
    margin_right::Real
    titles::Array{String,1}
    function Figure(;id::String=random_viewport_name(), num_cols::Int=1, num_rows::Int=1,
                    width::Int=200, height::Int=200,
                    trace_xmin::Real=0.0, trace_ymin::Real=0.0,
                    trace_width::Real=1.0, trace_height::Real=1.0,
                    margin_top::Real=0.0, margin_bottom::Real=0.0,
                    margin_left::Real=0.0, margin_right::Real=0.0, titles=Array{String,1}())
        new(id, num_cols, num_rows, width, height,
            trace_xmin, trace_ymin, trace_width, trace_height,
            margin_top, margin_bottom, margin_left, margin_right, titles)
    end

end

id(figure::Figure) = figure.id

id(subfigure::Pair{Figure,Int}) = "$(id(subfigure.first))_frame$(subfigure.second)"

function here(figure::Figure)
    setup = """
    <div id="$(figure.id)"></div>
    <script>
        var d3 = require("nbextensions/d3/d3.min");

        var svg = d3.select("#$(figure.id)").append("svg")
            .attr("width", $(figure.width))
            .attr("height", $(figure.height));

        // TODO delete me
        svg.append("rect")
            .attr("x", 0)
            .attr("y", 0)
            .attr("width", "100%")
            .attr("height", "100%")
            .style("stroke", "white")
            .style("fill", "white");

        var tile_width = $(figure.width) / $(figure.num_cols);
        var tile_height = $(figure.height) / $(figure.num_rows);
        var i = 1;
        for (var row=0; row<$(figure.num_rows); row++) {
            for (var col=0; col<$(figure.num_cols); col++) {
                var sub_svg = svg.append("svg")
                    .classed("subfigure", true)
			        .attr("id", "$(figure.id)_parent" + i)
                    .attr("x", col * tile_width)
                    .attr("y", row * tile_height)
                    .attr("width", tile_width)
                    .attr("height", tile_height);
                var inner_svg = sub_svg.append("svg")
                    .attr("x", $(figure.margin_left))
                    .attr("y", $(figure.margin_top))
                    .attr("width", tile_width - $(figure.margin_left) - $(figure.margin_right))
                    .attr("height", tile_height - $(figure.margin_top) - $(figure.margin_bottom));
                var frame = inner_svg.append("svg")
			        .attr("id", "$(figure.id)_frame" + i)
                    //.attr("preserveAspectRatio", "none")
                    .attr("viewBox", "$(figure.trace_xmin) $(figure.trace_ymin) $(figure.trace_width) $(figure.trace_height)");
                i += 1;
            }
        }
    """

    set_titles = map((i) -> """
        var parent = d3.select("#$(figure.id)_parent" + $i)
        parent.append("text")
                    .attr("x", tile_width / 2)
                    .attr("y", 16)
                    .attr("text-anchor", "middle")
                    .text("$(figure.titles[i])")
                    .style("font-size", 16);""", 1:length(figure.titles))
    javascript = join(vcat([setup], set_titles), "\n")
    HTML(javascript)
end


struct JupyterInlineRenderer
    name::String # The target name for Javascript
    dom_element_id::Nullable{String} # The DOM element where the JS code should render to
    comm::IJulia.Comm # communication object to Javascript
    configuration::Dict # gets sent to the JS renderer alongside the trace (the JS renderer is currently stateless)

    function JupyterInlineRenderer(name::String, configuration::Dict)
        comm = IJulia.Comm(name, data=Dict())
        new(name, Nullable{String}(), comm, configuration)
    end
end

#
#
#
#
#
#
#
#
#
#

function attach(renderer::JupyterInlineRenderer, dom_element_id::String)
    renderer.dom_element_id = dom_element_id
end

#function attach(renderer::JupyterInlineRenderer, figure::Figure)
    #renderer.dom_element_id = "$(figure.id)_frame1"
#end
#
#function attach(renderer::JupyterInlineRenderer, subfigure::Pair{Figure,Int})
    #renderer.dom_element_id = "$(subfigure.first.id)_frame$(subfigure.second)"
#end

function render(renderer::JupyterInlineRenderer, trace::Trace) # TODO handle DifferentiableTrace
    local id::String
    global active_viewport
    if (isnull(renderer.dom_element_id))
        if (active_viewport == "undefined")
            error("No viewport has been defined")
        end
        # use global active viewport
        id = active_viewport + "_frame1"
    else
        # use explicitly attached viewport
        id = get(renderer.dom_element_id)
    end
    IJulia.send_comm(renderer.comm, Dict("trace" => trace,
										 "dom_element_id" => id,
                                         "conf" => renderer.configuration))
end

macro javascript_str(s) display("text/javascript", s); end

CSS(str::String) = HTML("<style>$(str)</style>")

export CSS
export @javascript_str

export enable_inline
export JupyterInlineRenderer
export TiledJupyterInlineRenderer
export TiledJupyterInlineRenderer
#export inline
export viewport
export render
export attach

export Figure
export here
export id
export get_active_viewport
