define(['require'], function(require) {

    function load_ipython_extension(){
        console.info('Loading Gen extension');
    }

    function find_choice(trace, name) {
        if (name in trace.recorded) {
            return { value: trace.recorded[name], where: "recorded" }
        } else if (name in trace.interventions) {
            return { value: trace.interventions[name], where: "interventions"}
        } else if (name in trace.constraints) {
            return { value: trace.constraints[name], where: "constraints"}
        } else {
            return null
        }
    }

    function register_jupyter_renderer(name, render_function) {
        Jupyter.notebook.kernel.comm_manager.unregister_target(name);
        Jupyter.notebook.kernel.comm_manager.register_target(name, function(comm, msg) {
            comm.on_msg(function(msg) {
                var id = msg.content.data.dom_element_id;
                var trace = msg.content.data.trace;
                var conf = msg.content.data.conf;
                render_function(id, trace, conf);
            })
        });
        
    }


    return {
        load_ipython_extension: load_ipython_extension,
        find_choice: find_choice,
        register_jupyter_renderer: register_jupyter_renderer,
        recorded: "recorded",
        interventions: "interventions",
        constraints: "constraints"
    };
});
