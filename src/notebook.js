Gen = {

    find_choice: function(trace, name) {
        if (name in trace.recorded) {
            //console.log(name + " found in interventions");
            return { value: trace.recorded[name], where: "recorded" }
        } else if (name in trace.interventions) {
            //console.log(name + " found in interventions");
            return { value: trace.interventions[name], where: "interventions"}
        } else if (name in trace.constraints) {
            //console.log(name + " found in constraints");
            return { value: trace.constraints[name], where: "constraints"}
        } else {
            return null
        }
    },

    register_jupyter_renderer: function(name, render_function) {
        
        Jupyter.notebook.kernel.comm_manager.unregister_target(name);
        Jupyter.notebook.kernel.comm_manager.register_target(name, function(comm, msg) {
            comm.on_msg(function(msg) {
                var id = msg.content.data.dom_element_id;
                var trace = msg.content.data.trace;
                render_function(id, trace);
            })
        });
        
    }
};
