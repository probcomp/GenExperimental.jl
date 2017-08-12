define(['require'], function(require) {

    function load_ipython_extension(){
        console.info('Loading Gen extension');
    }

    function register_jupyter_renderer(name, render_function) {
        Jupyter.notebook.kernel.comm_manager.unregister_target(name);
        Jupyter.notebook.kernel.comm_manager.register_target(name, function(comm, msg) {
            comm.on_msg(function(msg) {
                var id = msg.content.data.dom_element_id;
                var trace = msg.content.data.trace;
                var conf = msg.content.data.conf;
                var args = msg.content.data.args;
                render_function(id, trace, conf, args);
            })
        });
        
    }


    return {
        load_ipython_extension: load_ipython_extension,
        register_jupyter_renderer: register_jupyter_renderer
    };
});
