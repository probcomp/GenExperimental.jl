using Gen
using PyPlot

srand(1)

function logistic_regression(T::Trace, features::Array{Float64,1})
    score = (0.0 ~ "bias")
    for i=1:length(features)
        score += features[i] * (0.0 ~ "w$i")
    end
    output = flip(1.0 / (1.0 + exp(-score))) ~ "output"
end

function render_logistic_regression(features, outputs, bias, weights, fname)
    plt[:figure](figsize=(6, 3))
    xs = map((feat) -> feat[1], features)
    ys = map((feat) -> feat[2], features)
    plt[:subplot](1, 2, 1)
    colors = map((output) -> output ? "purple" : "orange", outputs)
    plt[:scatter](xs, ys, c=colors)
    plt[:title]("training data")
    plt[:subplot](1, 2, 2)
    w = 100
    h = 100
    grid = Array{Float64,2}(w, h)
    for (i, x) in enumerate(linspace(-3, 3, w))
        for (j, y) in enumerate(linspace(-3, 3, h))
            trace = Trace()
            intervene!(trace, "bias", bias)
            for k=1:num_features
                intervene!(trace, "w$k", weights[k])
            end
            constrain!(trace, "output", true)
            logistic_regression(trace, [x, y])
            grid[i, j] = exp(score(trace)) # prob true
            delete!(trace, "output")
        end
    end
    plt[:imshow](grid, interpolation="none", cmap="PuOr", origin="lower")
    plt[:title]("predictions")
    plt[:tight_layout]()
    plt[:savefig](fname)
end

function mlp(T::Trace, features::Array{Float64,1}, num_hidden::Int)
    score = (0.0 ~ "bias")
    for hidden=1:num_hidden
        loading = (0.0 ~ "bias-$hidden")
        for input=1:length(features)
            loading += features[input] * (0.0 ~ "weight-$hidden-$input")
        end
        activation = 1.0 / (1.0 + exp(-loading))
        score += activation * (0.0 ~ "weight-$hidden")
    end
    output = flip(1.0 / (1.0 + exp(-score))) ~ "output"
end

function render_mlp(features, outputs, output_bias, output_weights, hidden_bias, hidden_weights, fname)
    plt[:figure](figsize=(6, 3))
    xs = map((feat) -> feat[1], features)
    ys = map((feat) -> feat[2], features)
    plt[:subplot](1, 2, 1)
    ax = plt[:gca]()
    colors = map((output) -> output ? "purple" : "orange", outputs)
    plt[:scatter](xs, ys, c=colors, s=10)
    ax[:set_xlim]((-8, 8))
    ax[:set_ylim]((-8, 8))
    plt[:title]("training data")
    plt[:subplot](1, 2, 2)
    width = 100
    height = 100
    grid = Array{Float64,2}(width, height)
    for (i, x) in enumerate(linspace(-8, 8, width))
        for (j, y) in enumerate(linspace(-8, 8, height))
            trace = Trace()
            intervene!(trace, "bias", output_bias)
            for h=1:num_hidden
                intervene!(trace, "weight-$h", output_weights[h])
                intervene!(trace, "bias-$h", hidden_biases[h])
                for k=1:num_features
                    intervene!(trace, "weight-$h-$k", hidden_weights[h, k])
                end
            end
            constrain!(trace, "output", true)
            mlp(trace, [x, y], num_hidden)
            grid[i, j] = exp(score(trace)) # prob true
            delete!(trace, "output")
        end
    end
    plt[:imshow](grid', interpolation="none", cmap="PuOr", origin="lower", extent=[-8, 8, -8, 8])
    plt[:title]("predicted probability")
    plt[:tight_layout]()
    plt[:savefig](fname)
end


# synthetic training data set
num_train = 100
std = 1.
num_features = 2
features = []
outputs = []
for i=1:num_train
    if flip(0.5)
        push!(outputs, true)
        if flip(0.5)
            if flip(0.5)
                push!(features, [normal(-2.,std), normal(2.,std)])
            else
                push!(features, [normal(-5.,std), normal(-5.,std)])
            end
        else
            if flip(0.5)
                push!(features, [normal(2.,std), normal(-2.,std)])
            else
                push!(features, [normal(4.,std), normal(5.,std)])
            end
        end
    else
        push!(outputs, false)
        if flip(0.5)
            push!(features, [normal(2.,std), normal(2.,std)])
        else
            push!(features, [normal(-2.,std), normal(-2.,std)])
        end
    end
end 


# training logistic regression
bias = randn()
weights = randn(num_features)
step_size = 0.01
for iter=1:1
    log_probability = 0.0
    grad_bias = 0.0
    grad_weights = zeros(num_features)
    for i=1:num_train
        trace = Trace()
        parametrize!(trace, "bias", bias)
        for j=1:num_features
            parametrize!(trace, "w$j", weights[j])
        end
        constrain!(trace, "output", outputs[i])
        logistic_regression(trace, features[i])
        backprop(trace)
        log_probability += score(trace)
        grad_bias += d(trace, "bias")
        grad_weights += map((j) -> d(trace, "w$j"), 1:num_features)
    end
    
    println("objective: $log_probability")
    bias += step_size * grad_bias
    weights += step_size * grad_weights

    println(bias)
    println(weights)
    render_logistic_regression(features, outputs, bias, weights, "logreg_$iter.png")
end

# training multi-layer perceptron
num_hidden = 10
hidden_biases = randn(num_hidden)
hidden_weights = randn(num_hidden, num_features)
output_bias = randn()
output_weights = randn(num_hidden)
step_size = 0.01
for iter=1:1000
    log_probability = 0.0
    grad_hidden_biases = zeros(num_hidden)
    grad_hidden_weights = zeros(num_hidden, num_features)
    grad_output_bias = 0.0
    grad_output_weights = zeros(num_hidden)
    for i=1:num_train
        trace = Trace()
        parametrize!(trace, "bias", output_bias)
        for hidden=1:num_hidden
            parametrize!(trace, "weight-$hidden", output_weights[hidden])
            parametrize!(trace, "bias-$hidden", hidden_biases[hidden])
            for input=1:num_features
                parametrize!(trace, "weight-$hidden-$input", hidden_weights[hidden, input])
            end
        end
        constrain!(trace, "output", outputs[i])
        mlp(trace, features[i], num_hidden)
        backprop(trace)
        log_probability += score(trace)

        grad_hidden_biases += map((hidden) -> d(trace, "bias-$hidden"), 1:num_hidden)
        for hidden=1:num_hidden
            grad_hidden_weights[hidden,:] += map((input) -> d(trace, "weight-$hidden-$input"), 1:num_features)
        end
        grad_output_bias += d(trace, "bias")
        grad_output_weights += map((hidden) -> d(trace, "weight-$hidden"), 1:num_hidden)
    end
    
    println("objective: $log_probability")
    hidden_biases += step_size * grad_hidden_biases
    hidden_weights += step_size * grad_hidden_weights
    output_bias += step_size * grad_output_bias
    output_weights += step_size * grad_output_weights

    render_mlp(features, outputs,
               output_bias, output_weights, 
               hidden_biases, hidden_weights, "mlp_$iter.png")
end





