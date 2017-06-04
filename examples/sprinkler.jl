using Gen

# We set a seed to make the results reproducible:

srand(1)

# Probabilistic programs in Gen

# We begin by learning to use Gen to write probabilistic models, run traced
# executions of these models, and inspect and render (visualize) the traces of
# the programs to characterize the distribution on traces of he program.

# Writing a probabilistic program

# We will now write a simple probabilistic program in Gen, and trace its
# execution. A probabilistic program in Gen is simply a Julia function whose
# first argument is `T::Trace`, and in which some random choices have been tagged
# with names using the `~` syntax:

function sprinkler_model(T::Trace)
	cloudy = flip(0.5) ~ "cloudy"
	sprinkler = flip(cloudy ? 0.1 : 0.5) ~ "sprinkler"
	rain = flip(cloudy ? 0.8 : 0.2) ~ "rain"
	wetgrass = flip(
		if sprinkler
			rain ? 0.99 : 0.9
		else
			rain ? 0.9 : 0.01
		end) ~ "wetgrass"
end

# Tracing a probabilistic program

# We next create an empty trace object, and run the program, recording its
# random choices in the trace:

sprinkler_trace = Trace()
sprinkler_model(sprinkler_trace)

# `sprinkler_trace` now contains a random trace of the program
# `sprinkler_model`

# Retrieving values from the trace

println("cloudy=$(sprinkler_trace.vals["cloudy"])")
println("sprinkler=$(sprinkler_trace.vals["sprinkler"])")
println("rain=$(sprinkler_trace.vals["rain"])")
println("wetgrass=$(sprinkler_trace.vals["wetgrass"])")

# Problem 1.1

# List all possible traces for the `sprinker_model` program above, and the
# probability for each trace. Check that they sum to 1.0. Feel free to generate
# the list programatically.

# Problem 1.2

# Give a directed acyclic graph (DAG) expressing the conditional independencies
# between random choices in the above `sprinkler_model` program (i.e. the
# Bayesian network for the probabilistic model). Specify the DAG by its
# vertices and edges.

# A program with an unbounded number of random choices:

function foo(T::Trace, n::Int)
	if (flip(0.5) ~ "flip_$n")
		foo(T, n + 1)
	else
		n
	end
end

function recursion_program(T::Trace)
	return foo(T,0)
end

# Note that probabilitsic programs can use sub-routines, and these sub-routines
# can have traced random choices, along as they also have the T::Trace as their
# first argument

# One traced execution of recursion_program
println("\nFirst execution")
recursion_trace = Trace()
recursion_program(recursion_trace)
for key in keys(recursion_trace.vals)
	println("$key=$(recursion_trace.vals[key])")
end

# Another traced execution
println("\nSecond execution")
recursion_trace = Trace() 
recursion_program(recursion_trace)
for key in keys(recursion_trace.vals)
	println("$key=$(recursion_trace.vals[key])")
end

# Note that the number of random choices that were sampled is different between
# the two executions. Unlike Bayesian networks, probabilistic programs can
# represent models in which (1) the set of which random choices are sampled is
# itself a random object, and (2) the set of random choices is potentially
# countably infinite

# We next perform 20 independent traced executions of the program, and list the
# value of the output in each.

return_values = []
for i=1:20
	push!(return_values, recursion_program(Trace()))
end
println("return values: $return_values")

# Probabilistic programs can be parameterized by adding arguments.
# Here, we parameterize the recursion_program by the probability of each flip
# resulting in heads, and run it again:

function foo(T::Trace, n::Int, prob_heads::Float64)
	if (flip(prob_heads) ~ "flip_$n")
		foo(T, n + 1)
	else
		n
	end
end

function recursion_program(T::Trace, prob_heads::Float64)
	return foo(T,0, prob_heads)
end

return_values = []
for i=1:20
	push!(return_values, recursion_program(Trace()))
end
println("return values: $return_values")

# Unsurprisingly, changing the p parameter has a pretty drastic effect on the
# distribution of values seen in the output.


# Problem 1.3

# Recall that a trace is the set of random choices and their values.

# (a) List the set of possible traces for the recursion_example program above,
# the probability of each possible trace, and the value of output for each trace.
# Do the probabilities sum to 1?

# (b) Describe the marginal distribution of output.

# (c) Can you draw a Bayesian network for this program?


