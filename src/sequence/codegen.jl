 # given a probabilistic program in the following form:

using Gen

function isa_random_choice(expr)
   return (expr.head == :macrocall) && (length(expr.args) >= 1) && (expr.args[1] == Symbol("@name"))
end

function get_random_choice_name(expr)
    @assert expr.head == :macrocall
    if length(expr.args) != 3
        error("invalid random choice name length")
    end
    @assert isa(expr.args[3], Symbol)
    return expr.args[3]
end

function transform_expr(expr, score_symbol, score_set::Dict{Symbol,Symbol})
    if !isa(expr, Expr)
        new_expr = expr
    elseif isa_random_choice(expr)
        if length(expr.args) != 3
            error("invalid @name syntax (length(expr.args) = $(length(expr.args)))")
        end
        random_choice_expr = expr.args[2]
        if random_choice_expr.head != :call
            error("invalid @name syntax (random_choice_expr.head = $(random_choice_expr.head))")
        end
        generator_name = random_choice_expr.args[1]
        generator_args = random_choice_expr.args[2:end]
        name = get_random_choice_name(expr)
        if haskey(score_set, name)
            new_expr = Expr(:block,
                Expr(:(+=) , score_symbol, Expr(:call, :logpdf, generator_name, score_set[name], generator_args...)),
                score_set[name])
        else
            new_expr = Expr(:call, :rand, generator_name, generator_args...)
        end
    else
        if expr.head == :return
            error("no return statements allowed")
        end
        new_expr = Expr(expr.head, [transform_expr(subexpr, score_symbol, score_set) for subexpr in expr.args]...)
    end
    return new_expr
end

function process_function_sig(ast::Expr, query_function_name::Symbol, score_symbol::Symbol, score_set::Dict{Symbol,Symbol})
    if ast.head != :function
        error("the probabilistic program must be a function definition")
    end
    args = ast.args[1]
    @assert args.head == :tuple
    body = ast.args[2]
    new_args = Any[]
    for value_symbol in values(score_set)
        push!(new_args, value_symbol)
    end
    for arg in args.args
        push!(new_args, arg)
    end
    @assert body.head == :block
    new_block = Any[]
    for expr in body.args
        new_expr = transform_expr(expr, score_symbol, score_set)
        push!(new_block, new_expr)
    end
    result = gensym()
    new_body = Expr(:block, Expr(:(=), score_symbol, :(0.)), Expr(:(=), result, Expr(:block, new_block...)), Expr(:tuple, score_symbol, result))
    function_expr = Expr(:function, Expr(:call, query_function_name, new_args...), new_body)
    println("\n=========== OPTIMIZED QUERY FUNCTION ==============:")
    println(function_expr)
    function_expr
end

function transform(ast, pattern, query_function_name::Symbol)
    score_set = Dict{Symbol,Symbol}()
    for name in pattern
        score_set[name] = gensym()
    end
    score_symbol = gensym()
    process_function_sig(ast, query_function_name, score_symbol, score_set)
end

# map from generator type (i.e. MyGenerator) to a tuple of symbols defining a query that has been generated
global GENERATOR_PROGRAMS = Dict{Type,Expr}()
global GENERATOR_METHODS = Dict{Type,Dict{Tuple,Symbol}}()

macro make_generator(generator_name, program)

    # generate the type
    eval(quote
            struct $generator_name end
            GENERATOR_METHODS[$generator_name] = Dict{Tuple,Symbol}()
    end)
    
    # store the generator AST
    eval(Expr(:(=), Expr(:ref, :GENERATOR_PROGRAMS, generator_name), Expr(:quote, program)))

    nothing
end

macro query(generator_name, params, args...)
    scored = Dict{Symbol,Any}()
    for arg in args
        if arg.head != :(=) || length(arg.args) != 2
            error("invalid query syntax at argument: $arg")
        end
        scored[arg.args[1]] = arg.args[2]
    end
    generator_type = eval(generator_name)
    name_symbols = tuple(sort(collect(keys(scored)))...)
    if haskey(GENERATOR_METHODS, generator_type)
        if haskey(GENERATOR_METHODS[generator_type], name_symbols)
            query_function_name = GENERATOR_METHODS[generator_type][name_symbols]
        else
            query_function_name = gensym()
            GENERATOR_METHODS[generator_type][name_symbols] = query_function_name
            ast = GENERATOR_PROGRAMS[generator_type]
            transformed_ast = transform(ast, name_symbols, query_function_name)
            eval(quote
                $transformed_ast
            end)
        end
    else
        error("generator type $generator_name not yet defined")
    end
    new_args = Any[]
    for name in name_symbols
        push!(new_args, scored[name])
    end
    @assert params.head == :tuple
    for param in params.args
        push!(new_args, param)
    end
    Expr(:call, query_function_name, new_args...)
end

# define the generator by giving a probablistic program for it
# TODO suport for code-generaton of dynamic addresses? or addresses in for loops at least?
@make_generator(
    MyGenerator,
    function (mu::Float64)
        x = @name(normal(mu, 1), foo)
        if x > 0
            y = @name(normal(mu, 1), bar) + x
        else
            y = 1
        end
        z = @name(normal(x + y, 1), obs)
        z
    end)

# Perform queries of the form e.g. Prob(obs=0.5). Each query triggers a
# 'lightweight' evaluation of the generator's probabilistic program, scoring
# those variables given in the arguments to the query (e.g. obs). We avoid name
# lookups at runtime by generating functions optimized for answering each query
# pattern (i.e. set of scored random choices). Even if the same query pattern
# is used many times, the query function is only generated once.
(score, value) = @query(MyGenerator, (0.3,), obs=0.5)
println("score: $score, value: $value")
(score, value) = @query(MyGenerator, (0.3,), obs=0.5, foo=0.1)
println("score: $score, value: $value")
(score, value) = @query(MyGenerator, (0.3,), obs=0.5)
println("score: $score, value: $value")
