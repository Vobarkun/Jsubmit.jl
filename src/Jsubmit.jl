"""
    Jsubmit

Run Julia functions as Slurm batch jobs, with the arguments and results of a job kept in one
JLD2 file next to its batch script.

[`export_job`](@ref) writes (and optionally submits) a job, [`runjob`](@ref) is what the
generated script executes on the compute node, and [`loadoutputs`](@ref) reads the results
back.
"""
module Jsubmit

using JLD2, Dates, FileWatching

"""
    numpath(path; sep = "_") -> String

Append the first free number to `path`, keeping its extension, so that the result does not
name an existing file: `"run.jld2"` becomes `"run_1.jld2"`, or `"run_2.jld2"` if
`"run_1.jld2"` already exists.
"""
function numpath(path; sep = "_")
    name, ext = splitext(path)
    i = 1
    while isfile(name * "$sep$i" * ext) 
        i += 1
    end
    return name * "$sep$i" * ext
end

"""
    export_job(path, func, args, kwargs; kwargs...)

Write a job for `func(args...; kwargs...)` to `\$path.sh` (the Slurm batch script) and
`\$path.jld2` (its arguments), and optionally submit it. The job is run by
[`runjob`](@ref), which appends the result to the same `.jld2` file.

# Keyword arguments
- `deduplicate = true`: pick the first free numbered `path` via [`numpath`](@ref) instead of
  overwriting an existing job.
- `metadata = nothing`: stored alongside the arguments in the `.jld2` file.
- `writejld2 = true`: set to `false` to regenerate only the batch script.
- `submit = nothing`: `"owl"` or `"moa"` to `sbatch` the script over ssh on that cluster.
- `excludebadgpus = false`: exclude known-bad GPU nodes from the submission.
- `partition = "p16,p20,p32,p40"`: Slurm partitions to run on.
- `usecheckpoints = false`: pass the last stored output back to `func` as the `checkpoint`
  keyword, so a resubmitted job continues where the previous one stopped.
- `storecode = false`: store a [`codesnapshot`](@ref) of `func`'s package, taken when the job
  starts, as the extra output field `code`.
- remaining keyword arguments are forwarded to [`jobscript`](@ref).
"""
function export_job(path, func, args, kwargs; deduplicate = true, metadata = nothing, writejld2 = true, submit = nothing, excludebadgpus = false, partition = "p16,p20,p32,p40", usecheckpoints = false, storecode = true, jobkwargs...)
    if deduplicate path = splitext(numpath("$path.jld2"))[1] end
    jobname = splitdir(path)[2]
 
    println("Writing $path.sh"); flush(stdout)
    export_jobscript("$path.sh", jobname, func; partition, jobkwargs...)
    writejld2 && println("Writing $path.jld2"); flush(stdout)
    writejld2 && jldsave("$path.jld2"; args, kwargs, metadata, usecheckpoints, storecode)
    if !isnothing(submit)
        dir = abspath(splitdir(path)[1])
        excludecommand = excludebadgpus ? "--exclude=\$(sinfo -p $partition -o %40G%N | grep 'x\\-10\\|x\\-20\\|4000' | cut -c 41- | paste -sd ',' -)" : ""
        cmd = if submit == "owl"
            `ssh owl5 "source /etc/profile; module load slurm; cd $dir; sbatch $excludecommand $jobname.sh"`
        elseif submit == "moa"
            cmd = `ssh moa1 "cd $dir; sbatch $excludecommand $jobname.sh"`
        end
        run(pipeline(cmd, stderr = IOBuffer()))
    end
    return "$path.sh"
end

"""
    export_jobscript(path, args...; kwargs...)

Write the batch script built by [`jobscript`](@ref)`(args...; kwargs...)` to `path`.
"""
function export_jobscript(path, args...; kwargs...)
    open(path, "w") do io
        write(io, jobscript(args...; kwargs...))
    end
end

"""
    printlog(x...)

Print `x` to `stdout` prefixed with the current time and flush.
"""
function printlog(x...)
    println(rpad(string(now()), 23, " "), "  ", join(x, " ")...)
    flush(stdout)
end

"""
    outputfields(output) -> (names, values)

Split a job result into the datasets [`runjob`](@ref) should store it as.

| `output` | `names` |
|:---|:---|
| `(a = 1, b = 2)` | `["a", "b"]` |
| `(result, (loss = 0.1, n = 7))` | `["output", "loss", "n"]` |
| anything else | `["output"]` |

Only a two-element tuple whose *second* element is a `NamedTuple` takes the second form; a
plain tuple is stored as a single output.
"""
outputfields(output) = (["output"], Any[output])
outputfields(output::NamedTuple) = (collect(string.(keys(output))), Any[Base.values(output)...])
function outputfields(output::Tuple{Any, NamedTuple})
    names, vals = outputfields(output[2])
    return (["output"; names], pushfirst!(vals, output[1]))
end

"""
    codesnapshot(mod) -> Union{Dict{String, String}, Nothing}

Contents of the files under `src/` of the package defining `mod`, keyed by path relative to
the package root, plus the active environment's `Project.toml`/`Manifest.toml` (under
`env/`) and, if the package is a git repository, its commit and uncommitted diff (under
`git/`). Returns `nothing` if `mod` does not belong to a package.
"""
function codesnapshot(mod)
    root = pkgdir(Base.moduleroot(mod))
    isnothing(root) && return nothing
    files = Dict{String, String}()
    for (dir, _, fs) in walkdir(joinpath(root, "src")), f in fs
        p = joinpath(dir, f)
        files[relpath(p, root)] = read(p, String)
    end
    project = Base.active_project()
    if !isnothing(project)
        for f in ("Project.toml", "Manifest.toml")
            p = joinpath(dirname(project), f)
            isfile(p) && (files["env/$f"] = read(p, String))
        end
    end
    try
        files["git/commit"] = readchomp(pipeline(Cmd(`git rev-parse HEAD`; dir = root), stderr = devnull))
        files["git/diff"] = read(pipeline(Cmd(`git diff HEAD`; dir = root), stderr = devnull), String)
    catch
    end
    return files
end

"""
    runjob(func, path)

Run the job stored in the JLD2 file `path` and append its result to that same file. This is
what the generated batch script calls on the compute node.

The arguments are read back from `path` and `func(args...; kwargs...)` is called; if the job
was exported with `usecheckpoints = true`, the most recent stored output is passed along as
the `checkpoint` keyword. If it was exported with `storecode = true`, a
[`codesnapshot`](@ref) is taken before `func` is called and stored as the field `code`.

The result is written under a pidlock (`\$path.pid`), so concurrent array tasks do not
corrupt the file. It is split into datasets by [`outputfields`](@ref) and each one is
numbered: inside a Slurm array without checkpoints with `SLURM_ARRAY_TASK_ID`, otherwise
with the first free index shared by all fields. Read the results back with [`loadoutputs`](@ref).
"""
function runjob(func, path)
    printlog("Processing $(path) on $(gethostname()) with $(Threads.nthreads()) threads")

    m = parentmodule(func)
    if isdefined(m, :CUDA) && m.CUDA.functional()
        devnames = [m.CUDA.name(d) for d in m.CUDA.devices()]
        devstrings = ["$(sum(devnames .== n)) × $n" for n in unique(devnames)]
        printlog("Using $(join(devstrings, ", "))")
    end

    @load path args kwargs
    usecheckpoints, storecode = jldopen(path, "r") do f
        get(f, "usecheckpoints", false), get(f, "storecode", false)
    end
    code = storecode ? codesnapshot(parentmodule(func)) : nothing
    storecode && isnothing(code) && printlog("No package found for $(parentmodule(func)), not storing code")

    output = if usecheckpoints
        checkpoint = jldopen(path, "r") do file
            is = [parse(Int, k[length("output")+1:end]) for k in keys(file) if occursin(r"^output\d+$", k)]
            isempty(is) && return nothing
            i = maximum(is)
            printlog("Resuming from $(path)[\"output$(i)\"]")
            return file["output$i"]
        end
        func(args...; checkpoint, kwargs...)
    else
        func(args...; kwargs...)
    end

    FileWatching.mkpidlock(splitext(path)[1] * ".pid") do
        jldopen(path, "a+") do file
            names, vals = outputfields(output)
            if !isnothing(code)
                push!(names, "code"); push!(vals, code)
            end
            suffix = if !usecheckpoints && haskey(ENV, "SLURM_ARRAY_TASK_ID")
                ENV["SLURM_ARRAY_TASK_ID"]
            else
                i = 1
                while any(haskey(file, "$name$i") for name in names)
                    i += 1
                end
                string(i)
            end
            savekeys = ["$name$suffix" for name in names]
            printlog("Saving to $(path)[$(join(savekeys, ", "))]")
            for (key, value) in zip(savekeys, vals)
                file[key] = value
            end
        end
    end

    printlog("$(splitext(path)[1]) finished.")
end

"""
    jobscript(jobname, jobfunc; kwargs...) -> String

Build the Slurm batch script that runs `Jsubmit.runjob(jobfunc, "\$jobname.jld2")`, with
`jobfunc` addressed through its parent module, so that module must be importable from the
job's project directory.

# Keyword arguments
- `narray = nothing`: number of array tasks; `nothing` submits a single job. Array jobs log
  with `--open-mode=append`.
- `nconcurrent = 1`: how many array tasks may run at once. More than one gives every task
  its own log file, named after its array index.
- `exclusive = true`: request whole nodes.
- `partition = "p16,p20,p32,p40"`: Slurm partitions to run on.
- `ngpus = 0`, `gputype = ""`: GPU count and, optionally, the type to request.
- `memory = exclusive ? "0" : "20G"`: memory per node; `"0"` means all of it.
- `ncpus = nothing`: CPUs per task; `nothing` leaves the Slurm default.
"""
function jobscript(jobname, jobfunc;
        narray = nothing, nconcurrent = 1,
        exclusive = true, partition = "p16,p20,p32,p40",
        ngpus = 0, gputype = "", memory = exclusive ? "0" : "20G",
        ncpus = nothing
    )
    array = isnothing(narray) ? "" : "\n#SBATCH --array=1-$(narray)%$(nconcurrent)"

    function_string = string(nameof(jobfunc))
    module_string = string(parentmodule(jobfunc))
    command = "import Jsubmit, $module_string; Jsubmit.runjob($module_string.$function_string, \"$jobname.jld2\")"

    """
    #!/bin/bash

    #SBATCH --partition $partition    
    #SBATCH --dependency=singleton
    #SBATCH --time=24:00:00

    #SBATCH --nodes=1
    $(isnothing(ncpus) ? "" : "#SBATCH --cpus-per-task=$ncpus")
    #SBATCH --gpus=$gputype$(isempty(gputype) ? "" : ":")$ngpus
    #SBATCH --mem=$memory

    #SBATCH --job-name=$jobname
    #SBATCH --mail-type=NONE
    #SBATCH --output=%x$(nconcurrent > 1 ? "_%a" : "").out
    #SBATCH --error=%x$(nconcurrent > 1 ? "_%a" : "").out
    $(isnothing(narray) ? "" : "#SBATCH --open-mode=append")

    $(exclusive ? "#SBATCH --exclusive\n" : "")$array
    
    srun julia --project=@. --threads=auto -e \'$command\'
    """
end

"""
    loadoutputs(path; name = "output") -> Vector

Load every result [`runjob`](@ref) stored in the JLD2 file `path`, ordered by their index.
Pass `name` (a `String` or `Symbol`) to read one of the extra named fields a job returned
instead of the plain output, e.g. `loadoutputs(path; name = :loss)`.
"""
function loadoutputs(path; name = "output")
    name = string(name)
    jldopen(path, "r") do file
        ks = filter(k -> k == name || occursin(Regex("^$(name)\\d+\$"), k), keys(file))
        sortkey(k) = k == name ? -1 : parse(Int, k[length(name)+1:end])
        return [file[k] for k in sort(ks, by=sortkey)]
    end
end

export export_job

end