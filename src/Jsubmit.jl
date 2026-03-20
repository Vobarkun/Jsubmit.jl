module Jsubmit

using JLD2, Dates, FileWatching

function numpath(path; sep = "_")
    name, ext = splitext(path)
    i = 1
    while isfile(name * "$sep$i" * ext) 
        i += 1
    end
    return name * "$sep$i" * ext
end

function export_job(path, func, args, kwargs; deduplicate = true, metadata = nothing, writejld2 = true, submit = nothing, excludebadgpus = false, partition = "p16,p20,p32,p40", usecheckpoints = false, jobkwargs...)
    if deduplicate path = splitext(numpath("$path.jld2"))[1] end
    jobname = splitdir(path)[2]
 
    println("Writing $path.sh"); flush(stdout)
    export_jobscript("$path.sh", jobname, func; partition, jobkwargs...)
    writejld2 && println("Writing $path.jld2"); flush(stdout)
    writejld2 && jldsave("$path.jld2"; args, kwargs, metadata, usecheckpoints)
    if !isnothing(submit)
        dir = abspath(splitdir(path)[1])
        excludecommand = excludebadgpus ? "--exclude=\$(sinfo -p $partition -o %40G%N | grep 'x\\-10\\|x\\-20\\|x\\-a' | cut -c 41- | paste -sd ',' -)" : ""
        cmd = if submit == "owl"
            `ssh owl5 "source /etc/profile; module load slurm; cd $dir; sbatch $excludecommand $jobname.sh"`
        elseif submit == "moa"
            cmd = `ssh moa1 "cd $dir; sbatch $excludecommand $jobname.sh"`
        end
        run(pipeline(cmd, stderr = IOBuffer()))
    end
    nothing
end

function export_jobscript(path, args...; kwargs...)
    open(path, "w") do io
        write(io, jobscript(args...; kwargs...))
    end
end

function printlog(x...)
    println(rpad(string(now()), 23, " "), "  ", join(x, " ")...)
    flush(stdout)
end

function runjob(func, path)
    printlog("Processing $(path) on $(gethostname()) with $(Threads.nthreads()) threads")

    m = parentmodule(func)
    if isdefined(m, :CUDA) && m.CUDA.functional()
        devnames = [m.CUDA.name(d) for d in m.CUDA.devices()]
        devstrings = ["$(sum(devnames .== n)) × $n" for n in unique(devnames)]
        printlog("Using $(join(devstrings, ", "))")
    end

    @load path args kwargs
    usecheckpoints = jldopen(f -> get(f, "usecheckpoints", false), path, "r")

    output = if usecheckpoints
        checkpoint = jldopen(path, "r") do file
            i = 0
            while haskey(file, "output$(i+1)")
                i += 1
            end
            i != 0 && printlog("Resuming from $(path)[\"output$(i)\"]")
            i != 0 ? file["output$i"] : nothing
        end
        func(args...; checkpoint, kwargs...)
    else
        func(args...; kwargs...)
    end

    FileWatching.mkpidlock(splitext(path)[1] * ".pid") do
        jldopen(path, "a+") do file
            key = if haskey(ENV, "SLURM_ARRAY_TASK_ID")
                "output$(ENV["SLURM_ARRAY_TASK_ID"])"
            else
                i = 1
                while haskey(file, "output$i")
                    i += 1
                end
                "output$i"
            end
            printlog("Saving to $(path)[\"$(key)\"]")
            file[key] = output
        end
    end

    printlog("$(splitext(path)[1]) finished.")
end

function jobscript(jobname, jobfunc; 
        narray = nothing, nconcurrent = 1, 
        exclusive = true, partition = "p16,p20,p32,p40", 
        ngpus = 0, gputype = "", memory = exclusive ? "0" : "20G"
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
    #SBATCH --gpus=$gputype$(isempty(gputype) ? "" : ":")$ngpus
    #SBATCH --mem=$memory

    #SBATCH --job-name=$jobname
    #SBATCH --mail-type=NONE
    #SBATCH --output=%x$(nconcurrent > 1 ? "_%a" : "").out
    #SBATCH --error=%x$(nconcurrent > 1 ? "_%a" : "").out

    $(exclusive ? "#SBATCH --exclusive\n" : "")$array
    
    julia --project=@. --threads=auto -e \'$command\'
    """
end

function loadoutputs(path)
    jldopen(path, "r") do file
        ks = filter(k -> k == "output" || occursin(r"^output\d+$", k), keys(file))
        sortkey(k) = k == "output" ? -1 : parse(Int, k[7:end])
        return [file[k] for k in sort(ks, by=sortkey)]
    end
end

export export_job

end