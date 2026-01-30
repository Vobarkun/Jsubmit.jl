module Jsubmit

using JLD2, Dates

function export_job(path, func, args, kwargs; metadata = nothing, writejld2 = true, submit = nothing, jobkwargs...)
    jobname = splitdir(path)[2]

    export_jobscript("$path.sh", jobname, func; jobkwargs...)
    writejld2 && jldsave("$path.jld2"; args, kwargs, metadata)
    if !isnothing(submit)
        dir = abspath(splitdir(path)[1])
        cmd = if submit == "owl"
            `ssh owl5 "source /etc/profile; module load slurm; cd $dir; sbatch $jobname.sh"`
        elseif submit == "moa"
            cmd = `ssh moa1 "cd $dir; sbatch $jobname.sh"`
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
    @load path args kwargs

    output = func(args...; kwargs...)

    printlog("Saving to $(path)")
    try
        jldopen(file -> (file["output"] = output), path, "a+")
    catch
        jldsave(path; args, kwargs, output)
    end
    printlog("$(splitext(path)[1]) finished.")
end

function jobscript(jobname, jobfunc; 
        narray = nothing, nconcurrent = 1, 
        exclusive = true, partition = "short,p40,p32,p16", 
        ngpus = 0, gputype = "", 
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

    #SBATCH --job-name=$jobname
    
    #SBATCH --mail-type=NONE
    #SBATCH --output=%x$(nconcurrent > 1 ? "_%a" : "").out
    #SBATCH --error=%x$(nconcurrent > 1 ? "_%a" : "").out

    $(exclusive ? "#SBATCH --exclusive\n" : "")$array
    
    julia --project=@. --threads=auto -e \'$command\'
    """
end

export export_job

end