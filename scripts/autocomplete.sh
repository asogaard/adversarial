_submit()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="-h --help --gpu --train --tensorflow --devices --folds --config --patch --tag --username"

    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _submit -o default ./submit.sh

# @TODO: Conform to restructured code
_run()
{
    local cur prev opts
    COMPREPLY=()
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    opts="-h --help -i --input -o --output -c --config -p --patch --devices --folds -v --verbose -g --gpu --tensorflow --train --train-classifier --train-adversarial --optimise-classifier --optimise-adversarial --plot --tensorboard"

    if [[ ${cur} == -* ]] ; then
        COMPREPLY=( $(compgen -W "${opts}" -- ${cur}) )
        return 0
    fi
}
complete -F _run -o default ./run.py
