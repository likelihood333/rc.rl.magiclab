#!/usr/bin/env bash

export MAGICLAB_RL_LAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ENV_KIND="system"
if ! [[ -z "${CONDA_PREFIX}" ]]; then
    ENV_KIND="conda"
    python_exe=${CONDA_PREFIX}/bin/python
elif ! [[ -z "${VIRTUAL_ENV}" ]]; then
    ENV_KIND="venv"
    python_exe=${VIRTUAL_ENV}/bin/python
else
    python_exe=$(command -v python3)
    if [[ -z "${python_exe}" ]]; then
        echo "[Error] python3 is not found. Please activate a Python environment first."
        exit 1
    fi
    echo "[Warn] No conda/venv detected, fallback to ${python_exe}."
fi


# task env name autocomplete
_ut_rl_lab_python_argcomplete_wrapper() {
    local IFS=$'\013'
    local SUPPRESS_SPACE=0
    if compopt +o nospace 2> /dev/null; then
        SUPPRESS_SPACE=1
    fi

    COMPREPLY=( $(IFS="$IFS" \
                    COMP_LINE="$COMP_LINE" \
                    COMP_POINT="$COMP_POINT" \
                    COMP_TYPE="$COMP_TYPE" \
                    _ARGCOMPLETE=1 \
                    _ARGCOMPLETE_SUPPRESS_SPACE=$SUPPRESS_SPACE \
                    ${python_exe} ${MAGICLAB_RL_LAB_PATH}/scripts/rsl_rl/train.py 8>&1 9>&2 1>/dev/null 2>/dev/null) )
}
complete -o nospace -F _ut_rl_lab_python_argcomplete_wrapper "./magiclab_rl_lab.sh"


_ut_setup_conda_env() {
    if [[ -z "${CONDA_PREFIX}" ]]; then
        echo "[Info] Skip conda activation hook setup (non-conda environment)."
        return 0
    fi

    # copied from isaaclab/_isaac_sim/setup_conda_env.sh
    # add source magiclab_rl_lab.sh to conda activate.d
    printf '%s\n' '#!/usr/bin/env bash' '' \
        '# for Isaac Lab' \
        'export ISAACLAB_PATH='${ISAACLAB_PATH}'' \
        'alias isaaclab='${ISAACLAB_PATH}'/isaaclab.sh' \
        '' \
        '# show icon if not running headless' \
        'export RESOURCE_NAME="IsaacSim"' \
        '' \
        '# for magiclab_rl_lab' \
        'source '${MAGICLAB_RL_LAB_PATH}'/magiclab_rl_lab.sh' \
        '' > ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh

    # check if we have _isaac_sim directory -> if so that means binaries were installed.
    # we need to setup conda variables to load the binaries
    local isaacsim_setup_conda_env_script=${ISAACLAB_PATH}/_isaac_sim/setup_conda_env.sh

    if [ -f "${isaacsim_setup_conda_env_script}" ]; then
        # add variables to environment during activation
        printf '%s\n' \
            '# for Isaac Sim' \
            'source '${isaacsim_setup_conda_env_script}'' \
            '' >> ${CONDA_PREFIX}/etc/conda/activate.d/setenv.sh
    fi
}

# pass the arguments
case "$1" in
    -i|--install)
        git lfs install # ensure git lfs is installed
        ${python_exe} -m pip install -e ${MAGICLAB_RL_LAB_PATH}/source/magiclab_rl_lab/
        if [[ "${ENV_KIND}" == "conda" ]]; then
            _ut_setup_conda_env
            activate-global-python-argcomplete
        else
            echo "[Info] Non-conda environment detected. Skip conda hook setup."
            echo "[Info] If needed, install argcomplete completion manually:"
            echo "       ${python_exe} -m pip install argcomplete && activate-global-python-argcomplete"
        fi
        ;;
    -l|--list)
        shift
        ${python_exe} ${MAGICLAB_RL_LAB_PATH}/scripts/list_envs.py "$@"
        ;;
    -p|--play)
        shift
        ${python_exe} ${MAGICLAB_RL_LAB_PATH}/scripts/rsl_rl/play.py "$@"
        ;;
    -t|--train)
        shift
        ${python_exe} ${MAGICLAB_RL_LAB_PATH}/scripts/rsl_rl/train.py --headless "$@"
        ;;
    *) # unknown option
        ;;
esac
