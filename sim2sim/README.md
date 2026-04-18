### Install
```bash
conda env create -f magiclab_mujoco.yaml -n magiclab_sim2sim
```

### Run

Run simulator:
```bash
python deploy_mujoco/z1_12dof_prop_mujoco.py --config=deploy_mujoco/configs/z1_12dof.yaml
```

Packages to install for keyboard control:
`sudo python -m pip install absl-py keyboard `

Test command input program (in another terminal):
```bash
python deploy_mujoco/gamepad_reader_btp.py
```
