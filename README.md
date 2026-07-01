# plasma_etch_simulation

## **Imporant** - General instructions for working on any git repository
1) Craeting a new branch - whenever you want to add a new feature in a code base and about to file a PR, always create a branch with a this syntax \<yourname/abbrevation\>_\<purpose\>. For example if I want to add parsing feature, I will name my branch as 
utk_parsing. 
2) Rebasing - Since the main repository is continously evolving, daily rebase your current working
with the main branch, that will make filing PR a lot easier.
3) Filling a merge/pull request - Make sure you current branch is first rebased with the target (main) branch and then file the PR.
4) While filing a new branch, also make sure to update the [task section](## Tasks) section below. No need to update the current state section.

## General programming pattern to follow
- write functions with brief description of what they are doing, no need to get into details
- to demostrate what a function does or set of them do, explain in a jupyter document
- try to name the function which gives an idea of what it is trying to calculate
- during the development, it is ok to write functions in the jupyter notebook, for after the dev and testing is done, transfer the functions to approprate file and import them in the notebook
- name the notebook according the demonstration or results it is displaying
----------------------------------------------------------------------------------------------------

## Current status

Update so far 


## Tasks

### Task 1:
- Transfer the functions in medium_git.ipynb and particle_in_box.ipynb to src/particle_simulation.py
- Rename the functions such to get a more clear idea of what they intend to do, a good pattern to follow is write functions in a class which can provide further context : for example getAcc can be renamed to getAcceleration to get_acceleration. if you write something like
    ```
    class ParticleSim:
        def getAcceleration():
        pass
    ```
    that is even better, since it tells we want to know acceleration of particles.
- Make sure follow common cases here, i.e eitherUseThis also know as camelCase or use_this_style also knows as snake_case.

### Task 2 :
- No simulator or model is good without proof of it's validity. Find the real use case and data of the functions you have written and test them against the real world data, or an existing paper and check if your function returns the same observables.
- Define a clear set of observables that can be plotted in a graph and verified (for example velocity and temperature distribution) for general particle simulation as well as sheath formation ( which is our ultimate goal)
- Start with a simple 1-d sheath simulation, find relevent sources with verifiable data and then simulate you sheath model to check if it fits real world data

### Task 3 :
- Study the sheath formation with respect to distance from the wall and get a plot of electron and positive ion density, electric potenital and approximate value of sheath length. Also compare the paramters with existing litterature
- Write a function that simulates plasma and take following inputs
    - pressure, plasma gases ( or molecular mass of gas), electron temperature, Ion temepearture
    - outputs the potential and density of the gas with respect to space (1D and 2-D), time and sheat length
- Develop a 1-D (and later 2-D) dynamic model of plasma sheath where gas is supplied continuously with a given flow rate (the function paramters will be flow rate of gas, gas properties and temperature (this is not same as electron temperature)) 

## Task 4 : 
- (In case all previous tasks are done) Develop a model where plasma gasaccelerates via sheath potential and calculate corresponding flux 
- In addition to sheath potential, simulate gas flow with external potential applied 