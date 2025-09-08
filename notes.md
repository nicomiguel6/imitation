
## Debug/Concern/Notes Log

### Author


### Date


### Description


### Expected Behavior


----------------------------------------

### Author
Nico 

### Date
September 08, 2025

### Description
Concern regarding efficacy of Behavioral Cloning to represent expert policy.

### Expected Behavior
Using BC to train an expert policy from demonstrations could prove ineffective for more complicated contexts. Should we consider using more involved imitation learning methods?

Something to note: Tagliabue's work uses entirely expert MPC as demonstrator.

----------------------------------------

### Author
Nico

### Date
September 08, 2025

### Description
TODO: Begin by generating expert demonstrations. To do this, we can utilize Python's OSQP library for solving MPCs, or an existing pre-trained (but unknown to algorithm) policy. 

----------------------------------------
