Causal Particle Filters
=======================

## thrun_localize.py

Attempts to reproduce particle filter video shown by Sebastian Thrun in ai-class particle filter video.

What I've learned from this exercise:
 - Computing the likelihood for each particle is the most time-consuming part of the algorithm.
    This is especially because of the inefficient ray-tracing used to simulate the range finders.
 - The likelihood function is not obvious -- it requires some thought about the physics
    of the environment to write.
 - Components of the likelihood require softening to improve robustness.
 - The population can get stuck in an incorrect region of the map.
    To escape, we may want to resample some portion of the population from a uniform prior each iteration.
    This is equivalent to the kidnapped robot problem.
 - This simulation could run in real-time by making the observation-resampling step asynchronous,
    but keeping the evolution step synchronous. The expensive likelihood computations could be
    performed on a distributed computer and integrated into the central machine (with the UI)
    once they are ready. This would require extending the filter to allow retroactive integration
    of the information returning from the satellite machines.

