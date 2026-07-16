**Process Notes**

My fork: https://github.com/KathyHaem/probing-multilingual-dynamics

- unfortunately the processing of UD + UniMorph that this morphology probing needed has really poor documentation
- ⚠️ pay attention to path variables

- first had to run preprocessing (=saving embedding files) for *one layer of one model checkpoint* at a time
    - this seems to take several minutes per treebank (longer for larger models)
    - wrote a calling script to run a larger batch of those
    - they’re saving those embeddings to the `data/ud` folder, which is certainly…a choice
    - eventually I decided to implement batching for the pre-processing, which unfortunately was also quite a hassle. but I guess it is now faster to run.

- Running the probe itself
    - by default the neural probes get a maximum number of 5000 epochs and an early stopping patience of 50, with the first one stopping after 66 epochs. this training took a few minutes
    - then, they pick some number of dimensions, each time iterating through all of them to check the best performance jointly with the already-picked dimensions. this took 50 x 1.5 minutes per language per model per layer per attribute.
    - so I decided to be "super greedy" (over 'greedy') and implemented a version where it just takes the 50 best dimensions after the first iteration of checking performance. this is probably worse in some way but it does take out a factor of fifty soooo


-> i've run numbers for Llama3.2-3B and Aya-Expanse-8B I believe
-> did only POS, and three layers per model (first, middle, last)
    at some point I'd decided to use Number, Gender, and POS, but POS may well be enough (!)



Output location:
    `results/{model}/inter-layer-{layer}/{lang}/{attribute}`
    
    Pretty large JSON. Needed for use as CLA metric: List of the picked dimensions per language -> intersect
    
Still to do:
    
- [ ]  aggregate numbers from the outputs / test and debug the code that's meant to do that
- [ ]  COULD do it for more languages with a newer UD version but probably not worth the effort
