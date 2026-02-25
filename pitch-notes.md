### Motivation

Cross-lingual alignment has been considered an important concept for multilingual models etc.

However there is still limited knowledge on this for decoders, and at least some papers found very little performance improvement by increasing alignment.

There are additionally quite a few different ways of measuring CLA which may or may not lead practitioners to the same conclusions.

[if possible: We want to understand the phenomena better, including by looking at subspaces (ah yeah, maybe Lens is one of the things to reproduce)…or indeed if the subspaces are used to effect]

[another related idea: cross-test encoders for [Eklectic](https://www.arxiv.org/pdf/2502.21228) to argue that decoders are indeed less multilingual??]

### Research Question(s)

- How well do different measures of cross-lingual alignment correlate with each other?
- How well do they correlate with downstream task performance?
- Are any of them more useful than others for predicting performance?

- How would we actually measure the subspace view best?
[- Are models able to use cross-lingually aligned subspaces effectively?]


### Approach

- pick downstream tasks to evaluate
    - POS/NER quite likely due to their data availability
    - PPL? → MEH because that’s not comparable across languages
    - MCQA possibly, preferably pick sth recent
    - Jindrich suggested SIB-200 -- **Potential problem:** the prompt is in english so the correlation might be trivial
    - facebook/belebele dataset -- skúsiť bez promptu; merať najpravdepodobnejšiu odpoveď
    - Preklad (flores) -- možno top6 vs. ostané? + chrF
    - Global MMLU (amerikocentrické)
    - Include https://arxiv.org/pdf/2411.19799v1
- koreluje performance s alignmentom s Angličtinou/Čínštinou (+priemer, maximum, ...)?
- pick languages — basically done; someone picking it up may look at it again??
- potential CLA metrics
    - [x]  xSIM https://github.com/facebookresearch/LASER/tree/main/tasks/xsim (has three settings, of which “absolute” is supposed to be equivalent to standard cosine similarity)
      > `xsim` Cross-lingual similarity search, also called xsim, evaluates the similarity between sentence embeddings across languages. Given a test dataset of bitexts, translations are encoded into the multilingual sentence embedding space and cosine similarity between all embeddings are computed. For each test instance, if the two corresponding translations are not the closest, we count it as an error in order to compute an error rate on the whole test set.
    - [x] "Morphology probing" from Stańczak et al. (I have some results from this but unfortunately not aggregated (yet)) <--- **skúsiť toto**
    - [ ]  maybe [Chang et al., 2022, Geometry of MLM Reps](https://aclanthology.org/2022.emnlp-main.9) (repo looks pretty usable at first glance) → may be useful for subspace view
    - [ ]  [Jones et al., 2021, Massively Multilingual Analysis of Cross-linguality] (I already worked with this previously, uses retrieval acc and average margin score…Gromov-Hausdorff etc. are scores of isomorphy though, which may also be cool?)
    - [ ]  Discriminative Alignment Index https://arxiv.org/abs/2504.09378
    - [ ]  MEXA https://arxiv.org/abs/2410.05873 
    - [ ]  [Gaschi et al., 2023](https://aclanthology.org/2023.findings-acl.189)
    - [x]  [Del & Fishel, 2022](https://aclanthology.org/2022.aacl-main.15/): ANC—seems cool and simple, the description makes it sound more different from the others than the results do.
        → this implies it’s best to use a relatively large number of target languages!
- extracting embeddings:
    - just average over sentence (done)
    - prompting approach?
    - will I actually have any word level scores?!
- check at least some measures on two different datasets?!

### Evaluation

- correlations
    - with downstream
    - with each other
    - with target language data size if available! (or via proxy corpus)
- plot at least some metrics throughout layers for comparison with each other (done for a few, should be pretty reusable)
    - e.g., y-axis scaled depending on their theoretical range?

if they correlate highly with each other but not with downstream it implies it’s not thaaat useful of a concept for generative

if they don’t correlate highly with each other but some do with downstream that implies a recommendation for those metrics

My main project repo: https://github.com/KathyHaem/cla-metrics

### Gender bias

- genderované vety v genderovanom jazyku -> ktorá sa alignuje lepšie s negenderovanou? -> implicitný gender v negenderovanom jazyku
- validácia: logical inference so zámenom
- EuroGEST alebo custom datasety

- Nový EuroParl -- multiparallel, politická príslušnosť
- v ktorých prípadoch je zlý alignment? Téma, koncepy, ktoré sú reprezentované inak medzi jazykmi?

## Furter CLA experiments

- nízky alinment je useless (nič nám nepovie)
- čo má vplyv na koreláciu medzi cla a metrikami? (meta-analýza)
  - množstvo dát v training?
  - skript ("unikátny" pre jazyk? unikátny pre low-resource jazyk?)?
  - vlastnosti tokenizeru: počet tokenov (avg dĺžka) pre daný jazyk oproti EN -- **baseline metrika**
  - Beyond literal overlap článok -- **ďalší baseline**
chrF sa líši podľa cieloveho jazyku, trebalo by spriemerovať koreláciu v rámci cieľových jazykov

## Models to check out:

- meta-llama/Llama-3.1-8B
- mistralai/Ministral-3-14B-Base-2512
- google/gemma-3-12b-pt