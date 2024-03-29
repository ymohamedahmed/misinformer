# Misinformer: Investigating the Adversarial Case in Misinformation Detection

Abstract: Misinformation has been a permanent feature of the internet since its inception, yet in recent years, its rising ubiquity on social networks has added urgency to the issue of combating it. Swathes of human fact-checkers are tasked with policing content but cannot match the endless flows of information, as a result, research into automated solutions has been pursued for some number of years. As solutions are still being researched, the potential impact of real-world users on such models is relatively unknown and an attempt to elucidate this is the crux of this thesis. This work assumes that real-world users may behave maliciously and, by analogy to the field of Adversarial Machine Learning, seeks to understand this effect. This work contributes a novel method, denoted as ‘Targeted-Importance Scores’ (or ‘T-Scores’) for understanding the globally most important features to a model’s decision making. T-Scores are utilised in a novel, black-box, genetically-inspired adversarial attack, denoted as the Misinformer algorithm, that attempts to perturb an input in order to convince an attacked model that the input is in fact truthful. The Misinformer is shown to reduce the accuracy of the best-performing models developed, in this work, from ∼75% to ∼46% which is the frequency of the targeted class; in other words, the best models are reduced to the performance of simply classifying true for all data points. Finally, the Misinformer is shown to be resistant to the most-popular defense algorithm from the Adversarial Machine Learning literature.

    .
    ├── adversary               # Implementation of the adversaries
    ├── data                    # Loading and processing of the pheme dataset
    ├── experiments             # Code for the experiments used
    ├── models                  # Code for the baseline models, including aggregation, embedding, classification etc.
    ├── utils                   # Tools and utilities
    └── README.md

<p align="center">
  <img src="misinformer-example.svg" />
</p>
