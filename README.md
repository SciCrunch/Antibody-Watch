# Antibody-Watch
Antibody Watch: Text Mining Antibody Specificity from the Literature

Motivation: Antibodies are widely used experimental reagents to test expression of proteins. However,
they might not always provide the intended tests when they do not specifically bind to the target proteins
that their providers designed them for, leading to unreliable research results. While many proposals have
been developed to deal with the problem of antibody specificity, they may not scale well to deal with the
millions of antibodies that have ever been designed and used in research. In this study, we investigate the
feasibility of automatically generate a report to alert scientist users of problematic antibodies by extracting
statements about antibody specificity reported in the literature.

Results: Our goal is to construct an “Antibody Watch” knowledge base containing supporting statements
of problematic antibodies. We developed a deep neural network system called (ABSA)2 and tested its
performance with a corpus of more than two thousand articles that reported uses of antibodies. We
divided the problem into two tasks. Given an input article, the first task is to identify snippets about antibody
specificity and classify if the snippets report any antibody that is nonspecific, and thus problematic. The
second task is to link each of these snippets to one or more antibodies that the snippet referred to. Our
experimental evaluation shows that our system can accurately perform both classification and linking tasks
with weighted F-scores over 0.925 and 0.923, respectively, and 0.914 overall when combined to complete
the joint task.We leveraged the Research Resource Identifiers (RRID) to precisely identify antibodies linked
to the extracted specificity snippets. The result shows that it is feasible to construct a reliable knowledge
base about problematic antibodies by text mining.