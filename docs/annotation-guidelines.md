# Human annotation protocol

The released JSON files contain final reviewed judgments. Two Chinese native speakers independently judged the candidate endings; disagreements and unresolved items were subsequently reviewed. Items still unresolved remained `U`. Annotator identities and private working forms are not distributed.

## What is judged

Show up to three preceding utterances and the target utterance with its sentence-final particle masked. Judge each of the twelve candidates independently. Several endings may be acceptable. The task is neither recovering the original ending nor selecting the single best continuation. The original ending and model outputs are not shown during annotation or adjudication.

| Label | Instruction |
| :--- | :--- |
| A | The candidate is natural and coherent with the supplied context. |
| N | There is a clear grammatical, collocational or discourse incompatibility, or the resulting utterance is clearly unnatural. |
| U | Context, reference, prosody or dialect familiarity is insufficient to decide reliably. |

Keep the target body unchanged. Do not add or replace words, invent an event or participant relationship, or supply an extra clause to rescue an ending. Ordinary spoken ellipsis is permitted. Do not force a preferred acceptance rate or label all candidates of one expression type alike.

## Clarification for the later confirmation batch

The later batch emphasized a natural reading **supported by the supplied text**, without requiring the best ending. A possible reading that requires invented context is insufficient for A. A clear conflict supports N; unresolved structure, reference or prosody supports U.

This clarification is part of the study history. Keep `first_test` and `confirmation` separate in reporting; their labels should not be presented as exchangeable replications under an identical annotation protocol.

## Machine-readable conventions

All final mappings contain each of the twelve candidates in [label order](../data/labels/label_order.json). A/N/U become 1/0/-1 for modeling. U is excluded from the judgment loss and applicable label-wise metrics. An explicit N for an observed corpus ending suppresses that example's conflicting usage target during training. These conventions do not turn unobserved endings into negative human judgments.
